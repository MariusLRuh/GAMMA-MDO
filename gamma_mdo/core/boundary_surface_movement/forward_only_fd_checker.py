"""Forward-only centered finite-difference derivative checker.

The stock CSDL ``check_optimization_derivatives`` builds the analytical
derivative (adjoint) operations into the active graph *before* it runs the
finite-difference pass, and the FD pass then re-executes that augmented graph.
For a DAFoam chain this silently reruns the CL/CD adjoint solves during every
nominally forward-only perturbation, which is both wrong and enormously
expensive.

This module provides a checker that structurally cannot do that:

* the finite-difference sweep only ever calls a caller-supplied ``forward_fn``
  (primal only); no derivative graph exists while it runs;
* each perturbation is evaluated independently from the stored baseline design
  point -- never chained ``+h -> -h`` -- and an optional ``reset_fn`` restores
  the baseline mesh / converged flow state before each evaluation;
* the analytical derivatives are computed exactly once, afterwards;
* the forward-only stage is wrapped in a file-descriptor-level output guard that
  fails if the adjoint marker ``"Solving Linear Equation"`` is emitted (DAFoam
  prints it from the C/PETSc layer, below Python ``stdout``).

It prints the mandated stage markers so a batch log can be audited:

    START FORWARD-ONLY FD / END FORWARD-ONLY FD
    START ANALYTICAL ADJOINT / END ANALYTICAL ADJOINT
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import os
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np


DesignPoint = Mapping[str, np.ndarray]
ForwardFunction = Callable[[DesignPoint], Mapping[str, float]]
AnalyticalFunction = Callable[
    [], Mapping[tuple[str, str], np.ndarray]
]

ADJOINT_MARKER = "Solving Linear Equation"


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------
@contextmanager
def guard_against_adjoint_output(
    *,
    markers: Iterable[str] = (ADJOINT_MARKER,),
    raise_on_match: bool = True,
    echo: bool = True,
):
    """Capture fd-1 output, then re-emit it and fail if a marker appears.

    DAFoam/PETSc write ``"Solving Linear Equation"`` at the file-descriptor
    level, so Python ``stdout`` redirection would miss it.  This redirects the
    real fd 1 to a temporary file, restores it on exit, re-emits everything that
    was written (so logs are preserved), and raises if any adjoint marker was
    found while the guard was active.

    Parameters
    ----------
    markers
        Substrings whose presence in the captured output indicates an adjoint
        solve. Matching is plain substring containment on the decoded text.
    raise_on_match
        Raise when a marker is found. ``False`` still captures and re-emits the
        output, reducing the guard to a pass-through.
    echo
        Re-emit the captured output on exit. ``False`` discards it, so a marker
        can still fail the run but the text is lost.

    Yields
    ------
    None
        The guarded block runs with file descriptor 1 redirected.

    Raises
    ------
    RuntimeError
        On exit, when any marker appeared and ``raise_on_match`` is set. The
        descriptor is always restored first, so the failure cannot leave stdout
        redirected.
    """
    marker_list = [str(m) for m in markers]
    sys.stdout.flush()
    saved_fd = os.dup(1)
    temp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(temp.fileno(), 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        temp.seek(0)
        captured = temp.read().decode("utf-8", errors="replace")
        temp.close()
        if echo and captured:
            sys.stdout.write(captured)
            sys.stdout.flush()
        offenders = [m for m in marker_list if m in captured]
        if offenders and raise_on_match:
            raise RuntimeError(
                "Adjoint solve detected during the forward-only FD stage "
                f"(found {offenders!r}). The finite-difference pass must not "
                "execute analytical-derivative operations."
            )


# ---------------------------------------------------------------------------
# Scales and steps
# ---------------------------------------------------------------------------
def characteristic_scales(
    design_variable_specs: Mapping[str, tuple[int, ...]],
    scales: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Return a positive characteristic scale per design variable.

    Defaults to 1.0 for any variable without an explicit scale.  Steps are
    ``h_i = eta * s_i`` so a single ``eta`` sweep is meaningful across variables
    with very different physical magnitudes (translation vs. area vs. degrees).

    Parameters
    ----------
    design_variable_specs
        Mapping of design-variable name to shape. Only the keys are used; the
        shapes are ignored here.
    scales
        Explicit characteristic scale per variable. ``None``, or any absent
        name, yields ``1.0``.

    Returns
    -------
    dict of str to float
        One positive scale per design variable, keyed by name.

    Raises
    ------
    ValueError
        If any resolved scale is zero or negative.
    """
    resolved: dict[str, float] = {}
    for name in design_variable_specs:
        scale = 1.0 if scales is None else float(scales.get(name, 1.0))
        if scale <= 0.0:
            raise ValueError(f"Characteristic scale for {name!r} must be > 0.")
        resolved[name] = scale
    return resolved


# ---------------------------------------------------------------------------
# Centered finite differences (forward-only)
# ---------------------------------------------------------------------------
def _copy_design_point(baseline: DesignPoint) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=float).copy() for name, value in baseline.items()}


def centered_finite_difference(
    forward_fn: ForwardFunction,
    baseline: DesignPoint,
    *,
    output_names: tuple[str, ...],
    design_variable_specs: Mapping[str, tuple[int, ...]],
    eta: float,
    scales: Mapping[str, float],
    reset_fn: Optional[Callable[[], None]] = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Centered FD of every scalar output wrt every design-variable component.

    Every ``+h`` and ``-h`` evaluation starts from a fresh copy of ``baseline``
    and calls ``reset_fn`` (if given) first, so no perturbation warm-starts from
    another.  Returns ``{(output, dv): gradient}`` with ``gradient`` shaped like
    the design variable.

    Each component is perturbed independently, so the cost is two ``forward_fn``
    calls per scalar design-variable component.

    Parameters
    ----------
    forward_fn
        Primal-only callable mapping a design point to a mapping of output name
        to scalar value. It must not build or run derivative operations.
    baseline
        Design point every perturbation starts from. Copied per evaluation and
        never mutated.
    output_names
        Outputs to differentiate. Each must be a key of ``forward_fn``'s result
        and is read as a scalar via its first element.
    design_variable_specs
        Mapping of design-variable name to shape. A falsy shape is treated as
        a single scalar component.
    eta
        Dimensionless step multiplier; the per-variable step is
        ``eta * scales[dv]``.
    scales
        Characteristic scale per design variable, as produced by
        :func:`characteristic_scales`. An entry is required for every variable.
    reset_fn
        Optional callable invoked before each individual evaluation, used to
        restore a baseline mesh or converged flow state so perturbations stay
        independent.

    Returns
    -------
    dict
        Keyed by ``(output_name, design_variable_name)``, each value an array
        shaped like that design variable holding the centered-difference
        derivative of that output.

    Raises
    ------
    ValueError
        If any resolved step is zero or negative.
    """
    gradients: dict[tuple[str, str], np.ndarray] = {
        (output, dv): np.zeros(design_variable_specs[dv], dtype=float)
        for output in output_names
        for dv in design_variable_specs
    }
    for dv, shape in design_variable_specs.items():
        step = eta * scales[dv]
        if step <= 0.0:
            raise ValueError(f"Non-positive FD step for {dv!r}.")
        size = int(np.prod(shape)) if shape else 1
        flat_shape = shape if shape else (1,)
        for index in range(size):
            multi_index = np.unravel_index(index, flat_shape)

            plus = _copy_design_point(baseline)
            plus[dv] = np.asarray(plus[dv], dtype=float).reshape(flat_shape)
            plus[dv][multi_index] += step
            if reset_fn is not None:
                reset_fn()
            f_plus = forward_fn(plus)

            minus = _copy_design_point(baseline)
            minus[dv] = np.asarray(minus[dv], dtype=float).reshape(flat_shape)
            minus[dv][multi_index] -= step
            if reset_fn is not None:
                reset_fn()
            f_minus = forward_fn(minus)

            for output in output_names:
                derivative = (
                    float(np.asarray(f_plus[output]).reshape(-1)[0])
                    - float(np.asarray(f_minus[output]).reshape(-1)[0])
                ) / (2.0 * step)
                gradients[output, dv].reshape(flat_shape)[multi_index] = derivative
    return gradients


def run_forward_only_fd_sweep(
    forward_fn: ForwardFunction,
    baseline: DesignPoint,
    *,
    output_names: tuple[str, ...],
    design_variable_specs: Mapping[str, tuple[int, ...]],
    etas: tuple[float, ...],
    scales: Optional[Mapping[str, float]] = None,
    reset_fn: Optional[Callable[[], None]] = None,
    guard: bool = True,
) -> dict[float, dict[tuple[str, str], np.ndarray]]:
    """Run the guarded, forward-only centered-FD step sweep.

    Prints ``START FORWARD-ONLY FD`` and ``END FORWARD-ONLY FD`` around the
    sweep so a batch log can be audited.

    Parameters
    ----------
    forward_fn
        Primal-only callable, as in :func:`centered_finite_difference`.
    baseline
        Design point every perturbation starts from.
    output_names
        Outputs to differentiate.
    design_variable_specs
        Mapping of design-variable name to shape.
    etas
        Step multipliers to sweep. Each is evaluated as its own complete FD
        pass, so cost scales with the number of entries.
    scales
        Characteristic scale per variable, resolved through
        :func:`characteristic_scales`. ``None`` uses 1.0 for every variable.
    reset_fn
        Optional callable invoked before each individual evaluation.
    guard
        Wrap the whole sweep in :func:`guard_against_adjoint_output`, failing if
        an adjoint marker is emitted. ``False`` runs unguarded, which does not
        make an adjoint solve correct here, only undetected.

    Returns
    -------
    dict
        Keyed by ``eta``, each value the gradient mapping returned by
        :func:`centered_finite_difference` for that step.

    Raises
    ------
    RuntimeError
        When guarding is enabled and an adjoint marker appears during the
        sweep.
    """
    resolved_scales = characteristic_scales(design_variable_specs, scales)
    print("START FORWARD-ONLY FD", flush=True)
    sweep: dict[float, dict[tuple[str, str], np.ndarray]] = {}

    def _run_all() -> None:
        for eta in etas:
            sweep[eta] = centered_finite_difference(
                forward_fn,
                baseline,
                output_names=output_names,
                design_variable_specs=design_variable_specs,
                eta=eta,
                scales=resolved_scales,
                reset_fn=reset_fn,
            )

    if guard:
        with guard_against_adjoint_output():
            _run_all()
    else:
        _run_all()
    print("END FORWARD-ONLY FD", flush=True)
    return sweep


def run_analytical_once(
    analytical_fn: AnalyticalFunction,
) -> dict[tuple[str, str], np.ndarray]:
    """Compute analytical derivatives exactly once, inside adjoint markers.

    Prints ``START ANALYTICAL ADJOINT`` and ``END ANALYTICAL ADJOINT`` around
    the call. The closing marker is printed even if ``analytical_fn`` raises.

    Parameters
    ----------
    analytical_fn
        Zero-argument callable returning a mapping keyed by
        ``(output_name, design_variable_name)``. Called exactly once.

    Returns
    -------
    dict
        The same mapping with every value converted to a float array.
    """
    print("START ANALYTICAL ADJOINT", flush=True)
    try:
        analytical = {
            key: np.asarray(value, dtype=float)
            for key, value in analytical_fn().items()
        }
    finally:
        print("END ANALYTICAL ADJOINT", flush=True)
    return analytical


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------
@dataclass
class DerivativeComparison:
    """Analytical and finite-difference derivatives with their relative errors.

    A plain record produced by :func:`compare_and_report`. The dictionaries are
    stored by reference after conversion to float arrays, not deep-copied.

    Attributes
    ----------
    analytical
        Analytical derivatives keyed by
        ``(output_name, design_variable_name)``. Its keys define which entries
        the comparison covers.
    fd_by_eta
        Finite-difference gradients keyed by step multiplier, each an inner
        mapping with the same key convention.
    relative_error_by_eta
        Relative error per step, keyed by step multiplier and then by
        ``(output_name, design_variable_name)``. Populated by
        :func:`compare_and_report`; empty on a directly constructed instance.
    """

    analytical: dict[tuple[str, str], np.ndarray]
    fd_by_eta: dict[float, dict[tuple[str, str], np.ndarray]]
    relative_error_by_eta: dict[float, dict[tuple[str, str], float]] = field(
        default_factory=dict
    )

    def best(self) -> dict[tuple[str, str], tuple[float | None, float]]:
        """Return the best step and error for each derivative entry.

        "Best" means the smallest recorded relative error across the swept
        steps, which is a diagnostic of where the FD noise floor and truncation
        error balance, not a guarantee of correctness.

        Returns
        -------
        dict
            Keyed by ``(output_name, design_variable_name)``, each value the
            tuple ``(best_eta, best_relative_error)``. The step is ``float |
            None``: an analytical key with no recorded finite error yields
            ``(None, inf)``, which happens when no swept step produced a
            comparable finite value for that entry.
        """
        result: dict[tuple[str, str], tuple[float | None, float]] = {}
        keys = self.analytical.keys()
        for key in keys:
            best_eta = None
            best_error = math.inf
            for eta, errors in self.relative_error_by_eta.items():
                error = errors.get(key, math.inf)
                if error < best_error:
                    best_error = error
                    best_eta = eta
            result[key] = (best_eta, best_error)
        return result


def _relative_error(analytical: np.ndarray, fd: np.ndarray) -> float:
    analytical = np.asarray(analytical, dtype=float).reshape(-1)
    fd = np.asarray(fd, dtype=float).reshape(-1)
    denominator = max(np.linalg.norm(analytical), np.linalg.norm(fd), 1e-30)
    return float(np.linalg.norm(analytical - fd) / denominator)


def compare_and_report(
    analytical: Mapping[tuple[str, str], np.ndarray],
    fd_by_eta: Mapping[float, Mapping[tuple[str, str], np.ndarray]],
    *,
    print_results: bool = True,
) -> DerivativeComparison:
    """Assemble relative errors for the step sweep and optionally print them.

    The relative error of each entry is
    ``||analytical - fd|| / max(||analytical||, ||fd||, 1e-30)``, so it stays
    finite when both sides vanish.

    Parameters
    ----------
    analytical
        Analytical derivatives keyed by ``(output_name, design_variable_name)``.
        Its keys define which entries are compared.
    fd_by_eta
        Finite-difference gradients keyed by step, as returned by
        :func:`run_forward_only_fd_sweep`. A key absent from a step's mapping
        is skipped for that step.
    print_results
        Print the per-step error table and the per-entry best step. The printed
        ``converges`` or ``NO CLEAR REGIME`` label is a reading aid thresholded
        at a relative error of ``1e-1``, not a pass/fail criterion.

    Returns
    -------
    DerivativeComparison
        Holding the analytical derivatives, the per-step gradients, and the
        per-step relative errors.
    """
    comparison = DerivativeComparison(
        analytical={k: np.asarray(v, dtype=float) for k, v in analytical.items()},
        fd_by_eta={
            eta: {k: np.asarray(v, dtype=float) for k, v in grads.items()}
            for eta, grads in fd_by_eta.items()
        },
    )
    for eta, grads in comparison.fd_by_eta.items():
        comparison.relative_error_by_eta[eta] = {
            key: _relative_error(comparison.analytical[key], grads[key])
            for key in comparison.analytical
            if key in grads
        }

    if print_results:
        etas = sorted(comparison.fd_by_eta, reverse=True)
        keys = sorted(comparison.analytical)
        print("\n=== Forward-only centered-FD vs analytical derivatives ===")
        header = "  (output, dv)".ljust(34) + "".join(
            f"{eta:>14.1e}" for eta in etas
        )
        print(header)
        for key in keys:
            label = f"  ({key[0]}, {key[1]})".ljust(34)
            row = "".join(
                f"{comparison.relative_error_by_eta[eta].get(key, float('nan')):>14.3e}"
                for eta in etas
            )
            print(label + row)
        print()
        for key, (best_eta, best_error) in comparison.best().items():
            regime = "converges" if best_error < 1.0e-1 else "NO CLEAR REGIME"
            print(
                f"  ({key[0]}, {key[1]}): min rel error {best_error:.3e} at "
                f"eta {best_eta:.1e}  ({regime})"
            )
        print()
    return comparison


def check_derivatives_forward_first(
    forward_fn: ForwardFunction,
    analytical_fn: AnalyticalFunction,
    baseline: DesignPoint,
    *,
    output_names: tuple[str, ...],
    design_variable_specs: Mapping[str, tuple[int, ...]],
    etas: tuple[float, ...],
    scales: Optional[Mapping[str, float]] = None,
    reset_fn: Optional[Callable[[], None]] = None,
    guard: bool = True,
    print_results: bool = True,
) -> DerivativeComparison:
    """Full forward-first ladder: guarded FD sweep, then one analytical solve.

    The finite-difference pass runs to completion before ``analytical_fn`` is
    ever called, so no analytical-derivative operation exists while FD executes.

    Parameters
    ----------
    forward_fn
        Primal-only callable used for every finite-difference evaluation.
    analytical_fn
        Zero-argument callable producing the analytical derivatives, invoked
        exactly once and only after the sweep finishes.
    baseline
        Design point every perturbation starts from.
    output_names
        Outputs to differentiate.
    design_variable_specs
        Mapping of design-variable name to shape.
    etas
        Step multipliers to sweep.
    scales
        Characteristic scale per variable; ``None`` uses 1.0 for each.
    reset_fn
        Optional callable invoked before each individual evaluation.
    guard
        Fail the sweep if an adjoint marker is emitted during the FD stage.
    print_results
        Print the comparison table and per-entry best step.

    Returns
    -------
    DerivativeComparison
        The assembled comparison of analytical and finite-difference
        derivatives.
    """
    fd_by_eta = run_forward_only_fd_sweep(
        forward_fn,
        baseline,
        output_names=output_names,
        design_variable_specs=design_variable_specs,
        etas=etas,
        scales=scales,
        reset_fn=reset_fn,
        guard=guard,
    )
    analytical = run_analytical_once(analytical_fn)
    return compare_and_report(
        analytical, fd_by_eta, print_results=print_results
    )


# ---------------------------------------------------------------------------
# Degree/radian consistency
# ---------------------------------------------------------------------------
def degree_radian_relative_error(
    gradient_wrt_degrees: np.ndarray,
    gradient_wrt_radians: np.ndarray,
) -> float:
    """Relative error of ``dF/dtheta_deg`` vs ``(pi/180) dF/dtheta_rad``.

    Documents and independently verifies the rotation-unit convention required
    by the derivative study, without needing a second full model evaluation.

    Parameters
    ----------
    gradient_wrt_degrees
        Derivative with respect to an angle measured in degrees.
    gradient_wrt_radians
        Derivative with respect to the same angle measured in radians, scaled
        here by ``pi / 180`` to form the expected degree-based value.

    Returns
    -------
    float
        Relative error between the expected and supplied degree-based
        gradients, using the same norm ratio as the FD comparison. A small
        value indicates the two conventions agree; it is a diagnostic, not a
        correctness proof.
    """
    expected = (math.pi / 180.0) * np.asarray(
        gradient_wrt_radians, dtype=float
    )
    return _relative_error(expected, np.asarray(gradient_wrt_degrees, dtype=float))


__all__ = [
    "ADJOINT_MARKER",
    "DerivativeComparison",
    "guard_against_adjoint_output",
    "characteristic_scales",
    "centered_finite_difference",
    "run_forward_only_fd_sweep",
    "run_analytical_once",
    "compare_and_report",
    "check_derivatives_forward_first",
    "degree_radian_relative_error",
]
