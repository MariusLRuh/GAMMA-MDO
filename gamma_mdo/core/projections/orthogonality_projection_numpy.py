"""NumPy Newton projection of points onto B-spline patches.

Two solves live here: an interior solve on the orthogonality residual, and an
edge solve with one parametric coordinate pinned to a patch boundary. Both are
eager NumPy kernels that execute when called, either directly or from a CSDL
custom operation's ``compute``.

Neither solve raises on failure. Every point comes back inside a
:class:`SurfaceProjectionResult` carrying its residual, step norm, iteration
count, and convergence flag, so the caller can decide whether to retry, route
the point to another patch, or accept it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


try:
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy_factory import (
        make_bspline_evaluator_numpy,
    )
except Exception:  # pragma: no cover
    make_bspline_evaluator_numpy = None


@dataclass(frozen=True)
class OrthogonalityNewtonParams:
    """Tolerances and iteration limits for the orthogonality Newton solve.

    Attributes
    ----------
    max_iter
        Maximum Newton iterations per point.
    tol_res
        Residual norm below which a point is accepted as converged.
    tol_step
        Parametric step norm below which iteration stops.
    use_active_set
        Clamp parameters to the unit square and mask outward residual components
        at a clamped bound, instead of letting the step leave the patch.
    bound_eps
        Distance from 0 or 1 at which a parameter counts as on the boundary.
    snap_eps
        Distance within which a converged parameter is snapped exactly to 0 or 1.
    zero_residual_tol
        Residual treated as exactly zero, used for coincident query points.
    det_eps, diag_eps
        Floors guarding the 2x2 solve and its diagonal against singularity.
    """

    max_iter: int = 40
    tol_res: float = 1e-9
    tol_step: float = 1e-9
    use_active_set: bool = True
    bound_eps: float = 1e-8
    snap_eps: float = 1e-8
    zero_residual_tol: float = 0.0
    det_eps: float = 1e-30
    diag_eps: float = 1e-14


@dataclass
class SurfaceProjectionResult:
    """Per-point outcome of a surface Newton projection.

    Attributes
    ----------
    uv
        Final parametric coordinates, shape ``(num_points, 2)``: the last
        iterate, which for a point whose ``converged`` entry is ``False`` is not
        a solution. ``converged`` records the solver's decision for each point,
        and ``residual``, ``step_norm``, and ``iterations`` are the associated
        diagnostics.
    projected_points
        Surface points at ``uv``.
    residual
        Final residual norm per point.
    step_norm
        Final parametric step norm per point.
    dist2
        Squared distance from each query point to its projection.
    converged
        Boolean per point.
    iterations
        Newton iterations actually taken per point.
    boundary_clamped
        ``True`` where the final parameter lies on a bound while the
        **unmasked** residual still points outward. It is computed before the
        active set is applied, so it is independent of ``converged`` and does
        not by itself imply that the point converged. Such a point sits on an
        edge and may belong on a neighbouring patch, so the warm-start driver
        routes it into the retry path. ``None`` for candidates that lie on a
        boundary by construction.
    """

    uv: np.ndarray
    projected_points: np.ndarray
    residual: np.ndarray
    step_norm: np.ndarray
    dist2: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    # True where the final parameter sits on a bound (u or v at 0/1) while the
    # unmasked residual still points outward past it. Computed before the active
    # set is applied, so it is independent of `converged` and does not by itself
    # imply the point converged. Such points sit on a patch edge but may want to
    # slide across it onto a neighbouring patch; the warm-start driver routes them
    # into the retry path.
    # None for edge/point candidates that are on a boundary by construction.
    boundary_clamped: Optional[np.ndarray] = None


def _require_numpy_bspline_factory() -> None:
    if make_bspline_evaluator_numpy is None:
        raise ImportError(
            "make_bspline_evaluator_numpy not found; ensure "
            "compute_basis_matrix_numpy_factory.py is on path."
        )


def _snap_to_unit_box_bounds_numpy(u: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0.0:
        return u
    u = np.where(u <= eps, 0.0, u)
    u = np.where(u >= (1.0 - eps), 1.0, u)
    return u


def _active_mask_numpy(u: np.ndarray, residual: np.ndarray, params: OrthogonalityNewtonParams) -> np.ndarray:
    lower_active = u <= params.bound_eps
    upper_active = u >= (1.0 - params.bound_eps)
    block_lower = lower_active & (residual > 0.0)
    block_upper = upper_active & (residual < 0.0)
    zero_res = np.abs(residual) <= params.zero_residual_tol
    inactive = block_lower | block_upper | zero_res
    return ~inactive


def _solve_2x2_numpy(A: np.ndarray, b: np.ndarray, det_eps: float) -> np.ndarray:
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]

    x = np.zeros_like(b)
    good = np.abs(det) > det_eps
    if np.any(good):
        inv00 = A[good, 1, 1] / det[good]
        inv01 = -A[good, 0, 1] / det[good]
        inv10 = -A[good, 1, 0] / det[good]
        inv11 = A[good, 0, 0] / det[good]
        x[good, 0] = inv00 * b[good, 0] + inv01 * b[good, 1]
        x[good, 1] = inv10 * b[good, 0] + inv11 * b[good, 1]

    bad = np.where(~good)[0]
    for idx in bad:
        try:
            x[idx] = np.linalg.solve(A[idx], b[idx])
        except np.linalg.LinAlgError:
            x[idx] = 0.0

    return x


def _solve_reduced_newton_system(
    jacobian: np.ndarray,
    residual: np.ndarray,
    active: np.ndarray,
    diag_eps: float,
    det_eps: float,
) -> np.ndarray:
    delta = np.zeros_like(residual)

    both_active = active[:, 0] & active[:, 1]
    first_only = active[:, 0] & ~active[:, 1]
    second_only = ~active[:, 0] & active[:, 1]

    if np.any(both_active):
        delta[both_active] = _solve_2x2_numpy(
            jacobian[both_active],
            -residual[both_active],
            det_eps=det_eps,
        )

    if np.any(first_only):
        denom = jacobian[first_only, 0, 0]
        good = np.abs(denom) > diag_eps
        if np.any(good):
            idx = np.where(first_only)[0][good]
            delta[idx, 0] = -residual[idx, 0] / denom[good]

    if np.any(second_only):
        denom = jacobian[second_only, 1, 1]
        good = np.abs(denom) > diag_eps
        if np.any(good):
            idx = np.where(second_only)[0][good]
            delta[idx, 1] = -residual[idx, 1] / denom[good]

    return delta


def make_surface_orthogonality_evaluator_numpy(
    degrees: Tuple[int, ...],
    knot_vectors: Tuple[np.ndarray, ...],
):
    """Build an eager NumPy evaluator for the orthogonality residual and Jacobian.

    The residual is the surface tangent basis dotted with the offset from the
    query point, so it vanishes exactly when that offset is orthogonal to the
    surface.

    Parameters
    ----------
    degrees
        Per-direction B-spline degrees, ordered ``(u, v)``. Exactly two
        directions are supported; anything else raises ``ValueError``.
    knot_vectors
        Per-direction knot vectors, ordered to match ``degrees``.

    Returns
    -------
    callable
        Function of parametric coordinates and coefficients returning the residual
        and the derivatives the Newton step needs.
    """
    _require_numpy_bspline_factory()

    if len(degrees) != 2:
        raise ValueError(
            "This orthogonality solver currently supports 2D parametric "
            f"patches only; got {len(degrees)}D."
        )

    eval_S = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=None,
    )
    eval_du = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(1, 0),
    )
    eval_dv = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(0, 1),
    )
    eval_duu = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(2, 0),
    )
    eval_duv = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(1, 1),
    )
    eval_dvv = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(0, 2),
    )

    def evaluate(us: np.ndarray, coeffs: np.ndarray):
        S = eval_S(us, coeffs)
        Su = eval_du(us, coeffs)
        Sv = eval_dv(us, coeffs)
        Suu = eval_duu(us, coeffs)
        Suv = eval_duv(us, coeffs)
        Svv = eval_dvv(us, coeffs)
        return S, Su, Sv, Suu, Suv, Svv

    return evaluate


def _compute_residual_and_jacobian(
    points: np.ndarray,
    S: np.ndarray,
    Su: np.ndarray,
    Sv: np.ndarray,
    Suu: np.ndarray,
    Suv: np.ndarray,
    Svv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = S - points

    residual = np.empty((points.shape[0], 2), dtype=S.dtype)
    residual[:, 0] = np.einsum("ij,ij->i", diff, Su)
    residual[:, 1] = np.einsum("ij,ij->i", diff, Sv)

    jacobian = np.empty((points.shape[0], 2, 2), dtype=S.dtype)
    jacobian[:, 0, 0] = np.einsum("ij,ij->i", Su, Su) + np.einsum("ij,ij->i", diff, Suu)
    jacobian[:, 0, 1] = np.einsum("ij,ij->i", Su, Sv) + np.einsum("ij,ij->i", diff, Suv)
    jacobian[:, 1, 0] = jacobian[:, 0, 1]
    jacobian[:, 1, 1] = np.einsum("ij,ij->i", Sv, Sv) + np.einsum("ij,ij->i", diff, Svv)

    dist2 = np.einsum("ij,ij->i", diff, diff)
    return residual, jacobian, dist2


def project_points_orthogonality_newton_numpy(
    points: np.ndarray,
    u0s: np.ndarray,
    coeffs: np.ndarray,
    degrees: Tuple[int, ...],
    knot_vectors: Tuple[np.ndarray, ...],
    *,
    params: OrthogonalityNewtonParams = OrthogonalityNewtonParams(),
) -> SurfaceProjectionResult:
    """Project points onto a patch interior by Newton on the orthogonality residual.

    Runs eagerly when called.

    Parameters
    ----------
    points
        Query points, shape ``(M, physical_dimension)``.
    u0s
        Initial parametric guesses, shape ``(M, 2)``, one per query point. They
        are clipped into the unit square and snapped to its bounds before the
        first iteration.
    coeffs
        Control points for the single patch being solved on, with the trailing
        axis holding the physical dimension.
    degrees
        Per-direction B-spline degrees, ordered ``(u, v)``.
    knot_vectors
        Per-direction knot vectors, ordered to match ``degrees``.
    params
        Solve tolerances; see :class:`OrthogonalityNewtonParams`.

    Returns
    -------
    SurfaceProjectionResult
        Final coordinates and per-point convergence evidence, including
        ``boundary_clamped``. Points that did not converge are returned rather
        than raising, so the caller decides what to do with them.
    """
    points = np.asarray(points, dtype=float)
    u0s = np.asarray(u0s, dtype=float)

    if points.ndim != 2:
        raise ValueError(f"points must have shape (M, phys_dim); got {points.shape}")
    if u0s.ndim != 2 or u0s.shape[1] != 2:
        raise ValueError(f"u0s must have shape (M, 2); got {u0s.shape}")
    if points.shape[0] != u0s.shape[0]:
        raise ValueError("points and u0s must have the same batch size.")

    evaluate = make_surface_orthogonality_evaluator_numpy(degrees, knot_vectors)

    u = _snap_to_unit_box_bounds_numpy(np.clip(u0s.copy(), 0.0, 1.0), params.snap_eps)
    converged = np.zeros((points.shape[0],), dtype=bool)
    iterations = np.zeros((points.shape[0],), dtype=int)
    last_step_norm = np.zeros((points.shape[0],), dtype=float)

    for iter_num in range(params.max_iter):
        S, Su, Sv, Suu, Suv, Svv = evaluate(u, coeffs)
        residual, jacobian, _ = _compute_residual_and_jacobian(points, S, Su, Sv, Suu, Suv, Svv)

        if params.use_active_set:
            active = _active_mask_numpy(u, residual, params)
        else:
            active = np.ones_like(residual, dtype=bool)

        residual_masked = residual * active
        delta = _solve_reduced_newton_system(
            jacobian=jacobian,
            residual=residual_masked,
            active=active,
            diag_eps=params.diag_eps,
            det_eps=params.det_eps,
        )

        u_new = _snap_to_unit_box_bounds_numpy(np.clip(u + delta, 0.0, 1.0), params.snap_eps)
        residual_norm = np.linalg.norm(residual_masked, axis=1)
        step_norm = np.linalg.norm(delta, axis=1)
        conv_new = (residual_norm < params.tol_res) | (step_norm < params.tol_step)

        u = np.where(converged[:, None], u, u_new)
        iterations = np.where(converged, iterations, iter_num + 1)
        last_step_norm = np.where(converged, last_step_norm, step_norm)
        converged = converged | conv_new

        if converged.all():
            break

    S, Su, Sv, Suu, Suv, Svv = evaluate(u, coeffs)
    residual_vec, _, dist2 = _compute_residual_and_jacobian(points, S, Su, Sv, Suu, Suv, Svv)

    # Detect boundary clamping: a point pinned at a parametric bound (u or v at
    # 0/1) whose *unmasked* residual still points outward past that bound. The
    # active set (below) zeros that component, so the point reports converged
    # with a near-zero masked residual even though its true closest point may
    # lie across the edge on a neighbouring patch. Surfacing this is what lets
    # the warm-start driver retry such points instead of silently pinning them.
    # This mirrors the block_lower/block_upper logic in _active_mask_numpy.
    lower_on_bound = u <= params.bound_eps
    upper_on_bound = u >= (1.0 - params.bound_eps)
    outward_lower = lower_on_bound & (residual_vec > 0.0)
    outward_upper = upper_on_bound & (residual_vec < 0.0)
    boundary_clamped = np.any(outward_lower | outward_upper, axis=1)

    if params.use_active_set:
        active = _active_mask_numpy(u, residual_vec, params)
        residual = np.linalg.norm(residual_vec * active, axis=1)
    else:
        residual = np.linalg.norm(residual_vec, axis=1)

    return SurfaceProjectionResult(
        uv=u,
        projected_points=S,
        residual=residual,
        step_norm=last_step_norm,
        dist2=dist2,
        converged=converged,
        iterations=iterations,
        boundary_clamped=boundary_clamped,
    )


def project_points_on_surface_edge_newton_numpy(
    points: np.ndarray,
    t0s: np.ndarray,
    coeffs: np.ndarray,
    degrees: Tuple[int, ...],
    knot_vectors: Tuple[np.ndarray, ...],
    *,
    fixed_axis: int,
    fixed_value: float,
    params: OrthogonalityNewtonParams = OrthogonalityNewtonParams(),
) -> SurfaceProjectionResult:
    """Project points onto one parametric boundary edge of a patch.

    One parametric coordinate is held fixed at 0 or 1 and Newton runs on the
    remaining coordinate. This supplies the edge warm-start candidates used when a
    point's closest location may lie on or across a patch boundary.

    Parameters
    ----------
    points
        Query points, shape ``(M, physical_dimension)``.
    t0s
        Initial guesses for the free coordinate, flattened to shape ``(M,)``,
        one per query point.
    coeffs
        Control points for the single patch being solved on, with the trailing
        axis holding the physical dimension.
    degrees
        Per-direction B-spline degrees, ordered ``(u, v)``.
    knot_vectors
        Per-direction knot vectors, ordered to match ``degrees``.
    fixed_axis
        Which parametric axis is held fixed: ``0`` for u, ``1`` for v. Any other
        value raises ``ValueError``.
    fixed_value
        The value that axis is held at, ``0.0`` or ``1.0``, selecting which of
        the patch's four edges is solved on.
    params
        Solve tolerances; see :class:`OrthogonalityNewtonParams`.

    Returns
    -------
    SurfaceProjectionResult
        Final coordinates and per-point convergence evidence, with the fixed
        axis written back into ``uv`` alongside the solved free coordinate.
        ``boundary_clamped`` is ``None`` here, because these candidates are on a
        boundary by construction.
    """
    points = np.asarray(points, dtype=float)
    t0s = np.asarray(t0s, dtype=float).reshape(-1)

    if points.ndim != 2:
        raise ValueError(f"points must have shape (M, phys_dim); got {points.shape}")
    if points.shape[0] != t0s.shape[0]:
        raise ValueError("points and t0s must have the same batch size.")
    if fixed_axis not in (0, 1):
        raise ValueError(f"fixed_axis must be 0 or 1; got {fixed_axis}")

    _require_numpy_bspline_factory()

    eval_S = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=None,
    )

    if fixed_axis == 0:
        eval_dfree = make_bspline_evaluator_numpy(
            degrees=degrees,
            knot_vectors=knot_vectors,
            der_orders=(0, 1),
        )
        eval_d2free = make_bspline_evaluator_numpy(
            degrees=degrees,
            knot_vectors=knot_vectors,
            der_orders=(0, 2),
        )
        free_axis = 1
    else:
        eval_dfree = make_bspline_evaluator_numpy(
            degrees=degrees,
            knot_vectors=knot_vectors,
            der_orders=(1, 0),
        )
        eval_d2free = make_bspline_evaluator_numpy(
            degrees=degrees,
            knot_vectors=knot_vectors,
            der_orders=(2, 0),
        )
        free_axis = 0

    t = _snap_to_unit_box_bounds_numpy(np.clip(t0s.copy(), 0.0, 1.0), params.snap_eps)
    converged = np.zeros((points.shape[0],), dtype=bool)
    iterations = np.zeros((points.shape[0],), dtype=int)
    last_step = np.zeros((points.shape[0],), dtype=float)

    uv = np.empty((points.shape[0], 2), dtype=float)

    for iter_num in range(params.max_iter):
        uv[:, fixed_axis] = fixed_value
        uv[:, free_axis] = t

        S = eval_S(uv, coeffs)
        Sfree = eval_dfree(uv, coeffs)
        S2free = eval_d2free(uv, coeffs)

        diff = S - points
        residual = np.einsum("ij,ij->i", diff, Sfree)
        jacobian = np.einsum("ij,ij->i", Sfree, Sfree) + np.einsum("ij,ij->i", diff, S2free)

        if params.use_active_set:
            active = ~(
                ((t <= params.bound_eps) & (residual > 0.0))
                | ((t >= (1.0 - params.bound_eps)) & (residual < 0.0))
                | (np.abs(residual) <= params.zero_residual_tol)
            )
        else:
            active = np.ones_like(residual, dtype=bool)

        step = np.zeros_like(t)
        good = active & (np.abs(jacobian) > params.diag_eps)
        step[good] = -residual[good] / jacobian[good]

        t_new = _snap_to_unit_box_bounds_numpy(np.clip(t + step, 0.0, 1.0), params.snap_eps)
        residual_masked = np.abs(residual * active)
        step_norm = np.abs(step)
        conv_new = (residual_masked < params.tol_res) | (step_norm < params.tol_step)

        t = np.where(converged, t, t_new)
        iterations = np.where(converged, iterations, iter_num + 1)
        last_step = np.where(converged, last_step, step_norm)
        converged = converged | conv_new

        if converged.all():
            break

    uv[:, fixed_axis] = fixed_value
    uv[:, free_axis] = t
    S = eval_S(uv, coeffs)
    Sfree = eval_dfree(uv, coeffs)
    diff = S - points
    residual = np.abs(np.einsum("ij,ij->i", diff, Sfree))
    dist2 = np.einsum("ij,ij->i", diff, diff)

    return SurfaceProjectionResult(
        uv=uv,
        projected_points=S,
        residual=residual,
        step_norm=last_step,
        dist2=dist2,
        converged=converged,
        iterations=iterations,
    )
