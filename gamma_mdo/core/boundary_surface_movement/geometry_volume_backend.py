"""Rank-0 geometry-to-volume backend for the DAFoam MPI coupling.

From CSDL's perspective the entire geometry parameterization, surface
deformation, and volume deformation is a single explicit map

    d  ->  X(d),

where ``d`` are the geometric design variables and ``X`` are the deformed global
volume-mesh coordinates in the stable Gmsh ordering.  This module implements the
*live* side of that map as a backend that runs only on MPI rank 0, keeping every
CAD object, projection cache, seam set, connectivity table, and elasticity
factorization off the non-root ranks.

The public backend interface is deliberately CSDL-free (NumPy in, NumPy out):

    forward(design_variables)            -> X            (num_points, 3)
    compute_vjp(design_variables, Xbar)  -> {name: dbar}

Internally the reference implementation reuses the existing, already
differentiable CSDL mesh-motion pipeline.  Rather than forming the dense
Jacobian dX/dd, the reverse pass injects the volume-coordinate seed ``Xbar`` as
a constant CSDL input, contracts it against the volume output to a scalar
``s = sum(X * Xbar)``, and differentiates that scalar with respect to the design
variables.  The result ``ds/dd = Xbar^T dX/dd`` is exactly the matrix-free VJP,
propagated by CSDL through the elasticity transpose solve, the surface-motion
reverse pass, and the geometry parameterization.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    runtime_checkable,
)

import numpy as np

import csdl_alpha as csdl

if TYPE_CHECKING:
    from .geometry_model import GeometryModel


@runtime_checkable
class GeometryVolumeBackend(Protocol):
    """Contract used by :class:`GeometryVolumeOperation` on the root rank."""

    design_variable_names: tuple[str, ...]
    output_shape: tuple[int, int]

    def forward(
        self, design_variables: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Return global volume coordinates with shape ``(num_points, 3)``.

        Parameters
        ----------
        design_variables
            Mapping of design-variable name to value array. Implementations
            require every name in ``design_variable_names``.

        Returns
        -------
        numpy.ndarray
            Deformed global volume-mesh coordinates of shape
            ``(num_points, 3)`` in the stable Gmsh node ordering.
        """

    def compute_vjp(
        self,
        design_variables: Mapping[str, np.ndarray],
        volume_seed: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        """Return a VJP for every geometric design variable.

        Parameters
        ----------
        design_variables
            Design point the reverse product is taken about, using the same
            names as :meth:`forward`.
        volume_seed
            Cotangent on the volume coordinates, shaped ``(num_points, 3)``.

        Returns
        -------
        Mapping[str, numpy.ndarray]
            One entry per design variable, keyed by name, each shaped like that
            design variable.
        """


# ---------------------------------------------------------------------------
# Reusable CSDL-recorder engine
# ---------------------------------------------------------------------------
ModelBuilder = Callable[
    [csdl.Recorder], "tuple[dict[str, csdl.Variable], csdl.Variable]"
]


class CSDLRecorderBackend:
    """Wrap a CSDL model ``d -> X`` as a matrix-free forward/VJP backend.

    Parameters
    ----------
    model_builder
        Callable that, given a freshly started recorder, constructs the design
        variables and the volume-coordinate output and returns
        ``(design_variables, volume_output)``.  ``design_variables`` maps each
        design-variable name to its CSDL ``Variable``; ``volume_output`` is the
        ``(num_points, 3)`` deformed-coordinate ``Variable``.
    build_eagerly
        Build the model (and any expensive setup it triggers) inside ``__init__``
        when ``True``; otherwise defer until the first ``forward``/``compute_vjp``.

    Notes
    -----
    The backend owns a private recorder and :class:`PySimulator`.  The volume
    seed used for the reverse pass is an ordinary CSDL input, so the derivative
    graph is built once (cached by :meth:`PySimulator.compute_totals`) and reused
    across seeds and design points.  ``compute_vjp`` reruns the forward model
    whenever the requested design point differs from the cached one, so the
    reverse pass is always taken about the matching primal state.
    """

    def __init__(
        self,
        model_builder: ModelBuilder,
        *,
        build_eagerly: bool = False,
        seed_name: str = "_geometry_volume_seed",
    ):
        self._model_builder = model_builder
        self._seed_name = str(seed_name)
        self._recorder: csdl.Recorder | None = None
        self._simulator: Any = None
        self._design_variables: dict[str, csdl.Variable] = {}
        self._volume_output: csdl.Variable | None = None
        self._seed: csdl.Variable | None = None
        self._contraction: csdl.Variable | None = None
        self._design_variable_names: tuple[str, ...] = ()
        self._output_shape: tuple[int, int] | None = None
        self._cached_design_point: dict[str, np.ndarray] | None = None
        self._cached_coordinates: np.ndarray | None = None
        if build_eagerly:
            self._ensure_built()

    # -- construction -----------------------------------------------------------
    def _ensure_built(self) -> None:
        if self._recorder is not None:
            return
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        design_variables, volume_output = self._model_builder(recorder)
        if not design_variables:
            raise ValueError("model_builder returned no design variables.")
        output_value = np.asarray(volume_output.value, dtype=float)
        if output_value.ndim != 2 or output_value.shape[1] != 3:
            raise ValueError(
                "volume_output must have shape (num_points, 3); got "
                f"{output_value.shape}."
            )
        seed = csdl.Variable(
            name=self._seed_name,
            value=np.zeros(output_value.shape, dtype=float),
        )
        contraction = csdl.sum(volume_output * seed)
        recorder.stop()

        self._recorder = recorder
        self._design_variables = dict(design_variables)
        self._design_variable_names = tuple(self._design_variables.keys())
        self._volume_output = volume_output
        self._seed = seed
        self._contraction = contraction
        self._output_shape = (int(output_value.shape[0]), 3)
        self._simulator = csdl.experimental.PySimulator(recorder=recorder)
        # The build already ran one inline forward; record it as the cache.
        self._cached_design_point = {
            name: np.asarray(variable.value, dtype=float).copy()
            for name, variable in self._design_variables.items()
        }
        self._cached_coordinates = output_value.copy()

    # -- metadata ---------------------------------------------------------------
    @property
    def design_variable_names(self) -> tuple[str, ...]:
        """Names of the design variables, in model-construction order.

        Building the model on first access is a side effect, so reading this
        property can trigger the expensive one-time setup.

        Returns
        -------
        tuple of str
            Design-variable names in the order the model builder created them.
        """
        self._ensure_built()
        return self._design_variable_names

    @property
    def output_shape(self) -> tuple[int, int]:
        """Global shape of the volume-coordinate output.

        Building the model on first access is a side effect.

        Returns
        -------
        tuple of int
            ``(num_points, 3)``, taken from the built model's output.
        """
        self._ensure_built()
        assert self._output_shape is not None
        return self._output_shape

    def design_variable_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return each design variable's shape.

        Building the model on first access is a side effect.

        Returns
        -------
        dict of str to tuple of int
            Shape of every design variable, keyed by name. A new dictionary on
            each call.
        """
        self._ensure_built()
        return {
            name: tuple(variable.shape)
            for name, variable in self._design_variables.items()
        }

    # -- forward ----------------------------------------------------------------
    def _apply_design_variables(
        self, design_variables: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        self._ensure_built()
        missing = set(self._design_variable_names).difference(design_variables)
        if missing:
            raise KeyError(
                "Missing design variables: " + ", ".join(sorted(missing))
            )
        applied: dict[str, np.ndarray] = {}
        for name, variable in self._design_variables.items():
            value = np.asarray(design_variables[name], dtype=float)
            variable.value = value.reshape(variable.shape)
            applied[name] = variable.value.copy()
        return applied

    def _design_point_matches_cache(
        self, design_variables: Mapping[str, np.ndarray]
    ) -> bool:
        if self._cached_design_point is None:
            return False
        for name in self._design_variable_names:
            cached = self._cached_design_point[name]
            candidate = np.asarray(design_variables[name], dtype=float).reshape(
                cached.shape
            )
            if not np.array_equal(cached, candidate):
                return False
        return True

    def forward(
        self, design_variables: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Run the model at ``design_variables`` and return volume coordinates.

        Applies the values to the recorder's inputs, runs the simulator, and
        caches both the design point and the resulting coordinates so a
        following :meth:`compute_vjp` at the same point can skip the re-solve.

        Parameters
        ----------
        design_variables
            Mapping of design-variable name to value. Each value is cast to
            float and reshaped to the variable's declared shape, so a
            compatible flat array is accepted. Every declared name is required;
            extra entries are ignored.

        Returns
        -------
        numpy.ndarray
            Deformed coordinates of shape ``(num_points, 3)``. A copy, so the
            caller cannot mutate the cache.

        Raises
        ------
        KeyError
            If any declared design variable is absent.
        FloatingPointError
            If the model produced any nonfinite coordinate.
        """
        self._ensure_built()
        assert self._volume_output is not None and self._simulator is not None
        applied = self._apply_design_variables(design_variables)
        self._simulator.run()
        coordinates = np.asarray(
            self._volume_output.value, dtype=float
        ).reshape(self.output_shape).copy()
        if not np.all(np.isfinite(coordinates)):
            raise FloatingPointError(
                "Geometry-to-volume forward produced nonfinite coordinates."
            )
        self._cached_design_point = applied
        self._cached_coordinates = coordinates
        return coordinates

    # -- reverse ----------------------------------------------------------------
    def compute_vjp(
        self,
        design_variables: Mapping[str, np.ndarray],
        volume_seed: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Evaluate the matrix-free reverse product at a design point.

        The seed is written into an ordinary CSDL input and the scalar
        contraction ``sum(X * Xbar)`` is differentiated, so the result is
        ``Xbar^T dX/dd`` without forming the Jacobian. When the requested
        design point differs from the cached one the forward model is rerun
        first, so the reverse pass is always taken about the matching primal
        state.

        Parameters
        ----------
        design_variables
            Design point to differentiate about, using the same names and
            reshaping rules as :meth:`forward`.
        volume_seed
            Cotangent on the volume coordinates, reshaped to
            ``(num_points, 3)``.

        Returns
        -------
        dict of str to numpy.ndarray
            One cotangent per design variable, keyed by name and reshaped to
            that variable's shape.

        Raises
        ------
        KeyError
            If any declared design variable is absent.
        FloatingPointError
            If the seed contains nonfinite values, or a triggered forward
            re-solve produced nonfinite coordinates.
        """
        self._ensure_built()
        assert (
            self._simulator is not None
            and self._seed is not None
            and self._contraction is not None
        )
        if not self._design_point_matches_cache(design_variables):
            self.forward(design_variables)
        else:
            # Ensure the recorder inputs hold the cached design point exactly.
            self._apply_design_variables(design_variables)

        seed = np.asarray(volume_seed, dtype=float).reshape(self.output_shape)
        if not np.all(np.isfinite(seed)):
            raise FloatingPointError("Volume seed contains nonfinite values.")
        self._seed.value = seed

        derivatives = self._simulator.compute_totals(
            [self._contraction],
            list(self._design_variables.values()),
        )
        result: dict[str, np.ndarray] = {}
        for name, variable in self._design_variables.items():
            value = np.asarray(
                derivatives[self._contraction, variable], dtype=float
            )
            result[name] = value.reshape(variable.shape)
        return result


# ---------------------------------------------------------------------------
# Mesh-motion pipeline backend
# ---------------------------------------------------------------------------
class MeshMotionVolumeBackend(CSDLRecorderBackend):
    """Live geometry -> surface -> volume backend for rank 0.

    Builds a differentiable mesh-motion model (without any
    downstream aerodynamic analysis) inside a private recorder and exposes its
    final elasticity volume coordinates as ``X(d)``.  The heavy one-time setup
    (CAD import, baseline projection ownership, seam identification, elasticity
    assembly/factorization) happens on the first ``forward`` call and is then
    reused for every subsequent forward and reverse evaluation.
    """

    def __init__(
        self,
        input_files: Any,
        geometry_values: Mapping[str, float],
        pipeline_config: Any,
        *,
        parameterization_factory: Callable[
            [Mapping[str, csdl.Variable]], GeometryModel
        ],
        aerodynamic_volume_method: str = "elasticity",
        build_eagerly: bool = False,
    ):
        self._input_files = input_files
        self._geometry_values = dict(geometry_values)
        self._pipeline_config = pipeline_config
        self._parameterization_factory = parameterization_factory
        self._aerodynamic_volume_method = str(aerodynamic_volume_method)
        self._last_mesh_motion_result: Any = None
        super().__init__(self._build_model, build_eagerly=build_eagerly)

    @property
    def last_mesh_motion_result(self) -> Any:
        """The most recent :class:`MeshMotionResult` (root rank only).

        Populated when the private recorder is first built (which runs one
        inline forward, including the surface/volume mesh-quality gates) so the
        coupled driver can still emit its quality diagnostics.
        """
        return self._last_mesh_motion_result

    def _build_model(
        self, recorder: csdl.Recorder
    ) -> "tuple[dict[str, csdl.Variable], csdl.Variable]":
        # Defer the pipeline import until the private model is constructed.
        from gamma_mdo.core.boundary_surface_movement.mesh_motion_pipeline import (
            run_mesh_motion,
        )

        design_variables = {
            name: csdl.Variable(name=name, value=float(value))
            for name, value in self._geometry_values.items()
        }
        geometry_parameterization = self._parameterization_factory(
            design_variables
        )
        result = run_mesh_motion(
            recorder=recorder,
            input_files=self._input_files,
            geometry=geometry_parameterization,
            config=self._pipeline_config,
            aerodynamic_analysis=None,
            aerodynamic_volume_method=self._aerodynamic_volume_method,
        )
        method = self._aerodynamic_volume_method
        if method not in result.volume_coordinates:
            raise ValueError(
                f"Volume method {method!r} is not available; enable it in the "
                "VolumeMotion used to build this backend."
            )
        self._last_mesh_motion_result = result
        volume_output = result.volume_coordinates[method]
        return design_variables, volume_output


def read_gmsh_volume_point_count(volume_mesh_file: Any) -> int:
    """Cheaply read the number of nodes from a Gmsh 2.2 ``$Nodes`` block.

    Non-root ranks need the global point count to declare the CSDL operation's
    output shape without building the geometry backend.  Only the ``$Nodes``
    header line is read, so this stays a lightweight, login-node-safe operation
    even for a large mesh file.

    Parameters
    ----------
    volume_mesh_file
        Path to the mesh file; ``~`` is expanded and the path resolved. Decoding
        errors are ignored, so a file with stray bytes outside the header still
        reads.

    Returns
    -------
    int
        Number of nodes. A single-integer header is read as Gmsh 2.2; a header
        with two or more fields is read as MSH4, whose node total is the second
        field.

    Raises
    ------
    ValueError
        If the file has no ``$Nodes`` section, or its header line is empty and
        so matches neither layout.
    """
    from pathlib import Path

    path = Path(volume_mesh_file).expanduser().resolve()
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.strip() == "$Nodes":
                count_line = stream.readline().strip()
                # Gmsh 2.2 stores a single integer; MSH4 stores four integers.
                fields = count_line.split()
                if len(fields) == 1:
                    return int(fields[0])
                if len(fields) >= 2:
                    return int(fields[1])
                raise ValueError(
                    f"Unexpected $Nodes header {count_line!r} in {path}."
                )
    raise ValueError(f"No $Nodes section found in {path}.")


__all__ = [
    "GeometryVolumeBackend",
    "CSDLRecorderBackend",
    "MeshMotionVolumeBackend",
    "read_gmsh_volume_point_count",
]
