"""Differentiable evaluation of a function set at prescribed coordinates.

Unlike projection, the parametric coordinates are prescribed rather than
solved for: they are passed on every call. For a fixed set of coordinates the
map from coefficients to points is linear, but the reverse product is taken
with respect to both inputs, so it is not simply the transpose of one basis
matrix.

:class:`FunctionSetEvaluationModel` performs the eager NumPy work. Its
constructor caches patch metadata, B-spline space data, and row spans;
:meth:`FunctionSetEvaluationModel.evaluate` and
:meth:`FunctionSetEvaluationModel.compute_vjp` then run immediately whenever
they are called, whether directly or from a CSDL custom operation's
``compute``.

:class:`FunctionSetEvaluationOperation` and :class:`FunctionSetEvaluationVJP`
wrap the model for CSDL. Their ``evaluate`` declares graph inputs, outputs, and
derivatives; their ``compute`` performs the eager calculation.

Coefficients use the stacked convention: patches concatenated by row in
``model.patch_ids`` order, shape ``(total_control_points,
physical_dimension)``. Parametric coordinates have shape ``(N, 3)`` with
columns ``[patch_id, u, v]``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

_PACKAGES_DIR = Path(__file__).resolve().parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _package_name in ("CSDL_alpha", "lsdo_function_spaces"):
    _package_root = _PACKAGES_DIR / _package_name
    if _package_root.is_dir():
        _package_root_str = str(_package_root)
        if _package_root_str not in sys.path:
            sys.path.insert(0, _package_root_str)

try:
    import joblib  # noqa: F401
except Exception:
    _joblib_stub = types.ModuleType("joblib")

    def _delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    class _Parallel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, tasks):
            return [task() for task in tasks]

    _joblib_stub.delayed = _delayed
    _joblib_stub.Parallel = _Parallel
    sys.modules["joblib"] = _joblib_stub

import csdl_alpha as csdl
import numpy as np

from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy_factory import (
    apply_basis_stencil_numpy,
    compute_basis_stencil_numpy,
    make_bspline_space_cache,
)


@dataclass(frozen=True)
class EvaluationPatchInfo:
    """Cached per-patch data used when evaluating a function set.

    Attributes
    ----------
    patch_id
        Identifier of the patch inside the owning function set.
    degrees
        Per-parametric-direction B-spline degrees, ordered ``(u, v)``.
    knot_vectors
        Per-direction knot vectors, ordered to match ``degrees``.
    coefficient_shape
        Shape of this patch's coefficient array before stacking, with the
        trailing axis holding the physical dimension.
    start, stop
        Half-open row range this patch occupies in the stacked coefficient array.
    space_cache
        Opaque B-spline space cache reused across calls to build basis
        stencils for this patch. Built once per patch at construction.
    """

    patch_id: int
    degrees: tuple[int, ...]
    knot_vectors: tuple[np.ndarray, ...]
    coefficient_shape: tuple[int, ...]
    start: int
    stop: int
    space_cache: object


class FunctionSetEvaluationModel:
    """NumPy evaluator for a function set at caller-supplied parametric points.

    Construction caches only per-patch metadata: degrees, knot vectors,
    coefficient shapes, stacked row spans, and B-spline space caches. The
    parametric coordinates are not bound when the model is built; they are
    supplied per call to :meth:`evaluate` and :meth:`compute_vjp`, which build
    the basis stencils for those coordinates on each call. Evaluation is linear in the
    coefficients for a fixed set of coordinates, but it is nonlinear in the
    coordinates themselves, which is why the VJP returns cotangents for both.
    Both methods run eagerly in NumPy; the CSDL wrappers below hold a reference
    to this object.
    """

    def __init__(
        self,
        function_set,
        *,
        patch_indices: Optional[Iterable[int]] = None,
    ):
        if patch_indices is None:
            patch_indices = sorted(int(idx) for idx in function_set.functions.keys())
        else:
            patch_indices = [int(idx) for idx in patch_indices]

        self.patch_ids = tuple(patch_indices)
        self.patch_id_set = set(self.patch_ids)
        first_patch = function_set.functions[self.patch_ids[0]]
        self.physical_dimension = int(np.asarray(first_patch.coefficients.value).shape[-1])

        patch_infos: Dict[int, EvaluationPatchInfo] = {}
        start = 0
        for patch_id in self.patch_ids:
            fun = function_set.functions[int(patch_id)]
            coeffs = np.asarray(fun.coefficients.value, dtype=float)
            coeff_shape = tuple(int(v) for v in coeffs.shape)
            num_ctrl = int(np.prod(coeff_shape[:-1]))
            stop = start + num_ctrl

            degrees = tuple(int(v) for v in fun.space.degree)
            knot_vectors = tuple(np.asarray(k, dtype=float) for k in fun.space.knots)
            patch_infos[int(patch_id)] = EvaluationPatchInfo(
                patch_id=int(patch_id),
                degrees=degrees,
                knot_vectors=knot_vectors,
                coefficient_shape=coeff_shape,
                start=start,
                stop=stop,
                space_cache=make_bspline_space_cache(degrees, knot_vectors),
            )
            start = stop

        self.patch_infos = patch_infos
        self.total_num_control_points = start

    def _parse_parametric_coordinates(
        self,
        parametric_coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        parametric_coordinates = np.asarray(parametric_coordinates, dtype=float).reshape(-1, 3)
        patch_id = np.rint(parametric_coordinates[:, 0]).astype(int)
        uv = np.asarray(parametric_coordinates[:, 1:3], dtype=float)

        invalid = sorted(set(int(idx) for idx in patch_id) - self.patch_id_set)
        if invalid:
            raise ValueError(
                f"Encountered patch ids {invalid} that are not present in this evaluation model."
            )

        return patch_id, uv

    def evaluate(
        self,
        stacked_coefficients: np.ndarray,
        parametric_coordinates: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the surface at the given parametric coordinates.

        Runs eagerly when called.

        Parameters
        ----------
        stacked_coefficients
            Coefficients in the stacked convention, ordered by
            ``model.patch_ids``, shape
            ``(total_control_points, physical_dimension)``.
        parametric_coordinates
            Array of shape ``(N, 3)`` with columns ``[patch_id, u, v]``.

        Returns
        -------
        numpy.ndarray
            Points of shape ``(N, physical_dimension)``.
        """
        stacked_coefficients = np.asarray(stacked_coefficients, dtype=float)
        patch_id, uv = self._parse_parametric_coordinates(parametric_coordinates)

        output = np.zeros((uv.shape[0], self.physical_dimension), dtype=float)
        for current_patch_id in self.patch_ids:
            point_indices = np.where(patch_id == int(current_patch_id))[0]
            if point_indices.size == 0:
                continue

            info = self.patch_infos[int(current_patch_id)]
            coeffs = np.asarray(stacked_coefficients[info.start:info.stop], dtype=float).reshape(info.coefficient_shape)
            cols_0, w_0, _ = compute_basis_stencil_numpy(
                uv[point_indices],
                info.degrees,
                info.knot_vectors,
                cache=info.space_cache,
            )
            output[point_indices] = apply_basis_stencil_numpy(cols_0, w_0, coeffs)

        return output

    def compute_vjp(
        self,
        stacked_coefficients: np.ndarray,
        parametric_coordinates: np.ndarray,
        d_evaluated_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the reverse-mode product for one seed.

        Parameters
        ----------
        stacked_coefficients
            Coefficients in the stacked convention.
        parametric_coordinates
            Array of shape ``(N, 3)`` with columns ``[patch_id, u, v]``.
        d_evaluated_points
            Reverse seed with the shape of the evaluated points.

        Returns
        -------
        d_coefficients : numpy.ndarray
            Coefficient cotangent in the stacked convention.
        d_parametric_coordinates : numpy.ndarray
            Cotangent of shape ``(N, 3)``. Column 0 is the discrete patch ID
            and is always zero; columns 1 and 2 are the seed contracted with
            the surface tangents in ``u`` and ``v``.

        Notes
        -----
        Returned in the order ``(d_coefficients, d_parametric_coordinates)``.
        For prescribed coordinates the coefficient block is the transpose of
        the basis stencil, but the coordinate block uses surface tangents, so
        the full product is not a single transposed matrix.
        """
        stacked_coefficients = np.asarray(stacked_coefficients, dtype=float)
        d_evaluated_points = np.asarray(d_evaluated_points, dtype=float).reshape(-1, self.physical_dimension)

        d_coefficients = np.zeros_like(stacked_coefficients)
        d_parametric_coordinates = np.zeros_like(np.asarray(parametric_coordinates, dtype=float).reshape(-1, 3))

        if not np.any(d_evaluated_points):
            return d_coefficients, d_parametric_coordinates

        patch_id, uv = self._parse_parametric_coordinates(parametric_coordinates)

        for current_patch_id in self.patch_ids:
            point_indices = np.where(patch_id == int(current_patch_id))[0]
            if point_indices.size == 0:
                continue

            info = self.patch_infos[int(current_patch_id)]
            coeffs = np.asarray(stacked_coefficients[info.start:info.stop], dtype=float).reshape(info.coefficient_shape)
            uv_batch = uv[point_indices]
            cotangent_batch = d_evaluated_points[point_indices]

            cols_0, w_0, _ = compute_basis_stencil_numpy(
                uv_batch,
                info.degrees,
                info.knot_vectors,
                cache=info.space_cache,
            )
            cols_u, w_u, _ = compute_basis_stencil_numpy(
                uv_batch,
                info.degrees,
                info.knot_vectors,
                der_orders=(1, 0),
                cache=info.space_cache,
            )
            cols_v, w_v, _ = compute_basis_stencil_numpy(
                uv_batch,
                info.degrees,
                info.knot_vectors,
                der_orders=(0, 1),
                cache=info.space_cache,
            )

            Su = apply_basis_stencil_numpy(cols_u, w_u, coeffs)
            Sv = apply_basis_stencil_numpy(cols_v, w_v, coeffs)

            local_grad = np.zeros((info.stop - info.start, self.physical_dimension), dtype=float)
            np.add.at(
                local_grad,
                cols_0.ravel(),
                (w_0[:, :, None] * cotangent_batch[:, None, :]).reshape(-1, self.physical_dimension),
            )

            d_coefficients[info.start:info.stop] += local_grad
            d_parametric_coordinates[point_indices, 1] = np.einsum("ij,ij->i", cotangent_batch, Su)
            d_parametric_coordinates[point_indices, 2] = np.einsum("ij,ij->i", cotangent_batch, Sv)

        return d_coefficients, d_parametric_coordinates


class FunctionSetEvaluationVJP(csdl.experimental.CustomExplicitOperationBeta):
    """CSDL custom operation for the reverse product of evaluation.

    Produces cotangents for both the coefficients and the parametric
    coordinates.
    """

    def __init__(self, model: FunctionSetEvaluationModel):
        super().__init__()
        self.model = model

    def evaluate(self, inputs, d_outputs):
        """Declare the reverse seed and both cotangent outputs.

        Parameters
        ----------
        inputs
            Mapping of the forward operation's inputs, holding
            ``"coefficients"`` and ``"parametric_coordinates"``.
        d_outputs
            Mapping of reverse seeds, holding ``"evaluated_points"`` with the
            shape of the forward output.

        Returns
        -------
        dict of str to csdl_alpha.Variable
            Cotangents keyed by the differentiated input name:
            ``"coefficients"`` with the stacked coefficient shape, and
            ``"parametric_coordinates"`` with shape ``(N, 3)``. The mapping is
            keyed, not ordered.
        """
        coefficients = inputs["coefficients"]
        parametric_coordinates = inputs["parametric_coordinates"].reshape(-1, 3)
        d_evaluated_points = d_outputs["evaluated_points"].reshape(-1, self.model.physical_dimension)

        self.declare_input("coefficients", coefficients)
        self.declare_input("parametric_coordinates", parametric_coordinates)
        self.declare_input("d_evaluated_points", d_evaluated_points)

        d_coefficients = self.create_output("d_coefficients", coefficients.shape)
        d_parametric_coordinates = self.create_output("d_parametric_coordinates", parametric_coordinates.shape)

        return {
            "coefficients": d_coefficients,
            "parametric_coordinates": d_parametric_coordinates,
        }

    def compute(self, inputs, outputs):
        """Compute both cotangents eagerly and populate ``outputs``.

        Parameters
        ----------
        inputs
            Mapping holding ``"coefficients"``, ``"parametric_coordinates"``,
            and the seed ``"d_evaluated_points"``.
        outputs
            Output buffer written in place with ``"d_coefficients"`` and
            ``"d_parametric_coordinates"``.

        Returns
        -------
        None
            Results are written into ``outputs``.
        """
        coefficients = np.asarray(inputs["coefficients"], dtype=float)
        parametric_coordinates = np.asarray(inputs["parametric_coordinates"], dtype=float).reshape(-1, 3)
        d_evaluated_points = np.asarray(inputs["d_evaluated_points"], dtype=float).reshape(-1, self.model.physical_dimension)

        d_coefficients, d_parametric_coordinates = self.model.compute_vjp(
            coefficients,
            parametric_coordinates,
            d_evaluated_points,
        )
        outputs["d_coefficients"] = d_coefficients
        outputs["d_parametric_coordinates"] = d_parametric_coordinates


class FunctionSetEvaluationOperation(csdl.experimental.CustomExplicitOperationBeta):
    """CSDL custom operation wrapping function-set evaluation.

    ``evaluate`` declares the graph inputs, output, and derivatives;
    ``compute`` performs the eager calculation through
    :meth:`FunctionSetEvaluationModel.evaluate`.
    """

    def __init__(self, model: FunctionSetEvaluationModel):
        super().__init__()
        self.model = model

    def evaluate(self, coefficients, parametric_coordinates):
        """Declare the graph inputs and the evaluated-point output.

        Parameters
        ----------
        coefficients
            Stacked coefficient variable.
        parametric_coordinates
            Variable of shape ``(N, 3)`` with columns ``[patch_id, u, v]``.

        Returns
        -------
        csdl_alpha.Variable
            Points of shape ``(N, physical_dimension)``.
        """
        parametric_coordinates = parametric_coordinates.reshape(-1, 3)

        self.declare_input("coefficients", coefficients)
        self.declare_input("parametric_coordinates", parametric_coordinates)

        evaluated_points = self.create_output(
            "evaluated_points",
            (parametric_coordinates.shape[0], self.model.physical_dimension),
        )

        self.declare_vjp_function(
            FunctionSetEvaluationVJP,
            model=self.model,
        )

        return evaluated_points

    def compute(self, inputs, outputs):
        """Compute the evaluated points eagerly and populate ``outputs``.

        Parameters
        ----------
        inputs
            Mapping holding ``"coefficients"`` and ``"parametric_coordinates"``.
        outputs
            Output buffer written in place with ``"evaluated_points"``.

        Returns
        -------
        None
            The result is written into ``outputs``.
        """
        coefficients = np.asarray(inputs["coefficients"], dtype=float)
        parametric_coordinates = np.asarray(inputs["parametric_coordinates"], dtype=float).reshape(-1, 3)
        outputs["evaluated_points"] = self.model.evaluate(coefficients, parametric_coordinates)


if __name__ == "__main__":
    try:
        from .function_set_closest_distance_custom_op import (
            _build_demo_function_set,
            stack_function_set_coefficients,
        )
    except ImportError:
        from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
            _build_demo_function_set,
            stack_function_set_coefficients,
        )

    setup_recorder = csdl.Recorder(inline=True)
    setup_recorder.start()
    function_set = _build_demo_function_set()
    setup_recorder.stop()

    evaluation_model = FunctionSetEvaluationModel(function_set=function_set)
    stacked_coefficients = stack_function_set_coefficients(function_set, evaluation_model.patch_ids)
    parametric_coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    direct_points = np.asarray(
        function_set.evaluate(
            parametric_coordinates=[
                (int(row[0]), np.asarray(row[1:3], dtype=float).copy())
                for row in parametric_coordinates
            ],
            non_csdl=True,
        ),
        dtype=float,
    ).reshape(-1, 3)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    coefficients_csdl = csdl.Variable(name="evaluation_stacked_coefficients", value=stacked_coefficients)
    parametric_coordinates_csdl = csdl.Variable(
        name="evaluation_parametric_coordinates",
        value=parametric_coordinates,
    )
    coefficients_csdl.set_as_design_variable()
    parametric_coordinates_csdl.set_as_design_variable()

    evaluation_op = FunctionSetEvaluationOperation(model=evaluation_model)
    evaluated_points = evaluation_op.evaluate(
        coefficients=coefficients_csdl,
        parametric_coordinates=parametric_coordinates_csdl,
    )
    weights = np.linspace(0.15, 1.15, evaluated_points.shape[0] * evaluated_points.shape[1], dtype=float).reshape(
        evaluated_points.shape
    )
    weights[1::2] *= -1.0
    weights_csdl = csdl.Variable(name="evaluation_weights", value=weights)
    objective = csdl.sum(evaluated_points * weights_csdl)
    objective.name = "evaluation_sum"
    objective.set_as_objective()
    d_objective_d_parametric = csdl.derivative(objective, parametric_coordinates_csdl)

    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    sim.run()

    d_parametric = np.asarray(d_objective_d_parametric.value, dtype=float).reshape(parametric_coordinates.shape)

    print("")
    print("Evaluation baseline output:", evaluated_points.value)
    print(
        "Direct function-set consistency:",
        float(np.max(np.abs(np.asarray(evaluated_points.value, dtype=float) - direct_points))),
    )
    print("Patch-index derivative column:", d_parametric[:, 0])
    sim.check_optimization_derivatives(step_size=1e-7, raise_on_error=False)
