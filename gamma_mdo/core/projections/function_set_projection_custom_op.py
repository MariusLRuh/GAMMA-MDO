"""Differentiable projection returning points or parametric coordinates.

This is the sibling of the closest-distance operation: it runs the same
eager warm-started Newton projection but exposes the projected result rather
than a scalar measure, which is what the mesh-motion pipeline needs when it
has to place vertices back onto a deformed surface.

Coefficients use the stacked convention, ordered by ``model.patch_ids``.
Parametric output has shape ``(N, 3)`` with columns ``[patch_id, u, v]``; the
patch-ID column is discrete and has zero derivative. Physical output has shape
``(N, physical_dimension)``.

Derivatives come from the forward state via the implicit-function theorem, so
a point whose Newton solve did not converge carries no derivative guarantee.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Dict

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
)

try:
    from .function_set_closest_distance_custom_op import (
        FunctionSetProjectionModel,
        _get_current_forward_state,
        _is_point_candidate,
        _parse_fixed_axis,
        _store_forward_state,
        _solve_reduced_linear_system,
    )
except ImportError:
    from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
        FunctionSetProjectionModel,
        _get_current_forward_state,
        _is_point_candidate,
        _parse_fixed_axis,
        _store_forward_state,
        _solve_reduced_linear_system,
    )


def _build_projection_output(
    forward_state: Dict[str, object],
    *,
    return_parametric: bool,
    physical_dimension: int,
) -> np.ndarray:
    if return_parametric:
        patch_id = np.asarray(forward_state["patch_id"], dtype=float).reshape(-1, 1)
        uv = np.asarray(forward_state["uv"], dtype=float).reshape(-1, 2)
        return np.hstack([patch_id, uv])

    return np.asarray(forward_state["projected_points"], dtype=float).reshape(-1, physical_dimension)


def _compute_projection_vjp(
    model: FunctionSetProjectionModel,
    stacked_coefficients: np.ndarray,
    points: np.ndarray,
    d_outputs: np.ndarray,
    forward_state: Dict[str, object],
    *,
    return_parametric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stacked_coefficients = np.asarray(stacked_coefficients, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, model.physical_dimension)

    d_points = np.zeros_like(points)
    d_coefficients = np.zeros_like(stacked_coefficients)

    if return_parametric:
        d_outputs = np.asarray(d_outputs, dtype=float).reshape(-1, 3)
        if not np.any(d_outputs[:, 1:]):
            return d_points, d_coefficients
    else:
        d_outputs = np.asarray(d_outputs, dtype=float).reshape(-1, model.physical_dimension)
        if not np.any(d_outputs):
            return d_points, d_coefficients

    selected_patch_id = np.asarray(forward_state["patch_id"], dtype=int)
    selected_uv = np.asarray(forward_state["uv"], dtype=float)
    candidate_kind = list(forward_state["candidate_kind"])

    for patch_id in model.patch_ids:
        point_indices = np.where(selected_patch_id == int(patch_id))[0]
        if point_indices.size == 0:
            continue

        info = model.patch_infos[int(patch_id)]
        coeffs = np.asarray(stacked_coefficients[info.start:info.stop], dtype=float).reshape(info.coefficient_shape)

        uv = selected_uv[point_indices]
        point_batch = points[point_indices]
        kind_batch = [candidate_kind[idx] for idx in point_indices]

        cols_0, w_0, _ = compute_basis_stencil_numpy(
            uv,
            info.degrees,
            info.knot_vectors,
            cache=info.space_cache,
        )
        cols_u, w_u, _ = compute_basis_stencil_numpy(
            uv,
            info.degrees,
            info.knot_vectors,
            der_orders=(1, 0),
            cache=info.space_cache,
        )
        cols_v, w_v, _ = compute_basis_stencil_numpy(
            uv,
            info.degrees,
            info.knot_vectors,
            der_orders=(0, 1),
            cache=info.space_cache,
        )
        cols_uu, w_uu, _ = compute_basis_stencil_numpy(
            uv,
            info.degrees,
            info.knot_vectors,
            der_orders=(2, 0),
            cache=info.space_cache,
        )
        cols_uv, w_uv, _ = compute_basis_stencil_numpy(
            uv,
            info.degrees,
            info.knot_vectors,
            der_orders=(1, 1),
            cache=info.space_cache,
        )
        cols_vv, w_vv, _ = compute_basis_stencil_numpy(
            uv,
            info.degrees,
            info.knot_vectors,
            der_orders=(0, 2),
            cache=info.space_cache,
        )

        S = apply_basis_stencil_numpy(cols_0, w_0, coeffs)
        Su = apply_basis_stencil_numpy(cols_u, w_u, coeffs)
        Sv = apply_basis_stencil_numpy(cols_v, w_v, coeffs)
        Suu = apply_basis_stencil_numpy(cols_uu, w_uu, coeffs)
        Suv = apply_basis_stencil_numpy(cols_uv, w_uv, coeffs)
        Svv = apply_basis_stencil_numpy(cols_vv, w_vv, coeffs)

        residual_vector = S - point_batch

        residual = np.empty((point_indices.size, 2), dtype=float)
        residual[:, 0] = np.einsum("ij,ij->i", residual_vector, Su)
        residual[:, 1] = np.einsum("ij,ij->i", residual_vector, Sv)

        jacobian = np.empty((point_indices.size, 2, 2), dtype=float)
        jacobian[:, 0, 0] = np.einsum("ij,ij->i", Su, Su) + np.einsum("ij,ij->i", residual_vector, Suu)
        jacobian[:, 0, 1] = np.einsum("ij,ij->i", Su, Sv) + np.einsum("ij,ij->i", residual_vector, Suv)
        jacobian[:, 1, 0] = jacobian[:, 0, 1]
        jacobian[:, 1, 1] = np.einsum("ij,ij->i", Sv, Sv) + np.einsum("ij,ij->i", residual_vector, Svv)

        free_mask = np.ones_like(residual, dtype=bool)
        for local_index, kind in enumerate(kind_batch):
            fixed_axis = _parse_fixed_axis(kind)
            if fixed_axis is not None:
                free_mask[local_index, fixed_axis] = False
            if _is_point_candidate(kind):
                free_mask[local_index, :] = False

        lower = uv <= model.params.bound_eps
        upper = uv >= (1.0 - model.params.bound_eps)
        block_lower = lower & (residual > 0.0)
        block_upper = upper & (residual < 0.0)
        free_mask &= ~(block_lower | block_upper)

        if return_parametric:
            adjoint_rhs = np.asarray(d_outputs[point_indices, 1:3], dtype=float) * free_mask
            lambda_vec = _solve_reduced_linear_system(
                jacobian,
                adjoint_rhs,
                free_mask,
                diag_eps=model.params.diag_eps,
                det_eps=model.params.det_eps,
            )
            d_points[point_indices] = lambda_vec[:, 0, None] * Su + lambda_vec[:, 1, None] * Sv
            primary_term = -(lambda_vec[:, 0, None] * Su + lambda_vec[:, 1, None] * Sv)
        else:
            output_gradient = np.asarray(d_outputs[point_indices], dtype=float)
            dfdz = np.empty_like(residual)
            dfdz[:, 0] = np.einsum("ij,ij->i", output_gradient, Su)
            dfdz[:, 1] = np.einsum("ij,ij->i", output_gradient, Sv)
            adjoint_rhs = dfdz * free_mask
            lambda_vec = _solve_reduced_linear_system(
                jacobian,
                adjoint_rhs,
                free_mask,
                diag_eps=model.params.diag_eps,
                det_eps=model.params.det_eps,
            )
            d_points[point_indices] = lambda_vec[:, 0, None] * Su + lambda_vec[:, 1, None] * Sv
            primary_term = output_gradient - lambda_vec[:, 0, None] * Su - lambda_vec[:, 1, None] * Sv

        local_grad = np.zeros((info.stop - info.start, model.physical_dimension), dtype=float)

        np.add.at(
            local_grad,
            cols_0.ravel(),
            (w_0[:, :, None] * primary_term[:, None, :]).reshape(-1, model.physical_dimension),
        )

        if np.any(lambda_vec[:, 0]):
            u_term = -lambda_vec[:, 0, None, None] * w_u[:, :, None] * residual_vector[:, None, :]
            np.add.at(local_grad, cols_u.ravel(), u_term.reshape(-1, model.physical_dimension))

        if np.any(lambda_vec[:, 1]):
            v_term = -lambda_vec[:, 1, None, None] * w_v[:, :, None] * residual_vector[:, None, :]
            np.add.at(local_grad, cols_v.ravel(), v_term.reshape(-1, model.physical_dimension))

        d_coefficients[info.start:info.stop] += local_grad

    return d_points, d_coefficients


class FunctionSetProjectionVJP(csdl.experimental.CustomExplicitOperationBeta):
    """CSDL custom operation for the reverse product of the projection.

    Reads the forward state cached in ``shared_state`` by the paired
    :class:`FunctionSetProjectionOperation`, so it must run against the same
    coefficients.
    """

    def __init__(
        self,
        model: FunctionSetProjectionModel,
        shared_state: Dict[str, object],
        *,
        return_parametric: bool,
        output_name: str,
    ):
        super().__init__()
        self.model = model
        self.shared_state = shared_state
        self.return_parametric = bool(return_parametric)
        self.output_name = str(output_name)

    def evaluate(self, inputs, d_outputs):
        """Declare the reverse seed and the cotangent outputs.

        Parameters
        ----------
        inputs
            Mapping of the forward operation's inputs, holding
            ``"coefficients"`` and ``"points"``.
        d_outputs
            Mapping of reverse seeds, holding the forward operation's output
            name: ``"parametric_coordinates"`` when the operation returns
            parametric output, otherwise ``"projected_points"``.

        Returns
        -------
        dict of str to csdl_alpha.Variable
            Cotangents keyed by the differentiated input name:
            ``"coefficients"`` with the stacked coefficient shape, and
            ``"points"`` with shape ``(N, physical_dimension)``. The mapping is
            keyed, not ordered.
        """
        coefficients = inputs["coefficients"]
        points = inputs["points"].reshape(-1, self.model.physical_dimension)
        output_shape = (points.shape[0], 3) if self.return_parametric else (points.shape[0], self.model.physical_dimension)
        d_output = d_outputs[self.output_name].reshape(output_shape)

        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)
        self.declare_input("d_output", d_output)

        d_coefficients = self.create_output("d_coefficients", coefficients.shape)
        d_points = self.create_output("d_points", points.shape)

        return {
            "coefficients": d_coefficients,
            "points": d_points,
        }

    def compute(self, inputs, outputs):
        """Compute the cotangents eagerly and populate ``outputs``.

        Reuses the forward state cached by the forward operation when it
        matches the current coefficients and points, and otherwise re-solves
        the projection for them.

        Parameters
        ----------
        inputs
            Mapping holding ``"coefficients"``, ``"points"``, and the seed
            ``"d_output"``.
        outputs
            Output buffer written in place with ``"d_points"`` and
            ``"d_coefficients"``.

        Returns
        -------
        None
            Results are written into ``outputs``.
        """
        coefficients = np.asarray(inputs["coefficients"], dtype=float)
        points = np.asarray(inputs["points"], dtype=float).reshape(-1, self.model.physical_dimension)
        d_output = np.asarray(inputs["d_output"], dtype=float)
        forward_state = _get_current_forward_state(self.model, self.shared_state, coefficients, points)

        d_points, d_coefficients = _compute_projection_vjp(
            self.model,
            coefficients,
            points,
            d_output,
            forward_state,
            return_parametric=self.return_parametric,
        )
        outputs["d_points"] = d_points
        outputs["d_coefficients"] = d_coefficients


class FunctionSetProjectionOperation(csdl.experimental.CustomExplicitOperationBeta):
    """CSDL custom operation returning projected points or coordinates.

    Wraps the same eager NumPy projection kernel as the closest-distance
    operation, but exposes the projected result itself rather than a scalar
    distance measure. ``evaluate`` declares the graph inputs, output, and
    derivatives; ``compute`` performs the calculation.
    """

    def __init__(
        self,
        model: FunctionSetProjectionModel,
        *,
        return_parametric: bool = True,
    ):
        super().__init__()
        self.model = model
        self.return_parametric = bool(return_parametric)
        self.output_name = "parametric_coordinates" if self.return_parametric else "projected_points"
        self.shared_state: Dict[str, object] = {}

    def evaluate(self, coefficients, points):
        """Declare the graph inputs and the projection output.

        Parameters
        ----------
        coefficients
            Stacked coefficient variable.
        points
            Query-point variable.

        Returns
        -------
        csdl_alpha.Variable
            Physical points of shape ``(N, physical_dimension)``, or, when the
            operation was configured to return parametric output, an array of
            shape ``(N, 3)`` with columns ``[patch_id, u, v]`` whose patch-ID
            column has zero derivative.
        """
        points = points.reshape(-1, self.model.physical_dimension)

        self.declare_input("coefficients", coefficients)
        self.declare_input("points", points)

        output_shape = (points.shape[0], 3) if self.return_parametric else (points.shape[0], self.model.physical_dimension)
        output = self.create_output(self.output_name, output_shape)

        self.declare_vjp_function(
            FunctionSetProjectionVJP,
            model=self.model,
            shared_state=self.shared_state,
            return_parametric=self.return_parametric,
            output_name=self.output_name,
        )

        return output

    def compute(self, inputs, outputs):
        """Compute the projection eagerly and populate ``outputs``.

        The forward state is cached in ``shared_state`` for the reverse pass.

        Parameters
        ----------
        inputs
            Mapping holding ``"coefficients"`` and ``"points"``.
        outputs
            Output buffer written in place under this operation's output name,
            ``"parametric_coordinates"`` or ``"projected_points"``.

        Returns
        -------
        None
            The result is written into ``outputs``.
        """
        coefficients = np.asarray(inputs["coefficients"], dtype=float)
        points = np.asarray(inputs["points"], dtype=float).reshape(-1, self.model.physical_dimension)

        _, forward_state = self.model.project(coefficients, points)
        _store_forward_state(self.shared_state, coefficients, points, forward_state)
        outputs[self.output_name] = _build_projection_output(
            forward_state,
            return_parametric=self.return_parametric,
            physical_dimension=self.model.physical_dimension,
        )


if __name__ == "__main__":
    try:
        from .function_set_closest_distance_custom_op import (
            OrthogonalityNewtonParams,
            _build_demo_function_set,
            stack_function_set_coefficients,
        )
    except ImportError:
        from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
            OrthogonalityNewtonParams,
            _build_demo_function_set,
            stack_function_set_coefficients,
        )

    setup_recorder = csdl.Recorder(inline=True)
    setup_recorder.start()
    function_set = _build_demo_function_set()
    setup_recorder.stop()

    projection_model = FunctionSetProjectionModel(
        function_set,
        warm_start_nu=40,
        warm_start_nv=40,
        edge_map_num_samples=25,
        params=OrthogonalityNewtonParams(max_iter=30, tol_res=1e-12, tol_step=1e-12),
    )
    stacked_coefficients = stack_function_set_coefficients(function_set, projection_model.patch_ids)

    interior_parametric_coordinates = [
        (0, np.array([0.20, 0.25])),
        (0, np.array([0.35, 0.70])),
        (0, np.array([0.55, 0.40])),
        (0, np.array([0.70, 0.80])),
        (1, np.array([0.25, 0.20])),
        (1, np.array([0.40, 0.60])),
        (1, np.array([0.65, 0.35])),
        (1, np.array([0.80, 0.75])),
    ]
    interior_surface_points = np.asarray(
        function_set.evaluate(parametric_coordinates=interior_parametric_coordinates, non_csdl=True),
        dtype=float,
    ).reshape(-1, 3)
    query_points = interior_surface_points + np.array([0.0, 0.0, 0.12])

    def run_mode_verification(return_parametric: bool) -> None:
        """Print a forward and derivative check for one projection output mode.

        Parameters
        ----------
        return_parametric
            Verify the parametric output when ``True``, otherwise the physical
            projected points.

        Returns
        -------
        None
            Results are printed.
        """
        mode_name = "parametric" if return_parametric else "physical"
        output_shape = (query_points.shape[0], 3)
        weights = np.linspace(0.25, 1.25, np.prod(output_shape), dtype=float).reshape(output_shape)
        weights[1::2] *= -1.0

        recorder = csdl.Recorder(inline=True)
        recorder.start()

        coefficients_csdl = csdl.Variable(name=f"{mode_name}_stacked_coefficients", value=stacked_coefficients)
        points_csdl = csdl.Variable(name=f"{mode_name}_points_to_project", value=query_points)
        coefficients_csdl.set_as_design_variable()
        points_csdl.set_as_design_variable()

        projection_op = FunctionSetProjectionOperation(
            model=projection_model,
            return_parametric=return_parametric,
        )
        projected_output = projection_op.evaluate(
            coefficients=coefficients_csdl,
            points=points_csdl,
        )
        weights_csdl = csdl.Variable(name=f"{mode_name}_weights", value=weights)
        objective = csdl.sum(projected_output * weights_csdl)
        objective.name = f"{mode_name}_projection_sum"
        objective.set_as_objective()
        d_objective_d_points = csdl.derivative(objective, points_csdl)

        recorder.stop()
        sim = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
        sim.run()

        forward_state = projection_op.shared_state["forward"]
        print("")
        print(f"Projection mode: {mode_name}")
        print("Baseline output:", projected_output.value)
        print("Forward patch ids:", forward_state["patch_id"].tolist())
        print("Forward candidate kinds:", forward_state["candidate_kind"])
        print("Forward converged:", forward_state["converged"].tolist())
        print(
            "d_objective_d_points norm:",
            float(np.linalg.norm(np.asarray(d_objective_d_points.value, dtype=float))),
        )
        sim.check_optimization_derivatives(step_size=1e-7, raise_on_error=False)

    run_mode_verification(return_parametric=True)
    run_mode_verification(return_parametric=False)
