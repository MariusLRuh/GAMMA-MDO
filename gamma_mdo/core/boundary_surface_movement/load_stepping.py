"""Fixed-count graph continuation with an OML projection at every load step.

The topology stays fixed, but inverse-area graph weights and the factorization
are rebuilt from the previously projected mesh.  Together with the nonlinear
closest-point projection, this makes the increments genuinely path-dependent:

``geometry(t_k) -> exact seams -> L(x_(k-1)) increment -> project graph rows``.

Exact intersection rows already lie on the driving OML by construction and
are not sent through the closest-point solver a second time.

The number of steps is deliberately fixed while a CSDL graph is being built.
Adaptive step acceptance would make both the graph structure and its derivative
discontinuous across design iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.preprocessing import ProjectionMetadata, VertexEvaluationMetadata

from .current_graph_solve import (
    CurrentGraphModel,
    CurrentGraphSolveOperation,
)
from .constraints import enforce_symmetry_plane
from .elasticity import GraphLaplacianAssembler
from .motion import ElasticityMotionSolver
from .quadratic_distortion import (
    CurrentGraphDistortionModel,
    CurrentGraphDistortionSolveOperation,
    QuadraticDistortionAssembler,
    QuadraticDistortionConfig,
)
from .ngon_affine import (
    CurrentGraphNgonAffineModel,
    CurrentGraphNgonAffineSolveOperation,
    NgonAffineAssembler,
    NgonAffineConfig,
)
from .projection import (
    VertexBatch,
    combine_vertices,
    project_onto_oml,
    reevaluate_vertices,
)


@dataclass(frozen=True)
class GraphLoadStepResult:
    """Collect outputs from a fixed graph-continuation path.

    Attributes
    ----------
    final_mesh_vertices
        Final projected coordinates in complete mesh order.
    final_preprojected_mesh_vertices
        Final coordinates immediately before OML projection.
    final_deformation_vertices
        Final coordinates on the actively deformed subset.
    load_fractions
        Monotone continuation fractions ending at one.
    preprojected_mesh_history
        Preprojection coordinates from every increment.
    projected_mesh_history
        Reprojected coordinates from every increment.
    distortion_normalization_scale
        Optional quadratic-regularizer normalization.
    ngon_affine_normalization_scale
        Optional affine-residual regularizer normalization.
    final_reprojected_vertex_ids
        Global IDs passed through closest-point projection at the final step.
    final_nonconverged_vertex_ids
        Subset whose final closest-point solve did not converge.
    """

    final_mesh_vertices: csdl.Variable
    final_preprojected_mesh_vertices: csdl.Variable
    final_deformation_vertices: csdl.Variable
    load_fractions: tuple[float, ...]
    preprojected_mesh_history: tuple[csdl.Variable, ...]
    projected_mesh_history: tuple[csdl.Variable, ...]
    final_reprojected_vertex_ids: np.ndarray
    final_nonconverged_vertex_ids: np.ndarray
    distortion_normalization_scale: float | None = None
    distortion_redundancy: float | None = None
    distortion_num_ear_clipped: int = 0
    distortion_maximum_warp_ratio: float = 0.0
    ngon_affine_normalization_scale: float | None = None
    ngon_affine_num_elements: int = 0
    ngon_affine_num_modes: int = 0
    ngon_affine_maximum_warp_ratio: float = 0.0


def linear_load_fractions(num_steps: int) -> tuple[float, ...]:
    """Return evenly spaced fractions ending at one.

    Parameters
    ----------
    num_steps
        Positive integer number of continuation increments.

    Returns
    -------
    tuple[float, ...]
        Fractions ``(1/N, ..., 1)``.

    Raises
    ------
    ValueError
        If ``num_steps`` is not a positive integer.
    """
    count = int(num_steps)
    if count < 1 or count != num_steps:
        raise ValueError("num_steps must be a positive integer.")
    return tuple(float(item) / count for item in range(1, count + 1))


def run_graph_load_steps(
    *,
    motion: ElasticityMotionSolver,
    mesh,
    initial_deformation_vertices,
    deformation_vertex_ids,
    component_coefficient_steps: Sequence[Mapping[object, object]],
    projection_metadata: Sequence[ProjectionMetadata],
    reevaluation_metadata: Sequence[VertexEvaluationMetadata],
    projection_options: Mapping[str, object] | None = None,
    load_fractions: Sequence[float] | None = None,
    distortion_config: QuadraticDistortionConfig | None = None,
    ngon_affine_config: NgonAffineConfig | None = None,
    symmetry_plane_vertex_ids: np.ndarray | None = None,
    symmetry_plane_axis: int = 1,
) -> GraphLoadStepResult:
    """Unroll a fixed number of graph/projection continuation increments.

    Each entry of ``component_coefficient_steps`` is the complete component
    coefficient map at one monotonically increasing load fraction.  The motion
    geometry state is evaluated repeatedly.  Consecutive component-reference
    and seam-departure states are differenced, while the graph weights and
    factorization are rebuilt from the *previously projected* mesh.  This is
    the nonlinear continuation state.

    Seam rows are overwritten by the current exact intersection and bypass
    closest-point projection, so the bracketed solution is preserved without
    a redundant nonlinear solve.

    Parameters
    ----------
    motion
        Configured graph-Laplacian surface-motion solver.
    mesh
        Baseline surface mesh.
    initial_deformation_vertices
        Initial coordinates for the actively deformed rows.
    deformation_vertex_ids
        Global IDs aligned with ``initial_deformation_vertices``.
    component_coefficient_steps
        Complete component coefficient mapping at every load fraction.
    projection_metadata
        Component ownership used to reproject graph-moved rows.
    reevaluation_metadata
        Fixed parametric coordinates used for exact component reevaluation.
    projection_options
        Optional projection solver settings.
    load_fractions
        Optional explicit strictly increasing fractions ending at one.
    distortion_config
        Optional quadratic-distortion regularizer.
    ngon_affine_config
        Optional affine-residual polygon regularizer.
    symmetry_plane_vertex_ids
        Optional fixed set constrained to the symmetry plane.
    symmetry_plane_axis
        Coordinate axis normal to the symmetry plane.

    Returns
    -------
    GraphLoadStepResult
        Final meshes, continuation history, and regularizer diagnostics.

    Raises
    ------
    TypeError
        If ``motion`` is not the required graph-Laplacian solver.
    ValueError
        If continuation data, partitions, or regularizer choices are invalid.
    """
    if not isinstance(motion, ElasticityMotionSolver):
        raise TypeError("motion must be an ElasticityMotionSolver.")
    if not isinstance(motion.assembler, GraphLaplacianAssembler):
        raise TypeError(
            "run_graph_load_steps requires a GraphLaplacianAssembler."
        )
    coefficient_steps = tuple(component_coefficient_steps)
    if not coefficient_steps:
        raise ValueError("component_coefficient_steps must not be empty.")
    if (
        distortion_config is not None
        and distortion_config.lambda_dist > 0.0
        and ngon_affine_config is not None
        and ngon_affine_config.lambda_ngon > 0.0
    ):
        raise ValueError(
            "Distortion and n-gon affine regularization are mutually exclusive."
        )

    if load_fractions is None:
        fractions = linear_load_fractions(len(coefficient_steps))
    else:
        fractions = tuple(float(item) for item in load_fractions)
        if len(fractions) != len(coefficient_steps):
            raise ValueError(
                "load_fractions must align with component_coefficient_steps."
            )
        if (
            any(not np.isfinite(item) for item in fractions)
            or any(item <= 0.0 or item > 1.0 for item in fractions)
            or any(right <= left for left, right in zip(fractions, fractions[1:]))
            or abs(fractions[-1] - 1.0) > 1e-12
        ):
            raise ValueError(
                "load_fractions must be strictly increasing in (0, 1] and end at 1."
            )

    ids = np.asarray(deformation_vertex_ids, dtype=np.int64).reshape(-1)
    initial = np.asarray(
        getattr(initial_deformation_vertices, "value", initial_deformation_vertices),
        dtype=float,
    ).reshape((-1, 3))
    if initial.shape[0] != ids.size:
        raise ValueError(
            "initial_deformation_vertices and deformation_vertex_ids must align."
        )
    if np.unique(ids).size != ids.size:
        raise ValueError("deformation_vertex_ids must be unique.")
    plane_ids = np.asarray(
        (
            np.empty(0, dtype=np.int64)
            if symmetry_plane_vertex_ids is None
            else symmetry_plane_vertex_ids
        ),
        dtype=np.int64,
    ).reshape(-1)
    if np.any(plane_ids < 0) or np.any(
        plane_ids >= motion.initial_vertices.shape[0]
    ):
        raise ValueError(
            "symmetry_plane_vertex_ids contains an out-of-range vertex."
        )

    initial_variable = (
        initial_deformation_vertices
        if hasattr(initial_deformation_vertices, "value")
        else csdl.Variable(value=initial)
    )
    current_deformation = initial_variable * 1.0
    current_mesh = csdl.Variable(value=motion.initial_vertices)
    current_mesh = current_mesh.set(_row_slice(ids), current_deformation)
    previous_state = None

    deformation_row_by_id = {
        int(vertex_id): row for row, vertex_id in enumerate(ids)
    }
    try:
        free_deformation_rows = np.asarray(
            [deformation_row_by_id[int(vertex)] for vertex in motion.free_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "deformation_vertex_ids must contain every graph free vertex."
        ) from error

    distance_weighting = getattr(
        motion.assembler, "distance_weighting", None
    )
    graph_model = CurrentGraphModel(
        mesh,
        free_ids=motion.free_ids,
        prescribed_ids=motion.prescribed_ids,
        stiffening_exponent=motion.assembler.stiffening_exponent,
        area_floor=motion.assembler.area_floor,
        distance_weighting=distance_weighting,
        quad_diagonal_weight=motion.assembler.quad_diagonal_weight,
        quad_bracing_mode=motion.assembler.quad_bracing_mode,
    )
    distortion_model = None
    distortion_system = None
    ngon_affine_model = None
    ngon_affine_system = None
    if (
        distortion_config is not None
        and distortion_config.lambda_dist > 0.0
    ):
        distortion_system = QuadraticDistortionAssembler(
            distortion_config
        ).assemble(
            mesh,
            free_ids=motion.free_ids,
            prescribed_ids=motion.prescribed_ids,
        )
        constrained_free_dofs = np.empty(0, dtype=np.int64)
        if motion.symmetry_plane_ids is not None:
            constrained_free_dofs = np.asarray(
                [
                    3 * motion._free_local[int(vertex)] + 1
                    for vertex in motion.symmetry_plane_ids
                ],
                dtype=np.int64,
            )
        distortion_model = CurrentGraphDistortionModel(
            graph_model,
            distortion_system,
            lambda_dist=distortion_config.lambda_dist,
            constrained_free_dofs=constrained_free_dofs,
            baseline_vertices=motion.initial_vertices,
        )
    if (
        ngon_affine_config is not None
        and ngon_affine_config.lambda_ngon > 0.0
    ):
        ngon_affine_system = NgonAffineAssembler(
            ngon_affine_config
        ).assemble(
            mesh,
            free_ids=motion.free_ids,
            prescribed_ids=motion.prescribed_ids,
        )
        ngon_affine_model = CurrentGraphNgonAffineModel(
            graph_model,
            ngon_affine_system,
            lambda_ngon=ngon_affine_config.lambda_ngon,
            baseline_vertices=motion.initial_vertices,
        )
    graph_model_y = None
    ngon_affine_model_y = None
    if motion.symmetry_plane_ids is not None and distortion_model is None:
        graph_model_y = CurrentGraphModel(
            mesh,
            free_ids=motion.free_ids_y,
            prescribed_ids=motion.prescribed_ids_y,
            stiffening_exponent=motion.assembler.stiffening_exponent,
            area_floor=motion.assembler.area_floor,
            distance_weighting=distance_weighting,
            quad_diagonal_weight=motion.assembler.quad_diagonal_weight,
            quad_bracing_mode=motion.assembler.quad_bracing_mode,
        )
        if ngon_affine_model is not None:
            ngon_affine_system_y = NgonAffineAssembler(
                ngon_affine_config
            ).assemble(
                mesh,
                free_ids=motion.free_ids_y,
                prescribed_ids=motion.prescribed_ids_y,
            )
            ngon_affine_model_y = CurrentGraphNgonAffineModel(
                graph_model_y,
                ngon_affine_system_y,
                lambda_ngon=ngon_affine_config.lambda_ngon,
                baseline_vertices=motion.initial_vertices,
            )

    preprojected_history: list[csdl.Variable] = []
    projected_history: list[csdl.Variable] = []
    final_reprojected_ids = np.empty(0, dtype=np.int64)
    final_nonconverged_ids = np.empty(0, dtype=np.int64)
    for step_index, coefficient_map in enumerate(coefficient_steps):
        state = motion.build_load_step_state(
            component_coeffs=coefficient_map,
            query_component_coeffs=coefficient_map,
        )
        reference_increment = (
            state.free_reference
            if previous_state is None
            else state.free_reference - previous_state.free_reference
        )
        prescribed_increments = (
            state.prescribed_deviations
            if previous_state is None
            else tuple(
                current - previous
                for current, previous in zip(
                    state.prescribed_deviations,
                    previous_state.prescribed_deviations,
                )
            )
        )
        prescribed_matrix = _stack_columns(prescribed_increments)
        if distortion_model is None and ngon_affine_model is None:
            solved_blocks = CurrentGraphSolveOperation(graph_model).evaluate(
                current_mesh,
                prescribed_matrix,
            )
        elif ngon_affine_model is not None:
            previous_reference = (
                csdl.Variable(
                    value=np.zeros((motion.free_ids.size, 3), dtype=float)
                )
                if previous_state is None
                else previous_state.free_reference
            )
            accumulated_correction = (
                current_mesh[_row_slice(motion.free_ids)]
                - motion.initial_vertices[motion.free_ids]
                - previous_reference
            )
            current_free_matrix = _stack_block_rows(
                accumulated_correction,
                motion._free_reference_groups,
            )
            total_prescribed_matrix = _stack_columns(
                state.prescribed_deviations
            )
            solved_blocks = CurrentGraphNgonAffineSolveOperation(
                ngon_affine_model
            ).evaluate(
                current_mesh,
                prescribed_matrix,
                current_free_matrix,
                total_prescribed_matrix,
            )
        else:
            previous_reference = (
                csdl.Variable(
                    value=np.zeros((motion.free_ids.size, 3), dtype=float)
                )
                if previous_state is None
                else previous_state.free_reference
            )
            accumulated_correction = (
                current_mesh[_row_slice(motion.free_ids)]
                - motion.initial_vertices[motion.free_ids]
                - previous_reference
            )
            current_free_matrix = _stack_block_rows(
                accumulated_correction,
                motion._free_reference_groups,
            )
            total_prescribed_matrix = _stack_columns(
                state.prescribed_deviations
            )
            solved_blocks = CurrentGraphDistortionSolveOperation(
                distortion_model
            ).evaluate(
                current_mesh,
                prescribed_matrix,
                current_free_matrix,
                total_prescribed_matrix,
            )
        correction_xz = csdl.Variable(
            value=np.zeros((motion.free_ids.size, 3), dtype=float)
        )
        for block_index, group in enumerate(motion._free_reference_groups):
            columns = csdl.slice[
                :,
                3 * block_index : 3 * (block_index + 1),
            ]
            correction_xz = correction_xz.set(
                _row_slice(group.free_rows),
                solved_blocks[_row_slice(group.free_rows)][columns],
            )

        if graph_model_y is None:
            correction = correction_xz
        else:
            prescribed_increments_y = (
                state.prescribed_deviations_y
                if previous_state is None
                else tuple(
                    current - previous
                    for current, previous in zip(
                        state.prescribed_deviations_y,
                        previous_state.prescribed_deviations_y,
                    )
                )
            )
            prescribed_y = _stack_columns(
                tuple(
                    item[csdl.slice[:, 1:2]]
                    for item in prescribed_increments_y
                )
            )
            if ngon_affine_model_y is None:
                solved_y_blocks = CurrentGraphSolveOperation(
                    graph_model_y
                ).evaluate(
                    current_mesh,
                    prescribed_y,
                )
            else:
                previous_reference_y = (
                    csdl.Variable(
                        value=np.zeros((motion.free_ids_y.size, 1), dtype=float)
                    )
                    if previous_state is None
                    else previous_state.free_reference[
                        _row_slice(
                            np.asarray(
                                [
                                    motion._free_local[int(vertex)]
                                    for vertex in motion.free_ids_y
                                ],
                                dtype=np.int64,
                            )
                        )
                    ][csdl.slice[:, 1:2]]
                )
                accumulated_correction_y = (
                    current_mesh[_row_slice(motion.free_ids_y)][csdl.slice[:, 1:2]]
                    - motion.initial_vertices[motion.free_ids_y, 1:2]
                    - previous_reference_y
                )
                current_free_y = _stack_scalar_block_rows(
                    accumulated_correction_y,
                    motion._free_reference_groups_y,
                )
                total_prescribed_y = _stack_columns(
                    tuple(
                        item[csdl.slice[:, 1:2]]
                        for item in state.prescribed_deviations_y
                    )
                )
                solved_y_blocks = CurrentGraphNgonAffineSolveOperation(
                    ngon_affine_model_y
                ).evaluate(
                    current_mesh,
                    prescribed_y,
                    current_free_y,
                    total_prescribed_y,
                )
            correction_y_free = csdl.Variable(
                value=np.zeros((motion.free_ids_y.size, 1), dtype=float)
            )
            for block_index, group in enumerate(
                motion._free_reference_groups_y
            ):
                correction_y_free = correction_y_free.set(
                    _row_slice(group.free_rows),
                    solved_y_blocks[
                        _row_slice(group.free_rows)
                    ][csdl.slice[:, block_index : block_index + 1]],
                )
            correction_y = csdl.sparse.matmat(
                motion._scatter_y,
                correction_y_free,
            )
            correction = csdl.concatenate(
                (
                    correction_xz[csdl.slice[:, 0:1]],
                    correction_y,
                    correction_xz[csdl.slice[:, 2:3]],
                ),
                axis=1,
            )

        free_preprojected = (
            current_mesh[_row_slice(motion.free_ids)]
            + reference_increment
            + correction
        )
        preprojected_deformation = current_deformation.set(
            _row_slice(free_deformation_rows),
            free_preprojected,
        )
        preprojected_deformation = _set_exact_seams(
            preprojected_deformation,
            state.solutions,
            deformation_row_by_id,
        )

        reevaluated_batch = reevaluate_vertices(
            mesh=mesh,
            metadata=reevaluation_metadata,
            component_coefficients=coefficient_map,
        )
        preprojected_mesh = combine_vertices(
            oml_projected_vertices=VertexBatch(
                values=preprojected_deformation,
                vertex_ids=ids,
                num_mesh_vertices=motion.initial_vertices.shape[0],
            ),
            reevaluated_mesh_vertices=reevaluated_batch,
        )
        preprojected_mesh = enforce_symmetry_plane(
            preprojected_mesh,
            vertex_ids=plane_ids,
            axis=symmetry_plane_axis,
        )
        exact_seam_ids = _solution_vertex_ids(state.solutions)
        reproject_mask = ~np.isin(ids, exact_seam_ids)
        reprojected_ids = ids[reproject_mask]
        reprojected_rows = np.where(reproject_mask)[0].astype(np.int64)
        projected_deformation = preprojected_deformation
        if reprojected_ids.size:
            closest_batch = project_onto_oml(
                deformed_mesh_vertices=preprojected_deformation[
                    _row_slice(reprojected_rows)
                ],
                deformed_mesh_vertex_ids=reprojected_ids,
                projection_metadata=projection_metadata,
                component_coefficients=coefficient_map,
                projection_options=dict(projection_options or {}),
            )
            projected_deformation = projected_deformation.set(
                _row_slice(reprojected_rows), closest_batch.values
            )
            final_nonconverged_ids = reprojected_ids[
                ~np.asarray(closest_batch.converged, dtype=bool)
            ]
        else:
            final_nonconverged_ids = np.empty(0, dtype=np.int64)
        final_reprojected_ids = reprojected_ids
        projected_batch = VertexBatch(
            values=projected_deformation,
            vertex_ids=ids,
            num_mesh_vertices=motion.initial_vertices.shape[0],
        )

        projected_mesh = combine_vertices(
            oml_projected_vertices=projected_batch,
            reevaluated_mesh_vertices=reevaluated_batch,
        )
        projected_mesh = enforce_symmetry_plane(
            projected_mesh,
            vertex_ids=plane_ids,
            axis=symmetry_plane_axis,
        )
        preprojected_history.append(preprojected_mesh)
        projected_history.append(projected_mesh)
        current_deformation = projected_mesh[_row_slice(ids)]
        current_mesh = projected_mesh
        previous_state = state

    return GraphLoadStepResult(
        final_mesh_vertices=projected_history[-1],
        final_preprojected_mesh_vertices=preprojected_history[-1],
        final_deformation_vertices=current_deformation,
        load_fractions=fractions,
        preprojected_mesh_history=tuple(preprojected_history),
        projected_mesh_history=tuple(projected_history),
        distortion_normalization_scale=(
            None
            if distortion_model is None
            else distortion_model.normalization_scale
        ),
        distortion_redundancy=(
            None if distortion_model is None else distortion_model.redundancy
        ),
        distortion_num_ear_clipped=(
            0 if distortion_system is None else distortion_system.num_ear_clipped
        ),
        distortion_maximum_warp_ratio=(
            0.0
            if distortion_system is None
            else distortion_system.maximum_warp_ratio
        ),
        ngon_affine_normalization_scale=(
            None
            if ngon_affine_model is None
            else ngon_affine_model.normalization_scale
        ),
        ngon_affine_num_elements=(
            0
            if ngon_affine_system is None
            else ngon_affine_system.num_regularized_elements
        ),
        ngon_affine_num_modes=(
            0
            if ngon_affine_system is None
            else ngon_affine_system.num_hourglass_modes
        ),
        ngon_affine_maximum_warp_ratio=(
            0.0
            if ngon_affine_system is None
            else ngon_affine_system.maximum_warp_ratio
        ),
        final_reprojected_vertex_ids=final_reprojected_ids,
        final_nonconverged_vertex_ids=final_nonconverged_ids,
    )


def _stack_columns(values):
    if not values:
        raise ValueError("At least one component block is required.")
    return values[0] if len(values) == 1 else csdl.concatenate(values, axis=1)


def _stack_block_rows(values, groups):
    """Place each component's free correction in its own 3-column block."""
    blocks = []
    for group in groups:
        block = csdl.Variable(value=np.zeros(values.shape, dtype=float))
        block = block.set(
            _row_slice(group.free_rows),
            values[_row_slice(group.free_rows)],
        )
        blocks.append(block)
    return _stack_columns(tuple(blocks))


def _stack_scalar_block_rows(values, groups):
    """Place each component's scalar correction in its own column."""
    blocks = []
    for group in groups:
        block = csdl.Variable(value=np.zeros(values.shape, dtype=float))
        block = block.set(
            _row_slice(group.free_rows),
            values[_row_slice(group.free_rows)],
        )
        blocks.append(block)
    return _stack_columns(tuple(blocks))


def _set_exact_seams(values, solutions, row_by_id):
    output = values
    for solution in solutions:
        if solution.vertex_ids is None:
            continue
        selected_solution_rows = []
        output_rows = []
        for solution_row, vertex_id in enumerate(solution.vertex_ids):
            output_row = row_by_id.get(int(vertex_id))
            if output_row is not None:
                selected_solution_rows.append(solution_row)
                output_rows.append(output_row)
        if output_rows:
            output = output.set(
                _row_slice(np.asarray(output_rows, dtype=np.int64)),
                solution.deformed_vertices[
                    _row_slice(
                        np.asarray(selected_solution_rows, dtype=np.int64)
                    )
                ],
            )
    return output


def _solution_vertex_ids(solutions) -> np.ndarray:
    """Return the unique mesh IDs governed by exact intersection solves."""
    blocks = [
        np.asarray(solution.vertex_ids, dtype=np.int64).reshape(-1)
        for solution in solutions
        if solution.vertex_ids is not None
    ]
    if not blocks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(blocks)).astype(np.int64)


def _row_slice(rows):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, :]
    return csdl.slice[rows.tolist(), :]


__all__ = [
    "GraphLoadStepResult",
    "linear_load_fractions",
    "run_graph_load_steps",
]
