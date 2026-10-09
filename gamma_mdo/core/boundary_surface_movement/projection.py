"""Component-restricted differentiable projection and reevaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
)
from gamma_mdo.core.projections.function_set_evaluation_custom_op import (
    FunctionSetEvaluationModel,
    FunctionSetEvaluationOperation,
)
from gamma_mdo.core.projections.function_set_projection_custom_op import (
    FunctionSetProjectionOperation,
)
from gamma_mdo.preprocessing import (
    ProjectionMetadata,
    VertexEvaluationMetadata,
    project_mesh_onto_geometry,
)
from gamma_mdo.preprocessing.mesh_io import _as_mesh_data

from .geometry import component_patch_ids, stack_component_coefficients


@dataclass(frozen=True)
class VertexBatch:
    """Pair a CSDL vertex array with its original global mesh IDs.

    Parameters
    ----------
    values
        Vertex-coordinate rows owned by the batch.
    vertex_ids
        Global mesh indices aligned with ``values``.
    num_mesh_vertices
        Total number of vertices in the complete mesh.
    converged
        Optional Boolean convergence flags aligned with ``values``. This is
        populated for closest-point projection batches and absent for exact
        parametric reevaluation batches.
    """

    values: csdl.Variable
    vertex_ids: np.ndarray
    num_mesh_vertices: int
    converged: np.ndarray | None = None


def project_onto_oml(
    *,
    mesh_vertices=None,
    geometry=None,
    deformed_mesh_vertices=None,
    deformed_mesh_vertex_ids=None,
    projection_metadata: Sequence[ProjectionMetadata] | None = None,
    component_coefficients: Mapping[object, object] | None = None,
    projection_options: dict | None = None,
):
    """Project initial or deformed vertices onto the OML.

    The ``mesh_vertices`` form is a setup-time NumPy projection returning
    parametric coordinates.  The ``deformed_mesh_vertices`` form is a
    differentiable CSDL operation and requires explicit component ownership
    metadata.

    Parameters
    ----------
    mesh_vertices
        Optional setup-time coordinate array.
    geometry
        Geometry used by setup-time projection.
    deformed_mesh_vertices
        Optional differentiable coordinates to reproject.
    deformed_mesh_vertex_ids
        Global IDs aligned with ``deformed_mesh_vertices``.
    projection_metadata
        Component ownership and patch restrictions for differentiable
        projection.
    component_coefficients
        Optional deformed coefficient overrides keyed by component.
    projection_options
        Options forwarded to the projection implementation.

    Returns
    -------
    object or VertexBatch
        Setup metadata for the NumPy form, or projected CSDL vertices paired
        with global IDs for the differentiable form.

    Raises
    ------
    ValueError
        If the selected call form is incomplete or metadata does not cover all
        supplied vertices.
    """
    if mesh_vertices is not None:
        if deformed_mesh_vertices is not None:
            raise ValueError(
                "Use either setup-time or differentiable projection arguments."
            )
        if geometry is None:
            raise ValueError("geometry is required for setup-time projection.")
        return project_mesh_onto_geometry(
            mesh_vertices=mesh_vertices,
            geometry=geometry,
            projection_options=projection_options,
        )

    if deformed_mesh_vertices is None:
        raise ValueError("deformed_mesh_vertices is required.")
    ids = np.asarray(deformed_mesh_vertex_ids, dtype=np.int64).reshape(-1)
    if ids.size != deformed_mesh_vertices.shape[0]:
        raise ValueError(
            "deformed_mesh_vertex_ids must align with deformed_mesh_vertices."
        )
    metadata_list = list(projection_metadata or ())
    if not metadata_list:
        raise ValueError("projection_metadata must contain at least one group.")

    coefficient_map = dict(component_coefficients or {})
    local_row_by_id = {int(vertex_id): row for row, vertex_id in enumerate(ids)}
    projected = deformed_mesh_vertices * 1.0
    assigned = np.zeros(ids.size, dtype=bool)
    converged = np.zeros(ids.size, dtype=bool)

    for metadata in metadata_list:
        component = metadata.component
        coefficients = _component_mapping_get(
            coefficient_map,
            component,
            stack_component_coefficients(component),
        )
        global_ids = np.asarray(metadata.vertex_ids, dtype=np.int64)
        local_rows = np.asarray(
            [
                local_row_by_id[int(vertex_id)]
                for vertex_id in global_ids
                if int(vertex_id) in local_row_by_id
            ],
            dtype=np.int64,
        )
        if local_rows.size == 0:
            continue
        selected_global_ids = ids[local_rows]
        source_points = deformed_mesh_vertices[_row_slice(local_rows)]

        if metadata.parent_patch_ids is None:
            group_patch_ids = tuple(int(item) for item in metadata.allowed_patch_ids)
            # FunctionSetProjectionModel lays out its coefficient offsets over the
            # *selected* patches only (start=0 at the first selected patch), so a
            # strict subset must be handed the matching coefficient subset -- the
            # full component stack would misalign every patch and project onto the
            # wrong surface.
            group_coefficients = (
                coefficients
                if len(group_patch_ids) == len(component_patch_ids(component))
                else _coefficient_subset(component, coefficients, group_patch_ids)
            )
            projected_points, group_converged = _project_group(
                component,
                group_coefficients,
                source_points,
                patch_ids=group_patch_ids,
                projection_options=projection_options,
            )
            projected = projected.set(
                _row_slice(local_rows),
                projected_points,
            )
            converged[local_rows] = group_converged
        else:
            parent_by_id = {
                int(vertex_id): int(patch_id)
                for vertex_id, patch_id in zip(
                    metadata.vertex_ids,
                    metadata.parent_patch_ids,
                )
            }
            parent_patches = np.asarray(
                [parent_by_id[int(vertex_id)] for vertex_id in selected_global_ids],
                dtype=np.int64,
            )
            fixed_coordinates = None
            if metadata.fixed_parametric_coordinates is not None:
                fixed_by_id = {
                    int(vertex_id): values
                    for vertex_id, values in zip(
                        metadata.vertex_ids,
                        metadata.fixed_parametric_coordinates,
                    )
                }
                fixed_coordinates = np.asarray(
                    [fixed_by_id[int(vertex_id)] for vertex_id in selected_global_ids],
                    dtype=float,
                )
            chordwise_reference = None
            chordwise_weight = None
            if metadata.chordwise_reference is not None:
                reference_by_id = dict(zip(
                    metadata.vertex_ids, metadata.chordwise_reference
                ))
                weight_by_id = dict(zip(
                    metadata.vertex_ids, metadata.chordwise_restore_weight
                ))
                chordwise_reference = np.asarray(
                    [reference_by_id[int(vertex_id)] for vertex_id in selected_global_ids],
                    dtype=float,
                )
                chordwise_weight = np.asarray(
                    [weight_by_id[int(vertex_id)] for vertex_id in selected_global_ids],
                    dtype=float,
                )
            for patch_id in np.unique(parent_patches):
                group_local = np.where(parent_patches == patch_id)[0]
                output_rows = local_rows[group_local]
                patch_points = source_points[_row_slice(group_local)]
                patch_coefficients = _coefficient_subset(
                    component,
                    coefficients,
                    (int(patch_id),),
                )
                if fixed_coordinates is None and chordwise_reference is None:
                    projected_points, group_converged = _project_group(
                        component,
                        patch_coefficients,
                        patch_points,
                        patch_ids=(int(patch_id),),
                        projection_options=projection_options,
                    )
                else:
                    projected_coordinates, group_converged = _project_group(
                        component,
                        patch_coefficients,
                        patch_points,
                        patch_ids=(int(patch_id),),
                        projection_options=projection_options,
                        return_parametric=True,
                    )
                    if fixed_coordinates is not None:
                        group_fixed = fixed_coordinates[group_local]
                        for axis in (0, 1):
                            fixed_rows = np.where(np.isfinite(group_fixed[:, axis]))[0]
                            if fixed_rows.size:
                                projected_coordinates = projected_coordinates.set(
                                    _coordinate_slice(fixed_rows, axis + 1),
                                    group_fixed[fixed_rows, axis].reshape((-1, 1)),
                                )
                    if chordwise_reference is not None:
                        weight = chordwise_weight[group_local].reshape((-1, 1))
                        reference = chordwise_reference[group_local].reshape((-1, 1))
                        projected_v = projected_coordinates[csdl.slice[:, 2:3]]
                        restored_v = projected_v * (1.0 - weight) + weight * reference
                        projected_coordinates = projected_coordinates.set(
                            csdl.slice[:, 2:3], restored_v
                        )
                    projected_points = FunctionSetEvaluationOperation(
                        FunctionSetEvaluationModel(component)
                    ).evaluate(coefficients, projected_coordinates)
                projected = projected.set(
                    _row_slice(output_rows),
                    projected_points,
                )
                converged[output_rows] = group_converged
        assigned[local_rows] = True

    if np.any(~assigned):
        missing = ids[~assigned]
        raise ValueError(
            f"{missing.size} deformed vertices have no projection metadata; "
            f"first IDs: {missing[:10].tolist()}."
        )
    return VertexBatch(
        values=projected,
        vertex_ids=ids,
        num_mesh_vertices=int(np.max(ids) + 1) if ids.size else 0,
        converged=converged,
    )


def reevaluate_vertices(
    *,
    mesh,
    metadata: Sequence[VertexEvaluationMetadata],
    component_coefficients: Mapping[object, object] | None = None,
) -> VertexBatch:
    """Reevaluate fixed parametric coordinates on deformed components.

    Parameters
    ----------
    mesh
        Complete mesh used to determine the global vertex count.
    metadata
        Component, parametric-coordinate, and global-ID groups to evaluate.
    component_coefficients
        Optional deformed coefficient overrides keyed by component.

    Returns
    -------
    VertexBatch
        Reevaluated CSDL coordinates paired with their global IDs.
    """
    mesh_data = _as_mesh_data(mesh)
    coefficient_map = dict(component_coefficients or {})
    groups = []
    ids = []
    for item in metadata:
        model = FunctionSetEvaluationModel(item.component)
        coefficients = _component_mapping_get(
            coefficient_map,
            item.component,
            stack_component_coefficients(item.component),
        )
        groups.append(
            FunctionSetEvaluationOperation(model).evaluate(
                coefficients,
                item.parametric_coordinates,
            )
        )
        ids.append(np.asarray(item.vertex_ids, dtype=np.int64))

    if not groups:
        values = csdl.Variable(value=np.empty((0, 3), dtype=float))
        combined_ids = np.empty((0,), dtype=np.int64)
    else:
        values = groups[0] if len(groups) == 1 else csdl.vstack(tuple(groups))
        combined_ids = np.concatenate(ids)
    return VertexBatch(
        values=values,
        vertex_ids=combined_ids,
        num_mesh_vertices=mesh_data.vertices.shape[0],
    )


def combine_vertices(
    *,
    oml_projected_vertices: VertexBatch,
    reevaluated_mesh_vertices: VertexBatch,
) -> csdl.Variable:
    """Assemble projected and reevaluated batches in original mesh order.

    Parameters
    ----------
    oml_projected_vertices
        Vertices moved by closest-point OML projection.
    reevaluated_mesh_vertices
        Vertices moved by exact parametric reevaluation.

    Returns
    -------
    csdl.Variable
        Complete coordinate array in global mesh order.

    Raises
    ------
    ValueError
        If batches overlap or leave any global vertex uncovered.
    """
    num_vertices = max(
        oml_projected_vertices.num_mesh_vertices,
        reevaluated_mesh_vertices.num_mesh_vertices,
    )
    output = csdl.Variable(value=np.zeros((num_vertices, 3), dtype=float))
    covered = np.zeros(num_vertices, dtype=bool)
    for batch in (oml_projected_vertices, reevaluated_mesh_vertices):
        ids = np.asarray(batch.vertex_ids, dtype=np.int64)
        if np.any(covered[ids]):
            overlap = ids[covered[ids]]
            raise ValueError(f"Vertex batches overlap at IDs {overlap[:10].tolist()}.")
        output = output.set(_row_slice(ids), batch.values)
        covered[ids] = True
    if not np.all(covered):
        missing = np.where(~covered)[0]
        raise ValueError(f"Vertex batches do not cover IDs {missing[:10].tolist()}.")
    return output


def _project_group(
    component,
    coefficients,
    points,
    *,
    patch_ids,
    projection_options,
    return_parametric=False,
):
    options = dict(projection_options or {})
    options.setdefault("warm_start_nu", 30)
    options.setdefault("warm_start_nv", 30)
    model = FunctionSetProjectionModel(
        component,
        patch_indices=patch_ids,
        **options,
    )
    operation = FunctionSetProjectionOperation(
        model,
        return_parametric=return_parametric,
    )
    output = operation.evaluate(coefficients, points)
    try:
        converged = np.asarray(
            operation.shared_state["forward"]["converged"], dtype=bool
        ).reshape(-1)
    except KeyError as error:
        raise RuntimeError(
            "Projection convergence is unavailable. Use an inline CSDL "
            "recorder when building the mesh-motion pipeline."
        ) from error
    return output, converged


def _coefficient_subset(component, coefficients, selected_patch_ids):
    selected = set(int(item) for item in selected_patch_ids)
    blocks = []
    offset = 0
    for patch_id in component_patch_ids(component):
        shape = component.functions[patch_id].coefficients.shape
        count = int(np.prod(shape[:-1]))
        if patch_id in selected:
            row_slice = (
                slice(offset, offset + count)
                if isinstance(coefficients, np.ndarray)
                else csdl.slice[offset : offset + count, :]
            )
            blocks.append(coefficients[row_slice])
        offset += count
    if not blocks:
        raise ValueError("No selected patch coefficients.")
    if len(blocks) == 1:
        return blocks[0]
    if isinstance(coefficients, np.ndarray):
        return np.vstack(blocks)
    return csdl.vstack(tuple(blocks))


def _component_mapping_get(mapping, component, default):
    try:
        return mapping.get(component, default)
    except TypeError:
        return mapping.get(id(component), default)


def _row_slice(rows):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, :]
    return csdl.slice[rows.tolist(), :]


def _coordinate_slice(rows, column):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, column : column + 1]
    return csdl.slice[rows.tolist(), column : column + 1]


__all__ = [
    "VertexBatch",
    "combine_vertices",
    "project_onto_oml",
    "reevaluate_vertices",
]
