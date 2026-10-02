"""Setup-time preprocessing for differentiable boundary-surface movement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .intersections import _project_component_to_vertices
from .mesh_io import _as_mesh_data


@dataclass(frozen=True)
class ProjectionMetadata:
    """Projection restrictions for a group of mesh vertices.

    ``parent_patch_ids`` is populated for thin lifting surfaces, where every
    vertex must project back to its original B-spline patch.  When it is
    ``None``, projection may select any patch in ``allowed_patch_ids``.
    Finite entries in ``fixed_parametric_coordinates`` preserve a parent-patch
    boundary coordinate while allowing the other coordinate to slide.
    ``chordwise_reference`` and ``chordwise_restore_weight`` blend the
    projected second parametric coordinate with a fixed baseline value; the
    latter must lie in [0, 1].
    """

    component: object
    vertex_ids: np.ndarray
    allowed_patch_ids: tuple[int, ...]
    parent_patch_ids: np.ndarray | None = None
    fixed_parametric_coordinates: np.ndarray | None = None
    chordwise_reference: np.ndarray | None = None
    chordwise_restore_weight: np.ndarray | None = None
    name: str | None = None


@dataclass(frozen=True)
class VertexEvaluationMetadata:
    """Fixed parametric coordinates for reevaluated mesh vertices."""

    component: object
    vertex_ids: np.ndarray
    parametric_coordinates: np.ndarray
    name: str | None = None


@dataclass(frozen=True)
class ComponentProjectionData:
    """Closest setup projection for one component."""

    component: object
    distances: np.ndarray
    parametric_coordinates: np.ndarray


def project_mesh_onto_components(
    *,
    mesh_vertices,
    components: Sequence,
    projection_cache: dict | None = None,
    projection_options: dict | None = None,
) -> tuple[ComponentProjectionData, ...]:
    """Project vertices independently to each component.

    Independent component projection is more robust than a single union
    projection at small or degenerate features because every component keeps
    its own warm-start mesh.  The result also provides the distance matrix
    needed for deterministic component ownership.
    """
    vertices = _as_numpy_vertices(mesh_vertices)
    selection_token = np.arange(vertices.shape[0], dtype=np.int64).tobytes()
    output = []
    for component in components:
        result = _project_component_to_vertices(
            component,
            vertices,
            mesh_cache_token=_array_cache_token(vertices),
            selection_cache_token=selection_token,
            projection_cache=projection_cache,
            projection_options=projection_options,
        )
        output.append(
            ComponentProjectionData(
                component=component,
                distances=np.asarray(result.distances, dtype=float),
                parametric_coordinates=np.column_stack(
                    (result.patch_ids.astype(float), result.uv)
                ),
            )
        )
    return tuple(output)


def project_mesh_onto_geometry(
    *,
    mesh_vertices,
    geometry,
    projection_options: dict | None = None,
) -> np.ndarray:
    """Project initial mesh vertices and return ``[patch_id, u, v]`` rows.

    This setup-time operation intentionally runs in NumPy.  The returned
    coordinates are constants in the CSDL graph and are later used for exact
    reevaluation of vertices outside deformation regions.
    """
    vertices = _as_numpy_vertices(mesh_vertices)
    result = _project_component_to_vertices(
        geometry,
        vertices,
        mesh_cache_token=id(vertices),
        selection_cache_token=np.arange(vertices.shape[0], dtype=np.int64).tobytes(),
        projection_cache=None,
        projection_options=projection_options,
    )
    return np.column_stack((result.patch_ids.astype(float), result.uv))


def identify_vertices_by_components(
    *,
    mesh,
    components: Sequence,
    intersection_vertices: Sequence[np.ndarray] | None = None,
    projection_options: dict | None = None,
) -> tuple[np.ndarray, ...]:
    """Assign every mesh vertex to its closest component.

    Intersection nodes are assigned deterministically to the first component
    in ``components`` that is tied for the minimum distance.  Exact
    intersection handling later supersedes this ownership choice.
    """
    mesh_data = _as_mesh_data(mesh)
    vertices = np.asarray(mesh_data.vertices, dtype=float)
    component_list = list(components)
    if not component_list:
        raise ValueError("components must contain at least one component.")

    distances = []
    for component in component_list:
        result = _project_component_to_vertices(
            component,
            vertices,
            mesh_cache_token=id(mesh_data.vertices),
            selection_cache_token=np.arange(vertices.shape[0], dtype=np.int64).tobytes(),
            projection_cache=None,
            projection_options=projection_options,
        )
        distances.append(np.abs(result.distances))
    ownership = np.argmin(np.column_stack(distances), axis=1)
    return tuple(vertices[ownership == index] for index in range(len(component_list)))


def identify_deformation_vertices(
    *,
    component,
    component_vertices,
    intersection_vertices,
    mesh=None,
    span_fraction: float | None = None,
    length_fraction: float | None = None,
    axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select component vertices in a coordinate envelope around intersections.

    Exactly one of ``span_fraction`` or ``length_fraction`` must be supplied.
    The default axis is inferred from the component control-point extent:
    span uses the largest transverse (y/z) extent, while length uses the
    largest overall extent.
    """
    if (span_fraction is None) == (length_fraction is None):
        raise ValueError("Provide exactly one of span_fraction or length_fraction.")
    fraction = float(span_fraction if span_fraction is not None else length_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("deformation fraction must lie in [0, 1].")

    vertices = np.asarray(component_vertices, dtype=float).reshape((-1, 3))
    intersections = _concatenate_vertices(intersection_vertices)
    if intersections.size == 0 or vertices.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int64)

    component_points = _component_control_points(component)
    extents = np.ptp(component_points, axis=0)
    if axis is None:
        if span_fraction is not None:
            axis = 1 + int(np.argmax(extents[1:]))
        else:
            axis = int(np.argmax(extents))
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    radius = fraction * float(extents[axis])
    coordinate_distance = np.min(
        np.abs(vertices[:, axis, None] - intersections[None, :, axis]),
        axis=1,
    )
    selected = vertices[coordinate_distance <= radius + 1e-12]

    if mesh is None:
        # Backward-compatible fallback for callers whose component_vertices are
        # already ordered mesh subsets.
        selected_ids = np.where(coordinate_distance <= radius + 1e-12)[0]
        return selected, selected_ids.astype(np.int64)
    vertex_ids = _vertex_ids_from_coordinates(_as_mesh_data(mesh).vertices, selected)
    order = np.argsort(vertex_ids)
    return selected[order], vertex_ids[order]


def get_projection_metadata(
    *,
    component,
    vertices,
    vertex_ids,
    para_coords: np.ndarray | None,
    mesh=None,
    name: str | None = None,
    allowed_patch_ids: Sequence[int] | None = None,
    fix_patch_boundaries: bool = False,
    patch_boundary_tolerance: float = 1e-6,
    chordwise_reference: np.ndarray | None = None,
    chordwise_restore_weight: np.ndarray | None = None,
) -> ProjectionMetadata:
    """Build component/patch restrictions for differentiable projection.

    ``allowed_patch_ids`` narrows projection to a subset of the component's
    patches (e.g. one surface side); combined with ``para_coords=None`` this
    lets a node slide between those patches but never leave the subset.
    When supplied together, ``chordwise_reference`` and
    ``chordwise_restore_weight`` restore a constant fraction of each vertex's
    baseline second parametric coordinate after projection.
    """
    del vertices, mesh  # Included for a readable call site and future checks.
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    component_patches = tuple(sorted(int(key) for key in component.functions))
    if allowed_patch_ids is None:
        allowed_patch_ids = component_patches
    else:
        allowed_patch_ids = tuple(sorted(int(key) for key in allowed_patch_ids))
        unknown = sorted(set(allowed_patch_ids) - set(component_patches))
        if unknown:
            raise ValueError(
                f"allowed_patch_ids {unknown} are not patches of this component."
            )
    parent_patch_ids = None
    fixed_parametric_coordinates = None
    if para_coords is not None:
        coordinates = np.asarray(para_coords, dtype=float).reshape((-1, 3))
        if ids.size and int(np.max(ids)) >= coordinates.shape[0]:
            raise ValueError("vertex_ids index beyond para_coords.")
        parent_patch_ids = np.rint(coordinates[ids, 0]).astype(np.int64)
        invalid = sorted(set(parent_patch_ids.tolist()) - set(allowed_patch_ids))
        if invalid:
            raise ValueError(
                f"Initial parent patches {invalid} are not part of the requested component."
            )
        if fix_patch_boundaries:
            uv = coordinates[ids, 1:3]
            fixed_parametric_coordinates = np.full_like(uv, np.nan)
            fixed_parametric_coordinates[uv <= patch_boundary_tolerance] = 0.0
            fixed_parametric_coordinates[
                uv >= 1.0 - patch_boundary_tolerance
            ] = 1.0
            if np.any(np.all(np.isnan(fixed_parametric_coordinates), axis=1)):
                raise ValueError(
                    "fix_patch_boundaries requires every metadata row to lie "
                    "on at least one parametric patch boundary."
                )
    elif fix_patch_boundaries:
        raise ValueError("fix_patch_boundaries requires para_coords.")
    if (chordwise_reference is None) != (chordwise_restore_weight is None):
        raise ValueError("Chordwise reference and restore weight must be supplied together.")
    if chordwise_reference is not None:
        if parent_patch_ids is None:
            raise ValueError("Chordwise restoration requires parent parametric coordinates.")
        chordwise_reference = np.asarray(chordwise_reference, dtype=float).reshape(-1)
        chordwise_restore_weight = np.asarray(chordwise_restore_weight, dtype=float).reshape(-1)
        if chordwise_reference.size != ids.size or chordwise_restore_weight.size != ids.size:
            raise ValueError("Chordwise restoration arrays must align with vertex_ids.")
        if (not np.all(np.isfinite(chordwise_reference)) or
                not np.all(np.isfinite(chordwise_restore_weight)) or
                np.any(chordwise_restore_weight < 0.0) or
                np.any(chordwise_restore_weight > 1.0)):
            raise ValueError("Chordwise restoration values must be finite; weights must be in [0, 1].")
    return ProjectionMetadata(
        component=component,
        vertex_ids=ids,
        allowed_patch_ids=allowed_patch_ids,
        parent_patch_ids=parent_patch_ids,
        fixed_parametric_coordinates=fixed_parametric_coordinates,
        chordwise_reference=chordwise_reference,
        chordwise_restore_weight=chordwise_restore_weight,
        name=name,
    )


def identify_reevaluated_vertices(
    *,
    mesh,
    vertex_ids,
    parametric_coords,
    components: Sequence,
    names: Sequence[str] | None = None,
) -> tuple[VertexEvaluationMetadata, ...]:
    """Return component-grouped metadata for all non-deformation vertices."""
    mesh_data = _as_mesh_data(mesh)
    excluded = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    keep_mask = np.ones(mesh_data.vertices.shape[0], dtype=bool)
    keep_mask[excluded] = False
    kept_ids = np.where(keep_mask)[0].astype(np.int64)
    coordinates = np.asarray(parametric_coords, dtype=float).reshape((-1, 3))
    if coordinates.shape[0] != mesh_data.vertices.shape[0]:
        raise ValueError("parametric_coords must contain one row per mesh vertex.")

    component_list = list(components)
    if names is None:
        names = [None] * len(component_list)
    if len(names) != len(component_list):
        raise ValueError("names and components must have the same length.")

    metadata = []
    assigned = np.zeros(kept_ids.size, dtype=bool)
    kept_patch_ids = np.rint(coordinates[kept_ids, 0]).astype(np.int64)
    for component, name in zip(component_list, names):
        patch_ids = set(int(key) for key in component.functions)
        # Explicit dtype: np.asarray([]) is float64, which makes the
        # bitwise-and below raise when no vertex is reevaluated.
        local_mask = np.asarray(
            [patch_id in patch_ids for patch_id in kept_patch_ids],
            dtype=bool,
        )
        local_mask &= ~assigned
        local_ids = kept_ids[local_mask]
        if local_ids.size:
            metadata.append(
                VertexEvaluationMetadata(
                    component=component,
                    vertex_ids=local_ids,
                    parametric_coordinates=coordinates[local_ids],
                    name=name,
                )
            )
        assigned |= local_mask

    if np.any(~assigned):
        missing_ids = kept_ids[~assigned]
        missing_patches = sorted(set(kept_patch_ids[~assigned].tolist()))
        raise ValueError(
            f"{missing_ids.size} reevaluated vertices belong to unlisted patches "
            f"{missing_patches}."
        )
    return tuple(metadata)


def _as_numpy_vertices(vertices) -> np.ndarray:
    value = getattr(vertices, "value", vertices)
    return np.asarray(value, dtype=float).reshape((-1, 3))


def _array_cache_token(array: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    array = np.asarray(array)
    return (
        int(array.__array_interface__["data"][0]),
        tuple(array.shape),
        tuple(array.strides),
    )


def _component_control_points(component) -> np.ndarray:
    blocks = [
        np.asarray(component.functions[key].coefficients.value, dtype=float).reshape((-1, 3))
        for key in sorted(component.functions)
    ]
    return np.vstack(blocks)


def _concatenate_vertices(vertices) -> np.ndarray:
    if vertices is None:
        return np.empty((0, 3), dtype=float)
    if isinstance(vertices, np.ndarray):
        return np.asarray(vertices, dtype=float).reshape((-1, 3))
    blocks = [
        np.asarray(block, dtype=float).reshape((-1, 3))
        for block in vertices
        if np.asarray(block).size
    ]
    return np.vstack(blocks) if blocks else np.empty((0, 3), dtype=float)


def _vertex_ids_from_coordinates(
    mesh_vertices: np.ndarray,
    query_vertices: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    mesh_vertices = np.asarray(mesh_vertices, dtype=float).reshape((-1, 3))
    query_vertices = np.asarray(query_vertices, dtype=float).reshape((-1, 3))
    scale = max(float(tolerance), 1e-14)
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(mesh_vertices):
        key = tuple(np.rint(point / scale).astype(np.int64))
        lookup.setdefault(key, []).append(index)

    ids = []
    for point in query_vertices:
        key = tuple(np.rint(point / scale).astype(np.int64))
        candidates = lookup.get(key, [])
        if not candidates:
            raise ValueError("A requested component vertex is not present in the mesh.")
        distances = np.linalg.norm(mesh_vertices[candidates] - point, axis=1)
        closest = int(candidates[int(np.argmin(distances))])
        if float(np.min(distances)) > tolerance:
            raise ValueError("A requested component vertex does not match the mesh tolerance.")
        ids.append(closest)
    return np.asarray(ids, dtype=np.int64)


__all__ = [
    "ComponentProjectionData",
    "ProjectionMetadata",
    "VertexEvaluationMetadata",
    "get_projection_metadata",
    "identify_deformation_vertices",
    "identify_reevaluated_vertices",
    "identify_vertices_by_components",
    "project_mesh_onto_geometry",
    "project_mesh_onto_components",
]
