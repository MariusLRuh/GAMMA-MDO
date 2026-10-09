"""Component intersection node detection for surface meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh_io import _as_mesh_data


@dataclass(frozen=True)
class _ProjectionResult:
    """Closest-projection data for one component evaluated at mesh vertices."""

    distances: np.ndarray
    patch_ids: np.ndarray
    uv: np.ndarray


def identify_intersection_vertices(
    *,
    components,
    driving_component,
    mesh,
    intersection_tolerance: float = 1e-4,
    symmetry_mode: str = "auto",
    symmetry_axis: int | None = None,
    symmetry_plane_tolerance: float = 1e-8,
    projection_cache: dict | None = None,
    projection_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify mesh vertices that lie on a component intersection.

    Parameters
    ----------
    components : sequence
        Components whose common intersection should be detected. Each component
        is projected onto every mesh vertex, and vertices within
        ``intersection_tolerance`` of every component are selected.
    driving_component : object
        Component used to return parametric coordinates. It must be one of the
        entries in ``components``. Returned coordinates have shape ``(n, 3)`` as
        ``[patch_id, u, v]``, matching the FunctionSet evaluation convention
        used by the deformation pipeline.
    mesh : MeshData, mesh-like, str, or Path
        Surface mesh or mesh path.
    intersection_tolerance : float, optional
        Maximum closest-surface distance for a vertex to be considered on every
        component in the intersection.
    symmetry_mode : {"auto", "none", "positive", "negative"}, optional
        Controls which mesh vertices are projected. ``"auto"`` projects only
        the source half of meshes marked with ``metadata["symmetric"]`` and
        otherwise projects the full mesh. ``"positive"`` and ``"negative"``
        force one side of the symmetry plane. ``"none"`` projects all vertices.
    symmetry_axis : int, optional
        Coordinate axis normal to the symmetry plane. Defaults to mesh metadata
        ``"symmetry_axis"`` when available, otherwise the y-axis.
    symmetry_plane_tolerance : float, optional
        Distance tolerance used to include vertices on the symmetry plane in
        either half mesh.
    projection_cache : dict, optional
        Mutable cache reused across calls. Passing the same cache avoids
        repeated component-to-mesh projections for workflows that identify many
        component intersections on the same mesh.
    projection_options : dict, optional
        Options passed to this repo's ``FunctionSetProjectionModel`` for
        FunctionSet-like components.

    Returns
    -------
    parametric_coordinates : ndarray of shape (n, 3)
        Driving-component coordinates as ``[patch_id, u, v]``.
    vertices : ndarray of shape (n, 3)
        Mesh vertex coordinates selected as intersection nodes.
    vertex_ids : ndarray of shape (n,)
        Zero-based mesh vertex indices.

    Notes
    -----
    The algorithm projects the selected mesh vertices to each requested
    component once, then intersects the near-surface masks. For symmetric meshes
    this avoids projecting mirrored duplicates while still returning original
    zero-based vertex IDs from the full mesh.
    """
    component_list = _normalize_components(components)
    _require_driving_component(component_list, driving_component)
    tolerance = _validate_intersection_tolerance(intersection_tolerance)
    mesh_data = _as_mesh_data(mesh)
    all_vertices = np.asarray(mesh_data.vertices, dtype=float).reshape((-1, 3))
    projection_vertex_ids = _select_projection_vertex_ids(
        mesh_data,
        all_vertices,
        symmetry_mode=symmetry_mode,
        symmetry_axis=symmetry_axis,
        symmetry_plane_tolerance=symmetry_plane_tolerance,
    )
    vertices = all_vertices[projection_vertex_ids]
    mesh_cache_token = _array_cache_token(all_vertices)
    selection_cache_token = projection_vertex_ids.tobytes()

    projection_results = {
        id(component): _project_component_to_vertices(
            component,
            vertices,
            mesh_cache_token=mesh_cache_token,
            selection_cache_token=selection_cache_token,
            projection_cache=projection_cache,
            projection_options=projection_options,
        )
        for component in _unique_components(component_list)
    }

    intersection_mask = np.ones(vertices.shape[0], dtype=bool)
    for component in component_list:
        distances = projection_results[id(component)].distances
        intersection_mask &= np.isfinite(distances) & (np.abs(distances) <= tolerance)

    local_vertex_ids = np.where(intersection_mask)[0].astype(np.int64)
    vertex_ids = projection_vertex_ids[local_vertex_ids].astype(np.int64)
    driving_projection = projection_results[id(driving_component)]
    parametric_coordinates = np.column_stack(
        [
            driving_projection.patch_ids[local_vertex_ids].astype(float),
            driving_projection.uv[local_vertex_ids],
        ]
    )
    return parametric_coordinates, all_vertices[vertex_ids], vertex_ids


def _normalize_components(components) -> list:
    if components is None:
        raise ValueError("components must contain at least two component objects.")
    component_list = list(components)
    if len(component_list) < 2:
        raise ValueError("components must contain at least two component objects.")
    return component_list


def _require_driving_component(components: list, driving_component) -> None:
    if not any(component is driving_component for component in components):
        raise ValueError("driving_component must be one of the objects in components.")


def _validate_intersection_tolerance(intersection_tolerance: float) -> float:
    tolerance = float(intersection_tolerance)
    if tolerance < 0.0:
        raise ValueError("intersection_tolerance must be non-negative.")
    return tolerance


def _select_projection_vertex_ids(
    mesh_data,
    vertices: np.ndarray,
    *,
    symmetry_mode: str,
    symmetry_axis: int | None,
    symmetry_plane_tolerance: float,
) -> np.ndarray:
    metadata = dict(mesh_data.metadata or {})
    mode = str(symmetry_mode).lower()
    if mode not in {"auto", "none", "positive", "negative"}:
        raise ValueError(
            "symmetry_mode must be one of 'auto', 'none', 'positive', or "
            f"'negative'. Got {symmetry_mode!r}."
        )
    if mode == "auto":
        if not bool(metadata.get("symmetric", False)):
            mode = "none"
        else:
            mode = str(metadata.get("symmetry_selected_side", "positive")).lower()
            if mode not in {"positive", "negative"}:
                mode = "positive"
    if mode == "none":
        return np.arange(vertices.shape[0], dtype=np.int64)

    axis = int(metadata.get("symmetry_axis", 1) if symmetry_axis is None else symmetry_axis)
    if axis < 0 or axis >= vertices.shape[1]:
        raise ValueError(f"symmetry_axis must be in [0, {vertices.shape[1] - 1}]. Got {axis}.")
    tolerance = float(symmetry_plane_tolerance)
    if tolerance < 0.0:
        raise ValueError("symmetry_plane_tolerance must be non-negative.")

    coordinates = vertices[:, axis]
    if mode == "positive":
        mask = coordinates >= -tolerance
    else:
        mask = coordinates <= tolerance
    return np.where(mask)[0].astype(np.int64)


def _unique_components(components: list) -> list:
    seen_ids = set()
    unique = []
    for component in components:
        component_id = id(component)
        if component_id in seen_ids:
            continue
        seen_ids.add(component_id)
        unique.append(component)
    return unique


def _array_cache_token(array: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    array = np.asarray(array)
    return (
        int(array.__array_interface__["data"][0]),
        tuple(array.shape),
        tuple(array.strides),
    )


def _project_component_to_vertices(
    component,
    vertices: np.ndarray,
    *,
    mesh_cache_token: int,
    selection_cache_token: bytes,
    projection_cache: dict | None,
    projection_options: dict[str, Any] | None,
) -> _ProjectionResult:
    cache_key = (id(component), mesh_cache_token, selection_cache_token, vertices.shape)
    if projection_cache is not None and cache_key in projection_cache:
        return projection_cache[cache_key]

    if hasattr(component, "functions"):
        result = _project_with_function_set_projection_model(
            component,
            vertices,
            projection_options=projection_options,
        )
    else:
        result = _project_with_native_component_method(component, vertices)

    if projection_cache is not None:
        projection_cache[cache_key] = result
    return result


def _project_with_native_component_method(component, vertices: np.ndarray) -> _ProjectionResult | None:
    project = getattr(component, "project", None)
    if project is None:
        raise TypeError(
            "Non-FunctionSet components must expose a project(points) method "
            "returning (distances, state). FunctionSet-like components must "
            "expose a .functions mapping and are projected with GAMMA's "
            "FunctionSetProjectionModel."
        )

    for args, kwargs in (((vertices,), {}), ((), {"points": vertices})):
        try:
            projection_output = project(*args, **kwargs)
        except TypeError:
            continue
        return _normalize_projection_output(projection_output, vertices.shape[0])
    return None


def _project_with_function_set_projection_model(
    component,
    vertices: np.ndarray,
    *,
    projection_options: dict[str, Any] | None,
) -> _ProjectionResult:
    if not hasattr(component, "functions"):
        raise TypeError("Component must expose a .functions mapping.")

    try:
        FunctionSetProjectionModel, stack_function_set_coefficients = (
            _repo_projection_model_api()
        )
    except Exception as exc:  # pragma: no cover - depends on optional geometry stack.
        raise ImportError(
            "identify_intersection_vertices requires the projection stack for "
            "FunctionSet-like components."
        ) from exc

    options = dict(projection_options or {})
    model = FunctionSetProjectionModel(component, **options)
    coefficients = stack_function_set_coefficients(component, model.patch_ids)
    projection_output = model.project(coefficients, vertices)
    return _normalize_projection_output(projection_output, vertices.shape[0])


def _repo_projection_model_api():
    """Return GAMMA's NumPy projection model API, never an LFS projection class."""
    try:
        import gamma_mdo

        return (
            gamma_mdo.FunctionSetProjectionModel,
            _import_stack_function_set_coefficients(),
        )
    except Exception:
        from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
            FunctionSetProjectionModel,
        )

        return FunctionSetProjectionModel, _import_stack_function_set_coefficients()


def _import_stack_function_set_coefficients():
    from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
        stack_function_set_coefficients,
    )

    return stack_function_set_coefficients


def _normalize_projection_output(projection_output, num_vertices: int) -> _ProjectionResult:
    if not isinstance(projection_output, tuple) or len(projection_output) != 2:
        raise TypeError(
            "Component projection must return a tuple of (distances, state)."
        )

    distances, state = projection_output
    distances = np.asarray(distances, dtype=float).reshape(-1)
    if distances.shape[0] != num_vertices:
        raise ValueError(
            f"Projection returned {distances.shape[0]} distances for "
            f"{num_vertices} mesh vertices."
        )
    if not isinstance(state, dict):
        raise TypeError("Projection state must be a dict with 'patch_id' and 'uv' entries.")
    if "patch_id" not in state or "uv" not in state:
        raise ValueError("Projection state must contain 'patch_id' and 'uv' entries.")

    patch_ids = np.asarray(state["patch_id"], dtype=np.int64).reshape(-1)
    uv = np.asarray(state["uv"], dtype=float).reshape((-1, 2))
    if patch_ids.shape[0] != num_vertices or uv.shape[0] != num_vertices:
        raise ValueError(
            "Projection state entries 'patch_id' and 'uv' must have one row "
            "per mesh vertex."
        )

    return _ProjectionResult(distances=distances, patch_ids=patch_ids, uv=uv)
