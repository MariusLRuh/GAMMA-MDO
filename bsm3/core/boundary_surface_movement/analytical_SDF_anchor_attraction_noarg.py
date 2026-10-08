from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Iterable, Tuple

import numpy as np

try:
    import pyvista as pv  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    pv = None  # type: ignore[assignment]
    _PYVISTA_IMPORT_ERROR = exc
else:
    _PYVISTA_IMPORT_ERROR = None


Vec3 = Tuple[float, float, float]


def _require_pyvista():
    if pv is None:  # pragma: no cover
        message = "PyVista is required for contour extraction and visualization."
        if _PYVISTA_IMPORT_ERROR is not None:
            raise ImportError(message) from _PYVISTA_IMPORT_ERROR
        raise ImportError(message)
    return pv


def _new_uniform_grid(pv_mod):
    if hasattr(pv_mod, "UniformGrid"):
        return pv_mod.UniformGrid()
    if hasattr(pv_mod, "ImageData"):
        return pv_mod.ImageData()
    raise AttributeError("PyVista installation does not expose UniformGrid or ImageData.")


def _split_bounds(bounds: Iterable[Vec3]) -> Tuple[float, float, float, float, float, float]:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    return xmin, xmax, ymin, ymax, zmin, zmax


@dataclass(frozen=True)
class AircraftGeometry:
    fuselage_center: Vec3 = (0.0, 0.0, 0.0)
    fuselage_radius: float = 0.7
    fuselage_half_length: float = 4.2
    nose_length: float = 1.2
    nose_tip_offset: float = 0.0
    wing_center_base: Vec3 = (0.5, 0.0, 0.0)
    wing_offset: float = 0.0
    wing_rotation_deg: float = 0.0
    wing_half_size: Vec3 = (0.35, 5.5, 0.08)
    htail_center: Vec3 = (-3.4, 0.0, 0.2)
    htail_half_size: Vec3 = (0.3, 2.0, 0.04)
    vtail_center: Vec3 = (-3.4, 0.0, 1.35)
    vtail_half_size: Vec3 = (0.45, 0.1, 1.0)


@dataclass(frozen=True)
class AnchorAttractionSettings:
    iterations: int = 5
    fd_epsilon: float = 2e-6
    projection_epsilon: float = 1e-3
    anchor_curvature_percentile: float = 96.0
    anchor_curvature_floor: float = 0.08
    attraction_decay_length: float = 0.9  # physical units
    attraction_support_radius: float = 2.0  # physical units
    max_anchors_per_vertex: int = 12
    attraction_strength: float = 1.0
    max_step_fraction: float = 0.18
    anchor_move_scale: float = 0.15
    tangential_smoothing_step: float = 0.06
    smoothing_every: int = 1
    use_anchor_local_maxima: bool = True
    anchor_maxima_tolerance: float = 1e-10


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=float)


def _wing_center(geometry: AircraftGeometry) -> np.ndarray:
    base = np.asarray(geometry.wing_center_base, dtype=float).copy()
    base[0] += float(geometry.wing_offset)
    return base


def _wing_rotation_matrix(geometry: AircraftGeometry) -> np.ndarray:
    return rotation_matrix_y(np.radians(float(geometry.wing_rotation_deg)))


def sdf_capped_cylinder(points: np.ndarray, *, center: Vec3, radius: float, half_length: float) -> np.ndarray:
    local = points - np.asarray(center)
    radial = np.sqrt(local[:, 1] ** 2 + local[:, 2] ** 2)
    q = np.stack((radial - radius, np.abs(local[:, 0]) - half_length), axis=1)
    outside = np.maximum(q, 0.0)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside_dist + inside


def sdf_box(points: np.ndarray, *, center: Vec3, half_size: Vec3, rotation: np.ndarray | None = None) -> np.ndarray:
    center_arr = np.asarray(center, dtype=float)
    local = points - center_arr
    if rotation is not None:
        local = local @ rotation.T
    local = np.abs(local) - np.asarray(half_size, dtype=float)
    outside = np.maximum(local, 0.0)
    outside_dist = np.linalg.norm(outside, axis=1)
    inside = np.minimum(np.maximum(np.maximum(local[:, 0], local[:, 1]), local[:, 2]), 0.0)
    return outside_dist + inside


def sdf_cone(points: np.ndarray, *, apex: Vec3, axis: Vec3, height: float, base_radius: float) -> np.ndarray:
    axis_vec = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_vec)
    if norm == 0.0:
        raise ValueError("Cone axis must be non-zero.")
    axis_dir = axis_vec / norm
    height = float(height)
    base_radius = float(base_radius)
    local = points - np.asarray(apex)
    axial = local @ axis_dir
    radial_vec = local - np.outer(axial, axis_dir)
    radial = np.linalg.norm(radial_vec, axis=1)
    ab = np.array([height, base_radius], dtype=float)
    ab_len_sq = height * height + base_radius * base_radius
    p2 = np.column_stack((axial, radial))
    t = np.clip((p2 @ ab) / ab_len_sq, 0.0, 1.0)
    projection = t[:, None] * ab
    diff = p2 - projection
    dist_segment = np.linalg.norm(diff, axis=1)
    inv_norm = 1.0 / np.sqrt(ab_len_sq)
    lateral_signed = (base_radius * axial - height * radial) * inv_norm
    dist_apex_point = np.sqrt(radial * radial + axial * axial)
    dist_base_plane = axial - height
    result = np.empty_like(axial)
    mask_tip = axial <= 0.0
    result[mask_tip] = dist_apex_point[mask_tip]
    mask_base = axial >= height
    if np.any(mask_base):
        rad_base = radial[mask_base]
        parallel = dist_base_plane[mask_base]
        result[mask_base] = np.where(
            rad_base <= base_radius,
            np.abs(parallel),
            np.sqrt((rad_base - base_radius) ** 2 + parallel ** 2),
        )
    mask_mid = (~mask_tip) & (~mask_base)
    if np.any(mask_mid):
        axial_mid = axial[mask_mid]
        radial_mid = radial[mask_mid]
        inside_mask = base_radius * axial_mid >= height * radial_mid
        idx_mid = np.where(mask_mid)[0]
        if np.any(inside_mask):
            idx = idx_mid[inside_mask]
            min_surface = np.minimum.reduce([lateral_signed[idx], dist_apex_point[idx], height - axial[idx]])
            result[idx] = -min_surface
        if np.any(~inside_mask):
            idx = idx_mid[~inside_mask]
            result[idx] = dist_segment[idx]
    return result


def smooth_union(a: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    if rho <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / rho, 0.0, 1.0)
    return (1.0 - h) * b + h * a - rho * h * (1.0 - h)


def aircraft_sdf(points: np.ndarray, geometry: AircraftGeometry | None = None, smooth_radius: float = 0.0) -> np.ndarray:
    geometry = geometry or AircraftGeometry()
    fuselage = sdf_capped_cylinder(
        points,
        center=geometry.fuselage_center,
        radius=geometry.fuselage_radius,
        half_length=geometry.fuselage_half_length,
    )
    front_center_x = geometry.fuselage_center[0] + geometry.fuselage_half_length
    nose_apex = (
        front_center_x + geometry.nose_length + geometry.nose_tip_offset,
        geometry.fuselage_center[1],
        geometry.fuselage_center[2],
    )
    nose = sdf_cone(points, apex=nose_apex, axis=(-1.0, 0.0, 0.0), height=geometry.nose_length, base_radius=geometry.fuselage_radius)
    wing = sdf_box(points, center=tuple(_wing_center(geometry)), half_size=geometry.wing_half_size, rotation=_wing_rotation_matrix(geometry))
    htail = sdf_box(points, center=geometry.htail_center, half_size=geometry.htail_half_size)
    vtail = sdf_box(points, center=geometry.vtail_center, half_size=geometry.vtail_half_size)
    combined = smooth_union(fuselage, nose, 0.0)
    combined = smooth_union(combined, wing, smooth_radius)
    combined = smooth_union(combined, htail, smooth_radius)
    combined = smooth_union(combined, vtail, smooth_radius)
    return combined


def make_uniform_grid(bounds: Iterable[Vec3], resolution: Tuple[int, int, int], smooth_radius: float = 0.0,
                      geometry: AircraftGeometry | None = None) -> tuple[Any, np.ndarray]:
    resolution = tuple(max(2, int(n)) for n in resolution)
    xmin, xmax, ymin, ymax, zmin, zmax = _split_bounds(bounds)
    xs = np.linspace(xmin, xmax, resolution[0])
    ys = np.linspace(ymin, ymax, resolution[1])
    zs = np.linspace(zmin, zmax, resolution[2])
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    stacked = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    sdf_values = aircraft_sdf(stacked, geometry=geometry, smooth_radius=smooth_radius).reshape(resolution)
    spacing = ((xmax - xmin) / (resolution[0] - 1), (ymax - ymin) / (resolution[1] - 1), (zmax - zmin) / (resolution[2] - 1))
    pv_mod = _require_pyvista()
    grid = _new_uniform_grid(pv_mod)
    grid.dimensions = resolution
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = spacing
    grid["sdf"] = np.asfortranarray(sdf_values).ravel(order="F")
    return grid, sdf_values


def extract_zero_level_contour(grid: Any, iso_subdivisions: int = 0) -> Any:
    pv_mod = _require_pyvista()
    surface = pv_mod.wrap(grid).contour(isosurfaces=[0.0], scalars="sdf")
    if iso_subdivisions > 0:
        surface = surface.subdivide(int(max(0, iso_subdivisions)), subfilter="loop")
    return surface


def _build_vertex_adjacency(mesh: Any) -> list[np.ndarray]:
    n_points = mesh.n_points
    faces = np.asarray(mesh.faces, dtype=int)
    adjacency = [set() for _ in range(n_points)]
    idx = 0
    while idx < len(faces):
        nverts = faces[idx]
        idx += 1
        verts = faces[idx: idx + nverts]
        idx += nverts
        for vi in verts:
            for vj in verts:
                if vi != vj:
                    adjacency[vi].add(vj)
    return [np.fromiter(neigh, dtype=int) if neigh else np.empty(0, dtype=int) for neigh in adjacency]


def _batched_vector_norm_derivatives(vectors: np.ndarray, *, eps: float = 1e-14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("Expected vectors with shape (num_points, dimension).")

    num_points, dimension = vectors.shape
    values = np.linalg.norm(vectors, axis=1)
    gradients = np.zeros_like(vectors)
    hessians = np.zeros((num_points, dimension, dimension), dtype=float)

    mask = values > eps
    if np.any(mask):
        vectors_masked = vectors[mask]
        values_masked = values[mask]
        gradients[mask] = vectors_masked / values_masked[:, None]
        eye = np.eye(dimension, dtype=float)[None, :, :]
        outer = np.einsum("ni,nj->nij", vectors_masked, vectors_masked)
        hessians[mask] = eye / values_masked[:, None, None] - outer / values_masked[:, None, None] ** 3

    return values, gradients, hessians


def _compose_from_intermediates(
    jacobians: np.ndarray,
    intermediate_hessians: np.ndarray,
    scalar_gradient: np.ndarray,
    scalar_hessian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.einsum("nka,nk->na", jacobians, scalar_gradient)
    hessian = np.einsum("nk,nkab->nab", scalar_gradient, intermediate_hessians)
    hessian += np.einsum("nka,nkl,nlb->nab", jacobians, scalar_hessian, jacobians)
    hessian = 0.5 * (hessian + np.transpose(hessian, (0, 2, 1)))
    return gradient, hessian


def _piecewise_box_like_sdf(
    q: np.ndarray,
    jacobians: np.ndarray,
    intermediate_hessians: np.ndarray,
    *,
    eps: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    num_points, num_components = q.shape

    values = np.empty(num_points, dtype=float)
    gradients = np.zeros((num_points, 3), dtype=float)
    hessians = np.zeros((num_points, 3, 3), dtype=float)

    outside_mask = np.any(q > 0.0, axis=1)
    if np.any(outside_mask):
        q_out = q[outside_mask]
        jac_out = jacobians[outside_mask]
        hess_out = intermediate_hessians[outside_mask]
        active = q_out > 0.0
        outside = np.where(active, q_out, 0.0)
        outside_norm = np.linalg.norm(outside, axis=1)
        values[outside_mask] = outside_norm

        good = outside_norm > eps
        if np.any(good):
            outside_good = outside[good]
            jac_good = jac_out[good]
            hess_good = hess_out[good]
            active_good = active[good].astype(float)
            norms = outside_norm[good]

            scalar_gradient = outside_good / norms[:, None]
            scalar_hessian = np.eye(num_components, dtype=float)[None, :, :] / norms[:, None, None]
            scalar_hessian -= np.einsum("ni,nj->nij", outside_good, outside_good) / norms[:, None, None] ** 3
            scalar_hessian *= np.einsum("ni,nj->nij", active_good, active_good)

            grad_good, hess_good_world = _compose_from_intermediates(
                jac_good,
                hess_good,
                scalar_gradient,
                scalar_hessian,
            )
            outside_indices = np.flatnonzero(outside_mask)[good]
            gradients[outside_indices] = grad_good
            hessians[outside_indices] = hess_good_world

    inside_mask = ~outside_mask
    if np.any(inside_mask):
        q_inside = q[inside_mask]
        jac_inside = jacobians[inside_mask]
        hess_inside = intermediate_hessians[inside_mask]
        closest_component = np.argmax(q_inside, axis=1)
        row_indices = np.arange(q_inside.shape[0], dtype=int)
        values[inside_mask] = q_inside[row_indices, closest_component]
        gradients[inside_mask] = jac_inside[row_indices, closest_component, :]
        hessians[inside_mask] = hess_inside[row_indices, closest_component, :, :]

    return values, gradients, hessians


def _radial_yz_value_gradient_hessian(local_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial, grad_yz, hess_yz = _batched_vector_norm_derivatives(local_points[:, 1:], eps=1e-14)
    gradients = np.zeros_like(local_points)
    gradients[:, 1:] = grad_yz
    hessians = np.zeros((local_points.shape[0], 3, 3), dtype=float)
    hessians[:, 1:, 1:] = hess_yz
    return radial, gradients, hessians


def _axis_radial_value_gradient_hessian(
    local_points: np.ndarray,
    axis_dir: np.ndarray,
    *,
    eps: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis_dir = np.asarray(axis_dir, dtype=float)
    projector = np.eye(3, dtype=float) - np.outer(axis_dir, axis_dir)
    axial = local_points @ axis_dir
    radial_vectors = local_points - np.outer(axial, axis_dir)
    radial = np.linalg.norm(radial_vectors, axis=1)

    axial_gradient = np.broadcast_to(axis_dir, local_points.shape).copy()
    radial_gradient = np.zeros_like(local_points)
    radial_hessian = np.zeros((local_points.shape[0], 3, 3), dtype=float)

    mask = radial > eps
    if np.any(mask):
        radial_gradient[mask] = radial_vectors[mask] / radial[mask, None]
        radial_hessian[mask] = projector[None, :, :] / radial[mask, None, None]
        radial_hessian[mask] -= (
            np.einsum("ni,nj->nij", radial_vectors[mask], radial_vectors[mask]) / radial[mask, None, None] ** 3
        )

    return axial, radial, axial_gradient, radial_gradient, radial_hessian


def _sdf_capped_cylinder_value_gradient_hessian(
    points: np.ndarray,
    *,
    center: Vec3,
    radius: float,
    half_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_arr = np.asarray(points, dtype=float)
    local_points = points_arr - np.asarray(center, dtype=float)
    radial, radial_gradient, radial_hessian = _radial_yz_value_gradient_hessian(local_points)

    q = np.stack((radial - float(radius), np.abs(local_points[:, 0]) - float(half_length)), axis=1)
    jacobians = np.zeros((points_arr.shape[0], 2, 3), dtype=float)
    jacobians[:, 0, :] = radial_gradient
    jacobians[:, 1, 0] = np.sign(local_points[:, 0])

    intermediate_hessians = np.zeros((points_arr.shape[0], 2, 3, 3), dtype=float)
    intermediate_hessians[:, 0, :, :] = radial_hessian
    return _piecewise_box_like_sdf(q, jacobians, intermediate_hessians)


def _sdf_box_value_gradient_hessian(
    points: np.ndarray,
    *,
    center: Vec3,
    half_size: Vec3,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_arr = np.asarray(points, dtype=float)
    rotation_matrix = np.eye(3, dtype=float) if rotation is None else np.asarray(rotation, dtype=float)
    local_points = (points_arr - np.asarray(center, dtype=float)) @ rotation_matrix.T
    q = np.abs(local_points) - np.asarray(half_size, dtype=float)
    jacobians = np.sign(local_points)[:, :, None] * rotation_matrix[None, :, :]
    intermediate_hessians = np.zeros((points_arr.shape[0], 3, 3, 3), dtype=float)
    return _piecewise_box_like_sdf(q, jacobians, intermediate_hessians)


def _sdf_cone_value_gradient_hessian(
    points: np.ndarray,
    *,
    apex: Vec3,
    axis: Vec3,
    height: float,
    base_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_vec = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm == 0.0:
        raise ValueError("Cone axis must be non-zero.")

    axis_dir = axis_vec / axis_norm
    height = float(height)
    base_radius = float(base_radius)
    points_arr = np.asarray(points, dtype=float)
    local_points = points_arr - np.asarray(apex, dtype=float)
    axial, radial, axial_gradient, radial_gradient, radial_hessian = _axis_radial_value_gradient_hessian(local_points, axis_dir)

    jacobians = np.zeros((points_arr.shape[0], 2, 3), dtype=float)
    jacobians[:, 0, :] = axial_gradient
    jacobians[:, 1, :] = radial_gradient

    intermediate_hessians = np.zeros((points_arr.shape[0], 2, 3, 3), dtype=float)
    intermediate_hessians[:, 1, :, :] = radial_hessian

    values = np.empty(points_arr.shape[0], dtype=float)
    scalar_gradient = np.zeros((points_arr.shape[0], 2), dtype=float)
    scalar_hessian = np.zeros((points_arr.shape[0], 2, 2), dtype=float)

    ab = np.array([height, base_radius], dtype=float)
    ab_len_sq = height * height + base_radius * base_radius
    ab_len = np.sqrt(ab_len_sq)

    tip_mask = axial <= 0.0
    if np.any(tip_mask):
        tip_vectors = np.column_stack((axial[tip_mask], radial[tip_mask]))
        tip_values, tip_gradient, tip_hessian = _batched_vector_norm_derivatives(tip_vectors)
        values[tip_mask] = tip_values
        scalar_gradient[tip_mask] = tip_gradient
        scalar_hessian[tip_mask] = tip_hessian

    base_mask = axial >= height
    if np.any(base_mask):
        base_indices = np.flatnonzero(base_mask)
        axial_base = axial[base_indices]
        radial_base = radial[base_indices]
        plane_distance = axial_base - height
        interior_base = radial_base <= base_radius

        if np.any(interior_base):
            idx = base_indices[interior_base]
            values[idx] = plane_distance[interior_base]
            scalar_gradient[idx, 0] = 1.0

        if np.any(~interior_base):
            idx = base_indices[~interior_base]
            rim_vectors = np.column_stack((axial[idx] - height, radial[idx] - base_radius))
            rim_values, rim_gradient, rim_hessian = _batched_vector_norm_derivatives(rim_vectors)
            values[idx] = rim_values
            scalar_gradient[idx] = rim_gradient
            scalar_hessian[idx] = rim_hessian

    mid_mask = (~tip_mask) & (~base_mask)
    if np.any(mid_mask):
        mid_indices = np.flatnonzero(mid_mask)
        axial_mid = axial[mid_indices]
        radial_mid = radial[mid_indices]
        inside_mid = base_radius * axial_mid >= height * radial_mid

        if np.any(inside_mid):
            idx = mid_indices[inside_mid]
            lateral_signed = (base_radius * axial[idx] - height * radial[idx]) / ab_len
            apex_distance = np.sqrt(axial[idx] ** 2 + radial[idx] ** 2)
            base_distance = height - axial[idx]
            candidates = np.stack((lateral_signed, apex_distance, base_distance), axis=1)
            closest_feature = np.argmin(candidates, axis=1)

            lateral_idx = idx[closest_feature == 0]
            if lateral_idx.size > 0:
                values[lateral_idx] = (height * radial[lateral_idx] - base_radius * axial[lateral_idx]) / ab_len
                scalar_gradient[lateral_idx, 0] = -base_radius / ab_len
                scalar_gradient[lateral_idx, 1] = height / ab_len

            apex_idx = idx[closest_feature == 1]
            if apex_idx.size > 0:
                apex_vectors = np.column_stack((axial[apex_idx], radial[apex_idx]))
                apex_values, apex_gradient, apex_hessian = _batched_vector_norm_derivatives(apex_vectors)
                values[apex_idx] = -apex_values
                scalar_gradient[apex_idx] = -apex_gradient
                scalar_hessian[apex_idx] = -apex_hessian

            base_idx = idx[closest_feature == 2]
            if base_idx.size > 0:
                values[base_idx] = axial[base_idx] - height
                scalar_gradient[base_idx, 0] = 1.0

        if np.any(~inside_mid):
            idx = mid_indices[~inside_mid]
            segment_parameter = (axial[idx] * ab[0] + radial[idx] * ab[1]) / ab_len_sq
            apex_idx = idx[segment_parameter <= 0.0]
            base_idx = idx[segment_parameter >= 1.0]
            lateral_idx = idx[(segment_parameter > 0.0) & (segment_parameter < 1.0)]

            if apex_idx.size > 0:
                apex_vectors = np.column_stack((axial[apex_idx], radial[apex_idx]))
                apex_values, apex_gradient, apex_hessian = _batched_vector_norm_derivatives(apex_vectors)
                values[apex_idx] = apex_values
                scalar_gradient[apex_idx] = apex_gradient
                scalar_hessian[apex_idx] = apex_hessian

            if base_idx.size > 0:
                rim_vectors = np.column_stack((axial[base_idx] - height, radial[base_idx] - base_radius))
                rim_values, rim_gradient, rim_hessian = _batched_vector_norm_derivatives(rim_vectors)
                values[base_idx] = rim_values
                scalar_gradient[base_idx] = rim_gradient
                scalar_hessian[base_idx] = rim_hessian

            if lateral_idx.size > 0:
                values[lateral_idx] = (height * radial[lateral_idx] - base_radius * axial[lateral_idx]) / ab_len
                scalar_gradient[lateral_idx, 0] = -base_radius / ab_len
                scalar_gradient[lateral_idx, 1] = height / ab_len

    gradients, hessians = _compose_from_intermediates(jacobians, intermediate_hessians, scalar_gradient, scalar_hessian)
    return values, gradients, hessians


def _smooth_union_value_gradient_hessian(
    a: np.ndarray,
    grad_a: np.ndarray,
    hess_a: np.ndarray,
    b: np.ndarray,
    grad_b: np.ndarray,
    hess_b: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rho <= 0.0:
        choose_a = a <= b
        values = np.where(choose_a, a, b)
        gradients = np.where(choose_a[:, None], grad_a, grad_b)
        hessians = np.where(choose_a[:, None, None], hess_a, hess_b)
        return values, gradients, hessians

    rho = float(rho)
    values = np.empty_like(a)
    gradients = np.zeros_like(grad_a)
    hessians = np.zeros_like(hess_a)

    diff = b - a
    blend_parameter = 0.5 + 0.5 * diff / rho
    choose_b = blend_parameter <= 0.0
    choose_a = blend_parameter >= 1.0
    blend_mask = (~choose_a) & (~choose_b)

    if np.any(choose_a):
        values[choose_a] = a[choose_a]
        gradients[choose_a] = grad_a[choose_a]
        hessians[choose_a] = hess_a[choose_a]

    if np.any(choose_b):
        values[choose_b] = b[choose_b]
        gradients[choose_b] = grad_b[choose_b]
        hessians[choose_b] = hess_b[choose_b]

    if np.any(blend_mask):
        blend_parameter = blend_parameter[blend_mask]
        values[blend_mask] = 0.5 * (a[blend_mask] + b[blend_mask]) - rho / 4.0 - diff[blend_mask] ** 2 / (4.0 * rho)
        gradients[blend_mask] = blend_parameter[:, None] * grad_a[blend_mask]
        gradients[blend_mask] += (1.0 - blend_parameter)[:, None] * grad_b[blend_mask]
        hessians[blend_mask] = blend_parameter[:, None, None] * hess_a[blend_mask]
        hessians[blend_mask] += (1.0 - blend_parameter)[:, None, None] * hess_b[blend_mask]
        grad_delta = grad_a[blend_mask] - grad_b[blend_mask]
        hessians[blend_mask] -= np.einsum("ni,nj->nij", grad_delta, grad_delta) / (2.0 * rho)

    hessians = 0.5 * (hessians + np.transpose(hessians, (0, 2, 1)))
    return values, gradients, hessians


def _aircraft_sdf_value_gradient_hessian(
    points: np.ndarray,
    *,
    geometry: AircraftGeometry,
    smooth_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fuselage, fuselage_gradient, fuselage_hessian = _sdf_capped_cylinder_value_gradient_hessian(
        points,
        center=geometry.fuselage_center,
        radius=geometry.fuselage_radius,
        half_length=geometry.fuselage_half_length,
    )

    front_center_x = geometry.fuselage_center[0] + geometry.fuselage_half_length
    nose_apex = (
        front_center_x + geometry.nose_length + geometry.nose_tip_offset,
        geometry.fuselage_center[1],
        geometry.fuselage_center[2],
    )
    nose, nose_gradient, nose_hessian = _sdf_cone_value_gradient_hessian(
        points,
        apex=nose_apex,
        axis=(-1.0, 0.0, 0.0),
        height=geometry.nose_length,
        base_radius=geometry.fuselage_radius,
    )

    wing, wing_gradient, wing_hessian = _sdf_box_value_gradient_hessian(
        points,
        center=tuple(_wing_center(geometry)),
        half_size=geometry.wing_half_size,
        rotation=_wing_rotation_matrix(geometry),
    )
    htail, htail_gradient, htail_hessian = _sdf_box_value_gradient_hessian(
        points,
        center=geometry.htail_center,
        half_size=geometry.htail_half_size,
    )
    vtail, vtail_gradient, vtail_hessian = _sdf_box_value_gradient_hessian(
        points,
        center=geometry.vtail_center,
        half_size=geometry.vtail_half_size,
    )

    values, gradients, hessians = _smooth_union_value_gradient_hessian(
        fuselage,
        fuselage_gradient,
        fuselage_hessian,
        nose,
        nose_gradient,
        nose_hessian,
        0.0,
    )
    values, gradients, hessians = _smooth_union_value_gradient_hessian(
        values,
        gradients,
        hessians,
        wing,
        wing_gradient,
        wing_hessian,
        smooth_radius,
    )
    values, gradients, hessians = _smooth_union_value_gradient_hessian(
        values,
        gradients,
        hessians,
        htail,
        htail_gradient,
        htail_hessian,
        smooth_radius,
    )
    values, gradients, hessians = _smooth_union_value_gradient_hessian(
        values,
        gradients,
        hessians,
        vtail,
        vtail_gradient,
        vtail_hessian,
        smooth_radius,
    )
    return values, gradients, hessians


def _sdf_with_gradient_finite_difference(
    points: np.ndarray,
    *,
    geometry: AircraftGeometry,
    smooth_radius: float = 0.0,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    phi = aircraft_sdf(points, geometry=geometry, smooth_radius=smooth_radius)
    grad = np.empty_like(points)
    for axis in range(3):
        direction = np.zeros(3, dtype=float)
        direction[axis] = epsilon
        forward = aircraft_sdf(points + direction, geometry=geometry, smooth_radius=smooth_radius)
        backward = aircraft_sdf(points - direction, geometry=geometry, smooth_radius=smooth_radius)
        grad[:, axis] = (forward - backward) / (2.0 * epsilon)
    return phi, grad


def _sdf_gradient_hessian_finite_difference(
    points: np.ndarray,
    *,
    geometry: AircraftGeometry,
    smooth_radius: float = 0.0,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, grad = _sdf_with_gradient_finite_difference(
        points,
        geometry=geometry,
        smooth_radius=smooth_radius,
        epsilon=epsilon,
    )
    num_points = points.shape[0]
    hess = np.empty((num_points, 3, 3), dtype=float)
    for axis in range(3):
        direction = np.zeros(3, dtype=float)
        direction[axis] = epsilon
        _, grad_plus = _sdf_with_gradient_finite_difference(
            points + direction,
            geometry=geometry,
            smooth_radius=smooth_radius,
            epsilon=epsilon,
        )
        _, grad_minus = _sdf_with_gradient_finite_difference(
            points - direction,
            geometry=geometry,
            smooth_radius=smooth_radius,
            epsilon=epsilon,
        )
        hess[:, :, axis] = (grad_plus - grad_minus) / (2.0 * epsilon)
    hess = 0.5 * (hess + np.transpose(hess, (0, 2, 1)))
    return phi, grad, hess


def sdf_with_gradient(points: np.ndarray, *, geometry: AircraftGeometry, smooth_radius: float = 0.0,
                      epsilon: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    del epsilon
    phi, grad, _ = _aircraft_sdf_value_gradient_hessian(points, geometry=geometry, smooth_radius=smooth_radius)
    return phi, grad


def sdf_gradient_hessian(points: np.ndarray, *, geometry: AircraftGeometry, smooth_radius: float = 0.0,
                         epsilon: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del epsilon
    return _aircraft_sdf_value_gradient_hessian(points, geometry=geometry, smooth_radius=smooth_radius)


def project_points_to_surface(points: np.ndarray, *, geometry: AircraftGeometry, smooth_radius: float = 0.0,
                              epsilon: float = 1e-3) -> np.ndarray:
    phi, grad = sdf_with_gradient(points, geometry=geometry, smooth_radius=smooth_radius, epsilon=epsilon)
    grad_norm_sq = np.sum(grad * grad, axis=1, keepdims=True)
    projected = points.copy()
    mask = grad_norm_sq[:, 0] > 1e-14
    if np.any(mask):
        projected[mask] = points[mask] - (phi[mask, None] * grad[mask]) / grad_norm_sq[mask]
    return projected


def _safe_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = ref - np.dot(ref, normal) * normal
    nrm = np.linalg.norm(t1)
    if nrm < 1e-14:
        ref = np.array([0.0, 0.0, 1.0])
        t1 = ref - np.dot(ref, normal) * normal
        nrm = np.linalg.norm(t1)
    t1 /= max(nrm, 1e-14)
    t2 = np.cross(normal, t1)
    t2 /= max(np.linalg.norm(t2), 1e-14)
    return t1, t2


def curvature_frame_from_hessian(grad: np.ndarray, hess: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm < 1e-12:
        zero = np.zeros(3, dtype=float)
        return 0.0, 0.0, zero, zero, np.eye(3), grad_norm
    normal = grad / grad_norm
    projector = np.eye(3) - np.outer(normal, normal)
    t1, t2 = _safe_tangent_basis(normal)
    basis = np.column_stack((t1, t2))
    B = -(basis.T @ hess @ basis) / grad_norm
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    e1 = basis @ eigvecs[:, 0]
    e2 = basis @ eigvecs[:, 1]
    e1 /= max(np.linalg.norm(e1), 1e-14)
    e2 /= max(np.linalg.norm(e2), 1e-14)
    return float(eigvals[0]), float(eigvals[1]), e1, e2, projector, grad_norm


def surface_laplacian_displacement(points: np.ndarray, adjacency: list[np.ndarray]) -> np.ndarray:
    displacement = np.zeros_like(points)
    for vidx, neighbors in enumerate(adjacency):
        if neighbors.size == 0:
            continue
        displacement[vidx] = points[neighbors].mean(axis=0) - points[vidx]
    return displacement


def _mean_incident_edge_lengths(points: np.ndarray, adjacency: list[np.ndarray]) -> np.ndarray:
    h = np.zeros(points.shape[0], dtype=float)
    for vidx, neighbors in enumerate(adjacency):
        if neighbors.size == 0:
            continue
        h[vidx] = np.linalg.norm(points[neighbors] - points[vidx], axis=1).mean()
    fallback = h[h > 0.0].mean() if np.any(h > 0.0) else 1.0
    h[h <= 0.0] = fallback
    return h


def principal_curvature_data(points: np.ndarray, *, geometry: AircraftGeometry, smooth_radius: float,
                             fd_epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, grad, hess = sdf_gradient_hessian(points, geometry=geometry, smooth_radius=smooth_radius, epsilon=fd_epsilon)
    n = points.shape[0]
    projectors = np.zeros((n, 3, 3), dtype=float)
    k1 = np.zeros(n, dtype=float)
    k2 = np.zeros(n, dtype=float)
    curvature_mag = np.zeros(n, dtype=float)
    principal_dir = np.zeros((n, 3), dtype=float)
    for i in range(n):
        k1_i, k2_i, e1_i, _e2_i, projector_i, _ = curvature_frame_from_hessian(grad[i], hess[i])
        projectors[i] = projector_i
        k1[i] = k1_i
        k2[i] = k2_i
        curvature_mag[i] = max(abs(k1_i), abs(k2_i))
        principal_dir[i] = e1_i
    return k1, k2, curvature_mag, principal_dir, projectors


def anchor_mask_from_curvature(curvature_mag: np.ndarray, adjacency: list[np.ndarray], *, percentile: float,
                               floor: float, use_local_maxima: bool, maxima_tolerance: float) -> np.ndarray:
    if curvature_mag.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = max(float(floor), float(np.percentile(curvature_mag, percentile)))
    mask = curvature_mag >= threshold
    if use_local_maxima:
        maxima = np.zeros_like(mask)
        tol = float(maxima_tolerance)
        for i, neighbors in enumerate(adjacency):
            if neighbors.size == 0:
                maxima[i] = mask[i]
                continue
            neigh_max = curvature_mag[neighbors].max()
            maxima[i] = curvature_mag[i] >= neigh_max - tol
        mask &= maxima
    if not np.any(mask):
        idx = np.argmax(curvature_mag)
        mask[idx] = True
    return mask


def anchor_attraction_direction(points: np.ndarray, anchor_mask: np.ndarray, anchor_strengths: np.ndarray, *,
                                decay_length: float, support_radius: float, max_anchors_per_vertex: int) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    dirs = np.zeros((n, 3), dtype=float)
    influence = np.zeros(n, dtype=float)
    anchor_idx = np.flatnonzero(anchor_mask)
    if anchor_idx.size == 0:
        return dirs, influence
    anchor_points = points[anchor_idx]
    anchor_strengths = np.asarray(anchor_strengths[anchor_idx], dtype=float)
    decay_length = max(float(decay_length), 1e-8)
    support_radius = max(float(support_radius), 1e-8)
    max_anchors_per_vertex = max(1, int(max_anchors_per_vertex))
    for i in range(n):
        if anchor_mask[i]:
            continue
        vecs = anchor_points - points[i]
        dists = np.linalg.norm(vecs, axis=1)
        valid = (dists > 1e-12) & (dists <= support_radius)
        if not np.any(valid):
            continue
        idx_valid = np.flatnonzero(valid)
        if idx_valid.size > max_anchors_per_vertex:
            order = np.argsort(dists[idx_valid])[:max_anchors_per_vertex]
            idx_valid = idx_valid[order]
        d = dists[idx_valid]
        v = vecs[idx_valid] / d[:, None]
        w = np.exp(-d / decay_length) * np.maximum(anchor_strengths[idx_valid], 1e-12)
        weighted = np.sum(w[:, None] * v, axis=0)
        norm = np.linalg.norm(weighted)
        if norm > 1e-14:
            dirs[i] = weighted / norm
            influence[i] = np.clip(np.sum(w), 0.0, 1.0e6)
    return dirs, influence


def add_anchor_attraction_step(mesh: Any, *, geometry: AircraftGeometry, smooth_radius: float,
                               settings: AnchorAttractionSettings) -> Any:
    mesh = _require_pyvista().wrap(mesh).copy(deep=True)
    points = np.asarray(mesh.points, dtype=float)
    adjacency = _build_vertex_adjacency(mesh)
    last_anchor_mask = np.zeros(points.shape[0], dtype=bool)
    last_curvature_mag = np.zeros(points.shape[0], dtype=float)
    last_influence = np.zeros(points.shape[0], dtype=float)
    last_k1 = np.zeros(points.shape[0], dtype=float)
    last_k2 = np.zeros(points.shape[0], dtype=float)

    for iteration in range(int(max(0, settings.iterations))):
        mean_h = _mean_incident_edge_lengths(points, adjacency)
        k1, k2, curvature_mag, principal_dir, projectors = principal_curvature_data(
            points, geometry=geometry, smooth_radius=smooth_radius, fd_epsilon=settings.fd_epsilon
        )
        anchor_mask = anchor_mask_from_curvature(
            curvature_mag,
            adjacency,
            percentile=settings.anchor_curvature_percentile,
            floor=settings.anchor_curvature_floor,
            use_local_maxima=settings.use_anchor_local_maxima,
            maxima_tolerance=settings.anchor_maxima_tolerance,
        )
        anchor_strengths = curvature_mag / max(curvature_mag.max(), 1e-12)
        attr_dir, influence = anchor_attraction_direction(
            points,
            anchor_mask,
            anchor_strengths,
            decay_length=settings.attraction_decay_length,
            support_radius=settings.attraction_support_radius,
            max_anchors_per_vertex=settings.max_anchors_per_vertex,
        )
        # Mild principal-direction bias for anchors only, to keep them sliding in meaningful directions.
        direction = np.array(attr_dir, copy=True)
        anchor_bias = np.einsum('nij,nj->ni', projectors, principal_dir)
        anchor_bias_norm = np.linalg.norm(anchor_bias, axis=1)
        mask_bias = anchor_bias_norm > 1e-14
        anchor_bias[mask_bias] /= anchor_bias_norm[mask_bias, None]
        direction[anchor_mask] = settings.anchor_move_scale * anchor_bias[anchor_mask]
        direction_norm = np.linalg.norm(direction, axis=1)
        move_mask = direction_norm > 1e-14
        direction[move_mask] /= direction_norm[move_mask, None]
        step = settings.max_step_fraction * mean_h * settings.attraction_strength * np.tanh(influence)
        step[anchor_mask] = settings.anchor_move_scale * settings.max_step_fraction * mean_h[anchor_mask]
        points[move_mask] = points[move_mask] + step[move_mask, None] * direction[move_mask]
        points = project_points_to_surface(points, geometry=geometry, smooth_radius=smooth_radius, epsilon=settings.projection_epsilon)

        if settings.tangential_smoothing_step > 0.0 and settings.smoothing_every > 0 and ((iteration + 1) % settings.smoothing_every == 0):
            lap = surface_laplacian_displacement(points, adjacency)
            tangential_lap = np.einsum('nij,nj->ni', projectors, lap)
            smooth_scale = np.ones(points.shape[0], dtype=float)
            smooth_scale[anchor_mask] = 0.25
            smooth_scale *= (1.0 - 0.5 * np.tanh(influence))
            points = points + settings.tangential_smoothing_step * smooth_scale[:, None] * tangential_lap
            points = project_points_to_surface(points, geometry=geometry, smooth_radius=smooth_radius, epsilon=settings.projection_epsilon)

        last_anchor_mask = anchor_mask
        last_curvature_mag = curvature_mag
        last_influence = np.tanh(influence)
        last_k1 = k1
        last_k2 = k2

    mesh.points = points
    mesh['anchor_mask'] = last_anchor_mask.astype(np.int8)
    mesh['anchor_influence'] = last_influence
    mesh['curvature_magnitude'] = last_curvature_mag
    mesh['principal_curvature_1'] = last_k1
    mesh['principal_curvature_2'] = last_k2
    return mesh


def main(
    *,
    resolution: int | Tuple[int, int, int] = 180,
    bounds: Iterable[float] = (-6.0, 6.0, -6.0, 6.0, -4.0, 6.0),
    smooth_radius: float = 0.18,
    iso_subdivisions: int = 0,
    nose_length: float | None = None,
    nose_tip_offset: float | None = None,
    show_grid: bool = False,
    wireframe: bool = False,
    export_mesh: str | None = None,
    off_screen: bool = False,
    save: str | None = None,
    geometry: AircraftGeometry | None = None,
    pre_smooth_iterations: int = 0,
    pre_smooth_step: float = 0.25,
    projection_epsilon: float = 1e-3,
    color_by: str = 'none',
    settings: AnchorAttractionSettings = AnchorAttractionSettings(),
    log_camera_updates: bool = False,
) -> Any:
    if isinstance(resolution, Sequence) and not isinstance(resolution, (str, bytes)):
        res_vals = tuple(int(n) for n in resolution)
        if len(res_vals) == 1:
            resolution_tuple = (res_vals[0],) * 3
        elif len(res_vals) == 3:
            resolution_tuple = res_vals
        else:
            raise ValueError('Resolution must be an int or a tuple of three ints.')
    else:
        resolution_tuple = (int(resolution),) * 3
    resolution_tuple = tuple(max(2, n) for n in resolution_tuple)

    bounds_seq = tuple(float(b) for b in bounds)
    if len(bounds_seq) != 6:
        raise ValueError('Bounds must contain exactly six values.')
    bounds_tuple = ((bounds_seq[0], bounds_seq[1]), (bounds_seq[2], bounds_seq[3]), (bounds_seq[4], bounds_seq[5]))

    base_geom = geometry or AircraftGeometry()
    updates: dict[str, float] = {}
    if nose_length is not None:
        updates['nose_length'] = float(nose_length)
    if nose_tip_offset is not None:
        updates['nose_tip_offset'] = float(nose_tip_offset)
    if updates:
        base_geom = replace(base_geom, **updates)

    pv_mod = _require_pyvista()
    grid, _ = make_uniform_grid(bounds_tuple, resolution_tuple, smooth_radius=smooth_radius, geometry=base_geom)
    mesh = extract_zero_level_contour(grid, iso_subdivisions=iso_subdivisions)

    if pre_smooth_iterations > 0:
        adjacency = _build_vertex_adjacency(mesh)
        pts = np.asarray(mesh.points, dtype=float)
        for _ in range(int(pre_smooth_iterations)):
            lap = surface_laplacian_displacement(pts, adjacency)
            pts = pts + pre_smooth_step * lap
            pts = project_points_to_surface(pts, geometry=base_geom, smooth_radius=smooth_radius, epsilon=projection_epsilon)
        mesh.points = pts

    mesh = add_anchor_attraction_step(mesh, geometry=base_geom, smooth_radius=smooth_radius, settings=settings)

    if export_mesh:
        mesh.save(export_mesh)

    pv_mod.OFF_SCREEN = bool(off_screen or save is not None)
    plotter = pv_mod.Plotter(off_screen=pv_mod.OFF_SCREEN)
    if color_by == 'none':
        plotter.add_mesh(mesh, color='skyblue', smooth_shading=True, specular=0.15, specular_power=12)
    else:
        plotter.add_mesh(mesh, scalars=color_by, smooth_shading=True, specular=0.15, specular_power=12, cmap='viridis', show_scalar_bar=True)
    if wireframe:
        plotter.add_mesh(mesh, style='wireframe', color='black', line_width=1.0, opacity=0.85)
    if show_grid:
        plotter.add_mesh(grid.outline(), color='black', line_width=1, opacity=0.2)
    plotter.add_axes(line_width=2)
    plotter.set_background('white')
    plotter.camera_position = [(1.3, 13.3, 0.05), (0.2, 0.0, 0.3), (0.0, 0.0, 1.0)]

    if log_camera_updates and not pv_mod.OFF_SCREEN:
        def _camera_vector(cam: Any, attr_name: str, vtk_getter: str) -> tuple[float, float, float] | None:
            value: Any | None = None
            if hasattr(cam, attr_name):
                candidate = getattr(cam, attr_name)
                if callable(candidate):
                    try:
                        value = candidate()
                    except TypeError:
                        value = None
                else:
                    value = candidate
            if value is None and hasattr(cam, vtk_getter):
                candidate = getattr(cam, vtk_getter)
                if callable(candidate):
                    value = candidate()
            if value is None:
                return None
            return tuple(float(v) for v in value)

        def _camera_logger(*_args: Any, **_kwargs: Any) -> None:
            camera = plotter.camera
            position = _camera_vector(camera, 'position', 'GetPosition')
            focal_point = _camera_vector(camera, 'focal_point', 'GetFocalPoint')
            view_up = _camera_vector(camera, 'view_up', 'GetViewUp')
            if position is None or focal_point is None or view_up is None:
                warnings.warn('Unable to read camera state; camera logging skipped.', RuntimeWarning)
                return
            print(f'[camera] position={position} focal_point={focal_point} view_up={view_up}')

        observer_added = False
        plotter_add = getattr(plotter, 'add_observer', None)
        if callable(plotter_add):
            plotter_add('EndInteractionEvent', _camera_logger)
            observer_added = True
        if observer_added:
            _camera_logger()

    plotter.show(auto_close=True, screenshot=save)
    return mesh


if __name__ == '__main__':
    # ============================
    # User inputs
    # ============================
    resolution = 180
    bounds = (-6.0, 6.0, -6.0, 6.0, -4.0, 6.0)
    smooth_radius = 0.18
    iso_subdivisions = 0

    nose_length = None
    nose_tip_offset = None
    geometry = None

    show_grid = False
    wireframe = True
    export_mesh = None
    off_screen = False
    save = None
    log_camera_updates = False

    pre_smooth_iterations = 0
    pre_smooth_step = 0.25
    projection_epsilon = 1e-3

    color_by = 'none'  # 'anchor_mask', 'anchor_influence', 'curvature_magnitude', 'principal_curvature_1', 'principal_curvature_2', 'none'

    settings = AnchorAttractionSettings(
        iterations=10,
        fd_epsilon=2e-3,
        projection_epsilon=projection_epsilon,
        anchor_curvature_percentile=95.0,
        anchor_curvature_floor=0.8,
        attraction_decay_length=0.01,
        attraction_support_radius=2.2,
        max_anchors_per_vertex=12,
        attraction_strength=0.5,
        max_step_fraction=0.18,
        anchor_move_scale=0.1,
        tangential_smoothing_step=0.25,
        smoothing_every=1,
        use_anchor_local_maxima=True,
        anchor_maxima_tolerance=1e-10,
    )

    mesh = main(
        resolution=resolution,
        bounds=bounds,
        smooth_radius=smooth_radius,
        iso_subdivisions=iso_subdivisions,
        nose_length=nose_length,
        nose_tip_offset=nose_tip_offset,
        show_grid=show_grid,
        wireframe=wireframe,
        export_mesh=export_mesh,
        off_screen=off_screen,
        save=save,
        geometry=geometry,
        pre_smooth_iterations=pre_smooth_iterations,
        pre_smooth_step=pre_smooth_step,
        projection_epsilon=projection_epsilon,
        color_by=color_by,
        settings=settings,
        log_camera_updates=log_camera_updates,
    )
