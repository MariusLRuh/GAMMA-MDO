from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import csdl_alpha as csdl
import lsdo_function_spaces as lfs

import bsm3
from bsm3.core.boundary_surface_movement.movement_test_csdl_static_seam import (
    _compute_graph_distances,
    _expand_vector_to_columns,
    _format_stats,
    _gaussian_support,
    _interaction_indicator_two_components,
    _make_ks_rho_schedule,
    _normalize_weights,
    _plot_surface_debug,
    _rotate_points_about_y_axis_numpy,
    _select_farthest_point_indices_symmetric,
    _sharp_feature_vertex_mask,
    _smooth_minimum_signed_distance_numpy,
    _stack_coefficients_csdl,
    _stack_coefficients_numpy,
    pv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MESH_PATH = SCRIPT_DIR / "wall_surface.pkl"
PLOT_FINAL_MESH = True
SAVE_FINAL_MESH_SCREENSHOT = True
FINAL_MESH_SCREENSHOT_PATH = SCRIPT_DIR / "embraer_175_hex_mesh_undeformed.png"
FINAL_MESH_SCREENSHOT_WINDOW_SIZE = (3840, 2160)
PLOT_STEP8_PREDICTION = False
PLOT_STEP8_DEFORMED_GEOMETRY_OVERLAY = False
PLOT_SEAM_RELAXATION_DEBUG = False
PLOT_SOFT_BRACKET_DEBUG = False
PLOT_SOFT_BRACKET_MESH_OVERLAY = False
ANIMATE_SOFT_BRACKET_DEBUG = False
SOFT_BRACKET_DEBUG_POINTS = 5
SOFT_BRACKET_DEBUG_SURFACE = "upper"
SOFT_BRACKET_DEBUG_GIF_PATH = SCRIPT_DIR / "soft_bracket_debug.gif"


def _load_polygon_surface_pickle(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    with path.open("rb") as stream:
        data = pickle.load(stream)

    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a dict, got {type(data)!r}.")
    if "points" not in data or "connectivity" not in data:
        raise KeyError("Expected pickle keys 'points' and 'connectivity'.")

    points = np.asarray(data["points"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (n, 3), got {points.shape}.")

    connectivity = [
        np.asarray(face, dtype=np.int64).reshape(-1)
        for face in data["connectivity"]
    ]
    if len(connectivity) == 0:
        raise ValueError("Surface connectivity is empty.")

    for face_index, face in enumerate(connectivity):
        if face.size < 3:
            raise ValueError(f"Face {face_index} has fewer than three vertices.")
        if np.any(face < 0) or np.any(face >= points.shape[0]):
            raise ValueError(f"Face {face_index} references an invalid point index.")

    return points, connectivity


def _make_polygon_faces(connectivity: list[np.ndarray]) -> np.ndarray:
    faces: list[int] = []
    for face in connectivity:
        faces.append(int(face.size))
        faces.extend(int(index) for index in face)
    return np.asarray(faces, dtype=np.int64)


def _triangulate_polygon_surface(connectivity: list[np.ndarray]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for face in connectivity:
        anchor = int(face[0])
        for local_index in range(1, face.size - 1):
            triangles.append(
                (
                    anchor,
                    int(face[local_index]),
                    int(face[local_index + 1]),
                )
            )
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _build_polygon_vertex_neighbors(
    num_vertices: int,
    connectivity: list[np.ndarray],
) -> list[np.ndarray]:
    neighbor_sets = [set() for _ in range(num_vertices)]
    for face in connectivity:
        for local_index, vertex_index in enumerate(face):
            next_vertex_index = int(face[(local_index + 1) % face.size])
            vertex_index = int(vertex_index)
            neighbor_sets[vertex_index].add(next_vertex_index)
            neighbor_sets[next_vertex_index].add(vertex_index)
    return [
        np.asarray(sorted(neighbors), dtype=np.int64)
        for neighbors in neighbor_sets
    ]


def _build_sparse_uniform_laplacian_matrix(
    num_vertices: int,
    adjacency: list[np.ndarray],
) -> sps.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for vertex_index, neighbors in enumerate(adjacency):
        if neighbors.size == 0:
            continue

        rows.append(vertex_index)
        cols.append(vertex_index)
        data.append(-1.0)

        neighbor_weight = 1.0 / float(neighbors.size)
        rows.extend([vertex_index] * int(neighbors.size))
        cols.extend(int(neighbor) for neighbor in neighbors)
        data.extend([neighbor_weight] * int(neighbors.size))

    return sps.csr_matrix(
        (data, (rows, cols)),
        shape=(num_vertices, num_vertices),
    )


def _pairwise_distance_matrix_csdl(
    points_a: csdl.Variable,
    points_b: csdl.Variable,
) -> csdl.Variable:
    points_a_squared = csdl.sum(points_a**2, axes=(1,))
    points_b_squared = csdl.sum(points_b**2, axes=(1,))
    points_a_squared_matrix = csdl.matmat(
        csdl.reshape(points_a_squared, (-1, 1)),
        np.ones((1, points_b.shape[0]), dtype=float),
    )
    points_b_squared_matrix = csdl.matmat(
        np.ones((points_a.shape[0], 1), dtype=float),
        csdl.reshape(points_b_squared, (1, -1)),
    )
    pairwise_dot_products = csdl.matmat(points_a, csdl.transpose(points_b))
    pairwise_distance_squared = (
        points_a_squared_matrix + points_b_squared_matrix - 2.0 * pairwise_dot_products
    )
    pairwise_distance_squared = csdl.maximum(
        pairwise_distance_squared,
        np.zeros((points_a.shape[0], points_b.shape[0]), dtype=float),
        rho=200.0,
    )
    return csdl.sqrt(pairwise_distance_squared + 1e-14)


def _evaluate_rbf_kernel_csdl(
    distances: csdl.Variable,
    *,
    kernel: str,
    kernel_scale: float,
) -> csdl.Variable:
    scale_safe = max(float(kernel_scale), 1e-12)
    scaled_distances = distances / scale_safe

    if kernel == "cubic":
        return distances**3
    if kernel == "gaussian":
        return csdl.exp(-(scaled_distances**2))
    if kernel == "multiquadric":
        return csdl.sqrt(1.0 + scaled_distances**2)
    raise ValueError(f"Unsupported RBF kernel: {kernel}")


def _build_rbf_polynomial_terms_csdl(points: csdl.Variable) -> csdl.Variable:
    return csdl.concatenate(
        (np.ones((points.shape[0], 1), dtype=float), points),
        axis=1,
    )


def _solve_multi_rhs_csdl(system_matrix: csdl.Variable, rhs_matrix: csdl.Variable) -> csdl.Variable:
    solution_columns = []
    print("Solving linear system with shape:", system_matrix.shape, rhs_matrix.shape)
    for column_index in range(rhs_matrix.shape[1]):
        solution_column = csdl.solve_linear(
            system_matrix,
            rhs_matrix[csdl.slice[:, column_index : column_index + 1]],
        )
        solution_columns.append(solution_column)
    if len(solution_columns) == 1:
        return solution_columns[0]
    return csdl.concatenate(tuple(solution_columns), axis=1)


def _fit_vector_rbf_csdl(
    training_points: csdl.Variable,
    training_displacements: csdl.Variable,
    *,
    fit_mode: str,
    kernel: str,
    kernel_scale: float,
    regularization: float,
    polynomial_regularization: float = 0.0,
    sample_weights: np.ndarray | None = None,
) -> tuple[csdl.Variable, csdl.Variable]:
    num_points = training_points.shape[0]
    kernel_matrix = _evaluate_rbf_kernel_csdl(
        _pairwise_distance_matrix_csdl(training_points, training_points),
        kernel=kernel,
        kernel_scale=kernel_scale,
    )
    polynomial_terms = _build_rbf_polynomial_terms_csdl(training_points)

    if fit_mode == "exact_interpolation":
        regularized_kernel_matrix = kernel_matrix + regularization * np.eye(num_points, dtype=float)
        top_block = csdl.concatenate((regularized_kernel_matrix, polynomial_terms), axis=1)
        bottom_block = csdl.concatenate(
            (csdl.transpose(polynomial_terms), np.zeros((4, 4), dtype=float)),
            axis=1,
        )
        system_matrix = csdl.concatenate((top_block, bottom_block), axis=0)
        rhs_matrix = csdl.concatenate(
            (training_displacements, np.zeros((4, training_displacements.shape[1]), dtype=float)),
            axis=0,
        )
        solution = _solve_multi_rhs_csdl(system_matrix, rhs_matrix)
        return solution[csdl.slice[:num_points, :]], solution[csdl.slice[num_points:, :]]

    if fit_mode == "weighted_least_squares":
        if sample_weights is None:
            sample_weights = np.ones((num_points,), dtype=float)
        else:
            sample_weights = np.asarray(sample_weights, dtype=float).reshape((num_points,))

        sqrt_sample_weights = np.sqrt(np.maximum(sample_weights, 0.0)).reshape((num_points, 1))
        weighted_kernel_matrix = sqrt_sample_weights * kernel_matrix
        weighted_polynomial_terms = sqrt_sample_weights * polynomial_terms
        weighted_targets = sqrt_sample_weights * training_displacements

        design_matrix = csdl.concatenate(
            (weighted_kernel_matrix, weighted_polynomial_terms),
            axis=1,
        )
        rhs_matrix = weighted_targets

        if regularization > 0.0:
            regularization_rows = np.hstack(
                [
                    np.sqrt(regularization) * np.eye(num_points, dtype=float),
                    np.zeros((num_points, 4), dtype=float),
                ]
            )
            design_matrix = csdl.concatenate((design_matrix, regularization_rows), axis=0)
            rhs_matrix = csdl.concatenate(
                (rhs_matrix, np.zeros((num_points, training_displacements.shape[1]), dtype=float)),
                axis=0,
            )

        if polynomial_regularization > 0.0:
            polynomial_rows = np.hstack(
                [
                    np.zeros((4, num_points), dtype=float),
                    np.sqrt(polynomial_regularization) * np.eye(4, dtype=float),
                ]
            )
            design_matrix = csdl.concatenate((design_matrix, polynomial_rows), axis=0)
            rhs_matrix = csdl.concatenate(
                (rhs_matrix, np.zeros((4, training_displacements.shape[1]), dtype=float)),
                axis=0,
            )

        normal_matrix = csdl.matmat(csdl.transpose(design_matrix), design_matrix)
        normal_rhs = csdl.matmat(csdl.transpose(design_matrix), rhs_matrix)
        solution = _solve_multi_rhs_csdl(normal_matrix, normal_rhs)
        return solution[csdl.slice[:num_points, :]], solution[csdl.slice[num_points:, :]]

    raise ValueError(f"Unsupported RBF fit mode: {fit_mode}")


def _evaluate_vector_rbf_csdl(
    query_points: csdl.Variable,
    training_points: csdl.Variable,
    weights: csdl.Variable,
    polynomial_coefficients: csdl.Variable,
    *,
    kernel: str,
    kernel_scale: float,
) -> csdl.Variable:
    kernel_values = _evaluate_rbf_kernel_csdl(
        _pairwise_distance_matrix_csdl(query_points, training_points),
        kernel=kernel,
        kernel_scale=kernel_scale,
    )
    polynomial_terms = _build_rbf_polynomial_terms_csdl(query_points)
    return csdl.matmat(kernel_values, weights) + csdl.matmat(
        polynomial_terms,
        polynomial_coefficients,
    )


def _signed_distance_gradient_from_ad_csdl(
    signed_distance: csdl.Variable,
    query_points: csdl.Variable,
) -> csdl.Variable:
    gradient_flat = csdl.derivative(csdl.sum(signed_distance), query_points)
    return csdl.reshape(gradient_flat, query_points.shape)


def _smooth_minimum_signed_distance_three_components_csdl(
    component_sdf_a: csdl.Variable,
    component_sdf_b: csdl.Variable,
    component_sdf_c: csdl.Variable,
    *,
    ks_rho: float,
) -> tuple[csdl.Variable, csdl.Variable, csdl.Variable, csdl.Variable]:
    component_sdf_a = csdl.reshape(component_sdf_a, (-1, 1))
    component_sdf_b = csdl.reshape(component_sdf_b, (-1, 1))
    component_sdf_c = csdl.reshape(component_sdf_c, (-1, 1))

    reference_ab = csdl.minimum(component_sdf_a, component_sdf_b, rho=ks_rho)
    reference = csdl.minimum(reference_ab, component_sdf_c, rho=ks_rho)
    shifted_a = component_sdf_a - reference
    shifted_b = component_sdf_b - reference
    shifted_c = component_sdf_c - reference
    exp_a = csdl.exp(-ks_rho * shifted_a)
    exp_b = csdl.exp(-ks_rho * shifted_b)
    exp_c = csdl.exp(-ks_rho * shifted_c)
    normalization = exp_a + exp_b + exp_c
    union_sdf = reference - csdl.log(normalization) / ks_rho
    return (
        csdl.reshape(union_sdf, (-1,)),
        exp_a / normalization,
        exp_b / normalization,
        exp_c / normalization,
    )


def _make_coordinate_key(point: np.ndarray, tolerance: float) -> tuple[int, ...]:
    tolerance = max(float(tolerance), 1e-14)
    return tuple(np.rint(np.asarray(point, dtype=float) / tolerance).astype(np.int64))


def _smoothstep_numpy(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _compact_smooth_support_numpy(
    distances: np.ndarray,
    *,
    radius: float,
    full_strength_radius: float = 0.0,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    support = np.zeros_like(distances, dtype=float)
    finite_mask = np.isfinite(distances)
    if not np.any(finite_mask):
        return support

    radius = max(float(radius), 1e-12)
    full_strength_radius = np.clip(float(full_strength_radius), 0.0, radius)
    transition_width = max(radius - full_strength_radius, 1e-12)
    transition_coordinate = (
        distances[finite_mask] - full_strength_radius
    ) / transition_width
    support[finite_mask] = 1.0 - _smoothstep_numpy(transition_coordinate)
    support[finite_mask & (distances <= full_strength_radius)] = 1.0
    support[finite_mask & (distances >= radius)] = 0.0
    return np.clip(support, 0.0, 1.0)
def _endpoint_feature_rows_from_global_indices(
    *,
    interpolation_indices: np.ndarray,
    endpoint_feature_indices: np.ndarray,
    endpoint_graph_distances: np.ndarray,
    support_radius: float,
    full_strength_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    interpolation_indices = np.asarray(interpolation_indices, dtype=np.int64).reshape(-1)
    endpoint_feature_indices = np.asarray(endpoint_feature_indices, dtype=np.int64).reshape(-1)
    endpoint_feature_mask = np.isin(interpolation_indices, endpoint_feature_indices)
    endpoint_feature_local_indices = np.where(endpoint_feature_mask)[0]
    support = _compact_smooth_support_numpy(
        endpoint_graph_distances[interpolation_indices[endpoint_feature_local_indices]],
        radius=support_radius,
        full_strength_radius=full_strength_radius,
    )
    return endpoint_feature_local_indices, support


def _endpoint_feature_solve_rows_from_global_indices(
    *,
    solve_vertex_local_indices: np.ndarray,
    endpoint_feature_indices: np.ndarray,
    endpoint_graph_distances: np.ndarray,
    support_radius: float,
    full_strength_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    endpoint_feature_indices = np.asarray(endpoint_feature_indices, dtype=np.int64).reshape(-1)
    if endpoint_feature_indices.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=float)

    solve_rows = np.asarray(solve_vertex_local_indices, dtype=np.int64).reshape(-1)[
        endpoint_feature_indices
    ]
    valid_mask = solve_rows >= 0
    solve_rows = solve_rows[valid_mask]
    endpoint_feature_indices = endpoint_feature_indices[valid_mask]
    support = _compact_smooth_support_numpy(
        endpoint_graph_distances[endpoint_feature_indices],
        radius=support_radius,
        full_strength_radius=full_strength_radius,
    )
    return solve_rows, support


def _rows_from_global_support(
    *,
    row_global_indices: np.ndarray,
    global_support: np.ndarray,
    support_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_global_indices = np.asarray(row_global_indices, dtype=np.int64).reshape(-1)
    global_support = np.asarray(global_support, dtype=float).reshape(-1)
    row_support = global_support[row_global_indices]
    row_local_indices = np.where(row_support > float(support_cutoff))[0]
    return row_local_indices, row_support[row_local_indices]


def _solve_rows_from_global_support(
    *,
    solve_vertex_indices: np.ndarray,
    global_support: np.ndarray,
    support_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    solve_vertex_indices = np.asarray(solve_vertex_indices, dtype=np.int64).reshape(-1)
    global_support = np.asarray(global_support, dtype=float).reshape(-1)
    solve_support = global_support[solve_vertex_indices]
    solve_local_indices = np.where(solve_support > float(support_cutoff))[0]
    return solve_local_indices, solve_support[solve_local_indices]


def _endpoint_feature_halo_global_support(
    *,
    component_distances: np.ndarray,
    solve_vertex_mask: np.ndarray,
    excluded_indices: np.ndarray,
    endpoint_feature_indices: np.ndarray,
    endpoint_graph_distances: np.ndarray,
    feature_graph_distances: np.ndarray,
    endpoint_support_radius: float,
    endpoint_full_strength_radius: float,
    halo_radius: float,
    halo_full_strength_radius: float,
    max_blend: float,
    support_cutoff: float,
    projection_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    component_distances = np.asarray(component_distances, dtype=float).reshape(-1)
    solve_vertex_mask = np.asarray(solve_vertex_mask, dtype=bool).reshape(-1)
    endpoint_feature_indices = np.asarray(endpoint_feature_indices, dtype=np.int64).reshape(-1)
    num_vertices = component_distances.size
    halo_support = np.zeros(num_vertices, dtype=float)
    if endpoint_feature_indices.size == 0:
        return np.empty((0,), dtype=np.int64), halo_support

    span_support = _compact_smooth_support_numpy(
        endpoint_graph_distances,
        radius=endpoint_support_radius,
        full_strength_radius=endpoint_full_strength_radius,
    )
    normal_support = _compact_smooth_support_numpy(
        feature_graph_distances,
        radius=halo_radius,
        full_strength_radius=halo_full_strength_radius,
    )
    halo_support = np.clip(float(max_blend), 0.0, 1.0) * span_support * normal_support
    halo_support[endpoint_feature_indices] = 0.0
    halo_support[np.asarray(excluded_indices, dtype=np.int64).reshape(-1)] = 0.0

    halo_mask = (
        solve_vertex_mask
        & (np.abs(component_distances) <= float(projection_tolerance))
        & (halo_support > float(support_cutoff))
    )
    halo_support[~halo_mask] = 0.0
    return np.where(halo_mask)[0], halo_support


def _endpoint_feature_global_indices(
    *,
    vertices: np.ndarray,
    vertex_neighbors: list[np.ndarray],
    component_distances: np.ndarray,
    component_state: dict[str, np.ndarray],
    sharp_feature_mask: np.ndarray,
    solve_vertex_mask: np.ndarray,
    excluded_indices: np.ndarray,
    endpoint_global_index: int,
    endpoint_parametric_coordinate: np.ndarray,
    endpoint_point: np.ndarray,
    endpoint_graph_distances: np.ndarray,
    support_radius: float,
    support_cutoff: float,
    projection_tolerance: float,
    parametric_boundary_tolerance: float,
    coordinate_tolerance: float,
) -> np.ndarray:
    endpoint_parametric_coordinate = np.asarray(
        endpoint_parametric_coordinate,
        dtype=float,
    ).reshape(3)
    vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
    component_distances = np.asarray(component_distances, dtype=float).reshape(-1)
    sharp_feature_mask = np.asarray(sharp_feature_mask, dtype=bool).reshape(-1)
    solve_vertex_mask = np.asarray(solve_vertex_mask, dtype=bool).reshape(-1)
    endpoint_global_index = int(endpoint_global_index)
    endpoint_patch_id = int(endpoint_parametric_coordinate[0])
    endpoint_uv = endpoint_parametric_coordinate[1:3]
    endpoint_point = np.asarray(endpoint_point, dtype=float).reshape(3)
    coordinate_tolerance = max(float(coordinate_tolerance), 1e-14)

    support = _compact_smooth_support_numpy(
        endpoint_graph_distances,
        radius=support_radius,
        full_strength_radius=0.0,
    )
    excluded_mask = np.zeros(vertices.shape[0], dtype=bool)
    excluded_mask[np.asarray(excluded_indices, dtype=np.int64).reshape(-1)] = True

    patch_ids = np.asarray(component_state["patch_id"], dtype=np.int64).reshape(-1)
    uv = np.asarray(component_state["uv"], dtype=float).reshape((-1, 2))
    same_patch_mask = patch_ids == endpoint_patch_id
    same_endpoint_patch_edge_mask = np.zeros(vertices.shape[0], dtype=bool)
    for axis_index in range(2):
        endpoint_coordinate = endpoint_uv[axis_index]
        if (
            endpoint_coordinate <= parametric_boundary_tolerance
            or endpoint_coordinate >= 1.0 - parametric_boundary_tolerance
        ):
            same_endpoint_patch_edge_mask |= (
                same_patch_mask
                & (
                    np.abs(uv[:, axis_index] - endpoint_coordinate)
                    <= parametric_boundary_tolerance
                )
            )

    any_parametric_boundary_mask = np.any(
        (uv <= parametric_boundary_tolerance)
        | (uv >= 1.0 - parametric_boundary_tolerance),
        axis=1,
    )
    same_feature_coordinate_mask = (
        np.abs(vertices[:, 2] - endpoint_point[2]) <= coordinate_tolerance
    )
    physical_feature_line_mask = (
        (same_endpoint_patch_edge_mask & same_feature_coordinate_mask)
        | (
            same_feature_coordinate_mask
            & (any_parametric_boundary_mask | sharp_feature_mask)
        )
    )

    endpoint_feature_candidate_mask = (
        solve_vertex_mask
        & ~excluded_mask
        & (np.abs(component_distances) <= float(projection_tolerance))
        & (support > float(support_cutoff))
        & physical_feature_line_mask
    )
    endpoint_feature_candidate_mask[endpoint_global_index] = True

    visited_mask = np.zeros(vertices.shape[0], dtype=bool)
    stack = [endpoint_global_index]
    visited_mask[endpoint_global_index] = True
    while stack:
        vertex_index = stack.pop()
        for neighbor_index in vertex_neighbors[vertex_index]:
            neighbor_index = int(neighbor_index)
            if visited_mask[neighbor_index] or not endpoint_feature_candidate_mask[neighbor_index]:
                continue
            visited_mask[neighbor_index] = True
            stack.append(neighbor_index)

    endpoint_feature_mask = visited_mask & ~excluded_mask
    return np.where(endpoint_feature_mask)[0]


def _endpoint_feature_direction_parametric_pair(
    *,
    endpoint_parametric_coordinate: np.ndarray,
    endpoint_feature_indices: np.ndarray,
    endpoint_graph_distances: np.ndarray,
    component_state: dict[str, np.ndarray],
    target_distance_fraction: float = 0.25,
) -> np.ndarray:
    endpoint_parametric_coordinate = np.asarray(
        endpoint_parametric_coordinate,
        dtype=float,
    ).reshape(3)
    endpoint_feature_indices = np.asarray(endpoint_feature_indices, dtype=np.int64).reshape(-1)
    if endpoint_feature_indices.size == 0:
        return np.vstack([endpoint_parametric_coordinate, endpoint_parametric_coordinate])

    graph_distances = np.asarray(endpoint_graph_distances, dtype=float).reshape(-1)
    candidate_distances = graph_distances[endpoint_feature_indices]
    finite_mask = np.isfinite(candidate_distances) & (candidate_distances > 1e-12)
    if not np.any(finite_mask):
        return np.vstack([endpoint_parametric_coordinate, endpoint_parametric_coordinate])

    candidate_indices = endpoint_feature_indices[finite_mask]
    candidate_distances = candidate_distances[finite_mask]
    target_distance = np.clip(
        float(target_distance_fraction),
        0.0,
        1.0,
    ) * float(np.max(candidate_distances))
    if target_distance <= 0.0:
        target_distance = float(np.min(candidate_distances))
    nearby_index = int(candidate_indices[np.argmin(np.abs(candidate_distances - target_distance))])
    nearby_parametric_coordinate = np.array(
        [
            int(np.asarray(component_state["patch_id"], dtype=np.int64).reshape(-1)[nearby_index]),
            *np.asarray(component_state["uv"], dtype=float).reshape((-1, 2))[nearby_index],
        ],
        dtype=float,
    )
    return np.vstack([endpoint_parametric_coordinate, nearby_parametric_coordinate])


def _infer_spanwise_uv_axis_from_root_endpoints(
    leading_parametric_coordinate: np.ndarray,
    trailing_parametric_coordinate: np.ndarray,
) -> int:
    leading_uv = np.asarray(leading_parametric_coordinate, dtype=float).reshape(3)[1:3]
    trailing_uv = np.asarray(trailing_parametric_coordinate, dtype=float).reshape(3)[1:3]
    chordwise_axis = int(np.argmax(np.abs(trailing_uv - leading_uv)))
    return 1 - chordwise_axis


def _build_half_mesh_symmetry_maps(
    vertices: np.ndarray,
    *,
    symmetry_axis: int,
    symmetry_plane_tolerance: float,
    mirror_tolerance: float,
    solve_side: str,
) -> dict[str, np.ndarray]:
    vertices = np.asarray(vertices, dtype=float)
    signed_coordinates = vertices[:, symmetry_axis]
    if solve_side == "positive":
        solve_vertex_mask = signed_coordinates >= -symmetry_plane_tolerance
        mirrored_vertex_mask = signed_coordinates < -symmetry_plane_tolerance
    elif solve_side == "negative":
        solve_vertex_mask = signed_coordinates <= symmetry_plane_tolerance
        mirrored_vertex_mask = signed_coordinates > symmetry_plane_tolerance
    else:
        raise ValueError(f"Unsupported solve_side: {solve_side!r}.")

    solve_vertex_indices = np.where(solve_vertex_mask)[0]
    solve_vertex_local_indices = -np.ones(vertices.shape[0], dtype=np.int64)
    solve_vertex_local_indices[solve_vertex_indices] = np.arange(
        solve_vertex_indices.size,
        dtype=np.int64,
    )

    key_to_solve_index: dict[tuple[int, ...], int] = {}
    for vertex_index in solve_vertex_indices:
        key_to_solve_index[_make_coordinate_key(vertices[vertex_index], mirror_tolerance)] = int(
            vertex_index
        )

    full_source_local_indices = np.empty(vertices.shape[0], dtype=np.int64)
    full_mirror_signs = np.ones((vertices.shape[0], vertices.shape[1]), dtype=float)
    full_mirror_signs[mirrored_vertex_mask, symmetry_axis] = -1.0
    missing_mirror_indices: list[int] = []
    for vertex_index, vertex in enumerate(vertices):
        if solve_vertex_local_indices[vertex_index] >= 0:
            full_source_local_indices[vertex_index] = solve_vertex_local_indices[vertex_index]
            continue

        mirrored_vertex = vertex.copy()
        mirrored_vertex[symmetry_axis] *= -1.0
        source_index = key_to_solve_index.get(
            _make_coordinate_key(mirrored_vertex, mirror_tolerance)
        )
        if source_index is None:
            missing_mirror_indices.append(vertex_index)
            full_source_local_indices[vertex_index] = -1
        else:
            full_source_local_indices[vertex_index] = solve_vertex_local_indices[source_index]

    if missing_mirror_indices:
        preview = missing_mirror_indices[:10]
        raise ValueError(
            "Could not find mirrored solve-side partners for "
            f"{len(missing_mirror_indices)} vertices; first missing indices: {preview}."
        )

    symmetry_plane_vertex_mask = np.abs(signed_coordinates) <= symmetry_plane_tolerance
    symmetry_plane_solve_local_indices = solve_vertex_local_indices[
        np.where(symmetry_plane_vertex_mask & solve_vertex_mask)[0]
    ]
    return {
        "solve_vertex_indices": solve_vertex_indices,
        "solve_vertex_mask": solve_vertex_mask,
        "solve_vertex_local_indices": solve_vertex_local_indices,
        "full_source_local_indices": full_source_local_indices,
        "full_mirror_signs": full_mirror_signs,
        "symmetry_plane_solve_local_indices": symmetry_plane_solve_local_indices,
    }


def _build_solve_vertex_neighbors(
    vertex_neighbors: list[np.ndarray],
    solve_vertex_indices: np.ndarray,
    solve_vertex_local_indices: np.ndarray,
) -> list[np.ndarray]:
    solve_vertex_neighbors: list[np.ndarray] = []
    for vertex_index in solve_vertex_indices:
        neighbor_local_indices = solve_vertex_local_indices[vertex_neighbors[vertex_index]]
        solve_vertex_neighbors.append(neighbor_local_indices[neighbor_local_indices >= 0])
    return solve_vertex_neighbors
def _walk_ordered_seam_path(
    seam_graph_neighbors: list[set[int]],
    start_row: int,
    first_row: int,
    end_row: int,
) -> np.ndarray | None:
    ordered_rows = [int(start_row), int(first_row)]
    previous_row = int(start_row)
    current_row = int(first_row)
    visited_rows = {int(start_row), int(first_row)}
    while current_row != end_row:
        next_rows = [
            row
            for row in sorted(seam_graph_neighbors[current_row])
            if row != previous_row and (row == end_row or row not in visited_rows)
        ]
        if not next_rows:
            return None
        previous_row = current_row
        current_row = int(next_rows[0])
        if current_row in visited_rows and current_row != end_row:
            return None
        ordered_rows.append(current_row)
        visited_rows.add(current_row)
    return np.asarray(ordered_rows, dtype=np.int64)


def _edge_pairs_and_ranges_from_ordered_paths(
    ordered_paths: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    edge_blocks = []
    chain_ranges = []
    start_edge = 0
    for ordered_path in ordered_paths:
        ordered_path = np.asarray(ordered_path, dtype=np.int64).reshape(-1)
        if ordered_path.size < 2:
            continue
        edge_block = np.column_stack([ordered_path[:-1], ordered_path[1:]]).astype(np.int64)
        edge_blocks.append(edge_block)
        chain_ranges.append([start_edge, start_edge + edge_block.shape[0]])
        start_edge += edge_block.shape[0]
    if not edge_blocks:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    return np.vstack(edge_blocks), np.asarray(chain_ranges, dtype=np.int64)


def _build_ordered_seam_edge_chains(
    seam_global_indices: np.ndarray,
    vertex_neighbors: list[np.ndarray],
    seam_points: np.ndarray,
    start_row: int | None,
    end_row: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    seam_global_indices = np.asarray(seam_global_indices, dtype=np.int64).reshape(-1)
    seam_points = np.asarray(seam_points, dtype=float).reshape((-1, 3))
    num_seam_points = seam_global_indices.size
    if num_seam_points < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    start_row_valid = start_row is not None and 0 <= int(start_row) < num_seam_points
    end_row_valid = end_row is not None and 0 <= int(end_row) < num_seam_points
    if start_row_valid and end_row_valid and int(start_row) != int(end_row):
        start_row = int(start_row)
        end_row = int(end_row)
        seam_row_by_global_index = {
            int(global_index): local_index
            for local_index, global_index in enumerate(seam_global_indices)
        }
        seam_graph_neighbors = [set() for _ in range(num_seam_points)]
        for global_index in seam_global_indices:
            row_i = seam_row_by_global_index[int(global_index)]
            for neighbor_global_index in vertex_neighbors[int(global_index)]:
                row_j = seam_row_by_global_index.get(int(neighbor_global_index))
                if row_j is not None and row_i != row_j:
                    seam_graph_neighbors[row_i].add(row_j)
                    seam_graph_neighbors[row_j].add(row_i)

        if all(len(neighbors) <= 2 for neighbors in seam_graph_neighbors):
            ordered_paths = []
            seen_path_keys = set()
            for first_row in sorted(seam_graph_neighbors[start_row]):
                ordered_path = _walk_ordered_seam_path(
                    seam_graph_neighbors,
                    start_row,
                    int(first_row),
                    end_row,
                )
                if ordered_path is None:
                    continue
                path_key = tuple(int(row) for row in ordered_path)
                reverse_key = tuple(reversed(path_key))
                if path_key in seen_path_keys or reverse_key in seen_path_keys:
                    continue
                ordered_paths.append(ordered_path)
                seen_path_keys.add(path_key)
            if ordered_paths:
                return _edge_pairs_and_ranges_from_ordered_paths(ordered_paths)

        endpoint_axis = seam_points[end_row] - seam_points[start_row]
        endpoint_axis_norm = np.linalg.norm(endpoint_axis)
        if endpoint_axis_norm > 1e-12:
            chordwise_coordinate = (seam_points - seam_points[start_row]) @ (
                endpoint_axis / endpoint_axis_norm
            )
            middle_rows = np.setdiff1d(
                np.arange(num_seam_points, dtype=np.int64),
                np.array([start_row, end_row], dtype=np.int64),
            )
            middle_order = middle_rows[
                np.lexsort(
                    (
                        seam_points[middle_rows, 2],
                        seam_points[middle_rows, 1],
                        chordwise_coordinate[middle_rows],
                    )
                )
            ]
            ordered_rows = np.concatenate(
                [
                    np.array([start_row], dtype=np.int64),
                    middle_order,
                    np.array([end_row], dtype=np.int64),
                ]
            )
            return _edge_pairs_and_ranges_from_ordered_paths([ordered_rows])

    sorted_rows = np.lexsort((seam_points[:, 2], seam_points[:, 1], seam_points[:, 0]))
    return _edge_pairs_and_ranges_from_ordered_paths([sorted_rows])


def _normalized_cumulative_arc_positions_for_chains(
    points: np.ndarray,
    edge_pairs: np.ndarray,
    chain_ranges: np.ndarray,
) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    edge_pairs = np.asarray(edge_pairs, dtype=np.int64).reshape((-1, 2))
    chain_ranges = np.asarray(chain_ranges, dtype=np.int64).reshape((-1, 2))
    if edge_pairs.shape[0] == 0 or chain_ranges.shape[0] == 0:
        return np.empty((0,), dtype=float), 1.0
    normalized_arc_positions = []
    total_curve_length = 0.0
    for range_start, range_end in chain_ranges:
        chain_edge_pairs = edge_pairs[int(range_start) : int(range_end)]
        if chain_edge_pairs.shape[0] == 0:
            continue
        edge_vectors = points[chain_edge_pairs[:, 1]] - points[chain_edge_pairs[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        curve_length = max(float(np.sum(edge_lengths)), 1e-12)
        total_curve_length += curve_length
        if edge_lengths.size > 1:
            normalized_arc_positions.append(np.cumsum(edge_lengths)[:-1] / curve_length)
    if not normalized_arc_positions:
        return np.empty((0,), dtype=float), max(total_curve_length, 1e-12)
    return np.concatenate(normalized_arc_positions), max(total_curve_length, 1e-12)


def _rows_for_global_indices(
    seam_global_indices: np.ndarray,
    anchor_global_indices: np.ndarray,
) -> np.ndarray:
    seam_row_by_global_index = {
        int(global_index): local_index
        for local_index, global_index in enumerate(np.asarray(seam_global_indices, dtype=np.int64))
    }
    rows = []
    for global_index in np.asarray(anchor_global_indices, dtype=np.int64).reshape(-1):
        row = seam_row_by_global_index.get(int(global_index))
        if row is not None:
            rows.append(row)
    if len(rows) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _enforce_symmetry_plane_csdl(
    vertices_csdl: csdl.Variable,
    symmetry_plane_local_indices: np.ndarray,
    *,
    symmetry_axis: int,
) -> csdl.Variable:
    if symmetry_plane_local_indices.size == 0:
        return vertices_csdl
    return vertices_csdl.set(
        csdl.slice[
            symmetry_plane_local_indices.tolist(),
            symmetry_axis : symmetry_axis + 1,
        ],
        np.zeros((symmetry_plane_local_indices.size, 1), dtype=float),
    )


def _mirror_solve_vertices_to_full_csdl(
    solve_vertices_csdl: csdl.Variable,
    full_source_local_indices: np.ndarray,
    full_mirror_signs: np.ndarray,
) -> csdl.Variable:
    row_indices = np.arange(full_source_local_indices.size, dtype=np.int64)
    selection_matrix = sps.csr_matrix(
        (
            np.ones(full_source_local_indices.size, dtype=float),
            (row_indices, np.asarray(full_source_local_indices, dtype=np.int64)),
        ),
        shape=(full_source_local_indices.size, solve_vertices_csdl.shape[0]),
    )
    return csdl.sparse.matmat(selection_matrix, solve_vertices_csdl) * full_mirror_signs


def _function_set_from_stacked_coefficients(
    template_function_set: lfs.FunctionSet,
    stacked_coefficients: np.ndarray,
    *,
    name=None,
) -> lfs.FunctionSet:
    stacked_coefficients = np.asarray(stacked_coefficients, dtype=float).reshape((-1, 3))
    offset = 0
    moved_functions = {}
    for key, function in template_function_set.functions.items():
        coefficient_shape = function.coefficients.shape
        num_rows = int(np.prod(coefficient_shape) // 3)
        moved_function_coefficients = stacked_coefficients[
            offset : offset + num_rows
        ].reshape(coefficient_shape)
        moved_functions[key] = lfs.Function(
            space=function.space,
            coefficients=moved_function_coefficients,
            name=function.name,
        )
        offset += num_rows

    if offset != stacked_coefficients.shape[0]:
        raise ValueError("Stacked coefficients do not match the template function set.")

    return lfs.FunctionSet(
        functions=moved_functions,
        function_names=template_function_set.function_names,
        name=name,
    )


def _single_patch_function_set(
    template_function_set: lfs.FunctionSet,
    patch_id: int,
    *,
    name=None,
) -> lfs.FunctionSet:
    patch_id = int(patch_id)
    return lfs.FunctionSet(
        functions={patch_id: template_function_set.functions[patch_id]},
        name=name,
    )


def _stack_coefficients_csdl_allow_single(function_set: lfs.FunctionSet) -> csdl.Variable:
    reshaped_coefficients = [
        csdl.reshape(function.coefficients, (-1, 3))
        for function in function_set.functions.values()
    ]
    if len(reshaped_coefficients) == 1:
        return reshaped_coefficients[0]
    return csdl.vstack(reshaped_coefficients)


def _project_points_to_function_set_csdl(
    points: csdl.Variable,
    coefficients: csdl.Variable,
    projection_model,
    evaluation_model,
) -> csdl.Variable:
    projection_operation = bsm3.FunctionSetProjectionOperation(
        model=projection_model,
    )
    parametric_coordinates = projection_operation.evaluate(
        coefficients=coefficients,
        points=points,
    )
    evaluation_operation = bsm3.FunctionSetEvaluationOperation(
        model=evaluation_model,
    )
    return evaluation_operation.evaluate(
        coefficients=coefficients,
        parametric_coordinates=parametric_coordinates,
    )


def _plot_surface_debug_with_geometry_overlay(
    surface_vertices: np.ndarray,
    *,
    mesh_faces: np.ndarray,
    interpolation_indices: np.ndarray,
    seam_seed_indices: np.ndarray,
    protected_feature_indices: np.ndarray,
    mesh_label: str,
    geometry_function_set=None,
    geometry_opacity: float = 0.35,
    geometry_color: str = "#D55E00",
    mesh_opacity: float = 0.70,
) -> None:
    if pv is None:
        raise ImportError("PyVista is required for plotting.")

    surface_vertices = np.asarray(surface_vertices, dtype=float)
    plotting_elements = [
        {
            "mesh": pv.PolyData(surface_vertices, mesh_faces),
            "kwargs": {
                "opacity": mesh_opacity,
                "show_edges": True,
                "label": mesh_label,
            },
        }
    ]
    if geometry_function_set is not None:
        plotting_elements = geometry_function_set.plot(
            point_types=["evaluated_points"],
            plot_types=["function"],
            opacity=geometry_opacity,
            color=geometry_color,
            additional_plotting_elements=plotting_elements,
            show=False,
        )

    if interpolation_indices.size > 0:
        plotting_elements.append(
            {
                "mesh": pv.PolyData(surface_vertices[interpolation_indices]),
                "kwargs": {
                    "color": "#2ca02c", #"orange",
                    "point_size": 7,
                    "render_points_as_spheres": True,
                    "label": "Interpolation vertices",
                },
            }
        )
    if seam_seed_indices.size > 0:
        plotting_elements.append(
            {
                "mesh": pv.PolyData(surface_vertices[seam_seed_indices]),
                "kwargs": {
                    "color": "red",
                    "point_size": 7,
                    "render_points_as_spheres": True,
                    "label": "Exact seam seeds",
                },
            }
        )
    # if protected_feature_indices.size > 0:
    #     plotting_elements.append(
    #         {
    #             "mesh": pv.PolyData(surface_vertices[protected_feature_indices]),
    #             "kwargs": {
    #                 "color": "royalblue",
    #                 "point_size": 8,
    #                 "render_points_as_spheres": True,
    #                 "label": "Protected wing features",
    #             },
    #         }
    #     )

    lfs.show_plot(plotting_elements=plotting_elements, title=mesh_label)
def _plot_soft_bracket_debug(
    geometry_function_set: lfs.FunctionSet,
    traces: list[dict],
    *,
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
    geometry_opacity: float = 0.25,
    mesh_opacity: float = 0.18,
    point_size: float = 10.0,
    title: str = "Soft bracket seam debug",
) -> None:
    if pv is None:
        raise ImportError("PyVista is required for plotting.")

    plotting_elements = geometry_function_set.plot(
        point_types=["evaluated_points"],
        plot_types=["function"],
        opacity=geometry_opacity,
        color="#D55E00",
        show=False,
    )
    if mesh_vertices is not None and mesh_faces is not None:
        plotting_elements.append(
            {
                "mesh": pv.PolyData(np.asarray(mesh_vertices, dtype=float), mesh_faces),
                "kwargs": {
                    "opacity": mesh_opacity,
                    "show_edges": True,
                    "color": "lightgray",
                    "label": "Mesh overlay",
                },
            }
        )

    plotter = pv.Plotter()
    plotter.show_axes()
    for element in plotting_elements:
        if isinstance(element, dict) and "mesh" in element:
            plotter.add_mesh(element["mesh"], **element.get("kwargs", {}))
        elif isinstance(element, tuple) and len(element) == 2:
            plotter.add_mesh(element[0], **element[1])
        elif isinstance(element, pv.Actor):
            plotter.add_actor(element)
        elif isinstance(element, pv.DataSet):
            plotter.add_mesh(element)

    lower_label_added = False
    upper_label_added = False
    final_label_added = False
    trajectory_label_added = False
    for trace in traces:
        label = str(trace.get("label", "seam"))
        lower_points = np.asarray(trace.get("initial_lower_points", []), dtype=float).reshape((-1, 3))
        upper_points = np.asarray(trace.get("initial_upper_points", []), dtype=float).reshape((-1, 3))
        final_points = np.asarray(trace.get("final_points", []), dtype=float).reshape((-1, 3))
        midpoint_history = np.asarray(trace.get("midpoint_points", []), dtype=float)
        # if lower_points.shape[0] > 0:
        #     plotter.add_mesh(
        #         pv.PolyData(lower_points),
        #         color="royalblue",
        #         point_size=point_size,
        #         render_points_as_spheres=True,
        #         label=None if lower_label_added else "Lower bracket endpoint",
        #     )
        #     lower_label_added = True
        if upper_points.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(upper_points),
                color="#2ca02c", # "orange",
                point_size=point_size,
                render_points_as_spheres=True,
                label=None if upper_label_added else "Upper bracket endpoint",
            )
            upper_label_added = True
        if midpoint_history.ndim == 3 and midpoint_history.shape[0] > 0:
            num_steps = midpoint_history.shape[0]
            for step_index in range(num_steps):
                opacity = 0.18 + 0.72 * (step_index + 1) / max(num_steps, 1)
                plotter.add_mesh(
                    pv.PolyData(midpoint_history[step_index]),
                    color="black",
                    opacity=opacity,
                    point_size=point_size * 0.75,
                    render_points_as_spheres=True,
                    label=(
                        f"{label} midpoint trajectory"
                        if not trajectory_label_added and step_index == num_steps - 1
                        else None
                    ),
                )
            for point_index in range(midpoint_history.shape[1]):
                trajectory_points = midpoint_history[:, point_index, :]
                if trajectory_points.shape[0] < 2:
                    continue
                lines = np.concatenate(
                    (
                        np.array([trajectory_points.shape[0]], dtype=np.int64),
                        np.arange(trajectory_points.shape[0], dtype=np.int64),
                    )
                )
                trajectory_mesh = pv.PolyData(trajectory_points)
                trajectory_mesh.lines = lines
                plotter.add_mesh(
                    trajectory_mesh,
                    color="black",
                    opacity=0.35,
                    line_width=2.0,
                )
            trajectory_label_added = True
        if final_points.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(final_points),
                color="red",
                point_size=point_size * 1.2,
                render_points_as_spheres=True,
                label=None if final_label_added else "Final seam point",
            )
            final_label_added = True

    plotter.add_legend()
    plotter.show(title=title)


def _sdf_sign_colors(sdf_values: np.ndarray) -> np.ndarray:
    sdf_values = np.asarray(sdf_values, dtype=float).reshape(-1)
    colors = np.broadcast_to(
        np.array([230, 120, 0], dtype=np.uint8),
        (sdf_values.size, 3),
    ).copy()
    colors[sdf_values < 0.0] = np.array([0, 90, 220], dtype=np.uint8)
    return colors


def _add_signed_sdf_points(
    plotter,
    points: np.ndarray,
    sdf_values: np.ndarray,
    *,
    point_size: float,
    label: str | None = None,
    name: str | None = None,
) -> None:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if points.shape[0] == 0:
        return
    point_cloud = pv.PolyData(points)
    point_cloud["sdf_sign_rgb"] = _sdf_sign_colors(sdf_values)
    plotter.add_mesh(
        point_cloud,
        scalars="sdf_sign_rgb",
        rgb=True,
        point_size=point_size,
        render_points_as_spheres=True,
        label=label,
        name=name,
    )


def _save_soft_bracket_sdf_animation(
    geometry_function_set: lfs.FunctionSet,
    traces: list[dict],
    *,
    gif_path: str | Path = "soft_bracket_debug.gif",
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
    geometry_opacity: float = 0.18,
    mesh_opacity: float = 0.12,
    point_size: float = 12.0,
    fps: float = 2.0,
    title: str = "Soft bracket SDF convergence",
) -> None:
    if pv is None:
        raise ImportError("PyVista is required for plotting.")

    max_frames = 0
    for trace in traces:
        midpoint_history = np.asarray(trace.get("midpoint_points", []), dtype=float)
        if midpoint_history.ndim == 3:
            max_frames = max(max_frames, midpoint_history.shape[0])
    if max_frames == 0:
        return
    trace_points = []
    for trace in traces:
        for key in (
            "midpoint_points",
            "bracket_lower_points",
            "bracket_upper_points",
        ):
            points = np.asarray(trace.get(key, []), dtype=float)
            if points.size > 0:
                trace_points.append(points.reshape((-1, 3)))
    trace_bounds_points = np.vstack(trace_points) if trace_points else np.zeros((1, 3))
    trace_diagonal = float(
        np.linalg.norm(np.max(trace_bounds_points, axis=0) - np.min(trace_bounds_points, axis=0))
    )
    midpoint_camera_lift = max(1e-5 * trace_diagonal, 1e-8)

    plotting_elements = geometry_function_set.plot(
        point_types=["evaluated_points"],
        plot_types=["function"],
        opacity=geometry_opacity,
        color="#D55E00",
        show=False,
    )
    if mesh_vertices is not None and mesh_faces is not None:
        plotting_elements.append(
            {
                "mesh": pv.PolyData(np.asarray(mesh_vertices, dtype=float), mesh_faces),
                "kwargs": {
                    "opacity": mesh_opacity,
                    "show_edges": True,
                    "color": "lightgray",
                },
            }
        )

    plotter = pv.Plotter(off_screen=True)
    for element in plotting_elements:
        if isinstance(element, dict) and "mesh" in element:
            plotter.add_mesh(element["mesh"], **element.get("kwargs", {}))
        elif isinstance(element, tuple) and len(element) == 2:
            plotter.add_mesh(element[0], **element[1])
        elif isinstance(element, pv.Actor):
            plotter.add_actor(element)
        elif isinstance(element, pv.DataSet):
            plotter.add_mesh(element)

    plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.parallel_projection = True
    plotter.open_gif(str(gif_path), fps=float(fps))

    for frame_index in range(max_frames):
        frame_actor_names = []
        for trace_index, trace in enumerate(traces):
            midpoint_history = np.asarray(trace.get("midpoint_points", []), dtype=float)
            lower_history = np.asarray(trace.get("bracket_lower_points", []), dtype=float)
            lower_sdf_history = np.asarray(trace.get("bracket_lower_sdf", []), dtype=float)
            upper_history = np.asarray(trace.get("bracket_upper_points", []), dtype=float)
            upper_sdf_history = np.asarray(trace.get("bracket_upper_sdf", []), dtype=float)
            if midpoint_history.ndim != 3 or midpoint_history.shape[0] == 0:
                continue

            local_frame_index = min(frame_index, midpoint_history.shape[0] - 1)
            lower_name = f"lower_bracket_{trace_index}"
            upper_name = f"upper_bracket_{trace_index}"
            midpoint_name = f"midpoint_{trace_index}"
            frame_actor_names.extend([lower_name, upper_name, midpoint_name])

            if lower_history.ndim == 3 and lower_history.shape[0] > 0:
                lower_frame = min(local_frame_index, lower_history.shape[0] - 1)
                _add_signed_sdf_points(
                    plotter,
                    lower_history[lower_frame],
                    lower_sdf_history[lower_frame],
                    point_size=point_size * 0.85,
                    name=lower_name,
                )
            if upper_history.ndim == 3 and upper_history.shape[0] > 0:
                upper_frame = min(local_frame_index, upper_history.shape[0] - 1)
                _add_signed_sdf_points(
                    plotter,
                    upper_history[upper_frame],
                    upper_sdf_history[upper_frame],
                    point_size=point_size * 0.85,
                    name=upper_name,
                )

            midpoint_points = np.asarray(
                midpoint_history[local_frame_index],
                dtype=float,
            ).reshape((-1, 3)).copy()
            midpoint_points[:, 2] += midpoint_camera_lift
            plotter.add_mesh(
                pv.PolyData(midpoint_points),
                color="red",
                point_size=point_size * 1.25,
                render_points_as_spheres=True,
                lighting=False,
                name=midpoint_name,
            )

        plotter.write_frame()
        for actor_name in frame_actor_names:
            plotter.remove_actor(actor_name)

    plotter.close()
    print(f"Saved soft bracket SDF animation: {gif_path}")


def _select_soft_bracket_debug_sample_indices(
    num_points: int,
    num_samples: int,
    *,
    seam_points: np.ndarray | None = None,
    leading_row: int | None = None,
    trailing_row: int | None = None,
    surface_side: str = "upper",
) -> np.ndarray:
    num_points = max(int(num_points), 0)
    num_samples = max(int(num_samples), 0)
    if num_points == 0 or num_samples == 0:
        return np.empty((0,), dtype=np.int64)

    fallback_indices = np.arange(num_points, dtype=np.int64)
    if num_points > num_samples:
        fallback_indices = np.unique(
            np.linspace(0, num_points - 1, num_samples, dtype=np.int64)
        )

    if seam_points is None or leading_row is None or trailing_row is None:
        return fallback_indices

    seam_points = np.asarray(seam_points, dtype=float).reshape((-1, 3))
    if seam_points.shape[0] != num_points:
        return fallback_indices
    leading_row = int(leading_row)
    trailing_row = int(trailing_row)
    if not (0 <= leading_row < num_points and 0 <= trailing_row < num_points):
        return fallback_indices
    if leading_row == trailing_row:
        return fallback_indices

    side = surface_side.lower()
    if side not in ("upper", "lower"):
        raise ValueError(f"Unsupported soft bracket debug surface side: {surface_side!r}.")

    leading_point = seam_points[leading_row]
    trailing_point = seam_points[trailing_row]
    chord_vector = trailing_point - leading_point
    chord_length_squared = float(chord_vector @ chord_vector)
    if chord_length_squared <= 1e-24:
        return fallback_indices

    chord_coordinate = (seam_points - leading_point) @ chord_vector / chord_length_squared
    chord_line_z = leading_point[2] + chord_coordinate * chord_vector[2]
    vertical_offset = seam_points[:, 2] - chord_line_z
    offset_tolerance = 1e-10 * max(float(np.linalg.norm(chord_vector)), 1.0)
    if side == "upper":
        side_mask = vertical_offset > offset_tolerance
    else:
        side_mask = vertical_offset < -offset_tolerance

    side_indices = np.where(side_mask)[0]
    if side_indices.size == 0:
        return fallback_indices
    side_coordinates = chord_coordinate[side_indices]
    ordered_side_indices = side_indices[np.argsort(side_coordinates)]
    if ordered_side_indices.size <= num_samples:
        return ordered_side_indices.astype(np.int64)

    sorted_coordinates = chord_coordinate[ordered_side_indices]
    target_coordinates = np.linspace(
        float(sorted_coordinates[0]),
        float(sorted_coordinates[-1]),
        num_samples,
    )
    selected_indices = []
    for target_coordinate in target_coordinates:
        available_mask = np.ones(ordered_side_indices.size, dtype=bool)
        if selected_indices:
            available_mask[np.isin(ordered_side_indices, selected_indices)] = False
        available_rows = ordered_side_indices[available_mask]
        available_coordinates = chord_coordinate[available_rows]
        selected_indices.append(
            int(available_rows[np.argmin(np.abs(available_coordinates - target_coordinate))])
        )
    return np.asarray(selected_indices, dtype=np.int64)
def _nearest_source_indices_numpy(points: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    source_points = np.asarray(source_points, dtype=float).reshape((-1, 3))
    if source_points.shape[0] == 0:
        raise ValueError("source_points must contain at least one point.")
    distances_squared = np.sum(
        (points[:, None, :] - source_points[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances_squared, axis=1).astype(np.int64)
def _gather_rows_csdl(source: csdl.Variable, row_indices: np.ndarray) -> csdl.Variable:
    row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    selection_matrix = sps.csr_matrix(
        (
            np.ones(row_indices.size, dtype=float),
            (np.arange(row_indices.size, dtype=np.int64), row_indices),
        ),
        shape=(row_indices.size, source.shape[0]),
    )
    return csdl.sparse.matmat(selection_matrix, source)


def _apply_endpoint_feature_training_drive_csdl(
    training_displacements: csdl.Variable,
    seam_displacements: csdl.Variable | None,
    feature_local_indices: np.ndarray,
    endpoint_seam_row: int | None,
    feature_support: np.ndarray,
) -> csdl.Variable:
    feature_local_indices = np.asarray(feature_local_indices, dtype=np.int64).reshape(-1)
    feature_support = np.asarray(feature_support, dtype=float).reshape(-1)
    if (
        seam_displacements is None
        or endpoint_seam_row is None
        or feature_local_indices.size == 0
    ):
        return training_displacements
    if feature_support.size != feature_local_indices.size:
        raise ValueError("Endpoint feature support count must match feature row count.")

    feature_index_list = feature_local_indices.tolist()
    current_feature_displacements = training_displacements[
        csdl.slice[feature_index_list, :]
    ]
    endpoint_displacement = _gather_rows_csdl(
        seam_displacements,
        np.array([int(endpoint_seam_row)], dtype=np.int64),
    )
    endpoint_displacement_rows = csdl.matmat(
        np.ones((feature_local_indices.size, 1), dtype=float),
        endpoint_displacement,
    )
    blend = _expand_vector_to_columns(np.clip(feature_support, 0.0, 1.0), 3)
    return training_displacements.set(
        csdl.slice[feature_index_list, :],
        (1.0 - blend) * current_feature_displacements
        + blend * endpoint_displacement_rows,
    )


def _blend_vertices_toward_endpoint_seam_displacement_csdl(
    reference_vertices: csdl.Variable,
    deformed_vertices: csdl.Variable,
    seam_displacements: csdl.Variable | None,
    feature_local_indices: np.ndarray,
    endpoint_seam_row: int | None,
    feature_support: np.ndarray,
) -> csdl.Variable:
    feature_local_indices = np.asarray(feature_local_indices, dtype=np.int64).reshape(-1)
    feature_support = np.asarray(feature_support, dtype=float).reshape(-1)
    if (
        seam_displacements is None
        or endpoint_seam_row is None
        or feature_local_indices.size == 0
    ):
        return deformed_vertices
    if feature_support.size != feature_local_indices.size:
        raise ValueError("Endpoint feature support count must match feature vertex count.")

    feature_index_list = feature_local_indices.tolist()
    current_vertices = deformed_vertices[csdl.slice[feature_index_list, :]]
    reference_feature_vertices = reference_vertices[csdl.slice[feature_index_list, :]]
    endpoint_displacement = _gather_rows_csdl(
        seam_displacements,
        np.array([int(endpoint_seam_row)], dtype=np.int64),
    )
    endpoint_displacement_rows = csdl.matmat(
        np.ones((feature_local_indices.size, 1), dtype=float),
        endpoint_displacement,
    )
    blend = _expand_vector_to_columns(np.clip(feature_support, 0.0, 1.0), 3)
    endpoint_driven_vertices = reference_feature_vertices + endpoint_displacement_rows
    return deformed_vertices.set(
        csdl.slice[feature_index_list, :],
        (1.0 - blend) * current_vertices + blend * endpoint_driven_vertices,
    )


def _blend_vertices_toward_nearest_seam_displacement_csdl(
    reference_vertices: csdl.Variable,
    deformed_vertices: csdl.Variable,
    seam_displacements: csdl.Variable | None,
    driven_local_indices: np.ndarray,
    nearest_seam_local_indices: np.ndarray,
    driven_support: np.ndarray,
) -> csdl.Variable:
    driven_local_indices = np.asarray(driven_local_indices, dtype=np.int64).reshape(-1)
    nearest_seam_local_indices = np.asarray(nearest_seam_local_indices, dtype=np.int64).reshape(-1)
    driven_support = np.asarray(driven_support, dtype=float).reshape(-1)
    if (
        seam_displacements is None
        or driven_local_indices.size == 0
        or nearest_seam_local_indices.size == 0
    ):
        return deformed_vertices
    if nearest_seam_local_indices.size != driven_local_indices.size:
        raise ValueError("Nearest seam index count must match driven vertex count.")
    if driven_support.size != driven_local_indices.size:
        raise ValueError("Seam driving support count must match driven vertex count.")

    driven_index_list = driven_local_indices.tolist()
    current_vertices = deformed_vertices[csdl.slice[driven_index_list, :]]
    reference_driven_vertices = reference_vertices[csdl.slice[driven_index_list, :]]
    nearest_seam_displacements = _gather_rows_csdl(
        seam_displacements,
        nearest_seam_local_indices,
    )
    blend = _expand_vector_to_columns(np.clip(driven_support, 0.0, 1.0), 3)
    driven_target_vertices = reference_driven_vertices + nearest_seam_displacements
    return deformed_vertices.set(
        csdl.slice[driven_index_list, :],
        (1.0 - blend) * current_vertices + blend * driven_target_vertices,
    )


def _relax_two_component_seam_points_csdl(
    initial_points: csdl.Variable,
    coefficients_a: csdl.Variable,
    coefficients_b: csdl.Variable,
    sdf_model_a,
    sdf_model_b,
    *,
    num_iterations: int,
    step_size: float,
    damping: float,
    label: str,
    print_residuals: bool = True,
    edge_pairs: np.ndarray | None = None,
    arc_chain_ranges: np.ndarray | None = None,
    baseline_normalized_arc_positions: np.ndarray | None = None,
    spacing_weight: float = 0.0,
    spacing_length_scale: float = 1.0,
    anchor_indices: np.ndarray | None = None,
    anchor_points: csdl.Variable | None = None,
    anchor_weight: float = 0.0,
    regularization_step_size: float = 0.0,
    regularization_mode: str = "projected_tangent",
    unified_objective_step_size: float | None = None,
    directional_endpoint_row_indices: np.ndarray | None = None,
    directional_endpoint_feature_points: csdl.Variable | None = None,
    directional_endpoint_iterations: int = 0,
    directional_endpoint_step_size: float | None = None,
    freeze_directional_endpoint_rows: bool = True,
) -> csdl.Variable:
    relaxed_points = initial_points
    damping = max(float(damping), 0.0)
    step_size = float(step_size)
    spacing_weight = max(float(spacing_weight), 0.0)
    anchor_weight = max(float(anchor_weight), 0.0)
    regularization_step_size = max(float(regularization_step_size), 0.0)
    unified_objective_step_size = (
        float(step_size)
        if unified_objective_step_size is None
        else max(float(unified_objective_step_size), 0.0)
    )
    directional_endpoint_step_size = (
        float(step_size)
        if directional_endpoint_step_size is None
        else max(float(directional_endpoint_step_size), 0.0)
    )
    spacing_length_scale = max(float(spacing_length_scale), 1e-12)
    edge_pairs = (
        np.empty((0, 2), dtype=np.int64)
        if edge_pairs is None
        else np.asarray(edge_pairs, dtype=np.int64).reshape((-1, 2))
    )
    arc_chain_ranges = (
        np.empty((0, 2), dtype=np.int64)
        if arc_chain_ranges is None
        else np.asarray(arc_chain_ranges, dtype=np.int64).reshape((-1, 2))
    )
    if arc_chain_ranges.shape[0] == 0 and edge_pairs.shape[0] > 0:
        arc_chain_ranges = np.array([[0, edge_pairs.shape[0]]], dtype=np.int64)
    baseline_normalized_arc_positions = (
        np.empty((0,), dtype=float)
        if baseline_normalized_arc_positions is None
        else np.asarray(baseline_normalized_arc_positions, dtype=float).reshape(-1)
    )
    regularization_mode = str(regularization_mode)
    use_unified_objective = regularization_mode == "unified_nonlinear_least_squares"
    use_regularization_terms = regularization_step_size > 0.0 or use_unified_objective
    num_arc_position_residuals = int(
        sum(
            max(int(range_end) - int(range_start) - 1, 0)
            for range_start, range_end in arc_chain_ranges
        )
    )
    anchor_indices = (
        np.empty((0,), dtype=np.int64)
        if anchor_indices is None
        else np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    )
    directional_endpoint_row_indices = (
        np.empty((0,), dtype=np.int64)
        if directional_endpoint_row_indices is None
        else np.asarray(directional_endpoint_row_indices, dtype=np.int64).reshape(-1)
    )
    use_directional_endpoint_solve = (
        directional_endpoint_row_indices.size == 2
        and directional_endpoint_feature_points is not None
        and directional_endpoint_feature_points.shape[0] >= 4
        and int(directional_endpoint_iterations) > 0
    )
    use_spacing_regularization = (
        use_regularization_terms
        and spacing_weight > 0.0
        and edge_pairs.shape[0] > 0
        and arc_chain_ranges.shape[0] > 0
        and baseline_normalized_arc_positions.shape[0] == num_arc_position_residuals
        and baseline_normalized_arc_positions.shape[0] > 0
    )
    use_anchor_regularization = (
        use_regularization_terms
        and anchor_weight > 0.0
        and anchor_indices.size > 0
        and anchor_points is not None
    )

    fixed_row_indices = np.empty((0,), dtype=np.int64)

    def set_rows_zero(rows_variable: csdl.Variable, row_indices: np.ndarray) -> csdl.Variable:
        row_indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if row_indices.size == 0:
            return rows_variable
        return rows_variable.set(
            csdl.slice[row_indices.tolist(), :],
            np.zeros((row_indices.size, rows_variable.shape[1]), dtype=float),
        )

    def print_residual(iteration_label: str, sdf_a: csdl.Variable, sdf_b: csdl.Variable) -> None:
        if not print_residuals:
            return
        sdf_a_value = np.asarray(sdf_a.value, dtype=float).reshape(-1)
        sdf_b_value = np.asarray(sdf_b.value, dtype=float).reshape(-1)
        residual_norm = np.sqrt(sdf_a_value**2 + sdf_b_value**2)
        rms_residual = np.sqrt(np.mean(residual_norm**2))
        print(
            f"{label} seam residual {iteration_label}: "
            f"max={np.max(residual_norm):.6e}, "
            f"mean={np.mean(residual_norm):.6e}, "
            f"rms={rms_residual:.6e}, "
            f"max_abs_sdf_a={np.max(np.abs(sdf_a_value)):.6e}, "
            f"max_abs_sdf_b={np.max(np.abs(sdf_b_value)):.6e}"
        )

    def print_regularization(iteration_label: str, points: csdl.Variable) -> None:
        if not print_residuals:
            return
        point_values = np.asarray(points.value, dtype=float).reshape((-1, 3))
        messages = []
        if use_spacing_regularization:
            cumulative_arc_positions = []
            for range_start, range_end in arc_chain_ranges:
                chain_edge_pairs = edge_pairs[int(range_start) : int(range_end)]
                if chain_edge_pairs.shape[0] <= 1:
                    continue
                edge_vectors = point_values[chain_edge_pairs[:, 1]] - point_values[
                    chain_edge_pairs[:, 0]
                ]
                edge_lengths = np.linalg.norm(edge_vectors, axis=1)
                cumulative_arc_positions.append(
                    np.cumsum(edge_lengths)[:-1] / max(float(np.sum(edge_lengths)), 1e-12)
                )
            spacing_residual = (
                np.concatenate(cumulative_arc_positions)
                - baseline_normalized_arc_positions
            )
            messages.append(
                "spacing_max="
                f"{np.max(np.abs(spacing_residual)):.6e}, "
                "spacing_rms="
                f"{np.sqrt(np.mean(spacing_residual**2)):.6e}"
            )
        if use_anchor_regularization:
            anchor_residual = (
                point_values[anchor_indices]
                - np.asarray(anchor_points.value, dtype=float).reshape((-1, 3))
            )
            anchor_norm = np.linalg.norm(anchor_residual, axis=1)
            messages.append(
                f"anchor_max={np.max(anchor_norm):.6e}, "
                f"anchor_mean={np.mean(anchor_norm):.6e}"
            )
        if messages:
            print(f"{label} seam regularization {iteration_label}: " + ", ".join(messages))

    def regularization_residual_blocks_csdl(points: csdl.Variable) -> list[csdl.Variable]:
        residual_blocks = []
        if use_spacing_regularization:
            baseline_offset = 0
            for range_start, range_end in arc_chain_ranges:
                chain_edge_pairs = edge_pairs[int(range_start) : int(range_end)]
                if chain_edge_pairs.shape[0] <= 1:
                    continue
                edge_start_points = _gather_rows_csdl(points, chain_edge_pairs[:, 0])
                edge_end_points = _gather_rows_csdl(points, chain_edge_pairs[:, 1])
                edge_vectors = edge_end_points - edge_start_points
                edge_lengths = csdl.sqrt(
                    csdl.sum(edge_vectors * edge_vectors, axes=(1,)) + 1e-14
                )
                cumulative_arc_matrix = np.tril(
                    np.ones(
                        (chain_edge_pairs.shape[0] - 1, chain_edge_pairs.shape[0]),
                        dtype=float,
                    )
                )
                cumulative_arc_lengths = csdl.reshape(
                    csdl.matmat(
                        cumulative_arc_matrix,
                        csdl.reshape(edge_lengths, (chain_edge_pairs.shape[0], 1)),
                    ),
                    (chain_edge_pairs.shape[0] - 1,),
                )
                normalized_arc_positions = cumulative_arc_lengths / (
                    csdl.sum(edge_lengths) + 1e-14
                )
                baseline_count = chain_edge_pairs.shape[0] - 1
                spacing_residual = (
                    normalized_arc_positions
                    - baseline_normalized_arc_positions[
                        baseline_offset : baseline_offset + baseline_count
                    ]
                )
                residual_blocks.append(
                    np.sqrt(spacing_weight) * spacing_length_scale * spacing_residual
                )
                baseline_offset += baseline_count
        if use_anchor_regularization:
            current_anchor_points = _gather_rows_csdl(points, anchor_indices)
            anchor_residual = current_anchor_points - anchor_points
            residual_blocks.append(
                csdl.reshape(
                    np.sqrt(anchor_weight) * anchor_residual,
                    (anchor_indices.size * points.shape[1],),
                )
            )
        return residual_blocks

    def regularization_loss_csdl(points: csdl.Variable):
        regularization_loss = 0.0
        for residual_block in regularization_residual_blocks_csdl(points):
            regularization_loss = regularization_loss + csdl.sum(residual_block * residual_block)
        return regularization_loss

    def regularization_gradient(points: csdl.Variable) -> csdl.Variable | None:
        if not (use_spacing_regularization or use_anchor_regularization):
            return None
        regularization_loss = regularization_loss_csdl(points)
        gradient_flat = csdl.derivative(regularization_loss, points)
        return csdl.reshape(gradient_flat, points.shape)

    def project_vectors_to_tangent(
        vectors: csdl.Variable,
        gradient_a: csdl.Variable,
        gradient_b: csdl.Variable,
        normal_matrix_aa: csdl.Variable,
        normal_matrix_ab: csdl.Variable,
        normal_matrix_bb: csdl.Variable,
        determinant: csdl.Variable,
    ) -> csdl.Variable:
        rhs_a = csdl.sum(gradient_a * vectors, axes=(1,))
        rhs_b = csdl.sum(gradient_b * vectors, axes=(1,))
        multiplier_a = (normal_matrix_bb * rhs_a - normal_matrix_ab * rhs_b) / determinant
        multiplier_b = (-normal_matrix_ab * rhs_a + normal_matrix_aa * rhs_b) / determinant
        normal_component = (
            _expand_vector_to_columns(multiplier_a, 3) * gradient_a
            + _expand_vector_to_columns(multiplier_b, 3) * gradient_b
        )
        return vectors - normal_component

    if use_directional_endpoint_solve:
        endpoint_feature_points = directional_endpoint_feature_points
        endpoint_base_feature_points = _gather_rows_csdl(
            endpoint_feature_points,
            np.array([0, 2], dtype=np.int64),
        )
        endpoint_nearby_feature_points = _gather_rows_csdl(
            endpoint_feature_points,
            np.array([1, 3], dtype=np.int64),
        )
        endpoint_direction_rows = endpoint_nearby_feature_points - endpoint_base_feature_points
        endpoint_direction_norms = csdl.sqrt(
            csdl.sum(endpoint_direction_rows * endpoint_direction_rows, axes=(1,)) + 1e-14
        )
        endpoint_direction_rows = endpoint_direction_rows / _expand_vector_to_columns(
            endpoint_direction_norms,
            relaxed_points.shape[1],
        )
        endpoint_chord_direction = endpoint_base_feature_points[csdl.slice[1, :]] - endpoint_base_feature_points[
            csdl.slice[0, :]
        ]
        endpoint_chord_direction_norm = csdl.sqrt(
            csdl.sum(endpoint_chord_direction * endpoint_chord_direction) + 1e-14
        )
        endpoint_chord_direction_unit = csdl.reshape(
            endpoint_chord_direction / endpoint_chord_direction_norm,
            (1, relaxed_points.shape[1]),
        )
        endpoint_chord_direction_rows = csdl.matmat(
            np.ones((directional_endpoint_row_indices.size, 1), dtype=float),
            endpoint_chord_direction_unit,
        )
        endpoint_chord_projection = csdl.sum(
            endpoint_chord_direction_rows * endpoint_direction_rows,
            axes=(1,),
        )
        endpoint_secondary_direction_rows = (
            endpoint_chord_direction_rows
            - _expand_vector_to_columns(endpoint_chord_projection, relaxed_points.shape[1])
            * endpoint_direction_rows
        )
        endpoint_secondary_direction_norms = csdl.sqrt(
            csdl.sum(
                endpoint_secondary_direction_rows * endpoint_secondary_direction_rows,
                axes=(1,),
            )
            + 1e-14
        )
        endpoint_secondary_direction_rows = endpoint_secondary_direction_rows / (
            _expand_vector_to_columns(endpoint_secondary_direction_norms, relaxed_points.shape[1])
        )
        endpoint_row_list = directional_endpoint_row_indices.tolist()
        endpoint_points = relaxed_points[csdl.slice[endpoint_row_list, :]]
        for endpoint_iteration in range(int(directional_endpoint_iterations)):
            endpoint_sdf_operation_a = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_a)
            endpoint_sdf_operation_b = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_b)
            endpoint_sdf_a = endpoint_sdf_operation_a.evaluate(coefficients_a, endpoint_points)
            endpoint_sdf_b = endpoint_sdf_operation_b.evaluate(coefficients_b, endpoint_points)
            endpoint_gradient_a = _signed_distance_gradient_from_ad_csdl(
                endpoint_sdf_a, endpoint_points
            )
            endpoint_gradient_b = _signed_distance_gradient_from_ad_csdl(
                endpoint_sdf_b, endpoint_points
            )
            directional_sensitivity_a = csdl.sum(
                endpoint_gradient_a * endpoint_direction_rows,
                axes=(1,),
            )
            directional_sensitivity_b = csdl.sum(
                endpoint_gradient_b * endpoint_direction_rows,
                axes=(1,),
            )
            secondary_sensitivity_a = csdl.sum(
                endpoint_gradient_a * endpoint_secondary_direction_rows,
                axes=(1,),
            )
            secondary_sensitivity_b = csdl.sum(
                endpoint_gradient_b * endpoint_secondary_direction_rows,
                axes=(1,),
            )
            normal_matrix_11 = (
                directional_sensitivity_a * directional_sensitivity_a
                + directional_sensitivity_b * directional_sensitivity_b
                + damping
            )
            normal_matrix_12 = (
                directional_sensitivity_a * secondary_sensitivity_a
                + directional_sensitivity_b * secondary_sensitivity_b
            )
            normal_matrix_22 = (
                secondary_sensitivity_a * secondary_sensitivity_a
                + secondary_sensitivity_b * secondary_sensitivity_b
                + damping
            )
            normal_rhs_1 = (
                endpoint_sdf_a * directional_sensitivity_a
                + endpoint_sdf_b * directional_sensitivity_b
            )
            normal_rhs_2 = (
                endpoint_sdf_a * secondary_sensitivity_a
                + endpoint_sdf_b * secondary_sensitivity_b
            )
            normal_determinant = (
                normal_matrix_11 * normal_matrix_22
                - normal_matrix_12 * normal_matrix_12
                + 1e-14
            )
            directional_step = (
                normal_matrix_22 * normal_rhs_1 - normal_matrix_12 * normal_rhs_2
            ) / normal_determinant
            secondary_step = (
                -normal_matrix_12 * normal_rhs_1 + normal_matrix_11 * normal_rhs_2
            ) / normal_determinant
            endpoint_correction = (
                _expand_vector_to_columns(directional_step, relaxed_points.shape[1])
                * endpoint_direction_rows
                + _expand_vector_to_columns(secondary_step, relaxed_points.shape[1])
                * endpoint_secondary_direction_rows
            )
            endpoint_points = endpoint_points - directional_endpoint_step_size * endpoint_correction
        relaxed_points = relaxed_points.set(
            csdl.slice[endpoint_row_list, :],
            endpoint_points,
        )
        fixed_row_indices = directional_endpoint_row_indices if freeze_directional_endpoint_rows else np.empty((0,), dtype=np.int64)
        if use_anchor_regularization:
            anchor_target_points = anchor_points
            anchor_row_lookup = {int(row_index): i for i, row_index in enumerate(anchor_indices)}
            directional_anchor_rows = []
            directional_anchor_target_rows = []
            for endpoint_local_index, seam_row in enumerate(directional_endpoint_row_indices):
                anchor_local_index = anchor_row_lookup.get(int(seam_row))
                if anchor_local_index is None:
                    continue
                directional_anchor_rows.append(anchor_local_index)
                directional_anchor_target_rows.append(endpoint_local_index)
            if directional_anchor_rows:
                anchor_target_points = anchor_target_points.set(
                    csdl.slice[directional_anchor_rows, :],
                    endpoint_points[
                        csdl.slice[directional_anchor_target_rows, :]
                    ],
                )
            anchor_points = anchor_target_points

    for iteration_index in range(int(num_iterations)):
        sdf_operation_a = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_a)
        sdf_operation_b = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_b)
        sdf_a = sdf_operation_a.evaluate(coefficients_a, relaxed_points)
        sdf_b = sdf_operation_b.evaluate(coefficients_b, relaxed_points)
        print_residual(f"iteration {iteration_index}", sdf_a, sdf_b)
        if use_spacing_regularization or use_anchor_regularization:
            print_regularization(f"iteration {iteration_index}", relaxed_points)

        if use_unified_objective:
            residual_blocks = [
                csdl.reshape(sdf_a, (relaxed_points.shape[0],)),
                csdl.reshape(sdf_b, (relaxed_points.shape[0],)),
            ]
            residual_blocks.extend(regularization_residual_blocks_csdl(relaxed_points))
            residual_vector = csdl.concatenate(tuple(residual_blocks), axis=0)
            residual_jacobian = csdl.derivative(residual_vector, relaxed_points)
            normal_matrix = (
                csdl.matmat(csdl.transpose(residual_jacobian), residual_jacobian)
                + damping
                * np.eye(relaxed_points.shape[0] * relaxed_points.shape[1], dtype=float)
            )
            normal_rhs = csdl.matmat(
                csdl.transpose(residual_jacobian),
                csdl.reshape(residual_vector, (residual_vector.shape[0], 1)),
            )
            step_flat = csdl.solve_linear(normal_matrix, normal_rhs)
            step = csdl.reshape(
                step_flat,
                relaxed_points.shape,
            )
            step = set_rows_zero(step, fixed_row_indices)
            relaxed_points = relaxed_points - unified_objective_step_size * step
            continue

        gradient_a = _signed_distance_gradient_from_ad_csdl(sdf_a, relaxed_points)
        gradient_b = _signed_distance_gradient_from_ad_csdl(sdf_b, relaxed_points)

        normal_matrix_aa = csdl.sum(gradient_a * gradient_a, axes=(1,)) + damping
        normal_matrix_ab = csdl.sum(gradient_a * gradient_b, axes=(1,))
        normal_matrix_bb = csdl.sum(gradient_b * gradient_b, axes=(1,)) + damping
        determinant = (
            normal_matrix_aa * normal_matrix_bb
            - normal_matrix_ab * normal_matrix_ab
            + 1e-14
        )

        multiplier_a = (normal_matrix_bb * sdf_a - normal_matrix_ab * sdf_b) / determinant
        multiplier_b = (-normal_matrix_ab * sdf_a + normal_matrix_aa * sdf_b) / determinant
        correction = (
            _expand_vector_to_columns(multiplier_a, 3) * gradient_a
            + _expand_vector_to_columns(multiplier_b, 3) * gradient_b
        )
        correction = set_rows_zero(correction, fixed_row_indices)
        if use_spacing_regularization or use_anchor_regularization:
            gradient_regularization = regularization_gradient(relaxed_points)
            if regularization_mode == "projected_tangent":
                gradient_regularization = project_vectors_to_tangent(
                    gradient_regularization,
                    gradient_a,
                    gradient_b,
                    normal_matrix_aa,
                    normal_matrix_ab,
                    normal_matrix_bb,
                    determinant,
                )
            elif regularization_mode != "unprojected":
                raise ValueError(f"Unsupported seam regularization mode: {regularization_mode}")
            gradient_regularization = set_rows_zero(gradient_regularization, fixed_row_indices)
            relaxed_points = (
                relaxed_points
                - step_size * correction
                - regularization_step_size * gradient_regularization
            )
        else:
            relaxed_points = relaxed_points - step_size * correction

    if (use_spacing_regularization or use_anchor_regularization) and not use_unified_objective:
        cleanup_sdf_operation_a = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_a)
        cleanup_sdf_operation_b = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_b)
        cleanup_sdf_a = cleanup_sdf_operation_a.evaluate(coefficients_a, relaxed_points)
        cleanup_sdf_b = cleanup_sdf_operation_b.evaluate(coefficients_b, relaxed_points)
        cleanup_gradient_a = _signed_distance_gradient_from_ad_csdl(cleanup_sdf_a, relaxed_points)
        cleanup_gradient_b = _signed_distance_gradient_from_ad_csdl(cleanup_sdf_b, relaxed_points)
        cleanup_normal_matrix_aa = csdl.sum(cleanup_gradient_a * cleanup_gradient_a, axes=(1,)) + damping
        cleanup_normal_matrix_ab = csdl.sum(cleanup_gradient_a * cleanup_gradient_b, axes=(1,))
        cleanup_normal_matrix_bb = csdl.sum(cleanup_gradient_b * cleanup_gradient_b, axes=(1,)) + damping
        cleanup_determinant = (
            cleanup_normal_matrix_aa * cleanup_normal_matrix_bb
            - cleanup_normal_matrix_ab * cleanup_normal_matrix_ab
            + 1e-14
        )
        cleanup_multiplier_a = (
            cleanup_normal_matrix_bb * cleanup_sdf_a - cleanup_normal_matrix_ab * cleanup_sdf_b
        ) / cleanup_determinant
        cleanup_multiplier_b = (
            -cleanup_normal_matrix_ab * cleanup_sdf_a + cleanup_normal_matrix_aa * cleanup_sdf_b
        ) / cleanup_determinant
        cleanup_correction = (
            _expand_vector_to_columns(cleanup_multiplier_a, 3) * cleanup_gradient_a
            + _expand_vector_to_columns(cleanup_multiplier_b, 3) * cleanup_gradient_b
        )
        cleanup_correction = set_rows_zero(cleanup_correction, fixed_row_indices)
        relaxed_points = relaxed_points - step_size * cleanup_correction

    final_sdf_operation_a = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_a)
    final_sdf_operation_b = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_b)
    final_sdf_a = final_sdf_operation_a.evaluate(coefficients_a, relaxed_points)
    final_sdf_b = final_sdf_operation_b.evaluate(coefficients_b, relaxed_points)
    print_residual("final", final_sdf_a, final_sdf_b)
    print_regularization("final", relaxed_points)

    return relaxed_points


def _wing_fuse_seam_points_from_parametric_bracket_csdl(
    moved_wing_coefficients: csdl.Variable,
    moved_fuse_coefficients: csdl.Variable,
    wing_evaluation_model,
    fuse_sdf_model,
    wing_parametric_coordinates: np.ndarray,
    *,
    spanwise_uv_axis: int,
    tolerance: float,
    max_iter: int,
    label: str,
    print_status: bool = True,
) -> csdl.Variable:
    wing_parametric_coordinates = np.asarray(
        wing_parametric_coordinates,
        dtype=float,
    ).reshape((-1, 3))
    if wing_parametric_coordinates.shape[0] == 0:
        return csdl.Variable(value=np.empty((0, 3), dtype=float))

    spanwise_uv_axis = int(spanwise_uv_axis)
    if spanwise_uv_axis not in (0, 1):
        raise ValueError(f"spanwise_uv_axis must be 0 or 1. Got {spanwise_uv_axis}.")
    span_column = 1 + spanwise_uv_axis
    fixed_column = 1 + (1 - spanwise_uv_axis)

    num_points = wing_parametric_coordinates.shape[0]
    span_initial = np.clip(wing_parametric_coordinates[:, span_column], 1e-8, 1.0 - 1e-8)
    span_coordinates = csdl.ImplicitVariable(value=span_initial)
    span_column_variable = csdl.reshape(span_coordinates, (num_points, 1))
    patch_id_column = wing_parametric_coordinates[:, 0:1]
    fixed_coordinate_column = wing_parametric_coordinates[:, fixed_column : fixed_column + 1]

    if spanwise_uv_axis == 0:
        solved_parametric_coordinates = csdl.concatenate(
            (patch_id_column, span_column_variable, fixed_coordinate_column),
            axis=1,
        )
    else:
        solved_parametric_coordinates = csdl.concatenate(
            (patch_id_column, fixed_coordinate_column, span_column_variable),
            axis=1,
        )

    wing_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
        model=wing_evaluation_model,
    )
    fuse_sdf_operation = bsm3.FunctionSetClosestDistanceOperation(model=fuse_sdf_model)
    seam_points = wing_evaluation_operation.evaluate(
        moved_wing_coefficients,
        solved_parametric_coordinates,
    )
    fuse_sdf_residual = fuse_sdf_operation.evaluate(
        moved_fuse_coefficients,
        seam_points,
    )

    solver = csdl.nonlinear_solvers.BracketedSearch(
        name=f"{label}_parametric_bracketed_root",
        print_status=print_status,
        tolerance=float(tolerance),
        max_iter=int(max_iter),
        residual_jac_kwargs={"elementwise": True},
    )
    solver.add_state(
        span_coordinates,
        fuse_sdf_residual,
        bracket=(
            np.zeros((num_points,), dtype=float),
            np.ones((num_points,), dtype=float),
        ),
        tolerance=float(tolerance),
    )
    solver.run()

    if spanwise_uv_axis == 0:
        final_parametric_coordinates = csdl.concatenate(
            (patch_id_column, csdl.reshape(span_coordinates, (num_points, 1)), fixed_coordinate_column),
            axis=1,
        )
    else:
        final_parametric_coordinates = csdl.concatenate(
            (patch_id_column, fixed_coordinate_column, csdl.reshape(span_coordinates, (num_points, 1))),
            axis=1,
        )
    final_wing_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
        model=wing_evaluation_model,
    )
    final_seam_points = final_wing_evaluation_operation.evaluate(
        moved_wing_coefficients,
        final_parametric_coordinates,
    )
    final_fuse_sdf_operation = bsm3.FunctionSetClosestDistanceOperation(model=fuse_sdf_model)
    final_fuse_sdf = final_fuse_sdf_operation.evaluate(
        moved_fuse_coefficients,
        final_seam_points,
    )
    if print_status:
        final_fuse_sdf_value = np.asarray(final_fuse_sdf.value, dtype=float).reshape(-1)
        print(
            f"{label} parametric seam bracketed root residual: "
            f"max_abs_fuse_sdf={np.max(np.abs(final_fuse_sdf_value)):.6e}, "
            f"mean_abs_fuse_sdf={np.mean(np.abs(final_fuse_sdf_value)):.6e}"
        )
    return final_seam_points


def _component_fuse_seam_points_from_soft_parametric_bracket_csdl(
    moved_component_coefficients: csdl.Variable,
    moved_fuse_coefficients: csdl.Variable,
    component_evaluation_model,
    fuse_sdf_model,
    component_parametric_coordinates: np.ndarray,
    *,
    spanwise_uv_axis: int,
    num_iterations: int,
    sign_sharpness: float,
    sdf_scale: float,
    label: str,
    print_status: bool = True,
    debug_sample_indices: np.ndarray | None = None,
    debug_trace: dict | None = None,
) -> csdl.Variable:
    component_parametric_coordinates = np.asarray(
        component_parametric_coordinates,
        dtype=float,
    ).reshape((-1, 3))
    if component_parametric_coordinates.shape[0] == 0:
        return csdl.Variable(value=np.empty((0, 3), dtype=float))

    spanwise_uv_axis = int(spanwise_uv_axis)
    if spanwise_uv_axis not in (0, 1):
        raise ValueError(f"spanwise_uv_axis must be 0 or 1. Got {spanwise_uv_axis}.")
    span_column = 1 + spanwise_uv_axis
    fixed_column = 1 + (1 - spanwise_uv_axis)

    num_points = component_parametric_coordinates.shape[0]
    lower_span = csdl.Variable(value=np.zeros((num_points,), dtype=float))
    upper_span = csdl.Variable(value=np.ones((num_points,), dtype=float))
    patch_id_column = component_parametric_coordinates[:, 0:1]
    fixed_coordinate_column = component_parametric_coordinates[:, fixed_column : fixed_column + 1]
    num_iterations = max(int(num_iterations), 0)
    sign_sharpness = max(float(sign_sharpness), 0.0)
    sdf_scale_safe = max(float(sdf_scale), 1e-12)
    debug_sample_indices = (
        np.empty((0,), dtype=np.int64)
        if debug_sample_indices is None
        else np.asarray(debug_sample_indices, dtype=np.int64).reshape(-1)
    )
    debug_sample_indices = debug_sample_indices[
        (debug_sample_indices >= 0) & (debug_sample_indices < num_points)
    ]
    debug_enabled = debug_trace is not None and debug_sample_indices.size > 0
    midpoint_debug_points: list[np.ndarray] = []
    midpoint_debug_sdf: list[np.ndarray] = []
    bracket_lower_debug_points: list[np.ndarray] = []
    bracket_lower_debug_sdf: list[np.ndarray] = []
    bracket_upper_debug_points: list[np.ndarray] = []
    bracket_upper_debug_sdf: list[np.ndarray] = []

    def evaluate_points_and_fuse_sdf_at_span(
        span_values: csdl.Variable,
    ) -> tuple[csdl.Variable, csdl.Variable]:
        span_column_variable = csdl.reshape(span_values, (num_points, 1))
        if spanwise_uv_axis == 0:
            parametric_coordinates = csdl.concatenate(
                (patch_id_column, span_column_variable, fixed_coordinate_column),
                axis=1,
            )
        else:
            parametric_coordinates = csdl.concatenate(
                (patch_id_column, fixed_coordinate_column, span_column_variable),
                axis=1,
            )
        component_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
            model=component_evaluation_model,
        )
        fuse_sdf_operation = bsm3.FunctionSetClosestDistanceOperation(model=fuse_sdf_model)
        points = component_evaluation_operation.evaluate(
            moved_component_coefficients,
            parametric_coordinates,
        )
        return points, fuse_sdf_operation.evaluate(moved_fuse_coefficients, points)

    lower_points, lower_sdf = evaluate_points_and_fuse_sdf_at_span(lower_span)
    initial_upper_points = None
    initial_upper_sdf = None
    if print_status:
        initial_upper_points, initial_upper_sdf = evaluate_points_and_fuse_sdf_at_span(upper_span)
        lower_sdf_value = np.asarray(lower_sdf.value, dtype=float).reshape(-1)
        upper_sdf_value = np.asarray(initial_upper_sdf.value, dtype=float).reshape(-1)
        failed_bracket_count = int(np.count_nonzero(lower_sdf_value * upper_sdf_value > 0.0))
        print(
            f"{label} soft parametric seam initial bracket: "
            f"failed_sign_changes={failed_bracket_count}/{num_points}, "
            f"max_abs_lower_sdf={np.max(np.abs(lower_sdf_value)):.6e}, "
            f"max_abs_upper_sdf={np.max(np.abs(upper_sdf_value)):.6e}"
        )
    if debug_enabled:
        if initial_upper_points is None:
            initial_upper_points, initial_upper_sdf = evaluate_points_and_fuse_sdf_at_span(upper_span)
        debug_trace.clear()
        debug_trace.update(
            {
                "label": label,
                "sample_indices": debug_sample_indices.copy(),
                "initial_lower_points": np.asarray(
                    lower_points.value,
                    dtype=float,
                )[debug_sample_indices],
                "initial_upper_points": np.asarray(
                    initial_upper_points.value,
                    dtype=float,
                )[debug_sample_indices],
                "initial_lower_sdf": np.asarray(
                    lower_sdf.value,
                    dtype=float,
                ).reshape(-1)[debug_sample_indices],
                "initial_upper_sdf": np.asarray(
                    initial_upper_sdf.value,
                    dtype=float,
                ).reshape(-1)[debug_sample_indices],
            }
        )
    for _ in range(num_iterations):
        if debug_enabled:
            current_lower_points, current_lower_sdf = evaluate_points_and_fuse_sdf_at_span(
                lower_span
            )
            current_upper_points, current_upper_sdf = evaluate_points_and_fuse_sdf_at_span(
                upper_span
            )
            bracket_lower_debug_points.append(
                np.asarray(current_lower_points.value, dtype=float)[debug_sample_indices]
            )
            bracket_lower_debug_sdf.append(
                np.asarray(current_lower_sdf.value, dtype=float).reshape(-1)[
                    debug_sample_indices
                ]
            )
            bracket_upper_debug_points.append(
                np.asarray(current_upper_points.value, dtype=float)[debug_sample_indices]
            )
            bracket_upper_debug_sdf.append(
                np.asarray(current_upper_sdf.value, dtype=float).reshape(-1)[
                    debug_sample_indices
                ]
            )
        middle_span = 0.5 * (lower_span + upper_span)
        middle_points, middle_sdf = evaluate_points_and_fuse_sdf_at_span(middle_span)
        if debug_enabled:
            midpoint_debug_points.append(
                np.asarray(middle_points.value, dtype=float)[debug_sample_indices]
            )
            midpoint_debug_sdf.append(
                np.asarray(middle_sdf.value, dtype=float).reshape(-1)[debug_sample_indices]
            )
        sign_product = lower_sdf * middle_sdf
        product_scale = sdf_scale_safe * sdf_scale_safe
        sign_product_magnitude = csdl.sqrt(
            sign_product * sign_product + product_scale * product_scale
        )
        normalized_sign_product = sign_product / (product_scale + sign_product_magnitude)
        same_side_weight = 0.5 * (
            1.0 + csdl.tanh(sign_sharpness * normalized_sign_product)
        )
        lower_span = same_side_weight * middle_span + (1.0 - same_side_weight) * lower_span
        upper_span = same_side_weight * upper_span + (1.0 - same_side_weight) * middle_span
        _, lower_sdf = evaluate_points_and_fuse_sdf_at_span(lower_span)

    final_span = 0.5 * (lower_span + upper_span)
    final_span_column = csdl.reshape(final_span, (num_points, 1))
    if spanwise_uv_axis == 0:
        final_parametric_coordinates = csdl.concatenate(
            (patch_id_column, final_span_column, fixed_coordinate_column),
            axis=1,
        )
    else:
        final_parametric_coordinates = csdl.concatenate(
            (patch_id_column, fixed_coordinate_column, final_span_column),
            axis=1,
        )
    final_component_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
        model=component_evaluation_model,
    )
    final_seam_points = final_component_evaluation_operation.evaluate(
        moved_component_coefficients,
        final_parametric_coordinates,
    )
    final_fuse_sdf_operation = bsm3.FunctionSetClosestDistanceOperation(model=fuse_sdf_model)
    final_fuse_sdf = final_fuse_sdf_operation.evaluate(
        moved_fuse_coefficients,
        final_seam_points,
    )
    if print_status:
        final_fuse_sdf_value = np.asarray(final_fuse_sdf.value, dtype=float).reshape(-1)
        bracket_width_value = np.asarray((upper_span - lower_span).value, dtype=float).reshape(-1)
        print(
            f"{label} soft parametric seam bracket residual: "
            f"max_abs_fuse_sdf={np.max(np.abs(final_fuse_sdf_value)):.6e}, "
            f"mean_abs_fuse_sdf={np.mean(np.abs(final_fuse_sdf_value)):.6e}, "
            f"max_span_width={np.max(np.abs(bracket_width_value)):.6e}"
        )
    if debug_enabled:
        debug_trace.update(
            {
                "midpoint_points": np.asarray(midpoint_debug_points, dtype=float),
                "midpoint_sdf": np.asarray(midpoint_debug_sdf, dtype=float),
                "bracket_lower_points": np.asarray(bracket_lower_debug_points, dtype=float),
                "bracket_lower_sdf": np.asarray(bracket_lower_debug_sdf, dtype=float),
                "bracket_upper_points": np.asarray(bracket_upper_debug_points, dtype=float),
                "bracket_upper_sdf": np.asarray(bracket_upper_debug_sdf, dtype=float),
                "final_points": np.asarray(final_seam_points.value, dtype=float)[
                    debug_sample_indices
                ],
                "final_sdf": np.asarray(final_fuse_sdf.value, dtype=float).reshape(-1)[
                    debug_sample_indices
                ],
            }
        )
    return final_seam_points


def _print_two_component_target_sdf_diagnostic_csdl(
    label: str,
    target_points: csdl.Variable,
    coefficients_a: csdl.Variable,
    coefficients_b: csdl.Variable,
    sdf_model_a,
    sdf_model_b,
) -> None:
    if target_points.shape[0] == 0:
        return

    sdf_operation_a = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_a)
    sdf_operation_b = bsm3.FunctionSetClosestDistanceOperation(model=sdf_model_b)
    sdf_a = sdf_operation_a.evaluate(coefficients_a, target_points)
    sdf_b = sdf_operation_b.evaluate(coefficients_b, target_points)
    sdf_a_value = np.asarray(sdf_a.value, dtype=float).reshape(-1)
    sdf_b_value = np.asarray(sdf_b.value, dtype=float).reshape(-1)
    residual_norm = np.sqrt(sdf_a_value**2 + sdf_b_value**2)
    print(
        f"{label} corrected seam target residual before RBF: "
        f"max={np.max(residual_norm):.6e}, "
        f"mean={np.mean(residual_norm):.6e}, "
        f"rms={np.sqrt(np.mean(residual_norm**2)):.6e}, "
        f"max_abs_sdf_a={np.max(np.abs(sdf_a_value)):.6e}, "
        f"max_abs_sdf_b={np.max(np.abs(sdf_b_value)):.6e}"
    )


if __name__ == "__main__":
    from bsm3.core.boundary_surface_movement.embraer_175_geom_parameterization import evaluate_E175_geometry_parameterization

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    plot = PLOT_FINAL_MESH
    plot_step8_prediction = PLOT_STEP8_PREDICTION
    plot_step8_deformed_geometry_overlay = PLOT_STEP8_DEFORMED_GEOMETRY_OVERLAY
    plot_seam_relaxation_debug = PLOT_SEAM_RELAXATION_DEBUG
    plot_soft_bracket_debug = PLOT_SOFT_BRACKET_DEBUG
    plot_soft_bracket_mesh_overlay = PLOT_SOFT_BRACKET_MESH_OVERLAY
    animate_soft_bracket_debug = ANIMATE_SOFT_BRACKET_DEBUG
    soft_bracket_debug_enabled = plot_soft_bracket_debug or animate_soft_bracket_debug
    soft_bracket_debug_num_points = SOFT_BRACKET_DEBUG_POINTS
    soft_bracket_debug_surface = SOFT_BRACKET_DEBUG_SURFACE
    soft_bracket_debug_gif_path = SOFT_BRACKET_DEBUG_GIF_PATH
    plot_step8_mesh_opacity = 0.75
    plot_step8_deformed_geometry_opacity = 0.25
    mesh_path = MESH_PATH

    alpha = 1.0
    sigma_phi = 0.1
    master_component_name = "wing+horizontal_tail"
    rbf_fit_mode = "exact_interpolation"
    rbf_kernel = "cubic"
    rbf_kernel_scale_factor = 0.01
    rbf_regularization_factor = 1e-6
    rbf_polynomial_regularization_factor = 1e-15
    global_interpolation_sample_weight = 1.0
    seam_training_sample_weight = 1.0
    aft_fuse_symmetry_anchor_training_sample_weight = 1.0
    aft_fuse_symmetry_anchor_downsample_stride = 2
    aft_fuse_symmetry_neighbor_training_sample_weight = 0.35
    aft_fuse_symmetry_neighbor_blend = 0.35
    num_additional_interp_vertices = 750

    intersection_tolerance = 1e-3
    wing_surface_projection_tolerance = 1e-4
    aft_fuse_symmetry_anchors = True
    aft_fuse_symmetry_anchor_x_fraction = 0.2
    aft_fuse_symmetry_anchor_y_tolerance = 1e-10
    aft_fuse_symmetry_anchor_hard_enforcement = True
    aft_fuse_fixed_region = False
    aft_fuse_fixed_buffer_htail_chords = 0.4
    outboard_wing_parametric_fix = True
    outboard_wing_parametric_fix_normalized_y_min = 0.50
    outboard_htail_parametric_fix = True
    outboard_htail_parametric_fix_normalized_y_min = 0.0
    
    seam_support_sigma_factor = 0.15
    setup_seam_support_cutoff = 0.05
    seam_graph_radius_factor = 0.08
    translation_envelope_mac = 2.0
    rotation_envelope_degrees = 8.0
    half_mesh_symmetry_mode = True
    symmetry_axis = 1
    symmetry_plane_tolerance = 1e-10
    symmetry_mirror_tolerance = 1e-8
    symmetry_solve_side = "positive"
    
    
    fuselage_diameter_taper_before_empennage = True
    fuselage_diameter_taper_margin_htail_chords = 0.50
    fuselage_diameter_taper_length_htail_chords = 2.00
    
    exact_seam_sdf_relaxation = True
    exact_seam_wing_fuse_recompute_mode = "wing_parametric_soft_bracket"
    exact_seam_htail_fuse_recompute_mode = "htail_parametric_soft_bracket"
    exact_seam_parametric_root_tolerance = 1e-10
    exact_seam_parametric_root_max_iter = 100
    exact_seam_soft_bracket_iterations = 10
    exact_seam_soft_bracket_sign_sharpness = 50.0
    exact_seam_soft_bracket_sdf_scale = 1e-3
    exact_seam_sdf_relaxation_iterations = 8
    exact_seam_sdf_relaxation_step_size = 1.0
    exact_seam_sdf_relaxation_damping = 1e-4
    exact_seam_curve_regularization = True
    exact_seam_spacing_regularization_weight = 1e-5
    exact_seam_anchor_regularization_weight = 0.
    exact_seam_regularization_step_size = 0.25
    exact_seam_regularization_mode = "unprojected" # "projected_tangent" # "unified_nonlinear_least_squares" #  
    exact_seam_unified_objective_step_size = 0.5
    
    # master_5 keeps the soft wing/fuselage exact-seam recompute from
    # master_4, but otherwise uses the compact interpolation/training
    # assembly from master_3.  The extra master_4 seam-neighbor, LE/TE
    # feature, halo, and solve-vertex driving paths are disabled below.
    exact_seam_directional_endpoint_solve = False
    exact_seam_directional_endpoint_iterations = 6
    exact_seam_directional_endpoint_step_size = 1.0
    exact_seam_directional_endpoint_freeze = True
    exact_seam_endpoint_feature_driving = False
    exact_seam_endpoint_feature_graph_radius_factor = 0.06
    exact_seam_endpoint_feature_full_strength_graph_radius_factor = 0.015
    exact_seam_endpoint_feature_support_cutoff = 1e-3
    
    exact_seam_endpoint_feature_halo_driving = False
    exact_seam_endpoint_feature_halo_graph_radius_factor = 0.008
    exact_seam_endpoint_feature_halo_full_strength_graph_radius_factor = 0.0015
    exact_seam_endpoint_feature_halo_max_blend = 0.55
    exact_seam_endpoint_feature_halo_support_cutoff = 1e-3
    
    endpoint_feature_parametric_boundary_tolerance = 1e-6
    endpoint_feature_coordinate_tolerance = 1e-8
    exact_seam_neighbor_driving = True
    exact_seam_neighbor_graph_radius_factor = 0.02
    exact_seam_neighbor_support_cutoff = 1e-3
    exact_seam_neighbor_training = False
    exact_seam_neighbor_training_graph_radius_factor = 0.06
    max_exact_seam_neighbor_training_vertices = 1200
    exact_seam_solve_vertex_driving = False
    exact_seam_solve_vertex_driving_strength = 1.0

    step9_projection_mode = "setup_patch_eligibility"
    step9_smooth_union_ks_rho_start = 20.
    step9_smooth_union_ks_rho_end = 20.
    step9_smooth_union_num_iterations = 3 #  4 #  

    wing_feature_protection = True
    htail_feature_protection = True
    sharp_feature_angle_degrees = 110.0
    sharp_feature_include_boundary = True
    step9_pre_projection_laplacian_step = 0.2

    
    embraer_175 = lfs.import_file_patched(
        Path(__file__).with_name("e175.stp"),
        parallelize=False,
    )
    
    fuse_keys = np.arange(0, 8)
    wing_keys = np.arange(8, 20)
    htail_keys = np.arange(20, len(embraer_175.functions))
    fuse_function_set = lfs.FunctionSet(functions={key: embraer_175.functions[key] for key in fuse_keys})
    wing_function_set = lfs.FunctionSet(functions={key: embraer_175.functions[key] for key in wing_keys})
    htail_function_set = lfs.FunctionSet(functions={key: embraer_175.functions[key] for key in htail_keys})
    undeformed_fuse_coefficients = _stack_coefficients_numpy(fuse_function_set)
    undeformed_wing_coefficients = _stack_coefficients_numpy(wing_function_set)
    undeformed_htail_coefficients = _stack_coefficients_numpy(htail_function_set)

    ######################## GEOMETRY PARAMETERIZATION DESIGN VARIABLES ########################
    num_ffd_coefficients_chordwise = 8
    num_ffd_sections = 11
    num_semispan_ffd_sections = num_ffd_sections // 2 + 1

    twist_coeffs = csdl.Variable(shape=(8,), value=0. * np.ones(8) * np.pi/180, name='twist_b_spline_coefficients')
    twist_coeffs.set_as_design_variable(lower=-5*np.pi/180, upper=5*np.pi/180, scaler=15)
    delta_camber_percent_design_dof = csdl.Variable(
        shape=(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1),
        value=np.random.rand(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1) * 0.,
        name='delta_camber_percent_design_dof',
    )
    delta_thickness_percent_design_dof = csdl.Variable(
        shape=(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1),
        value=np.random.rand(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1) * 0.,
        name='delta_thickness_percent_design_dof',
    )
    delta_camber_percent_design_dof.set_as_design_variable(lower=-5.0, upper=5.0, scaler=0.2)
    # delta_thickness_percent_design_dof.set_as_design_variable(lower=-5.0, upper=5.0, scaler=0.2)

    wing_area_dv = csdl.Variable(shape=(1,), name='wing_area', value=np.array([70.])) #70
    wing_AR_dv = csdl.Variable(shape=(1,), name='wing_AR', value=np.array([8.4])) # 8.4
    wing_area_dv.set_as_design_variable(lower=0.75 * 70., upper=1.25 * 70., scaler=1/70.)
    wing_AR_dv.set_as_design_variable(lower=0.75 * 8.4, upper=1.25 * 8.4, scaler=1/8.4)

    htail_area_dv = csdl.Variable(shape=(1,), name='htail_area', value=np.array([23.25])) # 23.25
    htail_AR_dv = csdl.Variable(shape=(1,), name='htail_AR', value=np.array([4.3])) # 4.3
    htail_area_dv.set_as_design_variable(lower=0.75 * 23.25, upper=1.25 * 23.25, scaler=1/23.25)
    htail_AR_dv.set_as_design_variable(lower=0.75 * 4.3, upper=1.25 * 4.3, scaler=1/4.3)

    wing_translation_amount = csdl.Variable(name="wing_translation_amount", value=0.)
    wing_translation_amount.set_as_design_variable(lower=-2., upper=1.5, scaler=1.0)
    
    fuselage_diameter_scale_initial = 1.0
    fuselage_diameter_scale_lower = 0.75
    fuselage_diameter_scale_upper = 1.25
    fuselage_diameter_scale = csdl.Variable(name="fuselage_diameter_scale", value=fuselage_diameter_scale_initial)
    fuselage_diameter_scale.set_as_design_variable(lower=fuselage_diameter_scale_lower, upper=fuselage_diameter_scale_upper, scaler=1.0)
    
    htail_root_rotation_degrees = csdl.Variable(name="htail_root_rotation_degrees", value=0.0)
    htail_root_rotation_degrees.set_as_design_variable(lower=-8., upper=8., scaler=1/8.)

    cg_computation_inputs = evaluate_E175_geometry_parameterization(
        fuse_function_set=fuse_function_set,
        wing_function_set=wing_function_set,
        htail_function_set=htail_function_set,
        wing_area_dv=wing_area_dv,
        wing_AR_dv=wing_AR_dv,
        htail_area_dv=htail_area_dv,
        htail_AR_dv=htail_AR_dv,
        wing_twist_dvs=twist_coeffs,
        wing_camber_dvs=delta_camber_percent_design_dof,
        wing_thickness_dvs=delta_thickness_percent_design_dof,
        wing_translation_x=wing_translation_amount,
        htail_root_rotation_degrees=htail_root_rotation_degrees,
        fuselage_diameter_scale=fuselage_diameter_scale,
        num_ffd_sections=num_ffd_sections,
        num_ffd_coefficients_chordwise=num_ffd_coefficients_chordwise,
    )

    MAC = cg_computation_inputs.wing_MAC

    wing_coefficients_csdl = _stack_coefficients_csdl(wing_function_set)
    fuse_coefficients_csdl = _stack_coefficients_csdl(fuse_function_set)
    htail_coefficients_csdl = _stack_coefficients_csdl(htail_function_set)
    total_coefficients_csdl = _stack_coefficients_csdl(embraer_175)
    
    initial_wing_coefficients = undeformed_wing_coefficients.copy()
    initial_fuse_coefficients = undeformed_fuse_coefficients.copy()
    initial_htail_coefficients = undeformed_htail_coefficients.copy()
    initial_wing_function_set = _function_set_from_stacked_coefficients(
        wing_function_set,
        initial_wing_coefficients,
        name="initial_wing_function_set",
    )
    initial_fuse_function_set = _function_set_from_stacked_coefficients(
        fuse_function_set,
        initial_fuse_coefficients,
        name="initial_fuse_function_set",
    )
    initial_htail_function_set = _function_set_from_stacked_coefficients(
        htail_function_set,
        initial_htail_coefficients,
        name="initial_htail_function_set",
    )
    
    fuselage_centerline_point = np.array(
        [
            0.0,
            0.5
            * (
                float(np.min(initial_fuse_coefficients[:, 1]))
                + float(np.max(initial_fuse_coefficients[:, 1]))
            ),
            0.5
            * (
                float(np.min(initial_fuse_coefficients[:, 2]))
                + float(np.max(initial_fuse_coefficients[:, 2]))
            ),
        ],
        dtype=float,
    )
    fuselage_centerline_rows = np.broadcast_to(
        fuselage_centerline_point,
        initial_fuse_coefficients.shape,
    ).copy()
    fuselage_radial_mask = np.broadcast_to(
        np.array([0.0, 1.0, 1.0], dtype=float),
        initial_fuse_coefficients.shape,
    ).copy()

    vertices, polygon_connectivity = _load_polygon_surface_pickle(mesh_path)
    triangulated_surface_cells = _triangulate_polygon_surface(polygon_connectivity)
    mesh_faces = _make_polygon_faces(polygon_connectivity)

    # _plot_surface_debug(
    #         vertices,
    #         mesh_faces=mesh_faces,
    #         interpolation_indices=None,
    #         seam_seed_indices=None,
    #         protected_feature_indices=None,
    #         mesh_label="Initial mesh",
    #         screenshot_path=(
    #             FINAL_MESH_SCREENSHOT_PATH
    #             if SAVE_FINAL_MESH_SCREENSHOT
    #             else None
    #         ),
    #         screenshot_window_size=FINAL_MESH_SCREENSHOT_WINDOW_SIZE,
    #     )
    # exit("hi")

    num_vertices = vertices.shape[0]
    variable_indices = np.arange(num_vertices, dtype=np.int64)
    vertex_neighbors = _build_polygon_vertex_neighbors(num_vertices, polygon_connectivity)
    if half_mesh_symmetry_mode:
        symmetry_maps = _build_half_mesh_symmetry_maps(
            vertices,
            symmetry_axis=symmetry_axis,
            symmetry_plane_tolerance=symmetry_plane_tolerance,
            mirror_tolerance=symmetry_mirror_tolerance,
            solve_side=symmetry_solve_side,
        )
    else:
        symmetry_maps = {
            "solve_vertex_indices": variable_indices,
            "solve_vertex_mask": np.ones(num_vertices, dtype=bool),
            "solve_vertex_local_indices": variable_indices.copy(),
            "full_source_local_indices": variable_indices.copy(),
            "full_mirror_signs": np.ones_like(vertices, dtype=float),
            "symmetry_plane_solve_local_indices": np.empty((0,), dtype=np.int64),
        }
    solve_vertex_indices = symmetry_maps["solve_vertex_indices"]
    solve_vertex_mask = symmetry_maps["solve_vertex_mask"]
    solve_vertex_local_indices = symmetry_maps["solve_vertex_local_indices"]
    full_source_local_indices = symmetry_maps["full_source_local_indices"]
    full_mirror_signs = symmetry_maps["full_mirror_signs"]
    symmetry_plane_solve_local_indices = symmetry_maps["symmetry_plane_solve_local_indices"]
    solve_vertices = vertices[solve_vertex_indices]
    num_solve_vertices = solve_vertices.shape[0]
    solve_vertex_neighbors = _build_solve_vertex_neighbors(
        vertex_neighbors,
        solve_vertex_indices,
        solve_vertex_local_indices,
    )
    laplacian_matrix = _build_sparse_uniform_laplacian_matrix(num_solve_vertices, solve_vertex_neighbors)
    bounding_box_diagonal = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    seam_support_sigma = seam_support_sigma_factor * max(bounding_box_diagonal, 1e-12)
    seam_graph_radius = seam_graph_radius_factor * max(bounding_box_diagonal, 1e-12)
    exact_seam_neighbor_graph_radius = (
        exact_seam_neighbor_graph_radius_factor * max(bounding_box_diagonal, 1e-12)
    )
    exact_seam_endpoint_feature_graph_radius = (
        exact_seam_endpoint_feature_graph_radius_factor
        * max(bounding_box_diagonal, 1e-12)
    )
    exact_seam_endpoint_feature_full_strength_graph_radius = min(
        exact_seam_endpoint_feature_full_strength_graph_radius_factor
        * max(bounding_box_diagonal, 1e-12),
        exact_seam_endpoint_feature_graph_radius,
    )
    exact_seam_endpoint_feature_halo_graph_radius = (
        exact_seam_endpoint_feature_halo_graph_radius_factor
        * max(bounding_box_diagonal, 1e-12)
    )
    exact_seam_endpoint_feature_halo_full_strength_graph_radius = min(
        exact_seam_endpoint_feature_halo_full_strength_graph_radius_factor
        * max(bounding_box_diagonal, 1e-12),
        exact_seam_endpoint_feature_halo_graph_radius,
    )

    initial_wing_projection_model = bsm3.FunctionSetProjectionModel(
        initial_wing_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    initial_fuse_projection_model = bsm3.FunctionSetProjectionModel(
        initial_fuse_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    initial_htail_projection_model = bsm3.FunctionSetProjectionModel(
        initial_htail_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    wing_projection_model = bsm3.FunctionSetProjectionModel(
        wing_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    fuse_projection_model = bsm3.FunctionSetProjectionModel(
        fuse_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    htail_projection_model = bsm3.FunctionSetProjectionModel(
        htail_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    combined_projection_model = bsm3.FunctionSetProjectionModel(
        embraer_175,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
    )
    wing_sdf_model = bsm3.FunctionSetProjectionModel(
        wing_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
        output_mode="distance",
        sdf=True,
    )
    fuse_sdf_model = bsm3.FunctionSetProjectionModel(
        fuse_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
        output_mode="distance",
        sdf=True,
    )
    htail_sdf_model = bsm3.FunctionSetProjectionModel(
        htail_function_set,
        # warm_start_nu=40,
        # warm_start_nv=40,
        # edge_map_num_samples=25,
        output_mode="distance",
        sdf=True,
    )
    initial_wing_evaluation_model = bsm3.FunctionSetEvaluationModel(initial_wing_function_set)
    initial_fuse_evaluation_model = bsm3.FunctionSetEvaluationModel(initial_fuse_function_set)
    initial_htail_evaluation_model = bsm3.FunctionSetEvaluationModel(initial_htail_function_set)
    wing_evaluation_model = bsm3.FunctionSetEvaluationModel(wing_function_set)
    fuse_evaluation_model = bsm3.FunctionSetEvaluationModel(fuse_function_set)
    htail_evaluation_model = bsm3.FunctionSetEvaluationModel(htail_function_set)
    wing_patch_function_sets = {
        int(patch_id): _single_patch_function_set(
            wing_function_set,
            int(patch_id),
            name=f"wing_patch_{int(patch_id)}",
        )
        for patch_id in wing_keys
    }
    htail_patch_function_sets = {
        int(patch_id): _single_patch_function_set(
            htail_function_set,
            int(patch_id),
            name=f"htail_patch_{int(patch_id)}",
        )
        for patch_id in htail_keys
    }
    wing_patch_projection_models = {
        patch_id: bsm3.FunctionSetProjectionModel(function_set)
        for patch_id, function_set in wing_patch_function_sets.items()
    }
    htail_patch_projection_models = {
        patch_id: bsm3.FunctionSetProjectionModel(function_set)
        for patch_id, function_set in htail_patch_function_sets.items()
    }
    wing_patch_evaluation_models = {
        patch_id: bsm3.FunctionSetEvaluationModel(function_set)
        for patch_id, function_set in wing_patch_function_sets.items()
    }
    htail_patch_evaluation_models = {
        patch_id: bsm3.FunctionSetEvaluationModel(function_set)
        for patch_id, function_set in htail_patch_function_sets.items()
    }

    wing_projection_distances, wing_projection_state = initial_wing_projection_model.project(
        initial_wing_coefficients,
        vertices,
    )
    fuse_projection_distances, _ = initial_fuse_projection_model.project(
        initial_fuse_coefficients,
        vertices,
    )
    htail_projection_distances, htail_projection_state = initial_htail_projection_model.project(
        initial_htail_coefficients,
        vertices,
    )
    initial_component_distances = np.column_stack(
        [
            np.abs(wing_projection_distances),
            np.abs(fuse_projection_distances),
            np.abs(htail_projection_distances),
        ]
    )
    initial_component_ownership = np.argmin(initial_component_distances, axis=1)
    wing_owned_vertex_mask = initial_component_ownership == 0
    fuse_owned_vertex_mask = initial_component_ownership == 1
    htail_owned_vertex_mask = initial_component_ownership == 2

    wing_fuse_seam_seed_indices = np.where(
        (np.abs(wing_projection_distances) <= intersection_tolerance)
        & (np.abs(fuse_projection_distances) <= intersection_tolerance)
    )[0]
    wing_fuse_seam_seed_indices = wing_fuse_seam_seed_indices[
        solve_vertex_mask[wing_fuse_seam_seed_indices]
    ]
    if wing_fuse_seam_seed_indices.size == 0:
        raise ValueError("No exact seam seed vertices were identified from the initial wing/fuselage intersection.")

    htail_fuse_seam_seed_indices = np.where(
        (np.abs(htail_projection_distances) <= intersection_tolerance)
        & (np.abs(fuse_projection_distances) <= intersection_tolerance)
    )[0]
    htail_fuse_seam_seed_indices = htail_fuse_seam_seed_indices[
        solve_vertex_mask[htail_fuse_seam_seed_indices]
    ]
    if htail_fuse_seam_seed_indices.size == 0:
        raise ValueError(
            "No exact seam seed vertices were identified from the initial horizontal-tail/fuselage intersection."
        )

    seam_seed_indices = np.unique(
        np.concatenate([wing_fuse_seam_seed_indices, htail_fuse_seam_seed_indices])
    )
    outboard_wing_parametric_fixed_indices = np.empty((0,), dtype=np.int64)
    outboard_wing_parametric_fix_normalized_y = np.empty((0,), dtype=float)
    outboard_wing_parametric_fixed_parametric_coordinates = np.empty((0, 3), dtype=float)
    outboard_htail_parametric_fixed_indices = np.empty((0,), dtype=np.int64)
    outboard_htail_parametric_fix_normalized_y = np.empty((0,), dtype=float)
    outboard_htail_parametric_fixed_parametric_coordinates = np.empty((0, 3), dtype=float)

    seam_seed_wing_parametric_coordinates = np.column_stack(
        [
            wing_projection_state["patch_id"][wing_fuse_seam_seed_indices],
            wing_projection_state["uv"][wing_fuse_seam_seed_indices],
        ]
    )
    seam_seed_wing_points = initial_wing_evaluation_model.evaluate(
        initial_wing_coefficients,
        seam_seed_wing_parametric_coordinates,
    )
    seam_seed_leading_local = int(np.argmin(seam_seed_wing_points[:, 0]))
    seam_seed_trailing_local = int(np.argmax(seam_seed_wing_points[:, 0]))
    wing_fuse_le_global_index = int(wing_fuse_seam_seed_indices[seam_seed_leading_local])
    wing_fuse_te_global_index = int(wing_fuse_seam_seed_indices[seam_seed_trailing_local])
    root_leading_point = seam_seed_wing_points[seam_seed_leading_local : seam_seed_leading_local + 1]
    root_trailing_point = seam_seed_wing_points[
        seam_seed_trailing_local : seam_seed_trailing_local + 1
    ]
    root_chord_proxy = max(
        float(root_trailing_point[0, 0] - root_leading_point[0, 0]),
        1e-12,
    )
    root_quarter_chord = root_leading_point + 0.25 * (root_trailing_point - root_leading_point)
    wing_translation_bound = translation_envelope_mac * root_chord_proxy
    # wing_translation_amount.set_as_design_variable(
    #     lower=-2.5,
    #     upper=1.5,
    #     scaler=0.5,
    # )

    seam_seed_htail_parametric_coordinates = np.column_stack(
        [
            htail_projection_state["patch_id"][htail_fuse_seam_seed_indices],
            htail_projection_state["uv"][htail_fuse_seam_seed_indices],
        ]
    )
    seam_seed_htail_points = initial_htail_evaluation_model.evaluate(
        initial_htail_coefficients,
        seam_seed_htail_parametric_coordinates,
    )
    htail_root_leading_local = int(np.argmin(seam_seed_htail_points[:, 0]))
    htail_root_trailing_local = int(np.argmax(seam_seed_htail_points[:, 0]))
    htail_fuse_le_global_index = int(htail_fuse_seam_seed_indices[htail_root_leading_local])
    htail_fuse_te_global_index = int(htail_fuse_seam_seed_indices[htail_root_trailing_local])
    htail_root_leading_point = seam_seed_htail_points[
        htail_root_leading_local : htail_root_leading_local + 1
    ]
    htail_root_trailing_point = seam_seed_htail_points[
        htail_root_trailing_local : htail_root_trailing_local + 1
    ]
    htail_root_chord_proxy = max(
        float(htail_root_trailing_point[0, 0] - htail_root_leading_point[0, 0]),
        1e-12,
    )
    htail_root_quarter_chord = htail_root_leading_point + 0.25 * (
        htail_root_trailing_point - htail_root_leading_point
    )
    aft_fuse_fixed_vertex_indices = np.empty((0,), dtype=np.int64)
    aft_fuse_fixed_x_min = np.inf
    if aft_fuse_fixed_region:
        aft_fuse_fixed_x_min = float(
            htail_root_trailing_point[0, 0]
            + aft_fuse_fixed_buffer_htail_chords * htail_root_chord_proxy
        )
        aft_fuse_fixed_mask = (
            fuse_owned_vertex_mask
            & solve_vertex_mask
            & (vertices[:, 0] >= aft_fuse_fixed_x_min)
        )
        aft_fuse_fixed_mask[seam_seed_indices] = False
        aft_fuse_fixed_vertex_indices = np.where(aft_fuse_fixed_mask)[0]

    wing_fuse_graph_distances = _compute_graph_distances(
        vertices,
        vertex_neighbors,
        wing_fuse_seam_seed_indices,
    )
    htail_fuse_graph_distances = _compute_graph_distances(
        vertices,
        vertex_neighbors,
        htail_fuse_seam_seed_indices,
    )
    seam_graph_distances = _compute_graph_distances(vertices, vertex_neighbors, seam_seed_indices)
    sharp_feature_vertex_mask = _sharp_feature_vertex_mask(
        vertices,
        triangulated_surface_cells,
        feature_angle_degrees=sharp_feature_angle_degrees,
        include_boundary=sharp_feature_include_boundary,
    )
    endpoint_feature_coordinate_tolerance_setup = max(
        float(endpoint_feature_coordinate_tolerance),
        1e-8 * max(bounding_box_diagonal, 1e-12),
    )

    exact_seam_neighbor_training_graph_radius = (
        exact_seam_neighbor_training_graph_radius_factor
        * max(bounding_box_diagonal, 1e-12)
    )
    if exact_seam_neighbor_training:
        exact_seam_neighbor_training_mask = (
            solve_vertex_mask
            & (
                (wing_fuse_graph_distances <= exact_seam_neighbor_training_graph_radius)
                | (htail_fuse_graph_distances <= exact_seam_neighbor_training_graph_radius)
            )
        )
        exact_seam_neighbor_training_mask[seam_seed_indices] = False
        exact_seam_neighbor_training_mask[aft_fuse_fixed_vertex_indices] = False
        exact_seam_neighbor_training_indices = np.where(exact_seam_neighbor_training_mask)[0]
        if (
            max_exact_seam_neighbor_training_vertices > 0
            and exact_seam_neighbor_training_indices.size
            > max_exact_seam_neighbor_training_vertices
        ):
            exact_seam_neighbor_training_local_indices = _select_farthest_point_indices_symmetric(
                vertices[exact_seam_neighbor_training_indices],
                max_exact_seam_neighbor_training_vertices,
            )
            exact_seam_neighbor_training_indices = exact_seam_neighbor_training_indices[
                exact_seam_neighbor_training_local_indices
            ]
    else:
        exact_seam_neighbor_training_indices = np.empty((0,), dtype=np.int64)

    if outboard_wing_parametric_fix:
        outboard_wing_parametric_fix_threshold = np.clip(
            float(outboard_wing_parametric_fix_normalized_y_min),
            0.0,
            1.0,
        )
        candidate_outboard_wing_mask = (
            wing_owned_vertex_mask
            & (wing_projection_distances <= wing_surface_projection_tolerance)
            & (seam_graph_distances > seam_graph_radius)
            & solve_vertex_mask
        )
        candidate_outboard_wing_indices = np.where(candidate_outboard_wing_mask)[0]
        if candidate_outboard_wing_indices.size > 0:
            candidate_outboard_wing_abs_y = np.abs(vertices[candidate_outboard_wing_indices, 1])
            outboard_wing_abs_y_min = float(np.min(candidate_outboard_wing_abs_y))
            outboard_wing_abs_y_max = float(np.max(candidate_outboard_wing_abs_y))
            outboard_wing_abs_y_span = max(
                outboard_wing_abs_y_max - outboard_wing_abs_y_min,
                1e-12,
            )
            candidate_outboard_wing_normalized_y = (
                candidate_outboard_wing_abs_y - outboard_wing_abs_y_min
            ) / outboard_wing_abs_y_span
            retained_outboard_wing_mask = (
                candidate_outboard_wing_normalized_y >= outboard_wing_parametric_fix_threshold
            )
            outboard_wing_parametric_fixed_indices = candidate_outboard_wing_indices[
                retained_outboard_wing_mask
            ]
            outboard_wing_parametric_fix_normalized_y = (
                candidate_outboard_wing_normalized_y[retained_outboard_wing_mask]
            )
            outboard_wing_parametric_fixed_parametric_coordinates = np.column_stack(
                [
                    wing_projection_state["patch_id"][outboard_wing_parametric_fixed_indices],
                    wing_projection_state["uv"][outboard_wing_parametric_fixed_indices],
                ]
            )
    if outboard_htail_parametric_fix:
        outboard_htail_parametric_fix_threshold = np.clip(
            float(outboard_htail_parametric_fix_normalized_y_min),
            0.0,
            1.0,
        )
        candidate_outboard_htail_mask = (
            htail_owned_vertex_mask
            & (htail_projection_distances <= wing_surface_projection_tolerance)
            & (seam_graph_distances > seam_graph_radius)
            & solve_vertex_mask
        )
        candidate_outboard_htail_indices = np.where(candidate_outboard_htail_mask)[0]
        if candidate_outboard_htail_indices.size > 0:
            candidate_outboard_htail_abs_y = np.abs(vertices[candidate_outboard_htail_indices, 1])
            outboard_htail_abs_y_min = float(np.min(candidate_outboard_htail_abs_y))
            outboard_htail_abs_y_max = float(np.max(candidate_outboard_htail_abs_y))
            outboard_htail_abs_y_span = max(
                outboard_htail_abs_y_max - outboard_htail_abs_y_min,
                1e-12,
            )
            candidate_outboard_htail_normalized_y = (
                candidate_outboard_htail_abs_y - outboard_htail_abs_y_min
            ) / outboard_htail_abs_y_span
            retained_outboard_htail_mask = (
                candidate_outboard_htail_normalized_y >= outboard_htail_parametric_fix_threshold
            )
            outboard_htail_parametric_fixed_indices = candidate_outboard_htail_indices[
                retained_outboard_htail_mask
            ]
            outboard_htail_parametric_fix_normalized_y = (
                candidate_outboard_htail_normalized_y[retained_outboard_htail_mask]
            )
            outboard_htail_parametric_fixed_parametric_coordinates = np.column_stack(
                [
                    htail_projection_state["patch_id"][outboard_htail_parametric_fixed_indices],
                    htail_projection_state["uv"][outboard_htail_parametric_fixed_indices],
                ]
            )

    aft_fuse_symmetry_anchor_indices = np.empty((0,), dtype=np.int64)
    aft_fuse_symmetry_neighbor_indices = np.empty((0,), dtype=np.int64)
    if aft_fuse_symmetry_anchors:
        x_extent = max(float(np.max(vertices[:, 0]) - np.min(vertices[:, 0])), 1e-12)
        aft_fuse_symmetry_anchor_x_min = (
            float(np.max(vertices[:, 0])) - aft_fuse_symmetry_anchor_x_fraction * x_extent
        )
        aft_fuse_symmetry_anchor_mask = (
            fuse_owned_vertex_mask
            & (vertices[:, 0] >= aft_fuse_symmetry_anchor_x_min)
            & (np.abs(vertices[:, 1]) <= aft_fuse_symmetry_anchor_y_tolerance)
        )
        aft_fuse_symmetry_anchor_mask[seam_seed_indices] = False
        aft_fuse_symmetry_anchor_mask[aft_fuse_fixed_vertex_indices] = False
        aft_fuse_symmetry_anchor_candidate_indices = np.where(aft_fuse_symmetry_anchor_mask)[0]
        aft_fuse_symmetry_anchor_indices = aft_fuse_symmetry_anchor_candidate_indices
        if (
            aft_fuse_symmetry_anchor_downsample_stride > 1
            and aft_fuse_symmetry_anchor_candidate_indices.size > 2
        ):
            aft_fuse_candidate_points = vertices[aft_fuse_symmetry_anchor_candidate_indices]
            ordered_aft_fuse_anchor_local_indices = np.lexsort(
                (
                    aft_fuse_candidate_points[:, 0],
                    -aft_fuse_candidate_points[:, 2],
                )
            )
            ordered_aft_fuse_anchor_indices = aft_fuse_symmetry_anchor_candidate_indices[
                ordered_aft_fuse_anchor_local_indices
            ]
            aft_fuse_symmetry_anchor_indices = ordered_aft_fuse_anchor_indices[
                :: aft_fuse_symmetry_anchor_downsample_stride
            ]
            if aft_fuse_symmetry_anchor_indices[-1] != ordered_aft_fuse_anchor_indices[-1]:
                aft_fuse_symmetry_anchor_indices = np.concatenate(
                    [
                        aft_fuse_symmetry_anchor_indices,
                        ordered_aft_fuse_anchor_indices[-1:],
                    ]
                )
            aft_fuse_symmetry_anchor_indices = np.sort(
                np.unique(aft_fuse_symmetry_anchor_indices)
            )

    endpoint_feature_training_indices = np.empty((0,), dtype=np.int64)
    if exact_seam_endpoint_feature_driving:
        endpoint_feature_excluded_indices = np.unique(
            np.concatenate([seam_seed_indices, aft_fuse_symmetry_anchor_indices])
        )
        wing_fuse_le_graph_distances = _compute_graph_distances(
            vertices,
            vertex_neighbors,
            np.array([wing_fuse_le_global_index], dtype=np.int64),
        )
        wing_fuse_te_graph_distances = _compute_graph_distances(
            vertices,
            vertex_neighbors,
            np.array([wing_fuse_te_global_index], dtype=np.int64),
        )
        htail_fuse_le_graph_distances = _compute_graph_distances(
            vertices,
            vertex_neighbors,
            np.array([htail_fuse_le_global_index], dtype=np.int64),
        )
        htail_fuse_te_graph_distances = _compute_graph_distances(
            vertices,
            vertex_neighbors,
            np.array([htail_fuse_te_global_index], dtype=np.int64),
        )
        wing_fuse_le_endpoint_feature_indices = _endpoint_feature_global_indices(
            vertices=vertices,
            vertex_neighbors=vertex_neighbors,
            component_distances=wing_projection_distances,
            component_state=wing_projection_state,
            sharp_feature_mask=sharp_feature_vertex_mask,
            solve_vertex_mask=solve_vertex_mask,
            excluded_indices=endpoint_feature_excluded_indices,
            endpoint_global_index=wing_fuse_le_global_index,
            endpoint_parametric_coordinate=seam_seed_wing_parametric_coordinates[
                seam_seed_leading_local
            ],
            endpoint_point=root_leading_point[0],
            endpoint_graph_distances=wing_fuse_le_graph_distances,
            support_radius=exact_seam_endpoint_feature_graph_radius,
            support_cutoff=exact_seam_endpoint_feature_support_cutoff,
            projection_tolerance=wing_surface_projection_tolerance,
            parametric_boundary_tolerance=endpoint_feature_parametric_boundary_tolerance,
            coordinate_tolerance=endpoint_feature_coordinate_tolerance_setup,
        )
        wing_fuse_te_endpoint_feature_indices = _endpoint_feature_global_indices(
            vertices=vertices,
            vertex_neighbors=vertex_neighbors,
            component_distances=wing_projection_distances,
            component_state=wing_projection_state,
            sharp_feature_mask=sharp_feature_vertex_mask,
            solve_vertex_mask=solve_vertex_mask,
            excluded_indices=endpoint_feature_excluded_indices,
            endpoint_global_index=wing_fuse_te_global_index,
            endpoint_parametric_coordinate=seam_seed_wing_parametric_coordinates[
                seam_seed_trailing_local
            ],
            endpoint_point=root_trailing_point[0],
            endpoint_graph_distances=wing_fuse_te_graph_distances,
            support_radius=exact_seam_endpoint_feature_graph_radius,
            support_cutoff=exact_seam_endpoint_feature_support_cutoff,
            projection_tolerance=wing_surface_projection_tolerance,
            parametric_boundary_tolerance=endpoint_feature_parametric_boundary_tolerance,
            coordinate_tolerance=endpoint_feature_coordinate_tolerance_setup,
        )
        htail_fuse_le_endpoint_feature_indices = _endpoint_feature_global_indices(
            vertices=vertices,
            vertex_neighbors=vertex_neighbors,
            component_distances=htail_projection_distances,
            component_state=htail_projection_state,
            sharp_feature_mask=sharp_feature_vertex_mask,
            solve_vertex_mask=solve_vertex_mask,
            excluded_indices=endpoint_feature_excluded_indices,
            endpoint_global_index=htail_fuse_le_global_index,
            endpoint_parametric_coordinate=seam_seed_htail_parametric_coordinates[
                htail_root_leading_local
            ],
            endpoint_point=htail_root_leading_point[0],
            endpoint_graph_distances=htail_fuse_le_graph_distances,
            support_radius=exact_seam_endpoint_feature_graph_radius,
            support_cutoff=exact_seam_endpoint_feature_support_cutoff,
            projection_tolerance=wing_surface_projection_tolerance,
            parametric_boundary_tolerance=endpoint_feature_parametric_boundary_tolerance,
            coordinate_tolerance=endpoint_feature_coordinate_tolerance_setup,
        )
        htail_fuse_te_endpoint_feature_indices = _endpoint_feature_global_indices(
            vertices=vertices,
            vertex_neighbors=vertex_neighbors,
            component_distances=htail_projection_distances,
            component_state=htail_projection_state,
            sharp_feature_mask=sharp_feature_vertex_mask,
            solve_vertex_mask=solve_vertex_mask,
            excluded_indices=endpoint_feature_excluded_indices,
            endpoint_global_index=htail_fuse_te_global_index,
            endpoint_parametric_coordinate=seam_seed_htail_parametric_coordinates[
                htail_root_trailing_local
            ],
            endpoint_point=htail_root_trailing_point[0],
            endpoint_graph_distances=htail_fuse_te_graph_distances,
            support_radius=exact_seam_endpoint_feature_graph_radius,
            support_cutoff=exact_seam_endpoint_feature_support_cutoff,
            projection_tolerance=wing_surface_projection_tolerance,
            parametric_boundary_tolerance=endpoint_feature_parametric_boundary_tolerance,
            coordinate_tolerance=endpoint_feature_coordinate_tolerance_setup,
        )
        if exact_seam_endpoint_feature_halo_driving:
            wing_fuse_le_feature_graph_distances = _compute_graph_distances(
                vertices,
                vertex_neighbors,
                wing_fuse_le_endpoint_feature_indices,
            )
            wing_fuse_te_feature_graph_distances = _compute_graph_distances(
                vertices,
                vertex_neighbors,
                wing_fuse_te_endpoint_feature_indices,
            )
            htail_fuse_le_feature_graph_distances = _compute_graph_distances(
                vertices,
                vertex_neighbors,
                htail_fuse_le_endpoint_feature_indices,
            )
            htail_fuse_te_feature_graph_distances = _compute_graph_distances(
                vertices,
                vertex_neighbors,
                htail_fuse_te_endpoint_feature_indices,
            )
            (
                wing_fuse_le_endpoint_halo_indices,
                wing_fuse_le_endpoint_halo_support_global,
            ) = _endpoint_feature_halo_global_support(
                component_distances=wing_projection_distances,
                solve_vertex_mask=solve_vertex_mask,
                excluded_indices=endpoint_feature_excluded_indices,
                endpoint_feature_indices=wing_fuse_le_endpoint_feature_indices,
                endpoint_graph_distances=wing_fuse_le_graph_distances,
                feature_graph_distances=wing_fuse_le_feature_graph_distances,
                endpoint_support_radius=exact_seam_endpoint_feature_graph_radius,
                endpoint_full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
                halo_radius=exact_seam_endpoint_feature_halo_graph_radius,
                halo_full_strength_radius=exact_seam_endpoint_feature_halo_full_strength_graph_radius,
                max_blend=exact_seam_endpoint_feature_halo_max_blend,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
                projection_tolerance=wing_surface_projection_tolerance,
            )
            (
                wing_fuse_te_endpoint_halo_indices,
                wing_fuse_te_endpoint_halo_support_global,
            ) = _endpoint_feature_halo_global_support(
                component_distances=wing_projection_distances,
                solve_vertex_mask=solve_vertex_mask,
                excluded_indices=endpoint_feature_excluded_indices,
                endpoint_feature_indices=wing_fuse_te_endpoint_feature_indices,
                endpoint_graph_distances=wing_fuse_te_graph_distances,
                feature_graph_distances=wing_fuse_te_feature_graph_distances,
                endpoint_support_radius=exact_seam_endpoint_feature_graph_radius,
                endpoint_full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
                halo_radius=exact_seam_endpoint_feature_halo_graph_radius,
                halo_full_strength_radius=exact_seam_endpoint_feature_halo_full_strength_graph_radius,
                max_blend=exact_seam_endpoint_feature_halo_max_blend,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
                projection_tolerance=wing_surface_projection_tolerance,
            )
            (
                htail_fuse_le_endpoint_halo_indices,
                htail_fuse_le_endpoint_halo_support_global,
            ) = _endpoint_feature_halo_global_support(
                component_distances=htail_projection_distances,
                solve_vertex_mask=solve_vertex_mask,
                excluded_indices=endpoint_feature_excluded_indices,
                endpoint_feature_indices=htail_fuse_le_endpoint_feature_indices,
                endpoint_graph_distances=htail_fuse_le_graph_distances,
                feature_graph_distances=htail_fuse_le_feature_graph_distances,
                endpoint_support_radius=exact_seam_endpoint_feature_graph_radius,
                endpoint_full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
                halo_radius=exact_seam_endpoint_feature_halo_graph_radius,
                halo_full_strength_radius=exact_seam_endpoint_feature_halo_full_strength_graph_radius,
                max_blend=exact_seam_endpoint_feature_halo_max_blend,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
                projection_tolerance=wing_surface_projection_tolerance,
            )
            (
                htail_fuse_te_endpoint_halo_indices,
                htail_fuse_te_endpoint_halo_support_global,
            ) = _endpoint_feature_halo_global_support(
                component_distances=htail_projection_distances,
                solve_vertex_mask=solve_vertex_mask,
                excluded_indices=endpoint_feature_excluded_indices,
                endpoint_feature_indices=htail_fuse_te_endpoint_feature_indices,
                endpoint_graph_distances=htail_fuse_te_graph_distances,
                feature_graph_distances=htail_fuse_te_feature_graph_distances,
                endpoint_support_radius=exact_seam_endpoint_feature_graph_radius,
                endpoint_full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
                halo_radius=exact_seam_endpoint_feature_halo_graph_radius,
                halo_full_strength_radius=exact_seam_endpoint_feature_halo_full_strength_graph_radius,
                max_blend=exact_seam_endpoint_feature_halo_max_blend,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
                projection_tolerance=wing_surface_projection_tolerance,
            )
        else:
            wing_fuse_le_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
            wing_fuse_te_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
            htail_fuse_le_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
            htail_fuse_te_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
            wing_fuse_le_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
            wing_fuse_te_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
            htail_fuse_le_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
            htail_fuse_te_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
        endpoint_feature_training_indices = np.unique(
            np.concatenate(
                [
                    wing_fuse_le_endpoint_feature_indices,
                    wing_fuse_te_endpoint_feature_indices,
                    htail_fuse_le_endpoint_feature_indices,
                    htail_fuse_te_endpoint_feature_indices,
                    wing_fuse_le_endpoint_halo_indices,
                    wing_fuse_te_endpoint_halo_indices,
                    htail_fuse_le_endpoint_halo_indices,
                    htail_fuse_te_endpoint_halo_indices,
                ]
            )
        )
    else:
        wing_fuse_le_graph_distances = np.full(num_vertices, np.inf, dtype=float)
        wing_fuse_te_graph_distances = np.full(num_vertices, np.inf, dtype=float)
        htail_fuse_le_graph_distances = np.full(num_vertices, np.inf, dtype=float)
        htail_fuse_te_graph_distances = np.full(num_vertices, np.inf, dtype=float)
        wing_fuse_le_endpoint_feature_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_te_endpoint_feature_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_le_endpoint_feature_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_te_endpoint_feature_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_le_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_te_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_le_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_te_endpoint_halo_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_le_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
        wing_fuse_te_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
        htail_fuse_le_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)
        htail_fuse_te_endpoint_halo_support_global = np.zeros(num_vertices, dtype=float)

    interpolation_indices = np.unique(
        np.concatenate(
            [
                seam_seed_indices,
                aft_fuse_symmetry_anchor_indices,
                aft_fuse_symmetry_neighbor_indices,
                exact_seam_neighbor_training_indices,
                endpoint_feature_training_indices,
            ]
        )
    )
    remaining_variable_indices = np.setdiff1d(
        solve_vertex_indices,
        np.unique(
            np.concatenate(
                [
                    interpolation_indices,
                    aft_fuse_fixed_vertex_indices,
                ]
            )
        ),
    )
    num_global_additional_interp_vertices = max(
        num_additional_interp_vertices - exact_seam_neighbor_training_indices.size,
        0,
    )
    additional_interp_local = _select_farthest_point_indices_symmetric(
        vertices[remaining_variable_indices],
        min(num_global_additional_interp_vertices, remaining_variable_indices.size),
    )
    if additional_interp_local.size > 0:
        interpolation_indices = np.concatenate(
            [interpolation_indices, remaining_variable_indices[additional_interp_local]]
        )
    interpolation_indices = np.unique(interpolation_indices)
    interpolation_vertices = vertices[interpolation_indices]
    seam_interpolation_mask = np.isin(interpolation_indices, seam_seed_indices)
    wing_fuse_seam_interpolation_mask = np.isin(
        interpolation_indices,
        wing_fuse_seam_seed_indices,
    )
    htail_fuse_seam_interpolation_mask = np.isin(
        interpolation_indices,
        htail_fuse_seam_seed_indices,
    )
    aft_fuse_symmetry_anchor_interpolation_mask = np.isin(
        interpolation_indices,
        aft_fuse_symmetry_anchor_indices,
    )
    aft_fuse_symmetry_neighbor_interpolation_mask = np.isin(
        interpolation_indices,
        aft_fuse_symmetry_neighbor_indices,
    )
    wing_fuse_seam_interpolation_local_indices = np.where(wing_fuse_seam_interpolation_mask)[0]
    htail_fuse_seam_interpolation_local_indices = np.where(htail_fuse_seam_interpolation_mask)[0]
    aft_fuse_symmetry_anchor_local_indices = np.where(
        aft_fuse_symmetry_anchor_interpolation_mask
    )[0]
    aft_fuse_symmetry_neighbor_local_indices = np.where(
        aft_fuse_symmetry_neighbor_interpolation_mask
    )[0]

    wing_fuse_neighbor_support = np.exp(
        -(
            wing_fuse_graph_distances[interpolation_indices]
            / max(exact_seam_neighbor_graph_radius, 1e-12)
        )
        ** 2
    )
    htail_fuse_neighbor_support = np.exp(
        -(
            htail_fuse_graph_distances[interpolation_indices]
            / max(exact_seam_neighbor_graph_radius, 1e-12)
        )
        ** 2
    )
    wing_fuse_neighbor_support[wing_fuse_seam_interpolation_mask] = 1.0
    htail_fuse_neighbor_support[htail_fuse_seam_interpolation_mask] = 1.0
    wing_fuse_neighbor_mask = (
        (wing_fuse_neighbor_support >= exact_seam_neighbor_support_cutoff)
        & ~wing_fuse_seam_interpolation_mask
        & ~htail_fuse_seam_interpolation_mask
    )
    htail_fuse_neighbor_mask = (
        (htail_fuse_neighbor_support >= exact_seam_neighbor_support_cutoff)
        & ~htail_fuse_seam_interpolation_mask
        & ~wing_fuse_seam_interpolation_mask
    )
    wing_fuse_neighbor_local_indices = np.where(wing_fuse_neighbor_mask)[0]
    htail_fuse_neighbor_local_indices = np.where(htail_fuse_neighbor_mask)[0]
    wing_fuse_neighbor_nearest_seam_local_indices = np.empty((0,), dtype=np.int64)
    if wing_fuse_neighbor_local_indices.size > 0:
        wing_fuse_neighbor_nearest_seam_local_indices = _nearest_source_indices_numpy(
            interpolation_vertices[wing_fuse_neighbor_local_indices],
            interpolation_vertices[wing_fuse_seam_interpolation_local_indices],
        )
    htail_fuse_neighbor_nearest_seam_local_indices = np.empty((0,), dtype=np.int64)
    if htail_fuse_neighbor_local_indices.size > 0:
        htail_fuse_neighbor_nearest_seam_local_indices = _nearest_source_indices_numpy(
            interpolation_vertices[htail_fuse_neighbor_local_indices],
            interpolation_vertices[htail_fuse_seam_interpolation_local_indices],
        )

    interpolation_wing_distances, interpolation_wing_state = initial_wing_projection_model.project(
        initial_wing_coefficients,
        interpolation_vertices,
    )
    interpolation_fuse_distances, interpolation_fuse_state = initial_fuse_projection_model.project(
        initial_fuse_coefficients,
        interpolation_vertices,
    )
    interpolation_htail_distances, interpolation_htail_state = initial_htail_projection_model.project(
        initial_htail_coefficients,
        interpolation_vertices,
    )
    interpolation_wing_parametric_coordinates = np.column_stack(
        [
            interpolation_wing_state["patch_id"],
            interpolation_wing_state["uv"],
        ]
    )
    interpolation_fuse_parametric_coordinates = np.column_stack(
        [
            interpolation_fuse_state["patch_id"],
            interpolation_fuse_state["uv"],
        ]
    )
    interpolation_htail_parametric_coordinates = np.column_stack(
        [
            interpolation_htail_state["patch_id"],
            interpolation_htail_state["uv"],
        ]
    )
    interpolation_initial_wing_points = initial_wing_evaluation_model.evaluate(
        initial_wing_coefficients,
        interpolation_wing_parametric_coordinates,
    )
    interpolation_initial_fuse_points = initial_fuse_evaluation_model.evaluate(
        initial_fuse_coefficients,
        interpolation_fuse_parametric_coordinates,
    )
    interpolation_initial_htail_points = initial_htail_evaluation_model.evaluate(
        initial_htail_coefficients,
        interpolation_htail_parametric_coordinates,
    )
    interpolation_component_weights = _normalize_weights(
        np.exp(
            -alpha
            * np.column_stack(
                [
                    interpolation_wing_distances,
                    interpolation_fuse_distances,
                    interpolation_htail_distances,
                ]
            )
        )
    )
    interpolation_wing_weights = interpolation_component_weights[:, 0]
    interpolation_fuse_weights = interpolation_component_weights[:, 1]
    interpolation_htail_weights = interpolation_component_weights[:, 2]

    closest_initial_component = np.argmin(
        np.column_stack(
            [
                interpolation_wing_distances,
                interpolation_fuse_distances,
                interpolation_htail_distances,
            ]
        ),
        axis=1,
    )
    wing_owned_interpolation_mask = closest_initial_component == 0
    fuse_owned_interpolation_mask = closest_initial_component == 1
    htail_owned_interpolation_mask = closest_initial_component == 2
    wing_owned_interpolation_mask[seam_interpolation_mask] = False
    fuse_owned_interpolation_mask[seam_interpolation_mask] = False
    htail_owned_interpolation_mask[seam_interpolation_mask] = False
    wing_owned_interpolation_local_indices = np.where(wing_owned_interpolation_mask)[0]
    fuse_owned_interpolation_local_indices = np.where(fuse_owned_interpolation_mask)[0]
    htail_owned_interpolation_local_indices = np.where(htail_owned_interpolation_mask)[0]

    wing_fuse_seam_global_indices = interpolation_indices[
        wing_fuse_seam_interpolation_local_indices
    ]
    wing_fuse_seam_wing_parametric_coordinates = interpolation_wing_parametric_coordinates[
        wing_fuse_seam_interpolation_local_indices
    ]
    wing_fuse_seam_baseline_points = interpolation_vertices[
        wing_fuse_seam_interpolation_local_indices
    ]
    wing_fuse_anchor_seed_locals = np.unique(
        np.array([seam_seed_leading_local, seam_seed_trailing_local], dtype=np.int64)
    )
    wing_fuse_anchor_global_indices = wing_fuse_seam_seed_indices[
        wing_fuse_anchor_seed_locals
    ]
    wing_fuse_anchor_indices = _rows_for_global_indices(
        wing_fuse_seam_global_indices,
        wing_fuse_anchor_global_indices,
    )
    wing_fuse_le_seam_rows = _rows_for_global_indices(
        wing_fuse_seam_global_indices,
        np.array([wing_fuse_le_global_index], dtype=np.int64),
    )
    wing_fuse_te_seam_rows = _rows_for_global_indices(
        wing_fuse_seam_global_indices,
        np.array([wing_fuse_te_global_index], dtype=np.int64),
    )
    wing_fuse_le_seam_row = (
        int(wing_fuse_le_seam_rows[0]) if wing_fuse_le_seam_rows.size > 0 else None
    )
    wing_fuse_te_seam_row = (
        int(wing_fuse_te_seam_rows[0]) if wing_fuse_te_seam_rows.size > 0 else None
    )
    wing_fuse_seam_edge_pairs, wing_fuse_seam_arc_chain_ranges = _build_ordered_seam_edge_chains(
        wing_fuse_seam_global_indices,
        vertex_neighbors,
        wing_fuse_seam_baseline_points,
        wing_fuse_le_seam_row,
        wing_fuse_te_seam_row,
    )
    (
        wing_fuse_baseline_normalized_arc_positions,
        wing_fuse_baseline_curve_length,
    ) = _normalized_cumulative_arc_positions_for_chains(
        wing_fuse_seam_baseline_points,
        wing_fuse_seam_edge_pairs,
        wing_fuse_seam_arc_chain_ranges,
    )
    wing_fuse_anchor_parametric_coordinates = seam_seed_wing_parametric_coordinates[
        wing_fuse_anchor_seed_locals
    ]
    wing_fuse_spanwise_uv_axis = _infer_spanwise_uv_axis_from_root_endpoints(
        seam_seed_wing_parametric_coordinates[seam_seed_leading_local],
        seam_seed_wing_parametric_coordinates[seam_seed_trailing_local],
    )

    htail_fuse_seam_global_indices = interpolation_indices[
        htail_fuse_seam_interpolation_local_indices
    ]
    htail_fuse_seam_htail_parametric_coordinates = interpolation_htail_parametric_coordinates[
        htail_fuse_seam_interpolation_local_indices
    ]
    htail_fuse_seam_baseline_points = interpolation_vertices[
        htail_fuse_seam_interpolation_local_indices
    ]
    htail_fuse_anchor_seed_locals = np.unique(
        np.array([htail_root_leading_local, htail_root_trailing_local], dtype=np.int64)
    )
    htail_fuse_anchor_global_indices = htail_fuse_seam_seed_indices[
        htail_fuse_anchor_seed_locals
    ]
    htail_fuse_anchor_indices = _rows_for_global_indices(
        htail_fuse_seam_global_indices,
        htail_fuse_anchor_global_indices,
    )
    htail_fuse_le_seam_rows = _rows_for_global_indices(
        htail_fuse_seam_global_indices,
        np.array([htail_fuse_le_global_index], dtype=np.int64),
    )
    htail_fuse_te_seam_rows = _rows_for_global_indices(
        htail_fuse_seam_global_indices,
        np.array([htail_fuse_te_global_index], dtype=np.int64),
    )
    htail_fuse_le_seam_row = (
        int(htail_fuse_le_seam_rows[0]) if htail_fuse_le_seam_rows.size > 0 else None
    )
    htail_fuse_te_seam_row = (
        int(htail_fuse_te_seam_rows[0]) if htail_fuse_te_seam_rows.size > 0 else None
    )
    htail_fuse_seam_edge_pairs, htail_fuse_seam_arc_chain_ranges = _build_ordered_seam_edge_chains(
        htail_fuse_seam_global_indices,
        vertex_neighbors,
        htail_fuse_seam_baseline_points,
        htail_fuse_le_seam_row,
        htail_fuse_te_seam_row,
    )
    (
        htail_fuse_baseline_normalized_arc_positions,
        htail_fuse_baseline_curve_length,
    ) = _normalized_cumulative_arc_positions_for_chains(
        htail_fuse_seam_baseline_points,
        htail_fuse_seam_edge_pairs,
        htail_fuse_seam_arc_chain_ranges,
    )
    htail_fuse_anchor_parametric_coordinates = seam_seed_htail_parametric_coordinates[
        htail_fuse_anchor_seed_locals
    ]
    htail_fuse_spanwise_uv_axis = _infer_spanwise_uv_axis_from_root_endpoints(
        seam_seed_htail_parametric_coordinates[htail_root_leading_local],
        seam_seed_htail_parametric_coordinates[htail_root_trailing_local],
    )

    if exact_seam_endpoint_feature_driving:
        wing_fuse_le_feature_local_indices, wing_fuse_le_feature_support = (
            _endpoint_feature_rows_from_global_indices(
                interpolation_indices=interpolation_indices,
                endpoint_feature_indices=wing_fuse_le_endpoint_feature_indices,
                endpoint_graph_distances=wing_fuse_le_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        wing_fuse_te_feature_local_indices, wing_fuse_te_feature_support = (
            _endpoint_feature_rows_from_global_indices(
                interpolation_indices=interpolation_indices,
                endpoint_feature_indices=wing_fuse_te_endpoint_feature_indices,
                endpoint_graph_distances=wing_fuse_te_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        htail_fuse_le_feature_local_indices, htail_fuse_le_feature_support = (
            _endpoint_feature_rows_from_global_indices(
                interpolation_indices=interpolation_indices,
                endpoint_feature_indices=htail_fuse_le_endpoint_feature_indices,
                endpoint_graph_distances=htail_fuse_le_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        htail_fuse_te_feature_local_indices, htail_fuse_te_feature_support = (
            _endpoint_feature_rows_from_global_indices(
                interpolation_indices=interpolation_indices,
                endpoint_feature_indices=htail_fuse_te_endpoint_feature_indices,
                endpoint_graph_distances=htail_fuse_te_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        wing_fuse_le_feature_solve_local_indices, wing_fuse_le_feature_solve_support = (
            _endpoint_feature_solve_rows_from_global_indices(
                solve_vertex_local_indices=solve_vertex_local_indices,
                endpoint_feature_indices=wing_fuse_le_endpoint_feature_indices,
                endpoint_graph_distances=wing_fuse_le_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        wing_fuse_te_feature_solve_local_indices, wing_fuse_te_feature_solve_support = (
            _endpoint_feature_solve_rows_from_global_indices(
                solve_vertex_local_indices=solve_vertex_local_indices,
                endpoint_feature_indices=wing_fuse_te_endpoint_feature_indices,
                endpoint_graph_distances=wing_fuse_te_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        htail_fuse_le_feature_solve_local_indices, htail_fuse_le_feature_solve_support = (
            _endpoint_feature_solve_rows_from_global_indices(
                solve_vertex_local_indices=solve_vertex_local_indices,
                endpoint_feature_indices=htail_fuse_le_endpoint_feature_indices,
                endpoint_graph_distances=htail_fuse_le_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        htail_fuse_te_feature_solve_local_indices, htail_fuse_te_feature_solve_support = (
            _endpoint_feature_solve_rows_from_global_indices(
                solve_vertex_local_indices=solve_vertex_local_indices,
                endpoint_feature_indices=htail_fuse_te_endpoint_feature_indices,
                endpoint_graph_distances=htail_fuse_te_graph_distances,
                support_radius=exact_seam_endpoint_feature_graph_radius,
                full_strength_radius=exact_seam_endpoint_feature_full_strength_graph_radius,
            )
        )
        wing_fuse_le_halo_local_indices, wing_fuse_le_halo_support = _rows_from_global_support(
            row_global_indices=interpolation_indices,
            global_support=wing_fuse_le_endpoint_halo_support_global,
            support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
        )
        wing_fuse_te_halo_local_indices, wing_fuse_te_halo_support = _rows_from_global_support(
            row_global_indices=interpolation_indices,
            global_support=wing_fuse_te_endpoint_halo_support_global,
            support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
        )
        htail_fuse_le_halo_local_indices, htail_fuse_le_halo_support = _rows_from_global_support(
            row_global_indices=interpolation_indices,
            global_support=htail_fuse_le_endpoint_halo_support_global,
            support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
        )
        htail_fuse_te_halo_local_indices, htail_fuse_te_halo_support = _rows_from_global_support(
            row_global_indices=interpolation_indices,
            global_support=htail_fuse_te_endpoint_halo_support_global,
            support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
        )
        wing_fuse_le_halo_solve_local_indices, wing_fuse_le_halo_solve_support = (
            _solve_rows_from_global_support(
                solve_vertex_indices=solve_vertex_indices,
                global_support=wing_fuse_le_endpoint_halo_support_global,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
            )
        )
        wing_fuse_te_halo_solve_local_indices, wing_fuse_te_halo_solve_support = (
            _solve_rows_from_global_support(
                solve_vertex_indices=solve_vertex_indices,
                global_support=wing_fuse_te_endpoint_halo_support_global,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
            )
        )
        htail_fuse_le_halo_solve_local_indices, htail_fuse_le_halo_solve_support = (
            _solve_rows_from_global_support(
                solve_vertex_indices=solve_vertex_indices,
                global_support=htail_fuse_le_endpoint_halo_support_global,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
            )
        )
        htail_fuse_te_halo_solve_local_indices, htail_fuse_te_halo_solve_support = (
            _solve_rows_from_global_support(
                solve_vertex_indices=solve_vertex_indices,
                global_support=htail_fuse_te_endpoint_halo_support_global,
                support_cutoff=exact_seam_endpoint_feature_halo_support_cutoff,
            )
        )
        wing_fuse_directional_endpoint_parametric_coordinates = np.vstack(
            [
                _endpoint_feature_direction_parametric_pair(
                    endpoint_parametric_coordinate=seam_seed_wing_parametric_coordinates[
                        seam_seed_leading_local
                    ],
                    endpoint_feature_indices=wing_fuse_le_endpoint_feature_indices,
                    endpoint_graph_distances=wing_fuse_le_graph_distances,
                    component_state=wing_projection_state,
                ),
                _endpoint_feature_direction_parametric_pair(
                    endpoint_parametric_coordinate=seam_seed_wing_parametric_coordinates[
                        seam_seed_trailing_local
                    ],
                    endpoint_feature_indices=wing_fuse_te_endpoint_feature_indices,
                    endpoint_graph_distances=wing_fuse_te_graph_distances,
                    component_state=wing_projection_state,
                ),
            ]
        )
        htail_fuse_directional_endpoint_parametric_coordinates = np.vstack(
            [
                _endpoint_feature_direction_parametric_pair(
                    endpoint_parametric_coordinate=seam_seed_htail_parametric_coordinates[
                        htail_root_leading_local
                    ],
                    endpoint_feature_indices=htail_fuse_le_endpoint_feature_indices,
                    endpoint_graph_distances=htail_fuse_le_graph_distances,
                    component_state=htail_projection_state,
                ),
                _endpoint_feature_direction_parametric_pair(
                    endpoint_parametric_coordinate=seam_seed_htail_parametric_coordinates[
                        htail_root_trailing_local
                    ],
                    endpoint_feature_indices=htail_fuse_te_endpoint_feature_indices,
                    endpoint_graph_distances=htail_fuse_te_graph_distances,
                    component_state=htail_projection_state,
                ),
            ]
        )
    else:
        wing_fuse_le_feature_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_te_feature_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_le_feature_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_te_feature_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_le_feature_support = np.empty((0,), dtype=float)
        wing_fuse_te_feature_support = np.empty((0,), dtype=float)
        htail_fuse_le_feature_support = np.empty((0,), dtype=float)
        htail_fuse_te_feature_support = np.empty((0,), dtype=float)
        wing_fuse_le_feature_solve_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_te_feature_solve_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_le_feature_solve_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_te_feature_solve_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_le_feature_solve_support = np.empty((0,), dtype=float)
        wing_fuse_te_feature_solve_support = np.empty((0,), dtype=float)
        htail_fuse_le_feature_solve_support = np.empty((0,), dtype=float)
        htail_fuse_te_feature_solve_support = np.empty((0,), dtype=float)
        wing_fuse_directional_endpoint_parametric_coordinates = np.vstack(
            [
                seam_seed_wing_parametric_coordinates[seam_seed_leading_local],
                seam_seed_wing_parametric_coordinates[seam_seed_leading_local],
                seam_seed_wing_parametric_coordinates[seam_seed_trailing_local],
                seam_seed_wing_parametric_coordinates[seam_seed_trailing_local],
            ]
        )
        htail_fuse_directional_endpoint_parametric_coordinates = np.vstack(
            [
                seam_seed_htail_parametric_coordinates[htail_root_leading_local],
                seam_seed_htail_parametric_coordinates[htail_root_leading_local],
                seam_seed_htail_parametric_coordinates[htail_root_trailing_local],
                seam_seed_htail_parametric_coordinates[htail_root_trailing_local],
            ]
        )
        wing_fuse_le_halo_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_te_halo_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_le_halo_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_te_halo_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_le_halo_support = np.empty((0,), dtype=float)
        wing_fuse_te_halo_support = np.empty((0,), dtype=float)
        htail_fuse_le_halo_support = np.empty((0,), dtype=float)
        htail_fuse_te_halo_support = np.empty((0,), dtype=float)
        wing_fuse_le_halo_solve_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_te_halo_solve_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_le_halo_solve_local_indices = np.empty((0,), dtype=np.int64)
        htail_fuse_te_halo_solve_local_indices = np.empty((0,), dtype=np.int64)
        wing_fuse_le_halo_solve_support = np.empty((0,), dtype=float)
        wing_fuse_te_halo_solve_support = np.empty((0,), dtype=float)
        htail_fuse_le_halo_solve_support = np.empty((0,), dtype=float)
        htail_fuse_te_halo_solve_support = np.empty((0,), dtype=float)

    fuselage_diameter_taper_x_end = float(np.min(seam_seed_htail_points[:, 0])) - (
        fuselage_diameter_taper_margin_htail_chords * htail_root_chord_proxy
    )
    fuselage_diameter_taper_length = (
        fuselage_diameter_taper_length_htail_chords * htail_root_chord_proxy
    )
    fuselage_diameter_taper_x_start = (
        fuselage_diameter_taper_x_end - fuselage_diameter_taper_length
    )
    if fuselage_diameter_taper_before_empennage:
        fuselage_diameter_taper_parameter = (
            fuselage_diameter_taper_x_end - initial_fuse_coefficients[:, 0]
        ) / max(fuselage_diameter_taper_length, 1e-12)
        fuselage_diameter_taper_weights = _smoothstep_numpy(
            fuselage_diameter_taper_parameter
        )
    else:
        fuselage_diameter_taper_weights = np.ones(
            initial_fuse_coefficients.shape[0],
            dtype=float,
        )
    fuselage_diameter_taper_rows = np.broadcast_to(
        fuselage_diameter_taper_weights.reshape((-1, 1)),
        initial_fuse_coefficients.shape,
    ).copy()

    def move_wing_coefficients_numpy(translation_x: float) -> np.ndarray:
        translated_coefficients = initial_wing_coefficients + np.array(
            [translation_x, 0.0, 0.0],
            dtype=float,
        )[None, :]
        return translated_coefficients

    def move_htail_coefficients_numpy(translation_x: float, rotation_degrees: float) -> np.ndarray:
        translated_coefficients = initial_htail_coefficients + np.array(
            [translation_x, 0.0, 0.0],
            dtype=float,
        )[None, :]
        translated_root_quarter_chord = htail_root_quarter_chord + np.array(
            [translation_x, 0.0, 0.0],
            dtype=float,
        ).reshape((1, 3))
        return _rotate_points_about_y_axis_numpy(
            translated_coefficients,
            rotation_degrees,
            translated_root_quarter_chord,
        )

    def move_fuse_coefficients_numpy(diameter_scale: float) -> np.ndarray:
        fuselage_radial_offsets = (
            initial_fuse_coefficients - fuselage_centerline_rows
        ) * fuselage_radial_mask * fuselage_diameter_taper_rows
        return initial_fuse_coefficients + (
            float(diameter_scale) - 1.0
        ) * fuselage_radial_offsets

    envelope_translation_values = np.array(
        [
            -translation_envelope_mac * root_chord_proxy,
            0.0,
            translation_envelope_mac * root_chord_proxy,
        ],
        dtype=float,
    )
    htail_envelope_translation_values = np.array(
        [
            -translation_envelope_mac * htail_root_chord_proxy,
            0.0,
            translation_envelope_mac * htail_root_chord_proxy,
        ],
        dtype=float,
    )
    envelope_rotation_values = np.array(
        [-rotation_envelope_degrees, rotation_envelope_degrees],
        dtype=float,
    )
    fuselage_diameter_envelope_values = np.array(
        [
            fuselage_diameter_scale_lower,
            fuselage_diameter_scale_upper,
        ],
        dtype=float,
    )

    seam_support_envelope = np.zeros(num_vertices, dtype=float)
    swept_states = [(0.0, 0.0, 0.0, fuselage_diameter_scale_initial)]
    # swept_states.extend(
    #     [
    #         (
    #             float(wing_translation_x),
    #             float(htail_translation_x),
    #             float(htail_rotation_degrees),
    #             float(fuselage_diameter_scale_setup),
    #         )
    #         for wing_translation_x in (envelope_translation_values[0], envelope_translation_values[2])
    #         for htail_translation_x in (
    #             htail_envelope_translation_values[0],
    #             htail_envelope_translation_values[2],
    #         )
    #         for htail_rotation_degrees in envelope_rotation_values
    #         for fuselage_diameter_scale_setup in fuselage_diameter_envelope_values
    #     ]
    # )
    for (
        wing_translation_x,
        htail_translation_x,
        htail_rotation_degrees,
        fuselage_diameter_scale_setup,
    ) in swept_states:
        moved_wing_coefficients_setup = move_wing_coefficients_numpy(
            wing_translation_x,
        )
        moved_htail_coefficients_setup = move_htail_coefficients_numpy(
            htail_translation_x,
            htail_rotation_degrees,
        )
        moved_fuse_coefficients_setup = move_fuse_coefficients_numpy(
            fuselage_diameter_scale_setup,
        )
        wing_sdf_values, _ = wing_sdf_model.project(moved_wing_coefficients_setup, vertices)
        fuse_sdf_values, _ = fuse_sdf_model.project(moved_fuse_coefficients_setup, vertices)
        htail_sdf_values, _ = htail_sdf_model.project(moved_htail_coefficients_setup, vertices)
        setup_wing_fuse_indicator = _interaction_indicator_two_components(
            np.column_stack([wing_sdf_values, fuse_sdf_values])
        )
        setup_htail_fuse_indicator = _interaction_indicator_two_components(
            np.column_stack([htail_sdf_values, fuse_sdf_values])
        )
        seam_support_envelope = np.maximum(
            seam_support_envelope,
            _gaussian_support(setup_wing_fuse_indicator, seam_support_sigma),
        )
        seam_support_envelope = np.maximum(
            seam_support_envelope,
            _gaussian_support(setup_htail_fuse_indicator, seam_support_sigma),
        )

    seam_graph_band_mask = seam_graph_distances <= seam_graph_radius
    seam_envelope_mask = seam_support_envelope >= setup_seam_support_cutoff
    seam_candidate_mask = seam_graph_band_mask & seam_envelope_mask
    seam_candidate_mask[seam_seed_indices] = True
    seam_candidate_indices = np.where(seam_candidate_mask)[0]
    wing_endpoint_feature_mask = np.zeros(num_vertices, dtype=bool)
    wing_endpoint_feature_mask[wing_fuse_le_endpoint_feature_indices] = True
    wing_endpoint_feature_mask[wing_fuse_te_endpoint_feature_indices] = True
    wing_endpoint_halo_mask = np.zeros(num_vertices, dtype=bool)
    wing_endpoint_halo_mask[wing_fuse_le_endpoint_halo_indices] = True
    wing_endpoint_halo_mask[wing_fuse_te_endpoint_halo_indices] = True
    htail_endpoint_feature_mask = np.zeros(num_vertices, dtype=bool)
    htail_endpoint_feature_mask[htail_fuse_le_endpoint_feature_indices] = True
    htail_endpoint_feature_mask[htail_fuse_te_endpoint_feature_indices] = True
    htail_endpoint_halo_mask = np.zeros(num_vertices, dtype=bool)
    htail_endpoint_halo_mask[htail_fuse_le_endpoint_halo_indices] = True
    htail_endpoint_halo_mask[htail_fuse_te_endpoint_halo_indices] = True
    protected_wing_feature_indices = np.empty((0,), dtype=np.int64)
    protected_wing_feature_parametric_coordinates = np.empty((0, 3), dtype=float)
    if wing_feature_protection:
        protected_wing_feature_mask = (
            sharp_feature_vertex_mask
            & (wing_projection_distances <= wing_surface_projection_tolerance)
            & ~seam_candidate_mask
            & ~wing_endpoint_feature_mask
            & ~wing_endpoint_halo_mask
            & solve_vertex_mask
        )
        protected_wing_feature_indices = np.where(protected_wing_feature_mask)[0]
        if protected_wing_feature_indices.size > 0:
            protected_wing_feature_parametric_coordinates = np.column_stack(
                [
                    wing_projection_state["patch_id"][protected_wing_feature_indices],
                    wing_projection_state["uv"][protected_wing_feature_indices],
                ]
            )

    protected_htail_feature_indices = np.empty((0,), dtype=np.int64)
    protected_htail_feature_parametric_coordinates = np.empty((0, 3), dtype=float)
    if htail_feature_protection:
        protected_htail_feature_mask = (
            sharp_feature_vertex_mask
            & (htail_projection_distances <= wing_surface_projection_tolerance)
            & ~htail_endpoint_feature_mask
            & ~htail_endpoint_halo_mask
            & ~seam_candidate_mask
            & solve_vertex_mask
        )
        protected_htail_feature_indices = np.where(protected_htail_feature_mask)[0]
        if protected_htail_feature_indices.size > 0:
            protected_htail_feature_parametric_coordinates = np.column_stack(
                [
                    htail_projection_state["patch_id"][protected_htail_feature_indices],
                    htail_projection_state["uv"][protected_htail_feature_indices],
                ]
            )

    protected_feature_indices = np.unique(
        np.concatenate([protected_wing_feature_indices, protected_htail_feature_indices])
    )
    protected_wing_feature_solve_local_indices = solve_vertex_local_indices[
        protected_wing_feature_indices
    ]
    protected_htail_feature_solve_local_indices = solve_vertex_local_indices[
        protected_htail_feature_indices
    ]
    aft_fuse_symmetry_anchor_solve_local_indices = solve_vertex_local_indices[
        aft_fuse_symmetry_anchor_indices
    ]
    aft_fuse_fixed_solve_local_indices = solve_vertex_local_indices[
        aft_fuse_fixed_vertex_indices
    ]
    outboard_wing_parametric_fixed_solve_local_indices = solve_vertex_local_indices[
        outboard_wing_parametric_fixed_indices
    ]
    outboard_htail_parametric_fixed_solve_local_indices = solve_vertex_local_indices[
        outboard_htail_parametric_fixed_indices
    ]
    constrained_stationary_indices = np.unique(
        np.concatenate(
            [
                protected_wing_feature_solve_local_indices,
                protected_htail_feature_solve_local_indices,
                aft_fuse_symmetry_anchor_solve_local_indices,
                aft_fuse_fixed_solve_local_indices,
                outboard_wing_parametric_fixed_solve_local_indices,
                outboard_htail_parametric_fixed_solve_local_indices,
            ]
        )
    )
    constrained_stationary_mask = np.zeros(num_solve_vertices, dtype=bool)
    constrained_stationary_mask[constrained_stationary_indices] = True
    unprotected_laplacian_scale = np.ones(num_solve_vertices, dtype=float)
    unprotected_laplacian_scale[constrained_stationary_indices] = 0.0
    step9_projection_excluded_solve_local_indices = np.unique(
        np.concatenate(
            [
                aft_fuse_fixed_solve_local_indices,
                outboard_wing_parametric_fixed_solve_local_indices,
                outboard_htail_parametric_fixed_solve_local_indices,
            ]
        )
    )
    step9_projection_active_solve_local_indices = np.setdiff1d(
        np.arange(num_solve_vertices, dtype=np.int64),
        step9_projection_excluded_solve_local_indices,
    )
    step9_projection_active_mask = np.zeros(num_solve_vertices, dtype=bool)
    step9_projection_active_mask[step9_projection_active_solve_local_indices] = True
    step9_wing_fuse_seam_solve_local_indices = solve_vertex_local_indices[
        wing_fuse_seam_seed_indices
    ]
    step9_htail_fuse_seam_solve_local_indices = solve_vertex_local_indices[
        htail_fuse_seam_seed_indices
    ]
    step9_wing_fuse_seam_solve_local_indices = step9_wing_fuse_seam_solve_local_indices[
        step9_wing_fuse_seam_solve_local_indices >= 0
    ]
    step9_htail_fuse_seam_solve_local_indices = step9_htail_fuse_seam_solve_local_indices[
        step9_htail_fuse_seam_solve_local_indices >= 0
    ]
    step9_wing_fuse_seam_solve_local_indices = np.intersect1d(
        step9_wing_fuse_seam_solve_local_indices,
        step9_projection_active_solve_local_indices,
    )
    step9_htail_fuse_seam_solve_local_indices = np.intersect1d(
        step9_htail_fuse_seam_solve_local_indices,
        step9_projection_active_solve_local_indices,
    )
    step9_exact_seam_solve_mask = np.zeros(num_solve_vertices, dtype=bool)
    step9_exact_seam_solve_mask[step9_wing_fuse_seam_solve_local_indices] = True
    step9_exact_seam_solve_mask[step9_htail_fuse_seam_solve_local_indices] = True

    solve_initial_component_ownership = initial_component_ownership[solve_vertex_indices]
    solve_wing_patch_ids = np.asarray(
        wing_projection_state["patch_id"][solve_vertex_indices],
        dtype=np.int64,
    )
    solve_htail_patch_ids = np.asarray(
        htail_projection_state["patch_id"][solve_vertex_indices],
        dtype=np.int64,
    )
    step9_fuse_projection_solve_local_indices = np.where(
        step9_projection_active_mask
        & ~step9_exact_seam_solve_mask
        & (solve_initial_component_ownership == 1)
    )[0]
    step9_wing_patch_projection_solve_local_indices = {}
    for patch_id in wing_patch_function_sets:
        patch_mask = (
            step9_projection_active_mask
            & ~step9_exact_seam_solve_mask
            & (solve_initial_component_ownership == 0)
            & (solve_wing_patch_ids == int(patch_id))
        )
        patch_rows = np.where(patch_mask)[0]
        if patch_rows.size > 0:
            step9_wing_patch_projection_solve_local_indices[int(patch_id)] = patch_rows
    step9_htail_patch_projection_solve_local_indices = {}
    for patch_id in htail_patch_function_sets:
        patch_mask = (
            step9_projection_active_mask
            & ~step9_exact_seam_solve_mask
            & (solve_initial_component_ownership == 2)
            & (solve_htail_patch_ids == int(patch_id))
        )
        patch_rows = np.where(patch_mask)[0]
        if patch_rows.size > 0:
            step9_htail_patch_projection_solve_local_indices[int(patch_id)] = patch_rows

    def _seam_patch_local_groups(
        seam_solve_local_indices: np.ndarray,
        patch_ids: np.ndarray,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        seam_solve_local_indices = np.asarray(
            seam_solve_local_indices,
            dtype=np.int64,
        ).reshape(-1)
        groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if seam_solve_local_indices.size == 0:
            return groups
        seam_patch_ids = patch_ids[seam_solve_local_indices]
        for patch_id in np.unique(seam_patch_ids):
            seam_local_rows = np.where(seam_patch_ids == int(patch_id))[0]
            groups[int(patch_id)] = (
                seam_local_rows.astype(np.int64),
                seam_solve_local_indices[seam_local_rows],
            )
        return groups

    step9_wing_fuse_wing_patch_groups = _seam_patch_local_groups(
        step9_wing_fuse_seam_solve_local_indices,
        solve_wing_patch_ids,
    )
    step9_htail_fuse_htail_patch_groups = _seam_patch_local_groups(
        step9_htail_fuse_seam_solve_local_indices,
        solve_htail_patch_ids,
    )

    wing_fuse_solve_seam_mask = np.isin(solve_vertex_indices, wing_fuse_seam_seed_indices)
    htail_fuse_solve_seam_mask = np.isin(solve_vertex_indices, htail_fuse_seam_seed_indices)
    wing_fuse_solve_neighbor_support = np.exp(
        -(
            wing_fuse_graph_distances[solve_vertex_indices]
            / max(exact_seam_neighbor_graph_radius, 1e-12)
        )
        ** 2
    )
    htail_fuse_solve_neighbor_support = np.exp(
        -(
            htail_fuse_graph_distances[solve_vertex_indices]
            / max(exact_seam_neighbor_graph_radius, 1e-12)
        )
        ** 2
    )
    wing_fuse_solve_neighbor_support[wing_fuse_solve_seam_mask] = 1.0
    htail_fuse_solve_neighbor_support[htail_fuse_solve_seam_mask] = 1.0
    solve_vertex_driving_strength = max(
        float(exact_seam_solve_vertex_driving_strength),
        0.0,
    )
    if exact_seam_solve_vertex_driving and solve_vertex_driving_strength > 0.0:
        wing_fuse_solve_driven_mask = (
            (wing_fuse_solve_neighbor_support >= exact_seam_neighbor_support_cutoff)
            & ~htail_fuse_solve_seam_mask
            & (~constrained_stationary_mask | wing_fuse_solve_seam_mask)
        )
        htail_fuse_solve_driven_mask = (
            (htail_fuse_solve_neighbor_support >= exact_seam_neighbor_support_cutoff)
            & ~wing_fuse_solve_seam_mask
            & (~constrained_stationary_mask | htail_fuse_solve_seam_mask)
        )
    else:
        wing_fuse_solve_driven_mask = np.zeros(num_solve_vertices, dtype=bool)
        htail_fuse_solve_driven_mask = np.zeros(num_solve_vertices, dtype=bool)
    wing_fuse_solve_driven_local_indices = np.where(wing_fuse_solve_driven_mask)[0]
    htail_fuse_solve_driven_local_indices = np.where(htail_fuse_solve_driven_mask)[0]
    wing_fuse_solve_driven_support = (
        solve_vertex_driving_strength
        * wing_fuse_solve_neighbor_support[wing_fuse_solve_driven_local_indices]
    )
    htail_fuse_solve_driven_support = (
        solve_vertex_driving_strength
        * htail_fuse_solve_neighbor_support[htail_fuse_solve_driven_local_indices]
    )
    wing_fuse_solve_nearest_seam_local_indices = np.empty((0,), dtype=np.int64)
    if wing_fuse_solve_driven_local_indices.size > 0:
        wing_fuse_solve_nearest_seam_local_indices = _nearest_source_indices_numpy(
            solve_vertices[wing_fuse_solve_driven_local_indices],
            wing_fuse_seam_baseline_points,
        )
    htail_fuse_solve_nearest_seam_local_indices = np.empty((0,), dtype=np.int64)
    if htail_fuse_solve_driven_local_indices.size > 0:
        htail_fuse_solve_nearest_seam_local_indices = _nearest_source_indices_numpy(
            solve_vertices[htail_fuse_solve_driven_local_indices],
            htail_fuse_seam_baseline_points,
        )

    length_scale = np.linalg.norm(
        np.max(interpolation_vertices, axis=0) - np.min(interpolation_vertices, axis=0)
    )
    rbf_kernel_scale = rbf_kernel_scale_factor * max(length_scale, 1e-12)
    if rbf_kernel == "cubic":
        kernel_regularization_scale = max(length_scale**3, 1.0)
    else:
        kernel_regularization_scale = 1.0
    rbf_regularization = rbf_regularization_factor * kernel_regularization_scale
    rbf_polynomial_regularization = rbf_polynomial_regularization_factor

    training_sample_weights = np.full(
        interpolation_vertices.shape[0],
        global_interpolation_sample_weight,
        dtype=float,
    )
    training_sample_weights[seam_interpolation_mask] *= seam_training_sample_weight
    training_sample_weights[
        aft_fuse_symmetry_anchor_interpolation_mask
    ] *= aft_fuse_symmetry_anchor_training_sample_weight
    training_sample_weights[
        aft_fuse_symmetry_neighbor_interpolation_mask
    ] *= aft_fuse_symmetry_neighbor_training_sample_weight

    vertices_csdl = csdl.Variable(name="solve_surface_vertices", value=solve_vertices)
    interpolation_vertices_csdl = csdl.Variable(
        name="interpolation_vertices",
        value=interpolation_vertices,
    )
    wing_evaluation_interpolation_operation = bsm3.FunctionSetEvaluationOperation(
        model=wing_evaluation_model
    )
    fuse_evaluation_interpolation_operation = bsm3.FunctionSetEvaluationOperation(
        model=fuse_evaluation_model
    )
    htail_evaluation_interpolation_operation = bsm3.FunctionSetEvaluationOperation(
        model=htail_evaluation_model
    )
    wing_evaluation_feature_operation = bsm3.FunctionSetEvaluationOperation(
        model=wing_evaluation_model
    )
    wing_evaluation_outboard_fixed_operation = bsm3.FunctionSetEvaluationOperation(
        model=wing_evaluation_model
    )
    htail_evaluation_outboard_fixed_operation = bsm3.FunctionSetEvaluationOperation(
        model=htail_evaluation_model
    )
    htail_evaluation_feature_operation = bsm3.FunctionSetEvaluationOperation(
        model=htail_evaluation_model
    )
    interpolation_wing_closest_distance_operation = bsm3.FunctionSetClosestDistanceOperation(
        model=wing_sdf_model
    )
    interpolation_fuse_closest_distance_operation = bsm3.FunctionSetClosestDistanceOperation(
        model=fuse_sdf_model
    )
    interpolation_htail_closest_distance_operation = bsm3.FunctionSetClosestDistanceOperation(
        model=htail_sdf_model
    )

    moved_wing_coefficients = wing_coefficients_csdl
    moved_htail_coefficients = htail_coefficients_csdl
    moved_fuse_coefficients = fuse_coefficients_csdl
    moved_total_coefficients = csdl.concatenate(
        (moved_fuse_coefficients, moved_wing_coefficients, moved_htail_coefficients),
        axis=0,
    )
    moved_wing_patch_coefficients = {
        patch_id: _stack_coefficients_csdl_allow_single(function_set)
        for patch_id, function_set in wing_patch_function_sets.items()
    }
    moved_htail_patch_coefficients = {
        patch_id: _stack_coefficients_csdl_allow_single(function_set)
        for patch_id, function_set in htail_patch_function_sets.items()
    }

    moved_interpolation_wing_points = wing_evaluation_interpolation_operation.evaluate(
        moved_wing_coefficients,
        interpolation_wing_parametric_coordinates,
    )
    interpolation_wing_displacements = (
        moved_interpolation_wing_points - interpolation_initial_wing_points
    )
    moved_interpolation_fuse_points = fuse_evaluation_interpolation_operation.evaluate(
        moved_fuse_coefficients,
        interpolation_fuse_parametric_coordinates,
    )
    interpolation_fuse_displacements = (
        moved_interpolation_fuse_points - interpolation_initial_fuse_points
    )
    moved_interpolation_htail_points = htail_evaluation_interpolation_operation.evaluate(
        moved_htail_coefficients,
        interpolation_htail_parametric_coordinates,
    )
    interpolation_htail_displacements = (
        moved_interpolation_htail_points - interpolation_initial_htail_points
    )

    provisional_interpolation_target_displacements = _expand_vector_to_columns(
        interpolation_wing_weights,
        3,
    ) * interpolation_wing_displacements + _expand_vector_to_columns(
        interpolation_fuse_weights,
        3,
    ) * interpolation_fuse_displacements + _expand_vector_to_columns(
        interpolation_htail_weights,
        3,
    ) * interpolation_htail_displacements
    provisional_interpolation_vertices = (
        interpolation_vertices_csdl + provisional_interpolation_target_displacements
    )

    interpolation_sdf_to_wing = interpolation_wing_closest_distance_operation.evaluate(
        moved_wing_coefficients,
        provisional_interpolation_vertices,
    )
    interpolation_sdf_to_fuse = interpolation_fuse_closest_distance_operation.evaluate(
        moved_fuse_coefficients,
        provisional_interpolation_vertices,
    )
    interpolation_sdf_to_htail = interpolation_htail_closest_distance_operation.evaluate(
        moved_htail_coefficients,
        provisional_interpolation_vertices,
    )
    sigma_phi_safe = max(float(sigma_phi), 1e-12)
    interpolation_participation_wing = csdl.exp(
        -(interpolation_sdf_to_wing**2) / (sigma_phi_safe**2)
    )
    interpolation_participation_fuse = csdl.exp(
        -(interpolation_sdf_to_fuse**2) / (sigma_phi_safe**2)
    )
    interpolation_participation_htail = csdl.exp(
        -(interpolation_sdf_to_htail**2) / (sigma_phi_safe**2)
    )
    interpolation_participation_sum = (
        interpolation_participation_wing
        + interpolation_participation_fuse
        + interpolation_participation_htail
    )
    interpolation_smooth_wing_weight = interpolation_participation_wing / (
        interpolation_participation_sum
    )
    interpolation_smooth_fuse_weight = interpolation_participation_fuse / (
        interpolation_participation_sum
    )
    interpolation_smooth_htail_weight = interpolation_participation_htail / (
        interpolation_participation_sum
    )
    interpolation_wing_fuse_interaction_indicator = (
        interpolation_sdf_to_wing**2 + interpolation_sdf_to_fuse**2
    ) ** 0.5
    interpolation_htail_fuse_interaction_indicator = (
        interpolation_sdf_to_htail**2 + interpolation_sdf_to_fuse**2
    ) ** 0.5
    wing_fuse_seam_support_interp = csdl.exp(
        -(
            interpolation_wing_fuse_interaction_indicator / max(seam_support_sigma, 1e-12)
        )
        ** 2
    )
    htail_fuse_seam_support_interp = csdl.exp(
        -(
            interpolation_htail_fuse_interaction_indicator / max(seam_support_sigma, 1e-12)
        )
        ** 2
    )
    if wing_fuse_seam_interpolation_local_indices.size > 0:
        wing_fuse_seam_support_interp = wing_fuse_seam_support_interp.set(
            csdl.slice[wing_fuse_seam_interpolation_local_indices.tolist()],
            1.0,
        )
    if htail_fuse_seam_interpolation_local_indices.size > 0:
        htail_fuse_seam_support_interp = htail_fuse_seam_support_interp.set(
            csdl.slice[htail_fuse_seam_interpolation_local_indices.tolist()],
            1.0,
        )

    soft_interpolation_target_displacements = (
        _expand_vector_to_columns(interpolation_smooth_wing_weight, 3)
        * interpolation_wing_displacements
        + _expand_vector_to_columns(interpolation_smooth_fuse_weight, 3)
        * interpolation_fuse_displacements
        + _expand_vector_to_columns(interpolation_smooth_htail_weight, 3)
        * interpolation_htail_displacements
    )
    wing_fuse_seam_target_displacements = (
        interpolation_wing_displacements + interpolation_fuse_displacements
    )
    htail_fuse_seam_target_displacements = (
        interpolation_htail_displacements + interpolation_fuse_displacements
    )
    corrected_interpolation_target_displacements = (
        _expand_vector_to_columns(1.0 - wing_fuse_seam_support_interp, 3)
        * soft_interpolation_target_displacements
        + _expand_vector_to_columns(wing_fuse_seam_support_interp, 3)
        * wing_fuse_seam_target_displacements
    )
    corrected_interpolation_target_displacements = (
        _expand_vector_to_columns(1.0 - htail_fuse_seam_support_interp, 3)
        * corrected_interpolation_target_displacements
        + _expand_vector_to_columns(htail_fuse_seam_support_interp, 3)
        * htail_fuse_seam_target_displacements
    )
    relaxed_wing_fuse_seam_displacements = None
    relaxed_htail_fuse_seam_displacements = None
    relaxed_wing_fuse_seam_points = None
    relaxed_htail_fuse_seam_points = None
    corrected_wing_fuse_seam_target_points = None
    corrected_htail_fuse_seam_target_points = None
    moved_wing_fuse_anchor_points = None
    moved_htail_fuse_anchor_points = None
    moved_wing_fuse_directional_endpoint_feature_points = None
    moved_htail_fuse_directional_endpoint_feature_points = None
    soft_bracket_debug_traces = []
    if exact_seam_sdf_relaxation:
        if wing_fuse_seam_interpolation_local_indices.size > 0:
            wing_fuse_seam_local_index_list = wing_fuse_seam_interpolation_local_indices.tolist()
            wing_fuse_initial_seam_points = interpolation_vertices_csdl[
                csdl.slice[wing_fuse_seam_local_index_list, :]
            ]
            wing_fuse_initial_relaxation_points = (
                wing_fuse_initial_seam_points
                + wing_fuse_seam_target_displacements[
                    csdl.slice[wing_fuse_seam_local_index_list, :]
                ]
            )
            use_wing_fuse_parametric_seam = (
                exact_seam_wing_fuse_recompute_mode
                in ("wing_parametric_bracketed_root", "wing_parametric_soft_bracket")
            )
            if (
                exact_seam_curve_regularization
                and wing_fuse_anchor_indices.size > 0
                and not use_wing_fuse_parametric_seam
            ):
                wing_anchor_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
                    model=wing_evaluation_model
                )
                moved_wing_fuse_anchor_points = wing_anchor_evaluation_operation.evaluate(
                    moved_wing_coefficients,
                    wing_fuse_anchor_parametric_coordinates,
                )
            if exact_seam_directional_endpoint_solve and not use_wing_fuse_parametric_seam:
                wing_directional_endpoint_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
                    model=wing_evaluation_model
                )
                moved_wing_fuse_directional_endpoint_feature_points = (
                    wing_directional_endpoint_evaluation_operation.evaluate(
                        moved_wing_coefficients,
                        wing_fuse_directional_endpoint_parametric_coordinates,
                    )
                )
            if exact_seam_wing_fuse_recompute_mode == "wing_parametric_bracketed_root":
                relaxed_wing_fuse_seam_points = _wing_fuse_seam_points_from_parametric_bracket_csdl(
                    moved_wing_coefficients,
                    moved_fuse_coefficients,
                    wing_evaluation_model,
                    fuse_sdf_model,
                    wing_fuse_seam_wing_parametric_coordinates,
                    spanwise_uv_axis=wing_fuse_spanwise_uv_axis,
                    tolerance=exact_seam_parametric_root_tolerance,
                    max_iter=exact_seam_parametric_root_max_iter,
                    label="wing/fuse",
                )
            elif exact_seam_wing_fuse_recompute_mode == "wing_parametric_soft_bracket":
                wing_fuse_soft_bracket_debug_trace = (
                    {"label": "wing/fuse"}
                    if soft_bracket_debug_enabled
                    else None
                )
                wing_fuse_soft_bracket_debug_indices = (
                    _select_soft_bracket_debug_sample_indices(
                        wing_fuse_seam_wing_parametric_coordinates.shape[0],
                        soft_bracket_debug_num_points,
                        seam_points=wing_fuse_seam_baseline_points,
                        leading_row=wing_fuse_le_seam_row,
                        trailing_row=wing_fuse_te_seam_row,
                        surface_side=soft_bracket_debug_surface,
                    )
                    if soft_bracket_debug_enabled
                    else None
                )
                relaxed_wing_fuse_seam_points = _component_fuse_seam_points_from_soft_parametric_bracket_csdl(
                    moved_wing_coefficients,
                    moved_fuse_coefficients,
                    wing_evaluation_model,
                    fuse_sdf_model,
                    wing_fuse_seam_wing_parametric_coordinates,
                    spanwise_uv_axis=wing_fuse_spanwise_uv_axis,
                    num_iterations=exact_seam_soft_bracket_iterations,
                    sign_sharpness=exact_seam_soft_bracket_sign_sharpness,
                    sdf_scale=exact_seam_soft_bracket_sdf_scale,
                    label="wing/fuse",
                    debug_sample_indices=wing_fuse_soft_bracket_debug_indices,
                    debug_trace=wing_fuse_soft_bracket_debug_trace,
                )
                if (
                    wing_fuse_soft_bracket_debug_trace
                    and "midpoint_points" in wing_fuse_soft_bracket_debug_trace
                ):
                    soft_bracket_debug_traces.append(wing_fuse_soft_bracket_debug_trace)
            elif exact_seam_wing_fuse_recompute_mode == "sdf_relaxation":
                relaxed_wing_fuse_seam_points = _relax_two_component_seam_points_csdl(
                    wing_fuse_initial_relaxation_points,
                    moved_wing_coefficients,
                    moved_fuse_coefficients,
                    wing_sdf_model,
                    fuse_sdf_model,
                    num_iterations=exact_seam_sdf_relaxation_iterations,
                    step_size=exact_seam_sdf_relaxation_step_size,
                    damping=exact_seam_sdf_relaxation_damping,
                    label="wing/fuse",
                    edge_pairs=wing_fuse_seam_edge_pairs,
                    arc_chain_ranges=wing_fuse_seam_arc_chain_ranges,
                    baseline_normalized_arc_positions=wing_fuse_baseline_normalized_arc_positions,
                    spacing_weight=(
                        exact_seam_spacing_regularization_weight
                        if exact_seam_curve_regularization
                        else 0.0
                    ),
                    spacing_length_scale=wing_fuse_baseline_curve_length,
                    anchor_indices=wing_fuse_anchor_indices,
                    anchor_points=moved_wing_fuse_anchor_points,
                    anchor_weight=(
                        exact_seam_anchor_regularization_weight
                        if exact_seam_curve_regularization
                        else 0.0
                    ),
                    regularization_step_size=(
                        exact_seam_regularization_step_size
                        if exact_seam_curve_regularization
                        else 0.0
                    ),
                    regularization_mode=exact_seam_regularization_mode,
                    unified_objective_step_size=exact_seam_unified_objective_step_size,
                    directional_endpoint_row_indices=np.array(
                        [wing_fuse_le_seam_row, wing_fuse_te_seam_row],
                        dtype=np.int64,
                    )
                    if (
                        exact_seam_directional_endpoint_solve
                        and wing_fuse_le_seam_row is not None
                        and wing_fuse_te_seam_row is not None
                    )
                    else None,
                    directional_endpoint_feature_points=(
                        moved_wing_fuse_directional_endpoint_feature_points
                        if exact_seam_directional_endpoint_solve
                        else None
                    ),
                    directional_endpoint_iterations=(
                        exact_seam_directional_endpoint_iterations
                        if exact_seam_directional_endpoint_solve
                        else 0
                    ),
                    directional_endpoint_step_size=(
                        exact_seam_directional_endpoint_step_size
                        if exact_seam_directional_endpoint_solve
                        else 0.0
                    ),
                    freeze_directional_endpoint_rows=exact_seam_directional_endpoint_freeze,
                )
            else:
                raise ValueError(
                    "Unsupported exact_seam_wing_fuse_recompute_mode: "
                    f"{exact_seam_wing_fuse_recompute_mode}"
                )
            relaxed_wing_fuse_seam_displacements = (
                relaxed_wing_fuse_seam_points - wing_fuse_initial_seam_points
            )
            corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
                csdl.slice[wing_fuse_seam_local_index_list, :],
                relaxed_wing_fuse_seam_displacements,
            )
        if htail_fuse_seam_interpolation_local_indices.size > 0:
            htail_fuse_seam_local_index_list = htail_fuse_seam_interpolation_local_indices.tolist()
            htail_fuse_initial_seam_points = interpolation_vertices_csdl[
                csdl.slice[htail_fuse_seam_local_index_list, :]
            ]
            htail_fuse_initial_relaxation_points = (
                htail_fuse_initial_seam_points
                + htail_fuse_seam_target_displacements[
                    csdl.slice[htail_fuse_seam_local_index_list, :]
                ]
            )
            use_htail_fuse_parametric_seam = (
                exact_seam_htail_fuse_recompute_mode
                in ("htail_parametric_soft_bracket",)
            )
            if (
                exact_seam_curve_regularization
                and htail_fuse_anchor_indices.size > 0
                and not use_htail_fuse_parametric_seam
            ):
                htail_anchor_evaluation_operation = bsm3.FunctionSetEvaluationOperation(
                    model=htail_evaluation_model
                )
                moved_htail_fuse_anchor_points = htail_anchor_evaluation_operation.evaluate(
                    moved_htail_coefficients,
                    htail_fuse_anchor_parametric_coordinates,
                )
            if exact_seam_directional_endpoint_solve and not use_htail_fuse_parametric_seam:
                htail_directional_endpoint_evaluation_operation = (
                    bsm3.FunctionSetEvaluationOperation(model=htail_evaluation_model)
                )
                moved_htail_fuse_directional_endpoint_feature_points = (
                    htail_directional_endpoint_evaluation_operation.evaluate(
                        moved_htail_coefficients,
                        htail_fuse_directional_endpoint_parametric_coordinates,
                    )
                )
            if exact_seam_htail_fuse_recompute_mode == "htail_parametric_soft_bracket":
                htail_fuse_soft_bracket_debug_trace = (
                    {"label": "htail/fuse"}
                    if soft_bracket_debug_enabled
                    else None
                )
                htail_fuse_soft_bracket_debug_indices = (
                    _select_soft_bracket_debug_sample_indices(
                        htail_fuse_seam_htail_parametric_coordinates.shape[0],
                        soft_bracket_debug_num_points,
                        seam_points=htail_fuse_seam_baseline_points,
                        leading_row=htail_fuse_le_seam_row,
                        trailing_row=htail_fuse_te_seam_row,
                        surface_side=soft_bracket_debug_surface,
                    )
                    if soft_bracket_debug_enabled
                    else None
                )
                relaxed_htail_fuse_seam_points = _component_fuse_seam_points_from_soft_parametric_bracket_csdl(
                    moved_htail_coefficients,
                    moved_fuse_coefficients,
                    htail_evaluation_model,
                    fuse_sdf_model,
                    htail_fuse_seam_htail_parametric_coordinates,
                    spanwise_uv_axis=htail_fuse_spanwise_uv_axis,
                    num_iterations=exact_seam_soft_bracket_iterations,
                    sign_sharpness=exact_seam_soft_bracket_sign_sharpness,
                    sdf_scale=exact_seam_soft_bracket_sdf_scale,
                    label="htail/fuse",
                    debug_sample_indices=htail_fuse_soft_bracket_debug_indices,
                    debug_trace=htail_fuse_soft_bracket_debug_trace,
                )
                if (
                    htail_fuse_soft_bracket_debug_trace
                    and "midpoint_points" in htail_fuse_soft_bracket_debug_trace
                ):
                    soft_bracket_debug_traces.append(htail_fuse_soft_bracket_debug_trace)
            elif exact_seam_htail_fuse_recompute_mode == "sdf_relaxation":
                relaxed_htail_fuse_seam_points = _relax_two_component_seam_points_csdl(
                    htail_fuse_initial_relaxation_points,
                    moved_htail_coefficients,
                    moved_fuse_coefficients,
                    htail_sdf_model,
                    fuse_sdf_model,
                    num_iterations=exact_seam_sdf_relaxation_iterations,
                    step_size=exact_seam_sdf_relaxation_step_size,
                    damping=exact_seam_sdf_relaxation_damping,
                    label="htail/fuse",
                    edge_pairs=htail_fuse_seam_edge_pairs,
                    arc_chain_ranges=htail_fuse_seam_arc_chain_ranges,
                    baseline_normalized_arc_positions=htail_fuse_baseline_normalized_arc_positions,
                    spacing_weight=(
                        exact_seam_spacing_regularization_weight
                        if exact_seam_curve_regularization
                        else 0.0
                    ),
                    spacing_length_scale=htail_fuse_baseline_curve_length,
                    anchor_indices=htail_fuse_anchor_indices,
                    anchor_points=moved_htail_fuse_anchor_points,
                    anchor_weight=(
                        exact_seam_anchor_regularization_weight
                        if exact_seam_curve_regularization
                        else 0.0
                    ),
                    regularization_step_size=(
                        exact_seam_regularization_step_size
                        if exact_seam_curve_regularization
                        else 0.0
                    ),
                    regularization_mode=exact_seam_regularization_mode,
                    unified_objective_step_size=exact_seam_unified_objective_step_size,
                    directional_endpoint_row_indices=np.array(
                        [htail_fuse_le_seam_row, htail_fuse_te_seam_row],
                        dtype=np.int64,
                    )
                    if (
                        exact_seam_directional_endpoint_solve
                        and htail_fuse_le_seam_row is not None
                        and htail_fuse_te_seam_row is not None
                    )
                    else None,
                    directional_endpoint_feature_points=(
                        moved_htail_fuse_directional_endpoint_feature_points
                        if exact_seam_directional_endpoint_solve
                        else None
                    ),
                    directional_endpoint_iterations=(
                        exact_seam_directional_endpoint_iterations
                        if exact_seam_directional_endpoint_solve
                        else 0
                    ),
                    directional_endpoint_step_size=(
                        exact_seam_directional_endpoint_step_size
                        if exact_seam_directional_endpoint_solve
                        else 0.0
                    ),
                    freeze_directional_endpoint_rows=exact_seam_directional_endpoint_freeze,
                )
            else:
                raise ValueError(
                    "Unsupported exact_seam_htail_fuse_recompute_mode: "
                    f"{exact_seam_htail_fuse_recompute_mode}"
                )
            relaxed_htail_fuse_seam_displacements = (
                relaxed_htail_fuse_seam_points - htail_fuse_initial_seam_points
            )
            corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
                csdl.slice[htail_fuse_seam_local_index_list, :],
                relaxed_htail_fuse_seam_displacements,
            )
    if wing_owned_interpolation_local_indices.size > 0:
        corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
            csdl.slice[wing_owned_interpolation_local_indices.tolist(), :],
            interpolation_wing_displacements[csdl.slice[wing_owned_interpolation_local_indices.tolist(), :]],
        )
    # Fuse-owned anchors should not be hard-overwritten here. Away from the
    # intersections, the smooth fuse weight already makes them follow the
    # fuselage diameter change. Near a wing/fuselage or tail/fuselage
    # intersection, hard-setting them to the fuse target would pin those
    # anchors when diameter_scale == 1 and fight the moving seam support.
    if htail_owned_interpolation_local_indices.size > 0:
        corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
            csdl.slice[htail_owned_interpolation_local_indices.tolist(), :],
            interpolation_htail_displacements[csdl.slice[htail_owned_interpolation_local_indices.tolist(), :]],
        )
    if (
        exact_seam_neighbor_driving
        and relaxed_wing_fuse_seam_displacements is not None
        and wing_fuse_neighbor_local_indices.size > 0
    ):
        wing_fuse_neighbor_local_index_list = wing_fuse_neighbor_local_indices.tolist()
        wing_fuse_neighbor_current_displacements = corrected_interpolation_target_displacements[
            csdl.slice[wing_fuse_neighbor_local_index_list, :]
        ]
        wing_fuse_neighbor_seam_displacements = _gather_rows_csdl(
            relaxed_wing_fuse_seam_displacements,
            wing_fuse_neighbor_nearest_seam_local_indices,
        )
        wing_fuse_neighbor_blend = _expand_vector_to_columns(
            wing_fuse_neighbor_support[wing_fuse_neighbor_local_indices],
            3,
        )
        corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
            csdl.slice[wing_fuse_neighbor_local_index_list, :],
            (
                (1.0 - wing_fuse_neighbor_blend)
                * wing_fuse_neighbor_current_displacements
                + wing_fuse_neighbor_blend * wing_fuse_neighbor_seam_displacements
            ),
        )
    if (
        exact_seam_neighbor_driving
        and relaxed_htail_fuse_seam_displacements is not None
        and htail_fuse_neighbor_local_indices.size > 0
    ):
        htail_fuse_neighbor_local_index_list = htail_fuse_neighbor_local_indices.tolist()
        htail_fuse_neighbor_current_displacements = corrected_interpolation_target_displacements[
            csdl.slice[htail_fuse_neighbor_local_index_list, :]
        ]
        htail_fuse_neighbor_seam_displacements = _gather_rows_csdl(
            relaxed_htail_fuse_seam_displacements,
            htail_fuse_neighbor_nearest_seam_local_indices,
        )
        htail_fuse_neighbor_blend = _expand_vector_to_columns(
            htail_fuse_neighbor_support[htail_fuse_neighbor_local_indices],
            3,
        )
        corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
            csdl.slice[htail_fuse_neighbor_local_index_list, :],
            (
                (1.0 - htail_fuse_neighbor_blend)
                * htail_fuse_neighbor_current_displacements
                + htail_fuse_neighbor_blend * htail_fuse_neighbor_seam_displacements
            ),
        )
    if exact_seam_endpoint_feature_driving:
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_le_feature_local_indices,
                wing_fuse_le_seam_row,
                wing_fuse_le_feature_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_te_feature_local_indices,
                wing_fuse_te_seam_row,
                wing_fuse_te_feature_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_le_feature_local_indices,
                htail_fuse_le_seam_row,
                htail_fuse_le_feature_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_te_feature_local_indices,
                htail_fuse_te_seam_row,
                htail_fuse_te_feature_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_le_halo_local_indices,
                wing_fuse_le_seam_row,
                wing_fuse_le_halo_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_te_halo_local_indices,
                wing_fuse_te_seam_row,
                wing_fuse_te_halo_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_le_halo_local_indices,
                htail_fuse_le_seam_row,
                htail_fuse_le_halo_support,
            )
        )
        corrected_interpolation_target_displacements = (
            _apply_endpoint_feature_training_drive_csdl(
                corrected_interpolation_target_displacements,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_te_halo_local_indices,
                htail_fuse_te_seam_row,
                htail_fuse_te_halo_support,
            )
        )
    if aft_fuse_symmetry_anchor_local_indices.size > 0:
        corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
            csdl.slice[aft_fuse_symmetry_anchor_local_indices.tolist(), :],
            interpolation_fuse_displacements[
                csdl.slice[aft_fuse_symmetry_anchor_local_indices.tolist(), :]
            ],
        )
    if aft_fuse_symmetry_neighbor_local_indices.size > 0:
        aft_fuse_symmetry_neighbor_index_list = aft_fuse_symmetry_neighbor_local_indices.tolist()
        current_aft_fuse_neighbor_displacements = corrected_interpolation_target_displacements[
            csdl.slice[aft_fuse_symmetry_neighbor_index_list, :]
        ]
        aft_fuse_neighbor_target_displacements = interpolation_fuse_displacements[
            csdl.slice[aft_fuse_symmetry_neighbor_index_list, :]
        ]
        aft_fuse_neighbor_blend_rows = _expand_vector_to_columns(
            np.full(
                aft_fuse_symmetry_neighbor_local_indices.size,
                aft_fuse_symmetry_neighbor_blend,
                dtype=float,
            ),
            3,
        )
        corrected_interpolation_target_displacements = corrected_interpolation_target_displacements.set(
            csdl.slice[aft_fuse_symmetry_neighbor_index_list, :],
            (1.0 - aft_fuse_neighbor_blend_rows) * current_aft_fuse_neighbor_displacements
            + aft_fuse_neighbor_blend_rows * aft_fuse_neighbor_target_displacements,
        )
    if wing_fuse_seam_interpolation_local_indices.size > 0:
        wing_fuse_seam_local_index_list = wing_fuse_seam_interpolation_local_indices.tolist()
        corrected_wing_fuse_seam_target_points = (
            interpolation_vertices_csdl[csdl.slice[wing_fuse_seam_local_index_list, :]]
            + corrected_interpolation_target_displacements[
                csdl.slice[wing_fuse_seam_local_index_list, :]
            ]
        )
        _print_two_component_target_sdf_diagnostic_csdl(
            "wing/fuse",
            corrected_wing_fuse_seam_target_points,
            moved_wing_coefficients,
            moved_fuse_coefficients,
            wing_sdf_model,
            fuse_sdf_model,
        )
    if htail_fuse_seam_interpolation_local_indices.size > 0:
        htail_fuse_seam_local_index_list = htail_fuse_seam_interpolation_local_indices.tolist()
        corrected_htail_fuse_seam_target_points = (
            interpolation_vertices_csdl[csdl.slice[htail_fuse_seam_local_index_list, :]]
            + corrected_interpolation_target_displacements[
                csdl.slice[htail_fuse_seam_local_index_list, :]
            ]
        )
        _print_two_component_target_sdf_diagnostic_csdl(
            "htail/fuse",
            corrected_htail_fuse_seam_target_points,
            moved_htail_coefficients,
            moved_fuse_coefficients,
            htail_sdf_model,
            fuse_sdf_model,
        )

    rbf_weights, rbf_polynomial_coefficients = _fit_vector_rbf_csdl(
        interpolation_vertices_csdl,
        corrected_interpolation_target_displacements,
        fit_mode=rbf_fit_mode,
        kernel=rbf_kernel,
        kernel_scale=rbf_kernel_scale,
        regularization=rbf_regularization,
        polynomial_regularization=rbf_polynomial_regularization,
        sample_weights=training_sample_weights,
    )

    predicted_interpolation_displacements = _evaluate_vector_rbf_csdl(
        interpolation_vertices_csdl,
        interpolation_vertices_csdl,
        rbf_weights,
        rbf_polynomial_coefficients,
        kernel=rbf_kernel,
        kernel_scale=rbf_kernel_scale,
    )
    interpolation_target_error = (
        predicted_interpolation_displacements - corrected_interpolation_target_displacements
    )

    predicted_variable_displacements = _evaluate_vector_rbf_csdl(
        vertices_csdl,
        interpolation_vertices_csdl,
        rbf_weights,
        rbf_polynomial_coefficients,
        kernel=rbf_kernel,
        kernel_scale=rbf_kernel_scale,
    )
    predicted_variable_vertices = vertices_csdl + predicted_variable_displacements
    if half_mesh_symmetry_mode:
        predicted_variable_vertices = _enforce_symmetry_plane_csdl(
            predicted_variable_vertices,
            symmetry_plane_solve_local_indices,
            symmetry_axis=symmetry_axis,
        )

    moved_aft_fuse_symmetry_anchor_vertices = None
    if (
        aft_fuse_symmetry_anchor_hard_enforcement
        and aft_fuse_symmetry_anchor_indices.size > 0
    ):
        if aft_fuse_symmetry_anchor_local_indices.size != aft_fuse_symmetry_anchor_indices.size:
            raise ValueError(
                "Aft fuselage symmetry anchors must all be included in the interpolation set."
            )
        moved_aft_fuse_symmetry_anchor_vertices = (
            interpolation_vertices_csdl[
                csdl.slice[aft_fuse_symmetry_anchor_local_indices.tolist(), :]
            ]
            + interpolation_fuse_displacements[
                csdl.slice[aft_fuse_symmetry_anchor_local_indices.tolist(), :]
            ]
        )
        predicted_variable_vertices = predicted_variable_vertices.set(
            csdl.slice[aft_fuse_symmetry_anchor_solve_local_indices.tolist(), :],
            moved_aft_fuse_symmetry_anchor_vertices,
        )
    moved_aft_fuse_fixed_vertices = None
    if aft_fuse_fixed_solve_local_indices.size > 0:
        moved_aft_fuse_fixed_vertices = vertices_csdl[
            csdl.slice[aft_fuse_fixed_solve_local_indices.tolist(), :]
        ]
        predicted_variable_vertices = predicted_variable_vertices.set(
            csdl.slice[aft_fuse_fixed_solve_local_indices.tolist(), :],
            moved_aft_fuse_fixed_vertices,
        )
    moved_outboard_wing_parametric_fixed_vertices = None
    if outboard_wing_parametric_fixed_indices.size > 0:
        moved_outboard_wing_parametric_fixed_vertices = (
            wing_evaluation_outboard_fixed_operation.evaluate(
                moved_wing_coefficients,
                outboard_wing_parametric_fixed_parametric_coordinates,
            )
        )
        predicted_variable_vertices = predicted_variable_vertices.set(
            csdl.slice[outboard_wing_parametric_fixed_solve_local_indices.tolist(), :],
            moved_outboard_wing_parametric_fixed_vertices,
        )
    moved_outboard_htail_parametric_fixed_vertices = None
    if outboard_htail_parametric_fixed_indices.size > 0:
        moved_outboard_htail_parametric_fixed_vertices = (
            htail_evaluation_outboard_fixed_operation.evaluate(
                moved_htail_coefficients,
                outboard_htail_parametric_fixed_parametric_coordinates,
            )
        )
        predicted_variable_vertices = predicted_variable_vertices.set(
            csdl.slice[outboard_htail_parametric_fixed_solve_local_indices.tolist(), :],
            moved_outboard_htail_parametric_fixed_vertices,
        )

    moved_protected_wing_feature_vertices = None
    moved_protected_htail_feature_vertices = None
    if protected_wing_feature_indices.size > 0:
        moved_protected_wing_feature_vertices = wing_evaluation_feature_operation.evaluate(
            moved_wing_coefficients,
            protected_wing_feature_parametric_coordinates,
        )
        predicted_variable_vertices = predicted_variable_vertices.set(
            csdl.slice[protected_wing_feature_solve_local_indices.tolist(), :],
            moved_protected_wing_feature_vertices,
        )
    if protected_htail_feature_indices.size > 0:
        moved_protected_htail_feature_vertices = htail_evaluation_feature_operation.evaluate(
            moved_htail_coefficients,
            protected_htail_feature_parametric_coordinates,
        )
        predicted_variable_vertices = predicted_variable_vertices.set(
            csdl.slice[protected_htail_feature_solve_local_indices.tolist(), :],
            moved_protected_htail_feature_vertices,
        )

    if exact_seam_solve_vertex_driving:
        # The exact seam rows constrain the RBF locally, but sparse off-seam
        # samples can still leave nearby solve vertices behind the moved seam.
        # This fixed-support blend propagates the relaxed seam displacement to
        # the local solve-vertex band without adding another nonlinear solve.
        predicted_variable_vertices = _blend_vertices_toward_nearest_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_wing_fuse_seam_displacements,
            wing_fuse_solve_driven_local_indices,
            wing_fuse_solve_nearest_seam_local_indices,
            wing_fuse_solve_driven_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_nearest_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_htail_fuse_seam_displacements,
            htail_fuse_solve_driven_local_indices,
            htail_fuse_solve_nearest_seam_local_indices,
            htail_fuse_solve_driven_support,
        )
        if half_mesh_symmetry_mode:
            predicted_variable_vertices = _enforce_symmetry_plane_csdl(
                predicted_variable_vertices,
                symmetry_plane_solve_local_indices,
                symmetry_axis=symmetry_axis,
            )
        predicted_variable_displacements = predicted_variable_vertices - vertices_csdl

    if exact_seam_endpoint_feature_driving:
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_wing_fuse_seam_displacements,
            wing_fuse_le_feature_solve_local_indices,
            wing_fuse_le_seam_row,
            wing_fuse_le_feature_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_wing_fuse_seam_displacements,
            wing_fuse_te_feature_solve_local_indices,
            wing_fuse_te_seam_row,
            wing_fuse_te_feature_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_htail_fuse_seam_displacements,
            htail_fuse_le_feature_solve_local_indices,
            htail_fuse_le_seam_row,
            htail_fuse_le_feature_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_htail_fuse_seam_displacements,
            htail_fuse_te_feature_solve_local_indices,
            htail_fuse_te_seam_row,
            htail_fuse_te_feature_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_wing_fuse_seam_displacements,
            wing_fuse_le_halo_solve_local_indices,
            wing_fuse_le_seam_row,
            wing_fuse_le_halo_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_wing_fuse_seam_displacements,
            wing_fuse_te_halo_solve_local_indices,
            wing_fuse_te_seam_row,
            wing_fuse_te_halo_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_htail_fuse_seam_displacements,
            htail_fuse_le_halo_solve_local_indices,
            htail_fuse_le_seam_row,
            htail_fuse_le_halo_solve_support,
        )
        predicted_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
            vertices_csdl,
            predicted_variable_vertices,
            relaxed_htail_fuse_seam_displacements,
            htail_fuse_te_halo_solve_local_indices,
            htail_fuse_te_seam_row,
            htail_fuse_te_halo_solve_support,
        )
        if half_mesh_symmetry_mode:
            predicted_variable_vertices = _enforce_symmetry_plane_csdl(
                predicted_variable_vertices,
                symmetry_plane_solve_local_indices,
                symmetry_axis=symmetry_axis,
            )
        predicted_variable_displacements = predicted_variable_vertices - vertices_csdl

    # predicted_wing_fuse_seam_points = None
    # if wing_fuse_seam_interpolation_local_indices.size > 0:
    #     wing_fuse_seam_global_indices = interpolation_indices[
    #         wing_fuse_seam_interpolation_local_indices
    #     ]
    #     wing_fuse_seam_solve_local_indices = solve_vertex_local_indices[
    #         wing_fuse_seam_global_indices
    #     ]
    #     predicted_wing_fuse_seam_points = predicted_variable_vertices[
    #         csdl.slice[wing_fuse_seam_solve_local_indices.tolist(), :]
    #     ]
    # predicted_htail_fuse_seam_points = None
    # if htail_fuse_seam_interpolation_local_indices.size > 0:
    #     htail_fuse_seam_global_indices = interpolation_indices[
    #         htail_fuse_seam_interpolation_local_indices
    #     ]
    #     htail_fuse_seam_solve_local_indices = solve_vertex_local_indices[
    #         htail_fuse_seam_global_indices
    #     ]
    #     predicted_htail_fuse_seam_points = predicted_variable_vertices[
    #         csdl.slice[htail_fuse_seam_solve_local_indices.tolist(), :]
    #     ]

    step8_geometry_function_set = None
    if plot_seam_relaxation_debug or soft_bracket_debug_enabled or (
        plot_step8_prediction and plot_step8_deformed_geometry_overlay
    ):
        step8_geometry_function_set = _function_set_from_stacked_coefficients(
            embraer_175,
            moved_total_coefficients.value,
            name="Step 8 deformed B-spline geometry",
        )

    if (
        soft_bracket_debug_enabled
        and step8_geometry_function_set is not None
        and len(soft_bracket_debug_traces) > 0
    ):
        soft_bracket_mesh_vertices = None
        if plot_soft_bracket_mesh_overlay:
            if half_mesh_symmetry_mode:
                soft_bracket_mesh_vertices = np.asarray(
                    _mirror_solve_vertices_to_full_csdl(
                        predicted_variable_vertices,
                        full_source_local_indices,
                        full_mirror_signs,
                    ).value,
                    dtype=float,
                )
            else:
                soft_bracket_mesh_vertices = np.asarray(
                    predicted_variable_vertices.value,
                    dtype=float,
                )
        if plot_soft_bracket_debug:
            _plot_soft_bracket_debug(
                step8_geometry_function_set,
                soft_bracket_debug_traces,
                mesh_vertices=soft_bracket_mesh_vertices,
                mesh_faces=mesh_faces if soft_bracket_mesh_vertices is not None else None,
                geometry_opacity=plot_step8_deformed_geometry_opacity,
                title="Soft bracket seam midpoint trajectories",
            )
        if animate_soft_bracket_debug:
            _save_soft_bracket_sdf_animation(
                step8_geometry_function_set,
                soft_bracket_debug_traces,
                gif_path=soft_bracket_debug_gif_path,
                mesh_vertices=soft_bracket_mesh_vertices,
                mesh_faces=mesh_faces if soft_bracket_mesh_vertices is not None else None,
                geometry_opacity=plot_step8_deformed_geometry_opacity,
                fps=2.0,
                title="Soft bracket SDF midpoint convergence",
            )

    # if plot_seam_relaxation_debug and step8_geometry_function_set is not None:
        # relaxed_wing_fuse_points_value = (
        #     None
        #     if relaxed_wing_fuse_seam_points is None
        #     else np.asarray(relaxed_wing_fuse_seam_points.value, dtype=float)
        # )
        # corrected_wing_fuse_points_value = (
        #     None
        #     if corrected_wing_fuse_seam_target_points is None
        #     else np.asarray(corrected_wing_fuse_seam_target_points.value, dtype=float)
        # )
        # predicted_wing_fuse_points_value = (
        #     None
        #     if predicted_wing_fuse_seam_points is None
        #     else np.asarray(predicted_wing_fuse_seam_points.value, dtype=float)
        # )
        # wing_fuse_anchor_points_value = (
        #     None
        #     if moved_wing_fuse_anchor_points is None
        #     else np.asarray(moved_wing_fuse_anchor_points.value, dtype=float)
        # )
        # relaxed_htail_fuse_points_value = (
        #     None
        #     if relaxed_htail_fuse_seam_points is None
        #     else np.asarray(relaxed_htail_fuse_seam_points.value, dtype=float)
        # )
        # corrected_htail_fuse_points_value = (
        #     None
        #     if corrected_htail_fuse_seam_target_points is None
        #     else np.asarray(corrected_htail_fuse_seam_target_points.value, dtype=float)
        # )
        # predicted_htail_fuse_points_value = (
        #     None
        #     if predicted_htail_fuse_seam_points is None
        #     else np.asarray(predicted_htail_fuse_seam_points.value, dtype=float)
        # )
        # htail_fuse_anchor_points_value = (
        #     None
        #     if moved_htail_fuse_anchor_points is None
        #     else np.asarray(moved_htail_fuse_anchor_points.value, dtype=float)
        # )
        # if (
        #     relaxed_wing_fuse_points_value is not None
        #     and corrected_wing_fuse_points_value is not None
        # ):
        #     _print_point_delta_diagnostic(
        #         "wing/fuse corrected target minus relaxed target",
        #         corrected_wing_fuse_points_value,
        #         relaxed_wing_fuse_points_value,
        #     )
        # if (
        #     predicted_wing_fuse_points_value is not None
        #     and corrected_wing_fuse_points_value is not None
        # ):
        #     _print_point_delta_diagnostic(
        #         "wing/fuse step-8 RBF seam minus corrected target",
        #         predicted_wing_fuse_points_value,
        #         corrected_wing_fuse_points_value,
        #     )
        # if (
        #     relaxed_htail_fuse_points_value is not None
        #     and corrected_htail_fuse_points_value is not None
        # ):
        #     _print_point_delta_diagnostic(
        #         "htail/fuse corrected target minus relaxed target",
        #         corrected_htail_fuse_points_value,
        #         relaxed_htail_fuse_points_value,
        #     )
        # if (
        #     predicted_htail_fuse_points_value is not None
        #     and corrected_htail_fuse_points_value is not None
        # ):
        #     _print_point_delta_diagnostic(
        #         "htail/fuse step-8 RBF seam minus corrected target",
        #         predicted_htail_fuse_points_value,
        #         corrected_htail_fuse_points_value,
        #     )
        # _plot_seam_relaxation_debug(
        #     step8_geometry_function_set,
        #     relaxed_wing_fuse_points=relaxed_wing_fuse_points_value,
        #     corrected_wing_fuse_points=corrected_wing_fuse_points_value,
        #     predicted_wing_fuse_points=predicted_wing_fuse_points_value,
        #     wing_fuse_anchor_points=wing_fuse_anchor_points_value,
        #     relaxed_htail_fuse_points=relaxed_htail_fuse_points_value,
        #     corrected_htail_fuse_points=corrected_htail_fuse_points_value,
        #     predicted_htail_fuse_points=predicted_htail_fuse_points_value,
        #     htail_fuse_anchor_points=htail_fuse_anchor_points_value,
        #     geometry_opacity=plot_step8_deformed_geometry_opacity,
        # )

    if plot_step8_prediction:
        if half_mesh_symmetry_mode:
            predicted_plot_vertices = np.asarray(
                _mirror_solve_vertices_to_full_csdl(
                    predicted_variable_vertices,
                    full_source_local_indices,
                    full_mirror_signs,
                ).value,
                dtype=float,
            )
        else:
            predicted_plot_vertices = np.asarray(predicted_variable_vertices.value, dtype=float)
        if plot_step8_deformed_geometry_overlay:
            _plot_surface_debug_with_geometry_overlay(
                predicted_plot_vertices,
                mesh_faces=mesh_faces,
                interpolation_indices=interpolation_indices,
                seam_seed_indices=seam_seed_indices,
                protected_feature_indices=protected_feature_indices,
                mesh_label="Single-RBF predicted mesh",
                geometry_function_set=step8_geometry_function_set,
                geometry_opacity=plot_step8_deformed_geometry_opacity,
                mesh_opacity=plot_step8_mesh_opacity,
            )
        else:
            _plot_surface_debug(
                predicted_plot_vertices,
                mesh_faces=mesh_faces,
                interpolation_indices=interpolation_indices,
                seam_seed_indices=seam_seed_indices,
                protected_feature_indices=protected_feature_indices,
                mesh_label="Single-RBF predicted mesh",
            )
        # exit()

    if step9_pre_projection_laplacian_step > 0.0:
        laplacian_displacement = csdl.sparse.matmat(laplacian_matrix, predicted_variable_vertices)
        step9_smoothed_variable_vertices = (
            predicted_variable_vertices
            + step9_pre_projection_laplacian_step
            * _expand_vector_to_columns(unprotected_laplacian_scale, 3)
            * laplacian_displacement
        )
        if moved_protected_wing_feature_vertices is not None:
            step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
                csdl.slice[protected_wing_feature_solve_local_indices.tolist(), :],
                moved_protected_wing_feature_vertices,
            )
        if moved_protected_htail_feature_vertices is not None:
            step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
                csdl.slice[protected_htail_feature_solve_local_indices.tolist(), :],
                moved_protected_htail_feature_vertices,
            )
        if moved_aft_fuse_symmetry_anchor_vertices is not None:
            step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
                csdl.slice[aft_fuse_symmetry_anchor_solve_local_indices.tolist(), :],
                moved_aft_fuse_symmetry_anchor_vertices,
            )
        if moved_aft_fuse_fixed_vertices is not None:
            step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
                csdl.slice[aft_fuse_fixed_solve_local_indices.tolist(), :],
                moved_aft_fuse_fixed_vertices,
            )
        if moved_outboard_wing_parametric_fixed_vertices is not None:
            step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
                csdl.slice[outboard_wing_parametric_fixed_solve_local_indices.tolist(), :],
                moved_outboard_wing_parametric_fixed_vertices,
            )
        if moved_outboard_htail_parametric_fixed_vertices is not None:
            step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
                csdl.slice[outboard_htail_parametric_fixed_solve_local_indices.tolist(), :],
                moved_outboard_htail_parametric_fixed_vertices,
            )
        if exact_seam_endpoint_feature_driving:
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_le_feature_solve_local_indices,
                wing_fuse_le_seam_row,
                wing_fuse_le_feature_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_te_feature_solve_local_indices,
                wing_fuse_te_seam_row,
                wing_fuse_te_feature_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_le_feature_solve_local_indices,
                htail_fuse_le_seam_row,
                htail_fuse_le_feature_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_te_feature_solve_local_indices,
                htail_fuse_te_seam_row,
                htail_fuse_te_feature_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_le_halo_solve_local_indices,
                wing_fuse_le_seam_row,
                wing_fuse_le_halo_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_wing_fuse_seam_displacements,
                wing_fuse_te_halo_solve_local_indices,
                wing_fuse_te_seam_row,
                wing_fuse_te_halo_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_le_halo_solve_local_indices,
                htail_fuse_le_seam_row,
                htail_fuse_le_halo_solve_support,
            )
            step9_smoothed_variable_vertices = _blend_vertices_toward_endpoint_seam_displacement_csdl(
                vertices_csdl,
                step9_smoothed_variable_vertices,
                relaxed_htail_fuse_seam_displacements,
                htail_fuse_te_halo_solve_local_indices,
                htail_fuse_te_seam_row,
                htail_fuse_te_halo_solve_support,
            )
    else:
        step9_smoothed_variable_vertices = predicted_variable_vertices
    if half_mesh_symmetry_mode:
        step9_smoothed_variable_vertices = _enforce_symmetry_plane_csdl(
            step9_smoothed_variable_vertices,
            symmetry_plane_solve_local_indices,
            symmetry_axis=symmetry_axis,
        )
    if moved_aft_fuse_fixed_vertices is not None:
        step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
            csdl.slice[aft_fuse_fixed_solve_local_indices.tolist(), :],
            moved_aft_fuse_fixed_vertices,
        )
    if moved_outboard_wing_parametric_fixed_vertices is not None:
        step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
            csdl.slice[outboard_wing_parametric_fixed_solve_local_indices.tolist(), :],
            moved_outboard_wing_parametric_fixed_vertices,
        )
    if moved_outboard_htail_parametric_fixed_vertices is not None:
        step9_smoothed_variable_vertices = step9_smoothed_variable_vertices.set(
            csdl.slice[outboard_htail_parametric_fixed_solve_local_indices.tolist(), :],
            moved_outboard_htail_parametric_fixed_vertices,
        )

    if step9_projection_mode != "setup_patch_eligibility":
        raise ValueError(f"Unsupported step9_projection_mode: {step9_projection_mode}")

    projected_variable_vertices = step9_smoothed_variable_vertices

    for patch_id, solve_local_indices in step9_wing_patch_projection_solve_local_indices.items():
        projected_patch_vertices = _project_points_to_function_set_csdl(
            _gather_rows_csdl(step9_smoothed_variable_vertices, solve_local_indices),
            moved_wing_patch_coefficients[patch_id],
            wing_patch_projection_models[patch_id],
            wing_patch_evaluation_models[patch_id],
        )
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[solve_local_indices.tolist(), :],
            projected_patch_vertices,
        )

    if step9_fuse_projection_solve_local_indices.size > 0:
        projected_fuse_vertices = _project_points_to_function_set_csdl(
            _gather_rows_csdl(
                step9_smoothed_variable_vertices,
                step9_fuse_projection_solve_local_indices,
            ),
            moved_fuse_coefficients,
            fuse_projection_model,
            fuse_evaluation_model,
        )
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[step9_fuse_projection_solve_local_indices.tolist(), :],
            projected_fuse_vertices,
        )

    for patch_id, solve_local_indices in step9_htail_patch_projection_solve_local_indices.items():
        projected_patch_vertices = _project_points_to_function_set_csdl(
            _gather_rows_csdl(step9_smoothed_variable_vertices, solve_local_indices),
            moved_htail_patch_coefficients[patch_id],
            htail_patch_projection_models[patch_id],
            htail_patch_evaluation_models[patch_id],
        )
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[solve_local_indices.tolist(), :],
            projected_patch_vertices,
        )

    if step9_wing_fuse_seam_solve_local_indices.size > 0:
        wing_fuse_fuse_projected_vertices = _project_points_to_function_set_csdl(
            _gather_rows_csdl(
                step9_smoothed_variable_vertices,
                step9_wing_fuse_seam_solve_local_indices,
            ),
            moved_fuse_coefficients,
            fuse_projection_model,
            fuse_evaluation_model,
        )
        for patch_id, (
            seam_local_rows,
            seam_solve_local_indices,
        ) in step9_wing_fuse_wing_patch_groups.items():
            wing_projected_vertices = _project_points_to_function_set_csdl(
                _gather_rows_csdl(step9_smoothed_variable_vertices, seam_solve_local_indices),
                moved_wing_patch_coefficients[patch_id],
                wing_patch_projection_models[patch_id],
                wing_patch_evaluation_models[patch_id],
            )
            fuse_projected_vertices = _gather_rows_csdl(
                wing_fuse_fuse_projected_vertices,
                seam_local_rows,
            )
            projected_variable_vertices = projected_variable_vertices.set(
                csdl.slice[seam_solve_local_indices.tolist(), :],
                0.5 * (wing_projected_vertices + fuse_projected_vertices),
            )

    if step9_htail_fuse_seam_solve_local_indices.size > 0:
        htail_fuse_fuse_projected_vertices = _project_points_to_function_set_csdl(
            _gather_rows_csdl(
                step9_smoothed_variable_vertices,
                step9_htail_fuse_seam_solve_local_indices,
            ),
            moved_fuse_coefficients,
            fuse_projection_model,
            fuse_evaluation_model,
        )
        for patch_id, (
            seam_local_rows,
            seam_solve_local_indices,
        ) in step9_htail_fuse_htail_patch_groups.items():
            htail_projected_vertices = _project_points_to_function_set_csdl(
                _gather_rows_csdl(step9_smoothed_variable_vertices, seam_solve_local_indices),
                moved_htail_patch_coefficients[patch_id],
                htail_patch_projection_models[patch_id],
                htail_patch_evaluation_models[patch_id],
            )
            fuse_projected_vertices = _gather_rows_csdl(
                htail_fuse_fuse_projected_vertices,
                seam_local_rows,
            )
            projected_variable_vertices = projected_variable_vertices.set(
                csdl.slice[seam_solve_local_indices.tolist(), :],
                0.5 * (htail_projected_vertices + fuse_projected_vertices),
            )

    
    # step9_smooth_union_rho_schedule = _make_ks_rho_schedule(
    #     step9_smooth_union_ks_rho_start,
    #     step9_smooth_union_ks_rho_end,
    #     step9_smooth_union_num_iterations,
    # )
    # projected_variable_vertices = step9_smoothed_variable_vertices
    # first_union_sdf_csdl = None
    # for iteration_index, ks_rho in enumerate(step9_smooth_union_rho_schedule):
    #     active_projected_variable_vertices = _gather_rows_csdl(
    #         projected_variable_vertices,
    #         step9_projection_active_solve_local_indices,
    #     )
    #     wing_sdf_iteration_operation = bsm3.FunctionSetClosestDistanceOperation(model=wing_sdf_model)
    #     fuse_sdf_iteration_operation = bsm3.FunctionSetClosestDistanceOperation(model=fuse_sdf_model)
    #     htail_sdf_iteration_operation = bsm3.FunctionSetClosestDistanceOperation(model=htail_sdf_model)

    #     sdf_to_wing = wing_sdf_iteration_operation.evaluate(
    #         moved_wing_coefficients,
    #         active_projected_variable_vertices,
    #     )
    #     sdf_to_fuse = fuse_sdf_iteration_operation.evaluate(
    #         moved_fuse_coefficients,
    #         active_projected_variable_vertices,
    #     )
    #     sdf_to_htail = htail_sdf_iteration_operation.evaluate(
    #         moved_htail_coefficients,
    #         active_projected_variable_vertices,
    #     )

    #     wing_sdf_gradient = _signed_distance_gradient_from_ad_csdl(
    #         sdf_to_wing,
    #         active_projected_variable_vertices,
    #     )
    #     fuse_sdf_gradient = _signed_distance_gradient_from_ad_csdl(
    #         sdf_to_fuse,
    #         active_projected_variable_vertices,
    #     )
    #     htail_sdf_gradient = _signed_distance_gradient_from_ad_csdl(
    #         sdf_to_htail,
    #         active_projected_variable_vertices,
    #     )
    #     union_sdf, wing_union_weight, fuse_union_weight, htail_union_weight = (
    #         _smooth_minimum_signed_distance_three_components_csdl(
    #             sdf_to_wing,
    #             sdf_to_fuse,
    #             sdf_to_htail,
    #             ks_rho=ks_rho,
    #         )
    #     )
    #     if iteration_index == 0:
    #         first_union_sdf_csdl = union_sdf
    #     union_sdf_gradient = (
    #         _expand_vector_to_columns(csdl.reshape(wing_union_weight, (-1,)), 3) * wing_sdf_gradient
    #         + _expand_vector_to_columns(csdl.reshape(fuse_union_weight, (-1,)), 3) * fuse_sdf_gradient
    #         + _expand_vector_to_columns(csdl.reshape(htail_union_weight, (-1,)), 3) * htail_sdf_gradient
    #     )
    #     union_gradient_norm_squared = csdl.sum(union_sdf_gradient**2, axes=(1,)) + 1e-12
    #     updated_active_projected_vertices = active_projected_variable_vertices - (
    #         _expand_vector_to_columns(union_sdf / union_gradient_norm_squared, 3)
    #         * union_sdf_gradient
    #     )
    #     projected_variable_vertices = projected_variable_vertices.set(
    #         csdl.slice[step9_projection_active_solve_local_indices.tolist(), :],
    #         updated_active_projected_vertices,
    #     )
    #     if half_mesh_symmetry_mode:
    #         projected_variable_vertices = _enforce_symmetry_plane_csdl(
    #             projected_variable_vertices,
    #             symmetry_plane_solve_local_indices,
    #             symmetry_axis=symmetry_axis,
    #         )
    #     if moved_aft_fuse_fixed_vertices is not None:
    #         projected_variable_vertices = projected_variable_vertices.set(
    #             csdl.slice[aft_fuse_fixed_solve_local_indices.tolist(), :],
    #             moved_aft_fuse_fixed_vertices,
    #         )
    #     if moved_outboard_wing_parametric_fixed_vertices is not None:
    #         projected_variable_vertices = projected_variable_vertices.set(
    #             csdl.slice[outboard_wing_parametric_fixed_solve_local_indices.tolist(), :],
    #             moved_outboard_wing_parametric_fixed_vertices,
    #         )
    #     if moved_outboard_htail_parametric_fixed_vertices is not None:
    #         projected_variable_vertices = projected_variable_vertices.set(
    #             csdl.slice[outboard_htail_parametric_fixed_solve_local_indices.tolist(), :],
    #             moved_outboard_htail_parametric_fixed_vertices,
    #         )
    #     union_sdf_value = union_sdf.value
    #     print("max union sdf after iteration", iteration_index, ":", np.max(union_sdf_value))
    #     print("mean union sdf after iteration", iteration_index, ":", np.mean(union_sdf_value))

    if moved_protected_wing_feature_vertices is not None:
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[protected_wing_feature_solve_local_indices.tolist(), :],
            moved_protected_wing_feature_vertices,
        )
    if moved_protected_htail_feature_vertices is not None:
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[protected_htail_feature_solve_local_indices.tolist(), :],
            moved_protected_htail_feature_vertices,
        )
    if moved_aft_fuse_symmetry_anchor_vertices is not None:
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[aft_fuse_symmetry_anchor_solve_local_indices.tolist(), :],
            moved_aft_fuse_symmetry_anchor_vertices,
        )
    if moved_aft_fuse_fixed_vertices is not None:
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[aft_fuse_fixed_solve_local_indices.tolist(), :],
            moved_aft_fuse_fixed_vertices,
        )
    if moved_outboard_wing_parametric_fixed_vertices is not None:
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[outboard_wing_parametric_fixed_solve_local_indices.tolist(), :],
            moved_outboard_wing_parametric_fixed_vertices,
        )
    if moved_outboard_htail_parametric_fixed_vertices is not None:
        projected_variable_vertices = projected_variable_vertices.set(
            csdl.slice[outboard_htail_parametric_fixed_solve_local_indices.tolist(), :],
            moved_outboard_htail_parametric_fixed_vertices,
        )
    if half_mesh_symmetry_mode:
        projected_variable_vertices = _enforce_symmetry_plane_csdl(
            projected_variable_vertices,
            symmetry_plane_solve_local_indices,
            symmetry_axis=symmetry_axis,
        )
        projected_variable_vertices = _mirror_solve_vertices_to_full_csdl(
            projected_variable_vertices,
            full_source_local_indices,
            full_mirror_signs,
        )
    projected_variable_vertices.add_name("surface_mesh_vertices")

    if plot:
        _plot_surface_debug(
            projected_variable_vertices.value,
            mesh_faces=mesh_faces,
            interpolation_indices=interpolation_indices,
            seam_seed_indices=seam_seed_indices,
            protected_feature_indices=protected_feature_indices,
            mesh_label="Updated mesh",
            screenshot_path=(
                FINAL_MESH_SCREENSHOT_PATH
                if SAVE_FINAL_MESH_SCREENSHOT
                else None
            ),
            screenshot_window_size=FINAL_MESH_SCREENSHOT_WINDOW_SIZE,
        )
    
    objective = csdl.average(projected_variable_vertices)**2
    objective.add_name("objective")
    objective.set_as_objective()

    sim = csdl.experimental.PySimulator(
        recorder=recorder,
    )   
    import time
    start_time = time.time()
    sim.check_optimization_derivatives()
    stop_time = time.time()
    print("Time for derivative check:", stop_time - start_time)
    exit("hi")

   