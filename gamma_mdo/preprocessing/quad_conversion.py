"""Triangle-to-quad conversion routines and quality gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh_io import MeshData, _make_mesh_data


@dataclass(frozen=True)
class QuadQualityGates:
    """Quality thresholds used when merging triangle pairs into quads.

    Attributes
    ----------
    max_normal_deviation_deg : float
        Maximum angle between adjacent triangle normals.
    min_angle_deg, max_angle_deg : float
        Allowed interior-angle range for the resulting quad.
    max_planarity_ratio : float
        Maximum out-of-plane distance normalized by characteristic edge length.
    max_aspect_ratio : float
        Maximum ratio between longest and shortest quad edges.
    min_scaled_jacobian : float
        Minimum corner scaled-Jacobian value.
    max_quality_score : float
        Maximum aggregate quality penalty. Lower scores are better.
    min_flip_triangle_angle_deg : float
        Minimum triangle angle allowed after a pre-merge edge flip.
    max_flip_triangle_aspect_ratio : float
        Maximum triangle aspect ratio allowed after an edge flip.
    max_flip_normal_deviation_deg : float
        Maximum normal deviation between flipped triangle pairs.
    max_flip_min_angle_degradation_deg : float
        Allowed degradation in the local minimum triangle angle during flips.
    max_edge_flip_passes : int
        Maximum passes through the candidate edge-flip set.
    max_edge_flips : int
        Maximum number of accepted pre-merge edge flips.
    """

    max_normal_deviation_deg: float = 20.0
    min_angle_deg: float = 35.0
    max_angle_deg: float = 145.0
    max_planarity_ratio: float = 0.03
    max_aspect_ratio: float = 4.0
    min_scaled_jacobian: float = 0.25
    max_quality_score: float = 0.22
    min_flip_triangle_angle_deg: float = 15.0
    max_flip_triangle_aspect_ratio: float = 5.0
    max_flip_normal_deviation_deg: float = 25.0
    max_flip_min_angle_degradation_deg: float = 1.0
    max_edge_flip_passes: int = 2
    max_edge_flips: int = 256


_QUAD_QUALITY_GATE_PRESETS = {
    "conservative": QuadQualityGates(),
    "balanced": QuadQualityGates(
        max_normal_deviation_deg=25.0,
        min_angle_deg=30.0,
        max_angle_deg=150.0,
        max_planarity_ratio=0.05,
        max_aspect_ratio=5.0,
        min_scaled_jacobian=0.18,
        max_quality_score=0.30,
        min_flip_triangle_angle_deg=12.0,
        max_flip_triangle_aspect_ratio=6.0,
        max_flip_normal_deviation_deg=30.0,
    ),
    "aggressive": QuadQualityGates(
        max_normal_deviation_deg=30.0,
        min_angle_deg=25.0,
        max_angle_deg=155.0,
        max_planarity_ratio=0.08,
        max_aspect_ratio=6.5,
        min_scaled_jacobian=0.12,
        max_quality_score=0.40,
        min_flip_triangle_angle_deg=10.0,
        max_flip_triangle_aspect_ratio=7.5,
        max_flip_normal_deviation_deg=35.0,
        max_flip_min_angle_degradation_deg=2.0,
        max_edge_flip_passes=3,
        max_edge_flips=512,
    ),
}


def quad_quality_gates(preset: str = "conservative", **overrides) -> QuadQualityGates:
    """Return quad conversion quality gates.

    Parameters
    ----------
    preset : {"conservative", "balanced", "aggressive"}, optional
        Named baseline threshold set.
    **overrides
        Field-level threshold overrides passed to :class:`QuadQualityGates`.

    Returns
    -------
    QuadQualityGates
        Immutable quality gate configuration.

    Notes
    -----
    ``num_quads`` in ``create_symmetric_mesh`` is an upper bound on selected
    candidates. Loosening these gates increases the candidate pool; tightening
    them decreases it.
    """
    try:
        gates = _QUAD_QUALITY_GATE_PRESETS[preset.lower()]
    except KeyError as exc:
        presets = ", ".join(sorted(_QUAD_QUALITY_GATE_PRESETS))
        raise ValueError(f"Unknown quad quality gate preset {preset!r}. Available presets: {presets}.") from exc

    if not overrides:
        return gates

    valid_fields = set(QuadQualityGates.__dataclass_fields__)
    unknown_fields = sorted(set(overrides) - valid_fields)
    if unknown_fields:
        valid = ", ".join(sorted(valid_fields))
        unknown = ", ".join(unknown_fields)
        raise ValueError(f"Unknown quad quality gate field(s): {unknown}. Valid fields: {valid}.")
    return QuadQualityGates(**{**gates.__dict__, **overrides})


def _convert_triangles_to_quads(
    mesh: MeshData,
    *,
    max_num_quads: int | None,
    quality_gates: QuadQualityGates,
) -> MeshData:
    """Merge eligible triangle pairs into quad cells without moving vertices."""
    triangles = np.asarray(mesh.cell_blocks.get("triangle", np.empty((0, 3), dtype=np.int64)), dtype=np.int64)
    if triangles.size == 0 or max_num_quads == 0:
        return mesh

    flipped_triangles, accepted_flip_count = _apply_premerge_edge_flips(
        mesh.vertices,
        triangles,
        quality_gates=quality_gates,
    )
    candidates = _triangle_pair_quad_candidates(
        mesh.vertices,
        flipped_triangles,
        quality_gates=quality_gates,
    )
    selected_candidates, matching_method = _select_quad_merge_candidates(
        candidates,
        max_num_quads=max_num_quads,
    )
    used_triangles = {
        triangle_index
        for _, triangle_a, triangle_b, _ in selected_candidates
        for triangle_index in (triangle_a, triangle_b)
    }
    quads = [quad for _, _, _, quad in selected_candidates]

    leftover_triangles = np.asarray(
        [
            triangle
            for index, triangle in enumerate(flipped_triangles)
            if index not in used_triangles
        ],
        dtype=np.int64,
    ).reshape((-1, 3))
    cell_blocks = {
        cell_type: np.asarray(cells, dtype=np.int64)
        for cell_type, cells in mesh.cell_blocks.items()
        if cell_type != "triangle"
    }
    if leftover_triangles.size > 0:
        cell_blocks["triangle"] = leftover_triangles
    if quads:
        existing_quads = np.asarray(cell_blocks.get("quad", np.empty((0, 4), dtype=np.int64)), dtype=np.int64)
        cell_blocks["quad"] = np.vstack([existing_quads, np.asarray(quads, dtype=np.int64)])

    metadata = {
        **(mesh.metadata or {}),
        "quad_dominant": True,
        "num_quads_created": len(quads),
        "num_triangles_leftover": int(leftover_triangles.shape[0]),
        "premerge_edge_flips": accepted_flip_count,
        "quad_matching_method": matching_method,
        "quad_candidate_count": len(candidates),
        "quad_quality_gates": quality_gates.__dict__,
    }
    return _make_mesh_data(
        vertices=mesh.vertices,
        cell_blocks=cell_blocks,
        node_ids=mesh.node_ids,
        element_ids=None,
        element_tags=None,
        metadata=metadata,
    )


def _triangle_pair_quad_candidates(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    quality_gates: QuadQualityGates,
) -> list[tuple[float, int, int, np.ndarray]]:
    """Build quality-scored quad candidates from adjacent triangle pairs."""
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_index, triangle in enumerate(triangles):
        for local_index in range(3):
            edge = tuple(sorted((int(triangle[local_index]), int(triangle[(local_index + 1) % 3]))))
            edge_to_triangles.setdefault(edge, []).append(triangle_index)

    normals = _triangle_unit_normals(vertices, triangles)
    candidates = []
    for adjacent_triangles in edge_to_triangles.values():
        if len(adjacent_triangles) != 2:
            continue
        triangle_a, triangle_b = adjacent_triangles
        candidate = _evaluate_triangle_pair_quad_candidate(
            vertices,
            triangles[triangle_a],
            triangles[triangle_b],
            normals[triangle_a],
            normals[triangle_b],
            quality_gates=quality_gates,
        )
        if candidate is None:
            continue
        score, quad = candidate
        candidates.append((score, triangle_a, triangle_b, quad))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _select_quad_merge_candidates(
    candidates: list[tuple[float, int, int, np.ndarray]],
    *,
    max_num_quads: int | None,
) -> tuple[list[tuple[float, int, int, np.ndarray]], str]:
    if not candidates:
        return [], "none"

    try:
        import networkx as nx
    except ImportError:
        selected = _select_greedy_quad_candidates(candidates)
        if max_num_quads is not None:
            selected = selected[:max_num_quads]
        return selected, "greedy"

    scores = np.asarray([candidate[0] for candidate in candidates], dtype=float)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_span = max(score_max - score_min, 1e-12)

    graph = nx.Graph()
    for candidate_index, (score, triangle_a, triangle_b, _) in enumerate(candidates):
        normalized_quality = (score_max - score) / score_span
        graph.add_edge(
            int(triangle_a),
            int(triangle_b),
            weight=1.0 + normalized_quality,
            candidate_index=candidate_index,
        )

    selected_indices = []
    for component_nodes in nx.connected_components(graph):
        component_graph = graph.subgraph(component_nodes)
        matching = nx.algorithms.matching.max_weight_matching(
            component_graph,
            maxcardinality=True,
            weight="weight",
        )
        for triangle_a, triangle_b in matching:
            selected_indices.append(graph.edges[triangle_a, triangle_b]["candidate_index"])

    selected = [candidates[index] for index in selected_indices]
    selected.sort(key=lambda item: item[0])
    if max_num_quads is not None:
        selected = selected[:max_num_quads]
    return selected, "max_weight_matching"


def _select_greedy_quad_candidates(
    candidates: list[tuple[float, int, int, np.ndarray]],
) -> list[tuple[float, int, int, np.ndarray]]:
    used_triangles: set[int] = set()
    selected = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        _, triangle_a, triangle_b, _ = candidate
        if triangle_a in used_triangles or triangle_b in used_triangles:
            continue
        selected.append(candidate)
        used_triangles.add(triangle_a)
        used_triangles.add(triangle_b)
    return selected


def _apply_premerge_edge_flips(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    quality_gates: QuadQualityGates,
) -> tuple[np.ndarray, int]:
    """Conservatively flip triangle edges when doing so improves quadability."""
    updated_triangles = np.asarray(triangles, dtype=np.int64).copy()
    if updated_triangles.shape[0] < 2:
        return updated_triangles, 0

    accepted_flip_count = 0
    for _ in range(quality_gates.max_edge_flip_passes):
        edge_to_triangles = _triangle_edge_to_indices(updated_triangles)
        made_change = False
        for edge_key, incident_triangles in edge_to_triangles.items():
            if len(incident_triangles) != 2:
                continue
            triangle_a_index, triangle_b_index = incident_triangles
            current_pair = (
                updated_triangles[triangle_a_index].copy(),
                updated_triangles[triangle_b_index].copy(),
            )
            current_normals = _triangle_unit_normals(vertices, np.vstack(current_pair))
            current_candidate = _evaluate_triangle_pair_quad_candidate(
                vertices,
                current_pair[0],
                current_pair[1],
                current_normals[0],
                current_normals[1],
                quality_gates=quality_gates,
            )
            if current_candidate is not None:
                continue

            proposed_pair = _proposed_flipped_triangle_pair(
                vertices,
                current_pair[0],
                current_pair[1],
                edge_key,
                edge_to_triangles,
            )
            if proposed_pair is None:
                continue
            if not _triangle_pair_flip_is_reasonable(
                vertices,
                current_pair,
                proposed_pair,
                quality_gates=quality_gates,
            ):
                continue

            patch_indices = _local_triangle_patch_indices(
                updated_triangles,
                edge_to_triangles,
                triangle_a_index,
                triangle_b_index,
            )
            patch_lookup = {
                triangle_index: local_index
                for local_index, triangle_index in enumerate(patch_indices)
            }
            current_patch = updated_triangles[patch_indices].copy()
            proposed_patch = current_patch.copy()
            proposed_patch[patch_lookup[triangle_a_index]] = proposed_pair[0]
            proposed_patch[patch_lookup[triangle_b_index]] = proposed_pair[1]

            current_objective = _patch_quadification_objective(
                vertices,
                current_patch,
                quality_gates=quality_gates,
            )
            proposed_objective = _patch_quadification_objective(
                vertices,
                proposed_patch,
                quality_gates=quality_gates,
            )
            if not _quadification_objective_is_better(current_objective, proposed_objective):
                continue

            updated_triangles[triangle_a_index] = proposed_pair[0]
            updated_triangles[triangle_b_index] = proposed_pair[1]
            accepted_flip_count += 1
            made_change = True
            break
        if not made_change or accepted_flip_count >= quality_gates.max_edge_flips:
            break
    return updated_triangles, accepted_flip_count


def _triangle_edge_to_indices(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_index, triangle in enumerate(np.asarray(triangles, dtype=np.int64)):
        for edge in _cycle_edges(triangle):
            edge_to_triangles.setdefault(tuple(sorted(edge)), []).append(triangle_index)
    return edge_to_triangles


def _proposed_flipped_triangle_pair(
    vertices: np.ndarray,
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    shared_edge: tuple[int, int],
    edge_to_triangles: dict[tuple[int, int], list[int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    normals = _triangle_unit_normals(vertices, np.vstack([triangle_a, triangle_b]))
    quad = _quad_from_triangle_pair(vertices, triangle_a, triangle_b, normals[0], normals[1])
    if quad is None:
        return None

    proposed_diagonal = tuple(sorted((int(quad[0]), int(quad[2]))))
    if proposed_diagonal == tuple(sorted(shared_edge)) or proposed_diagonal in edge_to_triangles:
        return None

    reference_normal = normals[0] + normals[1]
    if np.linalg.norm(reference_normal) <= 1e-14:
        return None

    proposed = (
        np.asarray([quad[0], quad[1], quad[2]], dtype=np.int64),
        np.asarray([quad[0], quad[2], quad[3]], dtype=np.int64),
    )
    oriented = []
    for triangle in proposed:
        area_vector = np.cross(
            vertices[triangle[1]] - vertices[triangle[0]],
            vertices[triangle[2]] - vertices[triangle[0]],
        )
        if np.dot(area_vector, reference_normal) < 0.0:
            triangle = triangle[[0, 2, 1]]
        if _triangle_area_magnitude(vertices, triangle) <= 1e-14:
            return None
        oriented.append(triangle)
    return oriented[0], oriented[1]


def _triangle_pair_flip_is_reasonable(
    vertices: np.ndarray,
    current_pair: tuple[np.ndarray, np.ndarray],
    proposed_pair: tuple[np.ndarray, np.ndarray],
    *,
    quality_gates: QuadQualityGates,
) -> bool:
    proposed_min_angles = np.asarray(
        [_triangle_min_angle(vertices, triangle) for triangle in proposed_pair],
        dtype=float,
    )
    if float(np.min(proposed_min_angles)) < quality_gates.min_flip_triangle_angle_deg:
        return False

    proposed_aspect_ratios = np.asarray(
        [_triangle_aspect_ratio(vertices, triangle) for triangle in proposed_pair],
        dtype=float,
    )
    if float(np.max(proposed_aspect_ratios)) > quality_gates.max_flip_triangle_aspect_ratio:
        return False

    current_min_angles = np.asarray(
        [_triangle_min_angle(vertices, triangle) for triangle in current_pair],
        dtype=float,
    )
    if (
        float(np.min(proposed_min_angles))
        + quality_gates.max_flip_min_angle_degradation_deg
        < float(np.min(current_min_angles))
    ):
        return False

    proposed_normals = _triangle_unit_normals(vertices, np.vstack(proposed_pair))
    if float(np.dot(proposed_normals[0], proposed_normals[1])) < np.cos(
        np.radians(quality_gates.max_flip_normal_deviation_deg)
    ):
        return False
    return True


def _local_triangle_patch_indices(
    triangles: np.ndarray,
    edge_to_triangles: dict[tuple[int, int], list[int]],
    triangle_a_index: int,
    triangle_b_index: int,
) -> list[int]:
    patch_indices = {triangle_a_index, triangle_b_index}
    for triangle_index in (triangle_a_index, triangle_b_index):
        for edge in _cycle_edges(triangles[triangle_index]):
            patch_indices.update(edge_to_triangles.get(tuple(sorted(edge)), []))
    return sorted(patch_indices)


def _patch_quadification_objective(
    vertices: np.ndarray,
    patch_triangles: np.ndarray,
    *,
    quality_gates: QuadQualityGates,
) -> tuple[int, float]:
    candidates = _triangle_pair_quad_candidates(
        vertices,
        patch_triangles,
        quality_gates=quality_gates,
    )
    selected, _ = _select_quad_merge_candidates(candidates, max_num_quads=None)
    if not selected:
        return 0, np.inf
    return len(selected), float(np.sum([candidate[0] for candidate in selected]))


def _quadification_objective_is_better(
    current: tuple[int, float],
    proposed: tuple[int, float],
) -> bool:
    if proposed[0] != current[0]:
        return proposed[0] > current[0]
    return proposed[1] < current[1] - 1e-12


def _evaluate_triangle_pair_quad_candidate(
    vertices: np.ndarray,
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
    *,
    quality_gates: QuadQualityGates,
) -> tuple[float, np.ndarray] | None:
    normal_alignment = float(np.dot(normal_a, normal_b))
    if normal_alignment < np.cos(np.radians(quality_gates.max_normal_deviation_deg)):
        return None

    quad = _quad_from_triangle_pair(vertices, triangle_a, triangle_b, normal_a, normal_b)
    if quad is None:
        return None

    interior_angles = _quad_interior_angles(vertices, quad)
    if (
        float(np.min(interior_angles)) < quality_gates.min_angle_deg
        or float(np.max(interior_angles)) > quality_gates.max_angle_deg
    ):
        return None
    if _quad_planarity_ratio(vertices, quad) > quality_gates.max_planarity_ratio:
        return None
    if _quad_aspect_ratio(vertices, quad) > quality_gates.max_aspect_ratio:
        return None
    if _quad_min_scaled_jacobian(vertices, quad) < quality_gates.min_scaled_jacobian:
        return None

    quad_normal = _unit_normal(_newell_polygon_normal(vertices[quad]))
    if np.linalg.norm(quad_normal) <= 1e-14:
        return None
    if np.dot(quad_normal, normal_a) <= 0.0 or np.dot(quad_normal, normal_b) <= 0.0:
        return None

    score = _quad_quality_score(vertices, quad, normal_a, normal_b)
    if score > quality_gates.max_quality_score:
        return None
    return score, quad


def _triangle_unit_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangle_points = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    normals = np.cross(
        triangle_points[:, 1, :] - triangle_points[:, 0, :],
        triangle_points[:, 2, :] - triangle_points[:, 0, :],
    )
    normal_norms = np.linalg.norm(normals, axis=1)
    unit_normals = np.zeros_like(normals)
    nonzero = normal_norms > 1e-14
    unit_normals[nonzero] = normals[nonzero] / normal_norms[nonzero, None]
    return unit_normals


def _triangle_area_magnitude(vertices: np.ndarray, triangle: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float)[np.asarray(triangle, dtype=np.int64)]
    return float(
        0.5
        * np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
    )


def _triangle_interior_angles(vertices: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)[np.asarray(triangle, dtype=np.int64)]
    angles = np.zeros((3,), dtype=float)
    for local_index in range(3):
        current_point = points[local_index]
        previous_edge = points[(local_index - 1) % 3] - current_point
        next_edge = points[(local_index + 1) % 3] - current_point
        previous_norm = np.linalg.norm(previous_edge)
        next_norm = np.linalg.norm(next_edge)
        if previous_norm <= 1e-14 or next_norm <= 1e-14:
            angles[local_index] = 0.0
            continue
        cosine = np.clip(
            np.dot(previous_edge, next_edge) / (previous_norm * next_norm),
            -1.0,
            1.0,
        )
        angles[local_index] = np.degrees(np.arccos(cosine))
    return angles


def _triangle_min_angle(vertices: np.ndarray, triangle: np.ndarray) -> float:
    return float(np.min(_triangle_interior_angles(vertices, triangle)))


def _triangle_aspect_ratio(vertices: np.ndarray, triangle: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float)[np.asarray(triangle, dtype=np.int64)]
    edge_lengths = np.asarray(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[0] - points[2]),
        ],
        dtype=float,
    )
    return float(np.max(edge_lengths) / max(float(np.min(edge_lengths)), 1e-14))


def _unit_normal(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-14:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _cycle_edges(cell: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(cell[index]), int(cell[(index + 1) % cell.shape[0]]))
        for index in range(cell.shape[0])
    ]


def _newell_polygon_normal(points: np.ndarray) -> np.ndarray:
    normal = np.zeros((3,), dtype=float)
    for index in range(points.shape[0]):
        normal += np.cross(points[index], points[(index + 1) % points.shape[0]])
    return normal


def _shared_edge(triangle_a: np.ndarray, triangle_b: np.ndarray) -> tuple[int, int] | None:
    shared_vertices = sorted(set(map(int, triangle_a)) & set(map(int, triangle_b)))
    if len(shared_vertices) != 2:
        return None
    return shared_vertices[0], shared_vertices[1]


def _quad_from_triangle_pair(
    vertices: np.ndarray,
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
) -> np.ndarray | None:
    shared_edge = _shared_edge(triangle_a, triangle_b)
    if shared_edge is None:
        return None

    boundary_graph: dict[int, list[int]] = {}
    for triangle in (triangle_a, triangle_b):
        for edge in _cycle_edges(np.asarray(triangle, dtype=np.int64)):
            if tuple(sorted(edge)) == shared_edge:
                continue
            boundary_graph.setdefault(edge[0], []).append(edge[1])
            boundary_graph.setdefault(edge[1], []).append(edge[0])

    if len(boundary_graph) != 4 or any(len(neighbors) != 2 for neighbors in boundary_graph.values()):
        return None

    opposite_vertices = [int(vertex) for vertex in triangle_a if int(vertex) not in set(shared_edge)]
    if len(opposite_vertices) != 1:
        return None

    ordered_vertices = [opposite_vertices[0]]
    previous_vertex = None
    current_vertex = ordered_vertices[0]
    while len(ordered_vertices) < 4:
        neighbors = boundary_graph[current_vertex]
        next_vertex = neighbors[0]
        if previous_vertex is not None and next_vertex == previous_vertex:
            next_vertex = neighbors[1]
        ordered_vertices.append(next_vertex)
        previous_vertex = current_vertex
        current_vertex = next_vertex

    if len(set(ordered_vertices)) != 4:
        return None

    quad = np.asarray(ordered_vertices, dtype=np.int64)
    quad_normal = _newell_polygon_normal(vertices[quad])
    if np.dot(quad_normal, normal_a + normal_b) < 0.0:
        quad = quad[::-1]
    return quad


def _quad_interior_angles(vertices: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad_points = np.asarray(vertices, dtype=float)[np.asarray(quad, dtype=np.int64)]
    angles = np.zeros((4,), dtype=float)
    for local_index in range(4):
        current_point = quad_points[local_index]
        previous_edge = quad_points[(local_index - 1) % 4] - current_point
        next_edge = quad_points[(local_index + 1) % 4] - current_point
        previous_norm = np.linalg.norm(previous_edge)
        next_norm = np.linalg.norm(next_edge)
        if previous_norm <= 1e-14 or next_norm <= 1e-14:
            angles[local_index] = 180.0
            continue
        cosine = np.clip(
            np.dot(previous_edge, next_edge) / (previous_norm * next_norm),
            -1.0,
            1.0,
        )
        angles[local_index] = np.degrees(np.arccos(cosine))
    return angles


def _quad_planarity_ratio(vertices: np.ndarray, quad: np.ndarray) -> float:
    quad_points = np.asarray(vertices, dtype=float)[np.asarray(quad, dtype=np.int64)]
    first_triangle_normal = np.cross(
        quad_points[1] - quad_points[0],
        quad_points[2] - quad_points[0],
    )
    normal_norm = np.linalg.norm(first_triangle_normal)
    if normal_norm <= 1e-14:
        return np.inf
    edge_lengths = np.linalg.norm(
        quad_points[np.arange(4)] - quad_points[np.roll(np.arange(4), -1)],
        axis=1,
    )
    characteristic_length = max(float(edge_lengths.mean()), 1e-14)
    unit_normal = first_triangle_normal / normal_norm
    signed_distances = np.abs((quad_points - quad_points[0]) @ unit_normal)
    return float(np.max(signed_distances) / characteristic_length)


def _quad_aspect_ratio(vertices: np.ndarray, quad: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float)[np.asarray(quad, dtype=np.int64)]
    edge_lengths = np.linalg.norm(points[np.roll(np.arange(4), -1)] - points, axis=1)
    return float(np.max(edge_lengths) / max(float(np.min(edge_lengths)), 1e-14))


def _quad_min_scaled_jacobian(vertices: np.ndarray, quad: np.ndarray) -> float:
    quad_points = np.asarray(vertices, dtype=float)[np.asarray(quad, dtype=np.int64)]
    scaled_jacobians = []
    for local_index in range(4):
        current_point = quad_points[local_index]
        previous_edge = quad_points[(local_index - 1) % 4] - current_point
        next_edge = quad_points[(local_index + 1) % 4] - current_point
        denominator = np.linalg.norm(previous_edge) * np.linalg.norm(next_edge)
        if denominator <= 1e-14:
            scaled_jacobians.append(0.0)
            continue
        scaled_jacobians.append(float(np.linalg.norm(np.cross(previous_edge, next_edge)) / denominator))
    return float(np.min(scaled_jacobians))


def _quad_quality_score(
    vertices: np.ndarray,
    quad: np.ndarray,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
) -> float:
    points = np.asarray(vertices, dtype=float)[np.asarray(quad, dtype=np.int64)]
    edge_lengths = np.linalg.norm(points[np.roll(np.arange(4), -1)] - points, axis=1)
    mean_edge_length = max(float(edge_lengths.mean()), 1e-14)
    diagonal_lengths = np.asarray(
        [
            np.linalg.norm(points[2] - points[0]),
            np.linalg.norm(points[3] - points[1]),
        ],
        dtype=float,
    )
    angle_penalty = float(np.mean(np.abs(_quad_interior_angles(vertices, quad) - 90.0)) / 90.0)
    edge_balance_penalty = float(np.std(edge_lengths) / mean_edge_length)
    diagonal_balance_penalty = float(
        abs(diagonal_lengths[0] - diagonal_lengths[1])
        / max(float(diagonal_lengths.mean()), 1e-14)
    )
    normal_alignment_penalty = float(1.0 - np.clip(np.dot(normal_a, normal_b), -1.0, 1.0))
    planarity_penalty = _quad_planarity_ratio(vertices, quad)
    return (
        0.45 * angle_penalty
        + 0.2 * edge_balance_penalty
        + 0.1 * diagonal_balance_penalty
        + 0.15 * normal_alignment_penalty
        + 0.1 * planarity_penalty
    )
