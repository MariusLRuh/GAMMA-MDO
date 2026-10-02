"""Warm-started multi-candidate point projection onto a function set.

A single Newton solve seeded from one patch is not reliable near patch
boundaries or interior C0 knot lines: the closest location may lie on an
edge, a crease, a crease crossing, or on a neighbouring patch entirely. This module builds several candidate seeds per query point,
solves each, and ranks them: a converged candidate always outranks a
non-converged one, and the closest converged candidate wins. When no candidate
converges, the point is not dropped — the minimum-residual candidate is kept as
a fallback, and its ``converged`` flag stays ``False`` so the caller can tell
the two cases apart.

It also handles the two awkward cases that motivated it. A boundary edge whose
sampled arc length has collapsed to nearly a point cannot support a
one-dimensional edge solve, so that solve is replaced by a single fixed-point
candidate. Points whose final parameter sits on a bound with an outward
unmasked residual are flagged ``boundary_clamped`` and retried, since such a
point usually belongs on the other side of that edge.

Everything here is eager NumPy: the kernels execute when called, either
directly or from a CSDL custom operation's ``compute``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .orthogonality_projection_numpy import (
    OrthogonalityNewtonParams,
    SurfaceProjectionResult,
    make_surface_orthogonality_evaluator_numpy,
    project_points_on_surface_edge_newton_numpy,
    project_points_orthogonality_newton_numpy,
)
from .warm_start_projections import (
    build_sampled_patches_mesh,
    warm_start_from_triangulation,
)


try:
    import lsdo_function_spaces as lfs
except Exception:  # pragma: no cover
    lfs = None

try:
    from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy_factory import (
        make_bspline_evaluator_numpy,
    )
except Exception:  # pragma: no cover
    make_bspline_evaluator_numpy = None


EdgeName = str
POINT_CANDIDATE_AXIS = -2


@dataclass(frozen=True)
class NeighborEdgeMap:
    """Topological link from one patch edge to the adjoining patch edge.

    Attributes
    ----------
    neighbor_patch, neighbor_edge
        The patch and named edge on the other side of this boundary.
    reverse_along_edge
        ``True`` when the two edges run in opposite parametric directions, so a
        coordinate must be flipped when crossing.
    """

    neighbor_patch: int
    neighbor_edge: EdgeName
    reverse_along_edge: bool = False


@dataclass
class WarmStartCandidateProjectionResult:
    """Per-point outcome of the warm-started multi-candidate projection.

    Attributes
    ----------
    patch_id
        Patch that won for each point.
    uv
        Final parametric coordinates on that patch. For a point whose
        ``converged`` entry is ``False`` these come from the minimum-residual
        fallback and are not a solution.
    projected_points
        Surface points at ``uv``.
    dist2
        Squared distance from each query point to its accepted projection.
    residual, converged, iterations
        Newton evidence for the accepted candidate.
    candidate_kind
        Which candidate type won for each point, for example a patch interior,
        a named boundary edge, an interior C0 line or crossing, a fixed point
        on a degenerate edge, or a neighbouring patch.
    warm_patch_id, warm_uv0
        The seed the winning solve started from.
    edge_map
        Patch-edge adjacency used during the search.
    """

    patch_id: np.ndarray
    uv: np.ndarray
    projected_points: np.ndarray
    dist2: np.ndarray
    residual: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    candidate_kind: List[str]
    warm_patch_id: np.ndarray
    warm_uv0: np.ndarray
    edge_map: Dict[Tuple[int, EdgeName], NeighborEdgeMap]


@dataclass(frozen=True)
class _CandidateSpec:
    point_index: int
    patch_id: int
    uv0: Tuple[float, float]
    fixed_axis: int
    fixed_value: float
    kind: str


def _require_numpy_bspline_factory() -> None:
    if make_bspline_evaluator_numpy is None:
        raise ImportError(
            "make_bspline_evaluator_numpy not found; ensure "
            "compute_basis_matrix_numpy_factory.py is on path."
        )


def _which_edge(u: float, v: float, eps: float) -> List[EdgeName]:
    edges: List[EdgeName] = []
    if u <= eps:
        edges.append("u0")
    if u >= 1.0 - eps:
        edges.append("u1")
    if v <= eps:
        edges.append("v0")
    if v >= 1.0 - eps:
        edges.append("v1")
    return edges


def _edges_by_distance(u: float, v: float) -> List[EdgeName]:
    distances = [
        (u, "u0"),
        (1.0 - u, "u1"),
        (v, "v0"),
        (1.0 - v, "v1"),
    ]
    distances.sort(key=lambda item: (item[0], item[1]))
    return [edge for _, edge in distances]


def _edge_distance(u: float, v: float, edge: EdgeName) -> float:
    if edge == "u0":
        return u
    if edge == "u1":
        return 1.0 - u
    if edge == "v0":
        return v
    if edge == "v1":
        return 1.0 - v
    raise ValueError(f"Unknown edge {edge}")


def _map_uv_across_edge(
    uv: np.ndarray,
    from_edge: EdgeName,
    to_edge: EdgeName,
    reverse_along_edge: bool,
) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])

    if from_edge in ("u0", "u1"):
        t = v
        if reverse_along_edge:
            t = 1.0 - t
        if to_edge == "u0":
            return np.array([0.0, t], dtype=float)
        if to_edge == "u1":
            return np.array([1.0, t], dtype=float)
        if to_edge == "v0":
            return np.array([t, 0.0], dtype=float)
        if to_edge == "v1":
            return np.array([t, 1.0], dtype=float)
    else:
        t = u
        if reverse_along_edge:
            t = 1.0 - t
        if to_edge == "u0":
            return np.array([0.0, t], dtype=float)
        if to_edge == "u1":
            return np.array([1.0, t], dtype=float)
        if to_edge == "v0":
            return np.array([t, 0.0], dtype=float)
        if to_edge == "v1":
            return np.array([t, 1.0], dtype=float)

    raise ValueError(f"Invalid edge mapping: {from_edge} -> {to_edge}")


def _edge_uv_samples(edge: EdgeName, num_samples: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_samples, dtype=float)
    if edge == "u0":
        return np.column_stack([np.zeros_like(t), t])
    if edge == "u1":
        return np.column_stack([np.ones_like(t), t])
    if edge == "v0":
        return np.column_stack([t, np.zeros_like(t)])
    if edge == "v1":
        return np.column_stack([t, np.ones_like(t)])
    raise ValueError(f"Unknown edge {edge}")


def _edge_to_constraint(edge: EdgeName) -> tuple[int, float]:
    if edge == "u0":
        return 0, 0.0
    if edge == "u1":
        return 0, 1.0
    if edge == "v0":
        return 1, 0.0
    if edge == "v1":
        return 1, 1.0
    raise ValueError(f"Unknown edge {edge}")


def _get_patch_metadata(function_set, patch_id: int):
    fun = function_set.functions[int(patch_id)]
    coeffs = np.asarray(fun.coefficients.value, dtype=float)
    degrees = tuple(int(v) for v in fun.space.degree)
    knot_vectors = tuple(np.asarray(k, dtype=float) for k in fun.space.knots)
    return coeffs, degrees, knot_vectors


def _append_candidate_spec(
    specs: List[_CandidateSpec],
    seen: set,
    *,
    point_index: int,
    candidate_patch: int,
    candidate_uv: np.ndarray,
    fixed_axis: int,
    fixed_value: float,
    kind: str,
) -> None:
    fixed_value_key = None if fixed_axis < 0 else round(float(fixed_value), 12)
    key = (
        int(candidate_patch),
        int(fixed_axis),
        fixed_value_key,
        round(float(candidate_uv[0]), 12),
        round(float(candidate_uv[1]), 12),
    )
    if key in seen:
        return
    seen.add(key)
    specs.append(
        _CandidateSpec(
            point_index=point_index,
            patch_id=int(candidate_patch),
            uv0=(float(candidate_uv[0]), float(candidate_uv[1])),
            fixed_axis=int(fixed_axis),
            fixed_value=float(fixed_value),
            kind=kind,
        )
    )


def _add_edge_candidate_specs(
    specs: List[_CandidateSpec],
    seen: set,
    *,
    point_index: int,
    patch_id: int,
    uv_seed: np.ndarray,
    edge: EdgeName,
    edge_map: Dict[Tuple[int, EdgeName], NeighborEdgeMap],
    degenerate_edge_map: Optional[Dict[Tuple[int, EdgeName], float]],
    include_current_patch_boundary: bool,
    include_neighbor_patch: bool,
    include_neighbor_boundary: bool,
    kind_prefix: str = "",
) -> None:
    current_edge_uv = _map_uv_across_edge(
        uv=uv_seed,
        from_edge=edge,
        to_edge=edge,
        reverse_along_edge=False,
    )
    fixed_axis, fixed_value = _edge_to_constraint(edge)
    edge_is_degenerate = degenerate_edge_map is not None and (patch_id, edge) in degenerate_edge_map

    if include_current_patch_boundary:
        if edge_is_degenerate:
            _append_candidate_spec(
                specs,
                seen,
                point_index=point_index,
                candidate_patch=patch_id,
                candidate_uv=current_edge_uv,
                fixed_axis=POINT_CANDIDATE_AXIS,
                fixed_value=np.nan,
                kind=f"{kind_prefix}current_{edge}_degenerate_point",
            )
        else:
            _append_candidate_spec(
                specs,
                seen,
                point_index=point_index,
                candidate_patch=patch_id,
                candidate_uv=current_edge_uv,
                fixed_axis=fixed_axis,
                fixed_value=fixed_value,
                kind=f"{kind_prefix}current_{edge}_boundary",
            )

    neighbor = edge_map.get((patch_id, edge))
    if neighbor is None:
        return

    mapped_uv = _map_uv_across_edge(
        uv=uv_seed,
        from_edge=edge,
        to_edge=neighbor.neighbor_edge,
        reverse_along_edge=neighbor.reverse_along_edge,
    )

    if include_neighbor_patch:
        _append_candidate_spec(
            specs,
            seen,
            point_index=point_index,
            candidate_patch=neighbor.neighbor_patch,
            candidate_uv=mapped_uv,
            fixed_axis=-1,
            fixed_value=np.nan,
            kind=f"{kind_prefix}neighbor_{edge}_patch",
        )

    if include_neighbor_boundary:
        nbr_axis, nbr_value = _edge_to_constraint(neighbor.neighbor_edge)
        neighbor_edge_is_degenerate = (
            degenerate_edge_map is not None
            and (neighbor.neighbor_patch, neighbor.neighbor_edge) in degenerate_edge_map
        )
        if neighbor_edge_is_degenerate:
            _append_candidate_spec(
                specs,
                seen,
                point_index=point_index,
                candidate_patch=neighbor.neighbor_patch,
                candidate_uv=mapped_uv,
                fixed_axis=POINT_CANDIDATE_AXIS,
                fixed_value=np.nan,
                kind=f"{kind_prefix}neighbor_{neighbor.neighbor_edge}_degenerate_point",
            )
        else:
            _append_candidate_spec(
                specs,
                seen,
                point_index=point_index,
                candidate_patch=neighbor.neighbor_patch,
                candidate_uv=mapped_uv,
                fixed_axis=nbr_axis,
                fixed_value=nbr_value,
                kind=f"{kind_prefix}neighbor_{neighbor.neighbor_edge}_boundary",
            )


def build_edge_neighbor_map(
    function_set,
    patch_indices: Optional[Iterable[int]] = None,
    *,
    num_samples: int = 41,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Dict[Tuple[int, EdgeName], NeighborEdgeMap]:
    """Determine which patch edges adjoin which, and in what direction.

    Matching is geometric: edges whose sampled points coincide within tolerance
    are treated as adjoining, and the sample ordering decides whether the shared
    edge runs forward or reversed.

    Parameters
    ----------
    function_set
        Function set whose patches are examined. Patch coefficients are read at
        call time, so the map describes the geometry as it stands now.
    patch_indices
        Patches to consider, in the order given. ``None`` uses every patch in
        ``function_set``.
    num_samples
        Number of points sampled along each of the four edges of every patch.
        Higher values sample the comparison more densely and cost more, and can
        expose an interior mismatch that a coarser grid steps over. Because the
        sample locations themselves move with the count, matching is not
        guaranteed to grow monotonically stricter as this rises.
    atol
        Absolute tolerance for deciding that two sampled edge points coincide.
    rtol
        Relative tolerance for the same test, scaled by the sampled geometry.

    Returns
    -------
    dict
        Maps ``(patch_id, edge_name)`` to a :class:`NeighborEdgeMap`. Edges with
        no match are simply absent, which is the normal case at an open boundary.
    """
    _require_numpy_bspline_factory()

    if patch_indices is None:
        patch_indices = function_set.functions.keys()
    patch_indices = [int(idx) for idx in patch_indices]

    edge_points: Dict[Tuple[int, EdgeName], np.ndarray] = {}
    edge_keys: List[Tuple[int, EdgeName]] = []
    all_points: List[np.ndarray] = []

    evaluator_cache = {}
    for patch_id in patch_indices:
        coeffs, degrees, knot_vectors = _get_patch_metadata(function_set, patch_id)
        if patch_id not in evaluator_cache:
            evaluator_cache[patch_id] = make_bspline_evaluator_numpy(
                degrees=degrees,
                knot_vectors=knot_vectors,
                der_orders=None,
            )
        eval_S = evaluator_cache[patch_id]

        for edge in ("u0", "u1", "v0", "v1"):
            uv = _edge_uv_samples(edge, num_samples)
            xyz = np.asarray(eval_S(uv, coeffs), dtype=float)
            key = (patch_id, edge)
            edge_keys.append(key)
            edge_points[key] = xyz
            all_points.append(xyz)

    if not all_points:
        return {}

    stacked = np.vstack(all_points)
    diag = np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0))
    tol = max(float(atol), float(rtol) * float(diag))

    candidates = []
    for i, key_a in enumerate(edge_keys):
        xyz_a = edge_points[key_a]
        for j in range(i + 1, len(edge_keys)):
            key_b = edge_keys[j]
            if key_a[0] == key_b[0]:
                continue
            xyz_b = edge_points[key_b]

            forward_end = max(
                np.linalg.norm(xyz_a[0] - xyz_b[0]),
                np.linalg.norm(xyz_a[-1] - xyz_b[-1]),
            )
            reverse_end = max(
                np.linalg.norm(xyz_a[0] - xyz_b[-1]),
                np.linalg.norm(xyz_a[-1] - xyz_b[0]),
            )
            if min(forward_end, reverse_end) > tol:
                continue

            forward_score = np.max(np.linalg.norm(xyz_a - xyz_b, axis=1))
            reverse_score = np.max(np.linalg.norm(xyz_a - xyz_b[::-1], axis=1))
            reverse = reverse_score < forward_score
            score = reverse_score if reverse else forward_score

            if score <= tol:
                candidates.append((score, key_a, key_b, reverse))

    candidates.sort(key=lambda item: item[0])

    assigned = set()
    edge_map: Dict[Tuple[int, EdgeName], NeighborEdgeMap] = {}
    for _, key_a, key_b, reverse in candidates:
        if key_a in assigned or key_b in assigned:
            continue
        edge_map[key_a] = NeighborEdgeMap(
            neighbor_patch=int(key_b[0]),
            neighbor_edge=key_b[1],
            reverse_along_edge=reverse,
        )
        edge_map[key_b] = NeighborEdgeMap(
            neighbor_patch=int(key_a[0]),
            neighbor_edge=key_a[1],
            reverse_along_edge=reverse,
        )
        assigned.add(key_a)
        assigned.add(key_b)

    return edge_map


def _compute_geometric_near_edges(
    function_set,
    patch_id: np.ndarray,
    uv0: np.ndarray,
    points: np.ndarray,
    *,
    eps_edge: float,
    gap_atol: float,
    cell_factor: float,
) -> List[List[EdgeName]]:
    """Decide which patch edges each warm-start seed lies near.

    An edge qualifies when the neighbouring patch across it could hold the true
    closest point.

    Unlike the parametric ``_which_edge`` band (``u <= eps_edge``), the test is
    *physical* and grid-independent: an edge is "near" when the seed's physical
    distance to it (``|S_u| * du_to_edge``) is within a band dominated by the
    projection gap ``g = ||point - S(seed)||``. Rationale: the neighbour patch
    touches the shared edge, so the nearest point it can offer is at least the
    seed's physical distance to that edge; if that already exceeds the current
    gap, the neighbour cannot win and can be skipped. When the warm start lands
    in the *wrong* patch, ``g`` is inflated precisely because ``S(seed)`` is far
    from the true closest point, which automatically widens the band enough to
    admit the correct neighbour — without any dependence on ``nu``/``nv``.

    A small floor tied to one sampling cell (``2*eps_edge ~ max(du, dv)``, times
    the local tangent magnitude) keeps coverage at least as generous as the old
    parametric band even when ``g -> 0`` (a node sitting essentially on a seam).
    """
    num_points = int(patch_id.shape[0])
    near_edges: List[List[EdgeName]] = [[] for _ in range(num_points)]
    if num_points == 0:
        return near_edges

    cell_scale = max(float(cell_factor), 0.0) * (2.0 * max(float(eps_edge), 0.0))
    atol = max(float(gap_atol), 0.0)

    groups: Dict[int, List[int]] = {}
    for i in range(num_points):
        groups.setdefault(int(patch_id[i]), []).append(i)

    for pid, index_list in groups.items():
        idx = np.asarray(index_list, dtype=int)
        coeffs, degrees, knot_vectors = _get_patch_metadata(function_set, pid)
        evaluate = make_surface_orthogonality_evaluator_numpy(degrees, knot_vectors)
        uv = np.clip(np.asarray(uv0[idx], dtype=float), 0.0, 1.0)
        S, Su, Sv, _, _, _ = evaluate(uv, coeffs)

        a = np.linalg.norm(Su, axis=1)  # |S_u|: physical length per unit u
        b = np.linalg.norm(Sv, axis=1)  # |S_v|: physical length per unit v
        gap = np.linalg.norm(np.asarray(points[idx], dtype=float) - S, axis=1)

        band_u = gap + atol + cell_scale * a
        band_v = gap + atol + cell_scale * b
        u = uv[:, 0]
        v = uv[:, 1]
        near_u0 = (a * u) <= band_u
        near_u1 = (a * (1.0 - u)) <= band_u
        near_v0 = (b * v) <= band_v
        near_v1 = (b * (1.0 - v)) <= band_v

        for local_index, global_index in enumerate(idx):
            edges: List[EdgeName] = []
            if near_u0[local_index]:
                edges.append("u0")
            if near_u1[local_index]:
                edges.append("u1")
            if near_v0[local_index]:
                edges.append("v0")
            if near_v1[local_index]:
                edges.append("v1")
            near_edges[int(global_index)] = edges

    return near_edges


def _build_candidate_specs(
    patch_id: np.ndarray,
    uv0: np.ndarray,
    edge_map: Dict[Tuple[int, EdgeName], NeighborEdgeMap],
    degenerate_edge_map: Optional[Dict[Tuple[int, EdgeName], float]],
    *,
    eps_edge: float,
    include_current_patch_boundary: bool,
    include_neighbor_patch: bool,
    include_neighbor_boundary: bool,
    near_edges: Optional[List[List[EdgeName]]] = None,
    crease_knots: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    span_knots: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[_CandidateSpec]:
    specs: List[_CandidateSpec] = []

    for point_index in range(patch_id.shape[0]):
        pid = int(patch_id[point_index])
        uv_seed = np.asarray(uv0[point_index], dtype=float)
        seen = set()
        edges_added = set()

        _append_candidate_spec(
            specs,
            seen,
            point_index=point_index,
            candidate_patch=pid,
            candidate_uv=uv_seed,
            fixed_axis=-1,
            fixed_value=np.nan,
            kind="warm_start_patch",
        )

        # A constrained minimum at an interior C0 line is not a root of
        # the two-dimensional orthogonality equations. Search the lines
        # bounding the seed span, and their crossings, explicitly. This is
        # a local candidate search, not a global closest-point guarantee.
        bounds = [[], []]
        if crease_knots is not None and span_knots is not None:
            for axis in (0, 1):
                knots = span_knots[pid][axis]
                span = np.clip(np.searchsorted(knots, uv_seed[axis], side="right") - 1,
                               0, len(knots) - 2)
                bounds[axis] = [k for k in knots[span:span + 2]
                                if k in crease_knots[pid][axis]]
                for knot in bounds[axis]:
                    seed = uv_seed.copy()
                    seed[axis] = knot
                    _append_candidate_spec(
                        specs, seen, point_index=point_index, candidate_patch=pid,
                        candidate_uv=seed, fixed_axis=axis, fixed_value=knot,
                        kind=f"current_{'uv'[axis]}_c0_line",
                    )
            for u in bounds[0]:
                for v in bounds[1]:
                    _append_candidate_spec(
                        specs, seen, point_index=point_index, candidate_patch=pid,
                        candidate_uv=np.array([u, v]), fixed_axis=POINT_CANDIDATE_AXIS,
                        fixed_value=np.nan, kind="current_c0_corner_point",
                    )

        if near_edges is not None:
            seed_edges = near_edges[point_index]
        else:
            seed_edges = _which_edge(float(uv_seed[0]), float(uv_seed[1]), eps_edge)
        for edge in seed_edges:
            edges_added.add(edge)
            _add_edge_candidate_specs(
                specs,
                seen,
                point_index=point_index,
                patch_id=pid,
                uv_seed=uv_seed,
                edge=edge,
                edge_map=edge_map,
                degenerate_edge_map=degenerate_edge_map,
                include_current_patch_boundary=include_current_patch_boundary,
                include_neighbor_patch=include_neighbor_patch,
                include_neighbor_boundary=include_neighbor_boundary,
            )

        if degenerate_edge_map is not None:
            degenerate_threshold = 2.0 * eps_edge
            for edge in _edges_by_distance(float(uv_seed[0]), float(uv_seed[1])):
                if edge in edges_added:
                    continue
                if (pid, edge) not in degenerate_edge_map:
                    continue
                if _edge_distance(float(uv_seed[0]), float(uv_seed[1]), edge) > degenerate_threshold:
                    break
                _add_edge_candidate_specs(
                    specs,
                    seen,
                    point_index=point_index,
                    patch_id=pid,
                    uv_seed=uv_seed,
                    edge=edge,
                    edge_map=edge_map,
                    degenerate_edge_map=degenerate_edge_map,
                    include_current_patch_boundary=include_current_patch_boundary,
                    include_neighbor_patch=include_neighbor_patch,
                    include_neighbor_boundary=include_neighbor_boundary,
                )

    return specs


def _build_retry_candidate_specs(
    warm_patch_id: np.ndarray,
    warm_uv0: np.ndarray,
    point_indices: np.ndarray,
    edge_map: Dict[Tuple[int, EdgeName], NeighborEdgeMap],
    degenerate_edge_map: Optional[Dict[Tuple[int, EdgeName], float]],
    *,
    retry_num_closest_edges: int,
    include_current_patch_boundary: bool,
    include_neighbor_patch: bool,
    include_neighbor_boundary: bool,
    near_edges: Optional[List[List[EdgeName]]] = None,
) -> List[_CandidateSpec]:
    specs: List[_CandidateSpec] = []
    num_edges = max(1, int(retry_num_closest_edges))

    for point_index in np.asarray(point_indices, dtype=int):
        pid = int(warm_patch_id[point_index])
        uv_seed = np.asarray(warm_uv0[point_index], dtype=float)
        seen = set()

        # The nearest few edges (parametric) are always tried. When a geometric
        # near-edge set is supplied it is unioned in: this is the retry-scoped
        # form of the cross-patch fix, safe here because every candidate is
        # vetted by the normal/distance gates before it can replace a selection.
        edges_to_try = list(_edges_by_distance(float(uv_seed[0]), float(uv_seed[1]))[:num_edges])
        if near_edges is not None:
            for edge in near_edges[point_index]:
                if edge not in edges_to_try:
                    edges_to_try.append(edge)

        for edge in edges_to_try:
            _add_edge_candidate_specs(
                specs,
                seen,
                point_index=point_index,
                patch_id=pid,
                uv_seed=uv_seed,
                edge=edge,
                edge_map=edge_map,
                degenerate_edge_map=degenerate_edge_map,
                include_current_patch_boundary=include_current_patch_boundary,
                include_neighbor_patch=include_neighbor_patch,
                include_neighbor_boundary=include_neighbor_boundary,
                kind_prefix="retry_",
            )

    return specs


_LOCAL_RETRY_DIRECTIONS = np.array(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ],
    dtype=float,
)


def _build_local_retry_candidate_specs(
    selected_patch_id: np.ndarray,
    selected_uv: np.ndarray,
    point_indices: np.ndarray,
    *,
    uv_step: float,
    scale_factors: Sequence[float] = (1.0, 4.0, 16.0),
) -> List[_CandidateSpec]:
    """Re-seed the same-patch Newton at several length scales.

    The stencil spans *several* scales rather than a single sampling cell.

    The old builder probed a single ``±uv_step`` stencil (~one cell) and even
    re-ran the identical seed via a ``[0, 0]`` offset. That cannot rescue a point
    whose Newton parked on a spurious boundary minimum while the true closest
    point sits well inside the patch: e.g. a node near a high-curvature skin
    junction whose warm start lands at ``v = 1`` but whose real projection is at
    ``v ~ 0.85`` (a full 0.15 away). Sweeping progressively coarser scales gives
    the Newton seeds deep enough in the interior to fall into that basin, while
    the min-``dist2`` selection keeps the genuinely closest result. This is the
    "make the local retry an actual refinement" fix; it stays on-patch, so it
    adds no cross-patch behaviour and cannot flip a node to another skin.
    """
    specs: List[_CandidateSpec] = []
    base = max(float(uv_step), 1e-12)
    scales = [s for s in scale_factors if float(s) > 0.0] or [1.0]

    for point_index in np.asarray(point_indices, dtype=int):
        pid = int(selected_patch_id[point_index])
        uv_seed = np.asarray(selected_uv[point_index], dtype=float)
        # Pre-mark the seed so offsets that clip back onto it (e.g. an outward
        # step at a patch bound) are skipped: re-running the seed that just
        # failed is exactly the wasted work the old [0, 0] offset caused.
        seen = {
            (pid, -1, None, round(float(uv_seed[0]), 12), round(float(uv_seed[1]), 12))
        }
        for scale in scales:
            step = base * float(scale)
            for direction in _LOCAL_RETRY_DIRECTIONS:
                candidate_uv = np.clip(uv_seed + step * direction, 0.0, 1.0)
                _append_candidate_spec(
                    specs,
                    seen,
                    point_index=point_index,
                    candidate_patch=pid,
                    candidate_uv=candidate_uv,
                    fixed_axis=-1,
                    fixed_value=0.0,
                    kind="retry_local_patch",
                )

    return specs


def _run_candidate_projections(
    function_set,
    points: np.ndarray,
    specs: Sequence[_CandidateSpec],
    params: OrthogonalityNewtonParams,
) -> Dict[str, object]:
    num_candidates = len(specs)
    if num_candidates == 0:
        raise ValueError("No candidate projection specs were generated.")

    point_index = np.array([spec.point_index for spec in specs], dtype=int)
    patch_id = np.array([spec.patch_id for spec in specs], dtype=int)
    uv0 = np.array([spec.uv0 for spec in specs], dtype=float)
    fixed_axis = np.array([spec.fixed_axis for spec in specs], dtype=int)
    fixed_value = np.array([spec.fixed_value for spec in specs], dtype=float)
    candidate_kind = [spec.kind for spec in specs]

    projected_points = np.zeros((num_candidates, points.shape[1]), dtype=float)
    final_uv = np.zeros((num_candidates, 2), dtype=float)
    dist2 = np.full((num_candidates,), np.inf, dtype=float)
    residual = np.full((num_candidates,), np.inf, dtype=float)
    converged = np.zeros((num_candidates,), dtype=bool)
    iterations = np.zeros((num_candidates,), dtype=int)
    boundary_clamped = np.zeros((num_candidates,), dtype=bool)

    groups: Dict[Tuple[int, int, float], List[int]] = {}
    for candidate_index in range(num_candidates):
        key = (
            int(patch_id[candidate_index]),
            int(fixed_axis[candidate_index]),
            float(fixed_value[candidate_index]) if fixed_axis[candidate_index] >= 0 else None,
        )
        groups.setdefault(key, []).append(candidate_index)

    patch_cache = {}
    for (candidate_patch, candidate_fixed_axis, candidate_fixed_value), group_indices in groups.items():
        if candidate_patch not in patch_cache:
            patch_cache[candidate_patch] = _get_patch_metadata(function_set, candidate_patch)
        coeffs, degrees, knot_vectors = patch_cache[candidate_patch]

        group_indices = np.asarray(group_indices, dtype=int)
        group_points = points[point_index[group_indices]]
        group_uv0 = uv0[group_indices]

        if candidate_fixed_axis < 0:
            if candidate_fixed_axis == POINT_CANDIDATE_AXIS:
                evaluator = make_bspline_evaluator_numpy(
                    degrees=degrees,
                    knot_vectors=knot_vectors,
                    der_orders=None,
                )
                point_xyz = np.asarray(evaluator(group_uv0, coeffs), dtype=float)
                diff = point_xyz - group_points
                result = SurfaceProjectionResult(
                    uv=group_uv0.copy(),
                    projected_points=point_xyz,
                    residual=np.zeros((group_indices.shape[0],), dtype=float),
                    step_norm=np.zeros((group_indices.shape[0],), dtype=float),
                    dist2=np.einsum("ij,ij->i", diff, diff),
                    converged=np.ones((group_indices.shape[0],), dtype=bool),
                    iterations=np.zeros((group_indices.shape[0],), dtype=int),
                )
            else:
                result = project_points_orthogonality_newton_numpy(
                    group_points,
                    group_uv0,
                    coeffs,
                    degrees,
                    knot_vectors,
                    params=params,
                )
        else:
            free_axis = 1 - candidate_fixed_axis
            result = project_points_on_surface_edge_newton_numpy(
                group_points,
                group_uv0[:, free_axis],
                coeffs,
                degrees,
                knot_vectors,
                fixed_axis=candidate_fixed_axis,
                fixed_value=candidate_fixed_value,
                params=params,
            )

        projected_points[group_indices] = result.projected_points
        final_uv[group_indices] = result.uv
        dist2[group_indices] = result.dist2
        residual[group_indices] = result.residual
        converged[group_indices] = result.converged
        iterations[group_indices] = result.iterations
        if result.boundary_clamped is not None:
            boundary_clamped[group_indices] = result.boundary_clamped

    return {
        "point_index": point_index,
        "patch_id": patch_id,
        "uv": final_uv,
        "projected_points": projected_points,
        "dist2": dist2,
        "residual": residual,
        "converged": converged,
        "iterations": iterations,
        "boundary_clamped": boundary_clamped,
        "candidate_kind": candidate_kind,
    }


def _is_candidate_better(
    converged: bool,
    residual: float,
    dist2: float,
    best_converged: bool,
    best_residual: float,
    best_dist2: float,
) -> bool:
    rank = 0 if converged else 1
    best_rank = 0 if best_converged else 1

    if rank < best_rank:
        return True
    if rank > best_rank:
        return False
    if rank == 0:
        if dist2 < best_dist2:
            return True
        if dist2 > best_dist2:
            return False
        return residual < best_residual
    if residual < best_residual:
        return True
    if residual > best_residual:
        return False
    return dist2 < best_dist2


def _compute_candidate_normal(
    function_set,
    patch_id: int,
    uv: np.ndarray,
) -> np.ndarray:
    _require_numpy_bspline_factory()
    coeffs, degrees, knot_vectors = _get_patch_metadata(function_set, int(patch_id))
    uv = np.asarray(uv, dtype=float).reshape(1, 2)
    evaluator_u = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(1, 0),
    )
    evaluator_v = make_bspline_evaluator_numpy(
        degrees=degrees,
        knot_vectors=knot_vectors,
        der_orders=(0, 1),
    )
    tangent_u = np.asarray(evaluator_u(uv, coeffs), dtype=float).reshape(1, -1)[0]
    tangent_v = np.asarray(evaluator_v(uv, coeffs), dtype=float).reshape(1, -1)[0]
    if tangent_u.size != 3 or tangent_v.size != 3:
        return np.zeros(3, dtype=float)
    normal = np.cross(tangent_u, tangent_v)
    norm = np.linalg.norm(normal)
    if norm <= 1e-14:
        return np.zeros(3, dtype=float)
    return normal / norm


def _is_retry_candidate_compatible(
    function_set,
    *,
    selected_patch_id: int,
    selected_uv: np.ndarray,
    selected_dist2: float,
    retry_patch_id: int,
    retry_uv: np.ndarray,
    retry_dist2: float,
    retry_accept_distance_factor: float,
    retry_accept_distance_atol: float,
    retry_normal_dot_min: float,
) -> bool:
    if not np.isfinite(retry_dist2):
        return False
    if not np.isfinite(selected_dist2):
        return True

    selected_distance = float(np.sqrt(max(float(selected_dist2), 0.0)))
    retry_distance = float(np.sqrt(max(float(retry_dist2), 0.0)))
    distance_limit = (
        max(float(retry_accept_distance_factor), 1.0) * selected_distance
        + max(float(retry_accept_distance_atol), 0.0)
    )
    if retry_distance > distance_limit:
        return False

    if int(selected_patch_id) == int(retry_patch_id):
        return True

    if retry_normal_dot_min <= -1.0:
        return True

    selected_normal = _compute_candidate_normal(
        function_set,
        int(selected_patch_id),
        selected_uv,
    )
    retry_normal = _compute_candidate_normal(
        function_set,
        int(retry_patch_id),
        retry_uv,
    )
    if np.linalg.norm(selected_normal) <= 0.0 or np.linalg.norm(retry_normal) <= 0.0:
        return True

    return float(np.dot(selected_normal, retry_normal)) >= float(retry_normal_dot_min)


def _select_best_candidates(candidate_results: Dict[str, object], num_points: int) -> np.ndarray:
    point_index = candidate_results["point_index"]
    dist2 = candidate_results["dist2"]
    residual = candidate_results["residual"]
    converged = candidate_results["converged"]

    best_candidate = np.full((num_points,), -1, dtype=int)
    best_rank = np.full((num_points,), 2, dtype=int)
    best_residual = np.full((num_points,), np.inf, dtype=float)
    best_dist2 = np.full((num_points,), np.inf, dtype=float)

    for candidate_index in range(point_index.shape[0]):
        original_point = point_index[candidate_index]
        rank = 0 if converged[candidate_index] else 1

        better = False
        if rank < best_rank[original_point]:
            better = True
        elif rank == best_rank[original_point]:
            if rank == 0:
                if dist2[candidate_index] < best_dist2[original_point]:
                    better = True
                elif (
                    dist2[candidate_index] == best_dist2[original_point]
                    and residual[candidate_index] < best_residual[original_point]
                ):
                    better = True
            else:
                if residual[candidate_index] < best_residual[original_point]:
                    better = True
                elif (
                    residual[candidate_index] == best_residual[original_point]
                    and dist2[candidate_index] < best_dist2[original_point]
                ):
                    better = True

        if better:
            best_candidate[original_point] = candidate_index
            best_rank[original_point] = rank
            best_residual[original_point] = residual[candidate_index]
            best_dist2[original_point] = dist2[candidate_index]

    return best_candidate


def project_points_with_warm_start_candidates_numpy(
    function_set,
    points: np.ndarray,
    *,
    patch_indices: Optional[Iterable[int]] = None,
    mesh=None,
    warm_start_nu: int = 100,
    warm_start_nv: int = 100,
    edge_map: Optional[Dict[Tuple[int, EdgeName], NeighborEdgeMap]] = None,
    degenerate_edge_map: Optional[Dict[Tuple[int, EdgeName], float]] = None,
    edge_map_num_samples: int = 41,
    edge_map_atol: float = 1e-6,
    edge_map_rtol: float = 1e-6,
    eps_edge: float = 0.005,
    use_geometric_near_edges: bool = False,
    use_geometric_retry_edges: bool = True,
    near_edge_gap_atol: float = 1e-9,
    near_edge_cell_factor: float = 1.0,
    retry_dist_outlier_ratio: float = 2.0,
    retry_dist_outlier_atol: float = 1e-4,
    local_search_all_points: bool = False,
    include_current_patch_boundary: bool = True,
    include_neighbor_patch: bool = True,
    include_neighbor_boundary: bool = True,
    retry_on_failure: bool = True,
    retry_num_closest_edges: int = 2,
    retry_local_search: bool = True,
    retry_local_step_factor: float = 2.0,
    retry_accept_distance_factor: float = 4.0,
    retry_accept_distance_atol: float = 1e-3,
    retry_normal_dot_min: float = -0.25,
    params: OrthogonalityNewtonParams = OrthogonalityNewtonParams(),
) -> WarmStartCandidateProjectionResult:
    """Project points by trying several warm-started candidates and keeping the best.

    Runs eagerly. For each query point the search assembles candidate seeds:
    the nearest tessellated vertex's own patch, optionally that patch's
    boundary edges, and optionally the neighbouring patch and its edges. Each
    candidate is solved with the Newton routines in
    :mod:`orthogonality_projection_numpy`.

    Candidates are ranked rather than compared on distance alone. A converged
    candidate always outranks a non-converged one. Among converged candidates
    the smaller squared distance wins, with residual as the tie-break; when no
    candidate converged, the smaller residual wins, with squared distance as
    the tie-break.

    For an edge listed in ``degenerate_edge_map``, the one-dimensional edge
    solve is replaced by a single fixed-point candidate, because a collapsed
    edge cannot support a solve along its length.

    A point flagged ``boundary_clamped``, or one that fails outright, is routed
    into the retry path controlled by the ``retry_*`` arguments: additional
    nearby edges, a local multi-scale search, and acceptance thresholds.
    ``boundary_clamped`` triggers a retry independently of ``converged``.

    A retry replaces the selection only if it passes two sequential gates. It
    must first outrank the selection under the same ordering used in the first
    pass: a converged candidate outranks a non-converged one; if both are
    converged the retry must have the strictly smaller ``dist2``, with residual
    breaking only an exact distance tie; if both are non-converged the smaller
    residual wins, with distance as the tie-break. It must then pass the
    compatibility check, which requires a finite distance within the
    factor/absolute cap and, when the retry lands on a different patch, the
    normal-alignment guard.

    The cap therefore only ever permits a farther retry that already won on
    rank — one that converged where the selection did not, or, among two
    non-converged candidates, one with a smaller residual. It does **not**
    authorize a farther retry when both candidates are converged: in that case
    the ranking gate has already rejected it.

    Parameters
    ----------
    function_set
        Function set to project onto. Patch coefficients are read at call time.
    points
        Query points, shape ``(M, physical_dimension)``. Any other rank raises
        ``ValueError``.
    patch_indices
        Patches to project onto, in the order given. ``None`` uses every patch
        in ``function_set``, sorted ascending.
    mesh
        Pre-built tessellation used for the warm start. ``None`` builds one with
        :func:`~bsm3.core.projections.warm_start_projections.build_sampled_patches_mesh`
        at ``warm_start_nu`` by ``warm_start_nv`` samples per patch. Passing a
        mesh avoids rebuilding it on every call.
    warm_start_nu, warm_start_nv
        Per-patch sampling resolution used only when ``mesh`` is ``None``.
        Denser sampling gives better seeds at a higher eager
        tessellation-construction cost, paid on each call that builds the mesh,
        since this function runs when called.
    edge_map
        Patch adjacency from :func:`build_edge_neighbor_map`. ``None`` builds one
        with the ``edge_map_*`` arguments below.
    degenerate_edge_map
        Maps ``(patch_id, edge_name)`` to measured arc length for edges that
        have collapsed. ``None`` disables fixed-point substitution.
    edge_map_num_samples, edge_map_atol, edge_map_rtol
        Sample count and coincidence tolerances forwarded to
        :func:`build_edge_neighbor_map`, used only when ``edge_map`` is ``None``.
    eps_edge
        Parametric half-width of the band that counts as "on an edge". A seed
        within ``eps_edge`` of a bound contributes that edge's candidates. It
        also sets the length scale of the local retry step and of the
        geometric near-edge floor.
    use_geometric_near_edges
        Use the physical, grid-independent near-edge test instead of the
        parametric ``eps_edge`` band when building the *first-pass* candidates.
        Off by default.
    use_geometric_retry_edges
        Use that same physical test when building the *retry* edge candidates.
        On by default.
    near_edge_gap_atol
        Absolute floor added to the projection-gap band in the geometric
        near-edge test, keeping the band non-empty when a point lies essentially
        on the surface.
    near_edge_cell_factor
        Multiplier on the one-sampling-cell floor (``2 * eps_edge`` times the
        local tangent magnitude) in that same test. Larger values admit more
        neighbouring patches.
    retry_dist_outlier_ratio, retry_dist_outlier_atol
        Wrong-basin detector. A point is retried when its Newton distance
        exceeds ``ratio * warm_start_distance + atol``, which catches candidates
        that converged to a spurious interior minimum without being
        boundary-clamped. A ratio of zero or less disables the detector.
    local_search_all_points
        Run the multi-scale local retry on every point rather than only on
        points needing retry. Off by default: it costs a full extra Newton sweep
        and the outlier detector already finds the suspect nodes.
    include_current_patch_boundary
        Include candidates on the seed patch's own boundary edges.
    include_neighbor_patch
        Include a candidate in the neighbouring patch's interior, mapped across
        the shared edge.
    include_neighbor_boundary
        Include a candidate on the neighbouring patch's matching boundary edge.
    retry_on_failure
        Enable the retry pass for points that did not converge, are
        boundary-clamped, or are distance outliers. When ``False`` the
        first-pass selection is final.
    retry_num_closest_edges
        How many of the nearest edges to add candidates for during retry.
    retry_local_search
        Enable the on-patch multi-scale local search within the retry pass.
    retry_local_step_factor
        Multiplier on ``eps_edge`` setting the local search's parametric step.
    retry_accept_distance_factor, retry_accept_distance_atol
        Distance cap applied *after* the retry has already outranked the
        selection: the retry is rejected unless its distance is within
        ``max(factor, 1.0) * selected_distance + max(atol, 0.0)``. Because the
        ranking gate runs first, this cap can only admit a farther retry that
        improved the convergence rank, or that has a smaller residual among two
        non-converged candidates. When both candidates are converged the
        ranking gate already requires the retry to be strictly closer, so the
        cap never lets a farther one through. Its role is to bound how far an
        otherwise-better retry may be.
    retry_normal_dot_min
        Minimum dot product between the selected and retry surface normals when
        the retry lands on a *different* patch, rejecting replacements that flip
        to the opposite side of a thin body. Values at or below ``-1.0`` disable
        the check; it never applies within one patch.
    params
        Newton tolerances forwarded to the per-candidate solves; see
        :class:`~bsm3.core.projections.orthogonality_projection_numpy.OrthogonalityNewtonParams`.

    Returns
    -------
    WarmStartCandidateProjectionResult
        The accepted candidate per point together with its convergence
        evidence and the seed it started from. A point for which no candidate
        converged is returned with its minimum-residual fallback and
        ``converged`` set to ``False`` rather than raising.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must have shape (M, phys_dim); got {points.shape}")

    if patch_indices is None:
        patch_indices = sorted(int(idx) for idx in function_set.functions.keys())
    else:
        patch_indices = [int(idx) for idx in patch_indices]

    if mesh is None:
        mesh = build_sampled_patches_mesh(
            function_set=function_set,
            patch_indices=patch_indices,
            Nu=warm_start_nu,
            Nv=warm_start_nv,
        )

    warm = warm_start_from_triangulation(mesh=mesh, points=points)

    if edge_map is None:
        edge_map = build_edge_neighbor_map(
            function_set=function_set,
            patch_indices=patch_indices,
            num_samples=edge_map_num_samples,
            atol=edge_map_atol,
            rtol=edge_map_rtol,
        )

    near_edges = None
    if use_geometric_near_edges or use_geometric_retry_edges:
        near_edges = _compute_geometric_near_edges(
            function_set,
            warm.patch_id,
            warm.uv0,
            points,
            eps_edge=eps_edge,
            gap_atol=near_edge_gap_atol,
            cell_factor=near_edge_cell_factor,
        )

    crease_knots, span_knots = {}, {}
    for pid in np.unique(warm.patch_id):
        _, degrees, knot_vectors = _get_patch_metadata(function_set, int(pid))
        unique_counts = [np.unique(k, return_counts=True) for k in knot_vectors]
        span_knots[int(pid)] = tuple(k for k, _ in unique_counts)
        crease_knots[int(pid)] = tuple(
            k[(n >= degree) & (k > knots[degree]) & (k < knots[-degree - 1])]
            for (k, n), degree, knots in zip(unique_counts, degrees, knot_vectors)
        )

    specs = _build_candidate_specs(
        patch_id=warm.patch_id,
        uv0=warm.uv0,
        crease_knots=crease_knots,
        span_knots=span_knots,
        edge_map=edge_map,
        degenerate_edge_map=degenerate_edge_map,
        eps_edge=eps_edge,
        include_current_patch_boundary=include_current_patch_boundary,
        include_neighbor_patch=include_neighbor_patch,
        include_neighbor_boundary=include_neighbor_boundary,
        # First pass stays conservative unless explicitly opted in: aggressive
        # cross-patch inclusion here floods the *ungated* min-dist2 selection and
        # can fold cells. The geometric edges are instead fed to the vetted retry.
        near_edges=near_edges if use_geometric_near_edges else None,
    )

    candidate_results = _run_candidate_projections(
        function_set=function_set,
        points=points,
        specs=specs,
        params=params,
    )
    best_candidate = _select_best_candidates(candidate_results, num_points=points.shape[0])

    selected_patch_id = candidate_results["patch_id"][best_candidate]
    selected_uv = candidate_results["uv"][best_candidate]
    selected_projected_points = candidate_results["projected_points"][best_candidate]
    selected_dist2 = candidate_results["dist2"][best_candidate]
    selected_residual = candidate_results["residual"][best_candidate]
    selected_converged = candidate_results["converged"][best_candidate]
    selected_iterations = candidate_results["iterations"][best_candidate]
    selected_boundary_clamped = candidate_results["boundary_clamped"][best_candidate]
    selected_kind = [candidate_results["candidate_kind"][idx] for idx in best_candidate]

    # Retry not only on genuine non-convergence but also on boundary clamping: a
    # point reported converged solely because the active set masked an outward
    # residual at a patch edge wanted to slide across that edge. Feeding it into
    # the retry path lets the neighbour-patch candidates compete. The retry only
    # *replaces* the current selection when a candidate is strictly better and
    # normal-compatible (_is_candidate_better / _is_retry_candidate_compatible),
    # so over-flagging clamped points that truly belong on the edge is harmless.
    # Distance-outlier detector (cheap, O(N), no extra projection): the warm
    # start already found the closest *triangle* distance for every point. If the
    # Newton converged to a point substantially farther than that, it fell into a
    # wrong basin (a spurious interior minimum) even though it "converged" and is
    # not boundary-clamped — these are the residual junction folds. Flagging them
    # by the distance ratio means the (comparatively expensive) multi-scale local
    # retry runs only on the handful of genuinely-suspect nodes, not on all of
    # them, so the triangulation-accelerated projection stays fast on large
    # meshes. The ratio is scale-invariant, so legitimately far-from-surface
    # nodes (where warm and Newton distances are both large) are not flagged.
    selected_dist = np.sqrt(np.maximum(selected_dist2, 0.0))
    warm_dist = np.sqrt(np.maximum(np.asarray(warm.dist2, dtype=float), 0.0))
    if retry_dist_outlier_ratio > 0.0:
        dist_outlier = selected_dist > (
            float(retry_dist_outlier_ratio) * warm_dist + float(retry_dist_outlier_atol)
        )
    else:
        dist_outlier = np.zeros(points.shape[0], dtype=bool)

    needs_retry = (~selected_converged) | selected_boundary_clamped | dist_outlier
    failed_points = np.where(needs_retry)[0]

    # The on-patch multi-scale local retry and the cross-patch edge candidates
    # both run on `failed_points` (non-converged, boundary-clamped, or a distance
    # outlier). `local_search_all_points` is an opt-in escape hatch that widens
    # the local retry to *every* point; it is off by default because it costs an
    # extra multi-scale Newton sweep over the whole mesh and the outlier detector
    # already catches the wrong-basin nodes for a tiny fraction of the work.
    if local_search_all_points:
        local_points = np.arange(points.shape[0], dtype=int)
    else:
        local_points = failed_points

    run_retry = retry_on_failure and (
        failed_points.size > 0 or (retry_local_search and local_points.size > 0)
    )
    if run_retry:
        retry_specs: List[_CandidateSpec] = []
        if retry_local_search and local_points.size > 0:
            retry_specs.extend(
                _build_local_retry_candidate_specs(
                    selected_patch_id,
                    selected_uv,
                    local_points,
                    uv_step=max(float(retry_local_step_factor), 0.0) * max(float(eps_edge), 1e-12),
                )
            )
        if failed_points.size > 0:
            retry_specs.extend(
                _build_retry_candidate_specs(
                    warm_patch_id=warm.patch_id,
                    warm_uv0=warm.uv0,
                    point_indices=failed_points,
                    edge_map=edge_map,
                    degenerate_edge_map=degenerate_edge_map,
                    retry_num_closest_edges=retry_num_closest_edges,
                    include_current_patch_boundary=include_current_patch_boundary,
                    include_neighbor_patch=include_neighbor_patch,
                    include_neighbor_boundary=include_neighbor_boundary,
                    near_edges=near_edges if use_geometric_retry_edges else None,
                )
            )

        if retry_specs:
            retry_results = _run_candidate_projections(
                function_set=function_set,
                points=points,
                specs=retry_specs,
                params=params,
            )
            retry_best = _select_best_candidates(retry_results, num_points=points.shape[0])

            retry_points = (
                np.union1d(failed_points, local_points)
                if local_search_all_points
                else failed_points
            )
            for point_index in retry_points:
                retry_index = retry_best[point_index]
                if retry_index < 0:
                    continue

                if _is_candidate_better(
                    converged=bool(retry_results["converged"][retry_index]),
                    residual=float(retry_results["residual"][retry_index]),
                    dist2=float(retry_results["dist2"][retry_index]),
                    best_converged=bool(selected_converged[point_index]),
                    best_residual=float(selected_residual[point_index]),
                    best_dist2=float(selected_dist2[point_index]),
                ) and _is_retry_candidate_compatible(
                    function_set,
                    selected_patch_id=int(selected_patch_id[point_index]),
                    selected_uv=selected_uv[point_index],
                    selected_dist2=float(selected_dist2[point_index]),
                    retry_patch_id=int(retry_results["patch_id"][retry_index]),
                    retry_uv=retry_results["uv"][retry_index],
                    retry_dist2=float(retry_results["dist2"][retry_index]),
                    retry_accept_distance_factor=retry_accept_distance_factor,
                    retry_accept_distance_atol=retry_accept_distance_atol,
                    retry_normal_dot_min=retry_normal_dot_min,
                ):
                    selected_patch_id[point_index] = retry_results["patch_id"][retry_index]
                    selected_uv[point_index] = retry_results["uv"][retry_index]
                    selected_projected_points[point_index] = retry_results["projected_points"][retry_index]
                    selected_dist2[point_index] = retry_results["dist2"][retry_index]
                    selected_residual[point_index] = retry_results["residual"][retry_index]
                    selected_converged[point_index] = retry_results["converged"][retry_index]
                    selected_iterations[point_index] = retry_results["iterations"][retry_index]
                    selected_kind[point_index] = retry_results["candidate_kind"][retry_index]

    return WarmStartCandidateProjectionResult(
        patch_id=selected_patch_id,
        uv=selected_uv,
        projected_points=selected_projected_points,
        dist2=selected_dist2,
        residual=selected_residual,
        converged=selected_converged,
        iterations=selected_iterations,
        candidate_kind=selected_kind,
        warm_patch_id=np.asarray(warm.patch_id, dtype=int),
        warm_uv0=np.asarray(warm.uv0, dtype=float),
        edge_map=edge_map,
    )
