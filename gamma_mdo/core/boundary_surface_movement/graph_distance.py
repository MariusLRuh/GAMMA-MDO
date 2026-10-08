"""Fixed reference-geodesic distance weighting for the surface graph solve.

Even after all surface inversions are removed, the largest fuselage strain stays
localized immediately aft of the wing trailing edge.  The inverse-area exponent
alone cannot control the *blending length* of that transition.  This module adds
an optional, purely setup-time secondary edge weight based on the reference-mesh
geodesic (physical-edge-length) graph distance from the moving intersection
seams:

    w_ij = w_ij^A * [1 + beta * g(d_ij)] ,   d_ij = (d_i + d_j) / 2 ,

with ``g(d) = exp(-d / L)`` (default) or a bounded rational ``(1 + d/L)^(-p)``.
``d_i`` is the multi-source shortest-path distance from the enabled seam seeds.

The distance field, the decay parameters, and therefore every edge multiplier
are *reference/setup constants*: they never depend on the current displacement,
so there is no design-dependent branch and the multiplier carries no derivative.
``beta = 0`` yields a multiplier of exactly ``1`` on every edge and reproduces
the plain inverse-area weighting bit-for-bit.

The multiplier is bounded and symmetric by construction:

* ``g`` decays monotonically from ``g(0) = 1`` to ``g(inf) = 0`` so the
  far-field multiplier is exactly ``1`` (already normalized);
* ``1 + beta*g`` is capped at ``cap >= 1``;
* endpoints enter symmetrically through ``(d_i + d_j) / 2``.

Unreachable vertices (outside the restricted band) have distance ``+inf``, so
``g`` is ``0`` and their edges keep multiplier ``1`` -- no stiffness leak and the
graph matrix stays SPD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from gamma_mdo.preprocessing.mesh_io import _as_mesh_data


class GraphDistanceSummary(TypedDict):
    """Setup-time statistics returned by :meth:`GraphDistanceWeighting.summary`.

    Every value is a plain Python scalar, so the summary is directly
    JSON-serializable. The decay *kind* is intentionally not included: it is a
    configuration choice rather than a measured statistic, and callers read it
    from the configuration object that built the weighting.

    Attributes
    ----------
    beta
        Secondary-weight strength actually in force. ``0.0`` makes every edge
        multiplier exactly one.
    length
        Decay length scale ``L`` in model length units.
    cap
        Upper bound applied to each edge multiplier.
    num_reachable_vertices
        Number of vertices with a finite distance from the enabled seam seeds.
        Vertices in a disconnected component stay infinite and are excluded.
    max_finite_distance
        Largest finite seed distance, or ``0.0`` when no vertex is reachable.
    multiplier_min, multiplier_median, multiplier_max
        Minimum, median, and maximum edge multiplier over the queried edges.
    """

    beta: float
    length: float
    cap: float
    num_reachable_vertices: int
    max_finite_distance: float
    multiplier_min: float
    multiplier_median: float
    multiplier_max: float


@dataclass(frozen=True)
class GraphDistanceWeighting:
    """Immutable per-vertex geodesic distance and its bounded edge multiplier.

    ``vertex_distance`` is ``+inf`` where a vertex is unreachable from the seeds
    over the restricted band graph.  ``edge_multipliers`` maps any array of
    ordered/unordered vertex-index pairs to the fixed, positive, symmetric
    multiplier ``min(1 + beta*g((d_a + d_b)/2), cap)``.

    Parameters
    ----------
    vertex_distance
        Reference geodesic distance from the nearest seed per vertex.
    beta
        Nonnegative multiplier amplitude.
    length
        Positive physical decay length.
    cap
        Upper bound, no smaller than one.
    decay
        ``"exp"`` or ``"rational"`` decay law.
    rational_power
        Positive power for rational decay.
    seed_ids
        Optional source-vertex IDs retained for diagnostics.
    """

    vertex_distance: np.ndarray
    beta: float
    length: float
    cap: float
    decay: str = "exp"
    rational_power: float = 1.0
    seed_ids: np.ndarray | None = None

    def __post_init__(self):
        """Validate the decay parameters.

        Raises
        ------
        ValueError
            If ``beta`` is negative, ``length`` is not positive, or another
            decay setting is outside its permitted range.
        """
        if float(self.beta) < 0.0:
            raise ValueError("beta must be non-negative.")
        if float(self.length) <= 0.0:
            raise ValueError("length must be positive.")
        if float(self.cap) < 1.0:
            raise ValueError("cap must be at least 1 (multipliers never fall below 1).")
        if self.decay not in ("exp", "rational"):
            raise ValueError("decay must be 'exp' or 'rational'.")
        if self.decay == "rational" and float(self.rational_power) <= 0.0:
            raise ValueError("rational_power must be positive.")

    def _decay(self, distance: np.ndarray) -> np.ndarray:
        distance = np.asarray(distance, dtype=float)
        if self.decay == "exp":
            # exp(-inf) == 0, so unreachable vertices decay to the far-field 1.
            return np.exp(-distance / float(self.length))
        # (1 + d/L)^(-p); (1 + inf)^(-p) == 0 with the same far-field limit.
        return np.power(1.0 + distance / float(self.length), -float(self.rational_power))

    def edge_multipliers(self, edge_vertices: np.ndarray) -> np.ndarray:
        """Compute the fixed multiplier for each supplied edge.

        Parameters
        ----------
        edge_vertices
            Integer endpoint array with shape ``(num_edges, 2)``.

        Returns
        -------
        numpy.ndarray
            Positive symmetric multipliers in the interval ``[1, cap]``.
        """
        edges = np.asarray(edge_vertices, dtype=np.int64).reshape((-1, 2))
        endpoint_distance = 0.5 * (
            self.vertex_distance[edges[:, 0]] + self.vertex_distance[edges[:, 1]]
        )
        multiplier = 1.0 + float(self.beta) * self._decay(endpoint_distance)
        capped = np.minimum(multiplier, float(self.cap))
        # A defensive floor keeps the SPD guarantee even under rounding.
        return np.maximum(capped, 1.0)

    def edge_multiplier_map(self, edge_keys) -> dict[tuple[int, int], float]:
        """Map vertex-pair keys to fixed edge multipliers.

        Parameters
        ----------
        edge_keys
            Sequence of two-vertex edge keys.

        Returns
        -------
        dict[tuple[int, int], float]
            Input keys paired with their computed multipliers.
        """
        keys = list(edge_keys)
        if not keys:
            return {}
        multipliers = self.edge_multipliers(np.asarray(keys, dtype=np.int64))
        return {tuple(int(v) for v in key): float(m) for key, m in zip(keys, multipliers)}

    def summary(self, edge_vertices: np.ndarray) -> GraphDistanceSummary:
        """Summarize distance multipliers over a set of edges.

        Parameters
        ----------
        edge_vertices
            Integer endpoint array with shape ``(num_edges, 2)``.

        Returns
        -------
        GraphDistanceSummary
            Configuration scalars together with reachable-distance and
            multiplier statistics. The decay *kind* is deliberately absent; read
            it from the configuration that built this weighting.
        """
        multipliers = self.edge_multipliers(edge_vertices)
        finite = self.vertex_distance[np.isfinite(self.vertex_distance)]
        return {
            "beta": float(self.beta),
            "length": float(self.length),
            "cap": float(self.cap),
            "num_reachable_vertices": int(finite.size),
            "max_finite_distance": float(np.max(finite)) if finite.size else 0.0,
            "multiplier_min": float(np.min(multipliers)),
            "multiplier_median": float(np.median(multipliers)),
            "multiplier_max": float(np.max(multipliers)),
        }


def compute_multisource_geodesic_distance(
    mesh,
    seed_ids: np.ndarray,
    *,
    restrict_vertex_ids: np.ndarray | None = None,
    edge_length_floor: float = 1e-12,
) -> np.ndarray:
    """Physical-edge-length multi-source Dijkstra on the reference mesh graph.

    Edges are the ring segments of every triangle/quad cell (the same graph the
    stiffness assembler uses).  When ``restrict_vertex_ids`` is given, only edges
    with *both* endpoints inside that band are kept, so shortest paths cannot
    shortcut through unrelated components; vertices outside the band remain at
    ``+inf``.  Distances use physical edge lengths because the CFD mesh is
    strongly graded.

    Parameters
    ----------
    mesh
        Reference surface mesh.
    seed_ids
        Source vertices for the multi-source shortest paths.
    restrict_vertex_ids
        Optional band within which both edge endpoints must lie.
    edge_length_floor
        Positive lower bound applied to physical edge lengths.

    Returns
    -------
    numpy.ndarray
        Minimum geodesic distance per mesh vertex; unreachable entries are
        infinite.

    Raises
    ------
    ValueError
        If seed IDs, restricted IDs, or the edge-length floor are invalid.
    """
    mesh_data = _as_mesh_data(mesh)
    points = np.asarray(mesh_data.vertices, dtype=float).reshape((-1, 3))
    num_vertices = points.shape[0]

    seeds = np.unique(np.asarray(seed_ids, dtype=np.int64).reshape(-1))
    if seeds.size == 0:
        raise ValueError("At least one distance seed is required.")
    if np.any(seeds < 0) or np.any(seeds >= num_vertices):
        raise ValueError("seed_ids contains an out-of-range vertex.")

    if restrict_vertex_ids is None:
        allowed = None
    else:
        allowed = np.zeros(num_vertices, dtype=bool)
        band = np.asarray(restrict_vertex_ids, dtype=np.int64).reshape(-1)
        if band.size and (np.any(band < 0) or np.any(band >= num_vertices)):
            raise ValueError("restrict_vertex_ids contains an out-of-range vertex.")
        allowed[band] = True

    length_floor = float(edge_length_floor)
    if not np.isfinite(length_floor) or length_floor <= 0.0:
        raise ValueError("edge_length_floor must be positive.")

    edge_length: dict[tuple[int, int], float] = {}
    for block in mesh_data.cell_blocks.values():
        cells = np.asarray(block, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] < 2:
            continue
        for cell in cells:
            count = cell.size
            for local_index in range(count):
                vertex_a = int(cell[local_index])
                vertex_b = int(cell[(local_index + 1) % count])
                if vertex_a == vertex_b:
                    continue
                if allowed is not None and not (allowed[vertex_a] and allowed[vertex_b]):
                    continue
                key = (vertex_a, vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)
                if key in edge_length:
                    continue
                edge_length[key] = max(
                    float(np.linalg.norm(points[vertex_a] - points[vertex_b])),
                    length_floor,
                )

    if not edge_length:
        return np.full(num_vertices, np.inf, dtype=float)

    keys = np.asarray(list(edge_length.keys()), dtype=np.int64)
    values = np.asarray(list(edge_length.values()), dtype=float)
    graph = sp.csr_matrix(
        (
            np.concatenate((values, values)),
            (
                np.concatenate((keys[:, 0], keys[:, 1])),
                np.concatenate((keys[:, 1], keys[:, 0])),
            ),
        ),
        shape=(num_vertices, num_vertices),
    )
    # ``min_only`` returns the per-vertex minimum over all seeds directly, which
    # avoids materializing a dense ``(num_seeds, num_vertices)`` distance matrix
    # on the strongly graded CFD mesh.
    return dijkstra(graph, directed=False, indices=seeds, min_only=True)


def build_graph_distance_weighting(
    mesh,
    seed_ids: np.ndarray,
    *,
    beta: float,
    length: float,
    cap: float = np.inf,
    decay: str = "exp",
    rational_power: float = 1.0,
    restrict_vertex_ids: np.ndarray | None = None,
) -> GraphDistanceWeighting:
    """Assemble fixed reference-geodesic distance weighting.

    Parameters
    ----------
    mesh
        Reference surface mesh.
    seed_ids
        Source vertices for the geodesic distance.
    beta
        Nonnegative multiplier amplitude.
    length
        Positive physical decay length.
    cap
        Upper multiplier bound.
    decay
        ``"exp"`` or ``"rational"`` decay law.
    rational_power
        Power used by rational decay.
    restrict_vertex_ids
        Optional vertex band used to restrict shortest paths.

    Returns
    -------
    GraphDistanceWeighting
        Immutable distances and multiplier configuration.
    """
    distance = compute_multisource_geodesic_distance(
        mesh,
        seed_ids,
        restrict_vertex_ids=restrict_vertex_ids,
    )
    return GraphDistanceWeighting(
        vertex_distance=distance,
        beta=float(beta),
        length=float(length),
        cap=float(cap),
        decay=decay,
        rational_power=float(rational_power),
        seed_ids=np.unique(np.asarray(seed_ids, dtype=np.int64).reshape(-1)),
    )


__all__ = [
    "GraphDistanceWeighting",
    "build_graph_distance_weighting",
    "compute_multisource_geodesic_distance",
]
