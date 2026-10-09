"""Reference-configuration stiffness assembly for mesh-motion propagation.

This module builds the linear system that the physics-based propagator solves
in place of the RBF field.  Milestone 1 uses an isotropic
displacement-difference graph energy

    Phi(u) = 1/2 * sum_{(i,j) in E} w_ij * || u_i - u_j ||^2 ,

whose stationarity under a prescribed-displacement (Dirichlet) split gives, per
Cartesian component,

    L_ff u_f = - L_fp u_p ,

with ``L`` the weighted graph Laplacian of the *reference* mesh.  The three
components decouple and share one factorization of ``L_ff`` (the SPD free-free
block); ``L_fp`` is a constant sparse coupling matrix consumed downstream by
``csdl.sparse.matmat``.

Assembly is setup-time NumPy/scipy: connectivity, edge weights, and the
factorization are constants in the CSDL graph (only ``u_p`` carries gradients).
The free/prescribed split is arbitrary vertex sets, so alternate configurations
(e.g. strut-adjacent bands) reuse the same assembler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from gamma_mdo.preprocessing.mesh_io import _as_mesh_data

from .spd_solve_custom_op import SPDFactor, factorize_spd


@dataclass(frozen=True)
class AssembledSystem:
    """Factored free-free block and constant free-prescribed coupling.

    ``free_ids`` and ``prescribed_ids`` fix the row/column ordering of the
    solve: ``u_f`` rows align with ``free_ids`` and ``u_p`` rows align with
    ``prescribed_ids``.  ``factor`` solves ``L_ff x = rhs`` (and, by symmetry,
    its adjoint); ``coupling`` is ``L_fp`` as a ``(n_free, n_prescribed)``
    sparse matrix.

    Parameters
    ----------
    free_ids
        Global vertex IDs aligned with free-system rows.
    prescribed_ids
        Global vertex IDs aligned with prescribed-system columns.
    factor
        Cached factorization of the free-free block.
    coupling
        Sparse free-prescribed block.
    """

    free_ids: np.ndarray
    prescribed_ids: np.ndarray
    factor: SPDFactor
    coupling: sp.spmatrix

    @property
    def num_free(self) -> int:
        """Return the number of free vertices.

        Returns
        -------
        int
            Number of free-system rows.
        """
        return int(self.free_ids.size)

    @property
    def num_prescribed(self) -> int:
        """Return the number of prescribed vertices.

        Returns
        -------
        int
            Number of prescribed-system columns.
        """
        return int(self.prescribed_ids.size)


@dataclass(frozen=True)

class StiffnessAssembler(Protocol):
    """Build a factored SPD free-free block and prescribed coupling.

    Notes
    -----
    Implementations preserve the supplied free and prescribed row order.
    """

    def assemble(
        self,
        mesh,
        *,
        free_ids: np.ndarray,
        prescribed_ids: np.ndarray,
    ) -> AssembledSystem:
        """Assemble and factor a partitioned stiffness system.

        Parameters
        ----------
        mesh
            Reference surface mesh.
        free_ids
            Unknown vertex IDs.
        prescribed_ids
            Dirichlet vertex IDs.

        Returns
        -------
        AssembledSystem
            Factored free-free block and free-prescribed coupling.
        """
        ...


class GraphLaplacianAssembler:
    """Displacement-difference stiffness: decoupled scalar graph Laplacian.

    Edge weights follow ``w_ij = sum_{f in faces(i,j)} A_f^{-chi}`` with
    ``A_f`` the reference face area.  ``chi = 0`` recovers uniform weights
    (``w_ij = 1`` per graph edge) -- the pure graph Laplacian / harmonic map,
    which is Milestone 1's validated default.  Positive ``chi`` stiffens small
    cells so they resist collapse under large compression.

    When ``quad_diagonal_weight`` is positive, ``quad_bracing_mode`` selects
    one quality-chosen diagonal, both diagonals, or a four-spoke virtual center.
    The center is eliminated analytically, so every mode affects only the
    deformation operator; none alters the aerodynamic mesh connectivity.

    Parameters
    ----------
    stiffening_exponent
        Nonnegative inverse-area exponent.
    area_floor
        Positive lower bound used in area-based weights.
    distance_weighting
        Optional fixed reference-geodesic edge multipliers.
    quad_diagonal_weight
        Nonnegative weight for auxiliary quadrilateral bracing.
    quad_bracing_mode
        Auxiliary bracing topology.
    """

    def __init__(
        self,
        *,
        stiffening_exponent: float = 0.0,
        area_floor: float = 1e-12,
        distance_weighting=None,
        quad_diagonal_weight: float = 0.0,
        quad_bracing_mode: str = "both_diagonals",
    ):
        if float(stiffening_exponent) < 0.0:
            raise ValueError("stiffening_exponent (chi) must be non-negative.")
        if float(area_floor) <= 0.0:
            raise ValueError("area_floor must be positive.")
        if (
            not np.isfinite(float(quad_diagonal_weight))
            or float(quad_diagonal_weight) < 0.0
        ):
            raise ValueError("quad_diagonal_weight must be finite and non-negative.")
        _validate_quad_bracing_mode(quad_bracing_mode)
        self.stiffening_exponent = float(stiffening_exponent)
        self.area_floor = float(area_floor)
        self.quad_diagonal_weight = float(quad_diagonal_weight)
        self.quad_bracing_mode = str(quad_bracing_mode)
        # Optional fixed reference-geodesic distance multiplier applied to every
        # edge weight.  ``None`` (or ``beta = 0``) is the exact no-op baseline.
        self.distance_weighting = distance_weighting

    def assemble(
        self,
        mesh,
        *,
        free_ids: np.ndarray,
        prescribed_ids: np.ndarray,
    ) -> AssembledSystem:
        """Assemble the partitioned graph-Laplacian system.

        Parameters
        ----------
        mesh
            Reference surface mesh.
        free_ids
            Unknown vertex IDs.
        prescribed_ids
            Dirichlet vertex IDs containing every exterior graph neighbor.

        Returns
        -------
        AssembledSystem
            Factored free-free Laplacian and free-prescribed block.

        Raises
        ------
        ValueError
            If partitions overlap, contain duplicates, or leave a graph
            neighbor unclassified.
        """
        mesh_data = _as_mesh_data(mesh)
        points = np.asarray(mesh_data.vertices, dtype=float).reshape((-1, 3))

        free_ids = np.asarray(free_ids, dtype=np.int64).reshape(-1)
        prescribed_ids = np.asarray(prescribed_ids, dtype=np.int64).reshape(-1)
        if np.intersect1d(free_ids, prescribed_ids).size:
            raise ValueError("free_ids and prescribed_ids must be disjoint.")
        if np.unique(free_ids).size != free_ids.size:
            raise ValueError("free_ids must not contain duplicates.")
        if np.unique(prescribed_ids).size != prescribed_ids.size:
            raise ValueError("prescribed_ids must not contain duplicates.")

        edge_weights = _edge_weights(
            mesh_data,
            points,
            stiffening_exponent=self.stiffening_exponent,
            area_floor=self.area_floor,
            quad_diagonal_weight=self.quad_diagonal_weight,
            quad_bracing_mode=self.quad_bracing_mode,
        )
        if self.distance_weighting is not None:
            # Multiply each reference edge weight by its fixed distance factor.
            # Distances are setup constants, so this preserves symmetry, keeps
            # every weight positive, and leaves the free-free block SPD.
            multipliers = self.distance_weighting.edge_multiplier_map(
                edge_weights.keys()
            )
            edge_weights = {
                key: weight * multipliers[key]
                for key, weight in edge_weights.items()
            }

        free_local = {int(vertex): row for row, vertex in enumerate(free_ids)}
        prescribed_local = {
            int(vertex): row for row, vertex in enumerate(prescribed_ids)
        }
        free_set = set(free_local)
        prescribed_set = set(prescribed_local)

        adjacency: dict[int, dict[int, float]] = {}
        for (vertex_a, vertex_b), weight in edge_weights.items():
            adjacency.setdefault(vertex_a, {})[vertex_b] = weight
            adjacency.setdefault(vertex_b, {})[vertex_a] = weight

        ff_rows: list[int] = []
        ff_cols: list[int] = []
        ff_values: list[float] = []
        fp_rows: list[int] = []
        fp_cols: list[int] = []
        fp_values: list[float] = []
        for global_free, local_free in free_local.items():
            diagonal = 0.0
            for neighbor, weight in adjacency.get(global_free, {}).items():
                diagonal += weight
                if neighbor in free_set:
                    ff_rows.append(local_free)
                    ff_cols.append(free_local[neighbor])
                    ff_values.append(-weight)
                elif neighbor in prescribed_set:
                    fp_rows.append(local_free)
                    fp_cols.append(prescribed_local[neighbor])
                    fp_values.append(-weight)
                else:
                    raise ValueError(
                        "Free vertex "
                        f"{global_free} has neighbor {neighbor} that is neither "
                        "free nor prescribed; the prescribed set must contain "
                        "every graph neighbor of the free set (no stiffness leak)."
                    )
            ff_rows.append(local_free)
            ff_cols.append(local_free)
            ff_values.append(diagonal)

        num_free = free_ids.size
        num_prescribed = prescribed_ids.size
        l_ff = sp.csc_matrix(
            (ff_values, (ff_rows, ff_cols)),
            shape=(num_free, num_free),
        )
        l_fp = sp.csr_matrix(
            (fp_values, (fp_rows, fp_cols)),
            shape=(num_free, num_prescribed),
        )
        return AssembledSystem(
            free_ids=free_ids,
            prescribed_ids=prescribed_ids,
            factor=factorize_spd(l_ff),
            coupling=l_fp,
        )


def graph_neighbors(
    mesh,
    vertex_ids: np.ndarray,
    *,
    include_quad_diagonals: bool = False,
    quad_bracing_mode: str | None = None,
) -> np.ndarray:
    """Return the mesh-graph neighbors of ``vertex_ids`` not in the set.

    Neighbors are defined by the ring edges of the triangle/quad cells.
    ``include_quad_diagonals`` retains the legacy request for both diagonals.
    ``quad_bracing_mode`` can instead select the single-diagonal,
    both-diagonal, or statically condensed virtual-center operator graph.  This
    is how the elastic prescribed set is grown from the free set: every
    stiffness neighbor of a free vertex must be prescribed so the free-free
    block has no stiffness leak.

    Parameters
    ----------
    mesh
        Reference surface mesh.
    vertex_ids
        Source vertices whose exterior neighbors are requested.
    include_quad_diagonals
        Legacy switch that requests both quadrilateral diagonals.
    quad_bracing_mode
        Optional explicit quadrilateral bracing topology.

    Returns
    -------
    numpy.ndarray
        Sorted unique neighboring vertex IDs outside ``vertex_ids``.
    """
    mesh_data = _as_mesh_data(mesh)
    if quad_bracing_mode is not None:
        _validate_quad_bracing_mode(quad_bracing_mode)
    elif include_quad_diagonals:
        quad_bracing_mode = "both_diagonals"
    points = np.asarray(mesh_data.vertices, dtype=float)
    source = set(int(vertex) for vertex in np.asarray(vertex_ids, dtype=np.int64).reshape(-1))
    neighbors: set[int] = set()
    for block in mesh_data.cell_blocks.values():
        cells = np.asarray(block, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] < 2:
            continue
        for cell in cells:
            ring = cell.tolist()
            count = len(ring)
            edges = [
                (ring[local_index], ring[(local_index + 1) % count])
                for local_index in range(count)
            ]
            if quad_bracing_mode is not None and count == 4:
                edges.extend(
                    (vertex_a, vertex_b)
                    for vertex_a, vertex_b, _ in _quad_brace_pairs(
                        points,
                        np.asarray(cell, dtype=np.int64),
                        mode=quad_bracing_mode,
                        weight=1.0,
                    )
                )
            for raw_a, raw_b in edges:
                vertex_a = int(raw_a)
                vertex_b = int(raw_b)
                if vertex_a == vertex_b:
                    continue
                if vertex_a in source and vertex_b not in source:
                    neighbors.add(vertex_b)
                elif vertex_b in source and vertex_a not in source:
                    neighbors.add(vertex_a)
    return np.array(sorted(neighbors), dtype=np.int64)


def element_neighbors(mesh, vertex_ids: np.ndarray) -> np.ndarray:
    """Return all co-element neighbors outside ``vertex_ids``.

    CST polygon integration produces coupling between every boundary vertex of
    a condensed polygon.  Its Dirichlet support therefore needs the complete
    one-element halo, not only the ring-edge halo used by the graph Laplacian.

    Parameters
    ----------
    mesh
        Reference surface mesh.
    vertex_ids
        Source vertices whose co-element halo is requested.

    Returns
    -------
    numpy.ndarray
        Sorted unique co-element vertex IDs outside ``vertex_ids``.
    """
    mesh_data = _as_mesh_data(mesh)
    source = set(
        int(vertex)
        for vertex in np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    )
    neighbors: set[int] = set()
    for block in mesh_data.cell_blocks.values():
        cells = np.asarray(block, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] < 3:
            continue
        for cell in cells:
            ring = [int(vertex) for vertex in cell]
            if any(vertex in source for vertex in ring):
                neighbors.update(vertex for vertex in ring if vertex not in source)
    return np.asarray(sorted(neighbors), dtype=np.int64)


def _edge_weights(
    mesh_data,
    points: np.ndarray,
    *,
    stiffening_exponent: float,
    area_floor: float,
    quad_diagonal_weight: float = 0.0,
    quad_bracing_mode: str = "both_diagonals",
) -> dict[tuple[int, int], float]:
    """Reference-mesh edge weights keyed by ordered vertex pairs.

    Physical edges are the ring segments of every polygon.  For ``chi = 0``
    each physical edge weight is 1; for ``chi > 0`` each physical edge
    accumulates ``A_f^{-chi}`` from its incident faces.  With
    ``quad_diagonal_weight = lambda``, quads additionally contribute the
    selected auxiliary bracing operator.
    """
    uniform = stiffening_exponent == 0.0
    physical_weights: dict[tuple[int, int], float] = {}
    diagonal_weights: dict[tuple[int, int], float] = {}
    for block in mesh_data.cell_blocks.values():
        cells = np.asarray(block, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] < 2:
            continue
        for cell in cells:
            if uniform:
                contribution = 1.0
            else:
                area = max(_polygon_area(points[cell]), area_floor)
                contribution = area ** (-stiffening_exponent)
            ring = cell.tolist()
            count = len(ring)
            for local_index in range(count):
                vertex_a = int(ring[local_index])
                vertex_b = int(ring[(local_index + 1) % count])
                if vertex_a == vertex_b:
                    continue
                key = (vertex_a, vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)
                if uniform:
                    physical_weights[key] = 1.0
                else:
                    physical_weights[key] = (
                        physical_weights.get(key, 0.0) + contribution
                    )
            if len(ring) == 4 and quad_diagonal_weight > 0.0:
                for vertex_a, vertex_b, brace_scale in _quad_brace_pairs(
                    points,
                    np.asarray(cell, dtype=np.int64),
                    mode=quad_bracing_mode,
                    weight=quad_diagonal_weight,
                ):
                    if vertex_a == vertex_b:
                        continue
                    key = (
                        (vertex_a, vertex_b)
                        if vertex_a < vertex_b
                        else (vertex_b, vertex_a)
                    )
                    diagonal_weights[key] = (
                        diagonal_weights.get(key, 0.0)
                        + brace_scale * contribution
                    )
    weights = physical_weights.copy()
    for key, contribution in diagonal_weights.items():
        weights[key] = weights.get(key, 0.0) + contribution
    return weights


_QUAD_BRACING_MODES = (
    "single_diagonal",
    "both_diagonals",
    "virtual_center",
)


def _validate_quad_bracing_mode(mode: str) -> None:
    if str(mode) not in _QUAD_BRACING_MODES:
        choices = ", ".join(_QUAD_BRACING_MODES)
        raise ValueError(f"quad_bracing_mode must be one of: {choices}.")


def _quad_brace_pairs(
    points: np.ndarray,
    cell: np.ndarray,
    *,
    mode: str,
    weight: float,
) -> tuple[tuple[int, int, float], ...]:
    """Return condensed corner-pair contributions for one quad.

    ``single_diagonal`` selects the baseline diagonal that maximizes the worse
    mean-ratio quality of its two triangles.  ``both_diagonals`` contributes
    both crossing diagonals.  ``virtual_center`` represents four equal spokes
    of weight ``2 * weight`` to a free center node.  Eliminating that center
    exactly gives all six corner pairs weight ``weight / 2``.
    """
    _validate_quad_bracing_mode(mode)
    vertices = tuple(int(vertex) for vertex in np.asarray(cell).reshape(-1))
    if len(vertices) != 4 or weight == 0.0:
        return ()
    if mode == "single_diagonal":
        option_02 = min(
            _triangle_mean_ratio(points[[vertices[0], vertices[1], vertices[2]]]),
            _triangle_mean_ratio(points[[vertices[0], vertices[2], vertices[3]]]),
        )
        option_13 = min(
            _triangle_mean_ratio(points[[vertices[0], vertices[1], vertices[3]]]),
            _triangle_mean_ratio(points[[vertices[1], vertices[2], vertices[3]]]),
        )
        diagonal = (
            (vertices[0], vertices[2])
            if option_02 >= option_13
            else (vertices[1], vertices[3])
        )
        return ((diagonal[0], diagonal[1], 2.0 * weight),)
    if mode == "both_diagonals":
        return (
            (vertices[0], vertices[2], 2.0 * weight),
            (vertices[1], vertices[3], 2.0 * weight),
        )
    pair_scale = 0.5 * weight
    return tuple(
        (vertices[left], vertices[right], pair_scale)
        for left in range(4)
        for right in range(left + 1, 4)
    )


def _triangle_mean_ratio(triangle: np.ndarray) -> float:
    triangle = np.asarray(triangle, dtype=float).reshape((3, 3))
    edges = np.roll(triangle, -1, axis=0) - triangle
    squared_lengths = np.einsum("ij,ij->i", edges, edges)
    denominator = float(np.sum(squared_lengths))
    if denominator <= 1e-30:
        return 0.0
    twice_area = float(np.linalg.norm(np.cross(edges[0], -edges[2])))
    return 2.0 * np.sqrt(3.0) * twice_area / denominator


def _polygon_area(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon, dtype=float).reshape((-1, 3))
    normal = np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
    return 0.5 * float(np.linalg.norm(normal))


__all__ = [
    "AssembledSystem",
    "GraphLaplacianAssembler",
    "StiffnessAssembler",
    "element_neighbors",
    "graph_neighbors",
]
