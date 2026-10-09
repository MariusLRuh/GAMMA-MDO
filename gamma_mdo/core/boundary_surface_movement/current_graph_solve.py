"""State-dependent graph-Laplacian solve and analytic implicit VJP.

Unlike :mod:`spd_solve_custom_op`, the matrix here is rebuilt from the current
projected mesh.  For inverse-area weights this makes a fixed-count load path
genuinely nonlinear:

``L_ff(x) u_f = -L_fp(x) u_p``.

The VJP differentiates the converged linear equilibrium, including the
dependence of every face-area weight on ``x``.  Connectivity and the
free/prescribed split remain fixed setup-time data.
"""

from __future__ import annotations

import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sp

from gamma_mdo.preprocessing.mesh_io import _as_mesh_data

from .elasticity import _quad_brace_pairs, _validate_quad_bracing_mode
from .spd_solve_custom_op import factorize_spd


class CurrentGraphModel:
    """Solve a fixed graph whose weights follow current polygon areas.

    Parameters
    ----------
    mesh
        Baseline mesh defining fixed cells and graph topology.
    free_ids
        Global unknown vertex IDs.
    prescribed_ids
        Global Dirichlet vertex IDs.
    stiffening_exponent
        Nonnegative inverse-current-area exponent.
    area_floor
        Positive lower bound used in area weights.
    distance_weighting
        Optional fixed reference-geodesic edge multipliers.
    quad_diagonal_weight
        Nonnegative weight for auxiliary quadrilateral bracing.
    quad_bracing_mode
        Auxiliary bracing topology.

    Raises
    ------
    ValueError
        If weights or partitions are invalid or graph vertices are
        unclassified.
    """

    def __init__(
        self,
        mesh,
        *,
        free_ids,
        prescribed_ids,
        stiffening_exponent: float,
        area_floor: float = 1e-12,
        distance_weighting=None,
        quad_diagonal_weight: float = 0.0,
        quad_bracing_mode: str = "both_diagonals",
    ):
        mesh_data = _as_mesh_data(mesh)
        baseline_points = np.asarray(mesh_data.vertices, dtype=float)
        self.num_vertices = int(mesh_data.vertices.shape[0])
        self.free_ids = np.asarray(free_ids, dtype=np.int64).reshape(-1)
        self.prescribed_ids = np.asarray(
            prescribed_ids, dtype=np.int64
        ).reshape(-1)
        self.stiffening_exponent = float(stiffening_exponent)
        self.area_floor = float(area_floor)
        self.quad_diagonal_weight = float(quad_diagonal_weight)
        self.quad_bracing_mode = str(quad_bracing_mode)
        # Optional fixed reference-geodesic distance multiplier.  It is a
        # setup-time constant per edge, so the current-area VJP only scales the
        # existing area derivative by it; the distance itself has no derivative.
        self.distance_weighting = distance_weighting
        if self.stiffening_exponent < 0.0:
            raise ValueError("stiffening_exponent must be non-negative.")
        if self.area_floor <= 0.0:
            raise ValueError("area_floor must be positive.")
        if (
            not np.isfinite(self.quad_diagonal_weight)
            or self.quad_diagonal_weight < 0.0
        ):
            raise ValueError("quad_diagonal_weight must be finite and non-negative.")
        _validate_quad_bracing_mode(self.quad_bracing_mode)
        if np.intersect1d(self.free_ids, self.prescribed_ids).size:
            raise ValueError("free_ids and prescribed_ids must be disjoint.")

        self._free_local = {
            int(vertex): row for row, vertex in enumerate(self.free_ids)
        }
        self._prescribed_local = {
            int(vertex): row for row, vertex in enumerate(self.prescribed_ids)
        }

        edge_index: dict[tuple[int, int], int] = {}
        edge_faces: list[list[int]] = []
        edge_face_scales: list[list[float]] = []
        edge_is_physical: list[bool] = []
        edge_diagonal_weights: list[float] = []
        cells: list[np.ndarray] = []
        cell_edges: list[np.ndarray] = []
        cell_edge_scales: list[np.ndarray] = []

        def register_edge(
            vertex_a: int,
            vertex_b: int,
            *,
            face_index: int,
            scale: float,
            physical: bool,
        ) -> int | None:
            """Register one face contribution to an undirected graph edge.

            Parameters
            ----------
            vertex_a, vertex_b
                Global edge endpoint IDs.
            face_index
                Cell supplying the weight contribution.
            scale
                Contribution multiplier.
            physical
                Whether the edge is a physical ring edge.

            Returns
            -------
            int or None
                Stable edge ID, or ``None`` for a zero-length topological edge.
            """
            if vertex_a == vertex_b:
                return None
            key = (
                (vertex_a, vertex_b)
                if vertex_a < vertex_b
                else (vertex_b, vertex_a)
            )
            edge_id = edge_index.get(key)
            if edge_id is None:
                edge_id = len(edge_index)
                edge_index[key] = edge_id
                edge_faces.append([])
                edge_face_scales.append([])
                edge_is_physical.append(False)
                edge_diagonal_weights.append(0.0)
            edge_faces[edge_id].append(face_index)
            edge_face_scales[edge_id].append(scale)
            if physical:
                edge_is_physical[edge_id] = True
            else:
                edge_diagonal_weights[edge_id] += scale
            return edge_id

        for block in mesh_data.cell_blocks.values():
            block_array = np.asarray(block, dtype=np.int64)
            if block_array.ndim != 2 or block_array.shape[1] < 2:
                continue
            for raw_cell in block_array:
                cell = np.asarray(raw_cell, dtype=np.int64)
                face_index = len(cells)
                local_edges = []
                local_scales = []
                for index, vertex_a in enumerate(cell):
                    vertex_b = int(cell[(index + 1) % cell.size])
                    vertex_a = int(vertex_a)
                    edge_id = register_edge(
                        vertex_a,
                        vertex_b,
                        face_index=face_index,
                        scale=1.0,
                        physical=True,
                    )
                    if edge_id is None:
                        continue
                    local_edges.append(edge_id)
                    local_scales.append(1.0)
                if cell.size == 4 and self.quad_diagonal_weight > 0.0:
                    for vertex_a, vertex_b, brace_scale in _quad_brace_pairs(
                        baseline_points,
                        cell,
                        mode=self.quad_bracing_mode,
                        weight=self.quad_diagonal_weight,
                    ):
                        edge_id = register_edge(
                            vertex_a,
                            vertex_b,
                            face_index=face_index,
                            scale=brace_scale,
                            physical=False,
                        )
                        if edge_id is None:
                            continue
                        local_edges.append(edge_id)
                        local_scales.append(brace_scale)
                cells.append(cell)
                cell_edges.append(np.asarray(local_edges, dtype=np.int64))
                cell_edge_scales.append(np.asarray(local_scales, dtype=float))

        ordered_edges = sorted(edge_index, key=edge_index.get)
        self.edge_vertices = np.asarray(ordered_edges, dtype=np.int64)
        self.edge_faces = tuple(
            np.asarray(items, dtype=np.int64) for items in edge_faces
        )
        self.edge_face_scales = tuple(
            np.asarray(items, dtype=float) for items in edge_face_scales
        )
        self.edge_uniform_weights = (
            np.asarray(edge_is_physical, dtype=float)
            + np.asarray(edge_diagonal_weights, dtype=float)
        )
        self.cells = tuple(cells)
        self.cell_edges = tuple(cell_edges)
        self.cell_edge_scales = tuple(cell_edge_scales)

        # Fixed per-edge distance multiplier (ones without a distance weighting),
        # aligned with ``edge_vertices``/``edge_faces`` so the area weights and
        # their derivative can be scaled edge-by-edge.
        if distance_weighting is None or self.edge_vertices.size == 0:
            self.edge_distance_multiplier = np.ones(
                self.edge_vertices.shape[0], dtype=float
            )
        else:
            self.edge_distance_multiplier = distance_weighting.edge_multipliers(
                self.edge_vertices
            )

        active_edges = []
        endpoint_free_rows = []
        endpoint_prescribed_rows = []
        for edge_id, (vertex_a, vertex_b) in enumerate(self.edge_vertices):
            free_a = self._free_local.get(int(vertex_a), -1)
            free_b = self._free_local.get(int(vertex_b), -1)
            if free_a < 0 and free_b < 0:
                continue
            prescribed_a = self._prescribed_local.get(int(vertex_a), -1)
            prescribed_b = self._prescribed_local.get(int(vertex_b), -1)
            if free_a < 0 and prescribed_a < 0:
                raise ValueError(
                    f"Free-set edge has unclassified vertex {int(vertex_a)}."
                )
            if free_b < 0 and prescribed_b < 0:
                raise ValueError(
                    f"Free-set edge has unclassified vertex {int(vertex_b)}."
                )
            active_edges.append(edge_id)
            endpoint_free_rows.append((free_a, free_b))
            endpoint_prescribed_rows.append((prescribed_a, prescribed_b))
        self.active_edges = np.asarray(active_edges, dtype=np.int64)
        self.endpoint_free_rows = np.asarray(
            endpoint_free_rows, dtype=np.int64
        )
        self.endpoint_prescribed_rows = np.asarray(
            endpoint_prescribed_rows, dtype=np.int64
        )

    def solve(self, current_vertices, prescribed_values, *, return_state=False):
        """Solve the current weighted graph equilibrium.

        Parameters
        ----------
        current_vertices
            Complete current coordinate array used to compute areas.
        prescribed_values
            Dirichlet values aligned with ``prescribed_ids``.
        return_state
            Whether to return the factor, coupling, weights, and areas.

        Returns
        -------
        numpy.ndarray or tuple
            Free values, optionally followed by the reusable solve state.
        """
        points, prescribed = self._validate_inputs(
            current_vertices, prescribed_values
        )
        weights, areas = self._weights(points)
        factor, coupling = self._assemble(weights)
        free_values = factor.solve(-coupling @ prescribed)
        if return_state:
            return free_values, factor, coupling, weights, areas
        return free_values

    def compute_vjp(
        self,
        current_vertices,
        prescribed_values,
        d_free_values,
    ):
        """Apply the analytic implicit VJP of the current graph solve.

        Parameters
        ----------
        current_vertices
            Complete current coordinate array.
        prescribed_values
            Dirichlet values aligned with ``prescribed_ids``.
        d_free_values
            Cotangent of the solved free values.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Cotangents for current coordinates and prescribed values.
        """
        points, prescribed = self._validate_inputs(
            current_vertices, prescribed_values
        )
        d_free = np.asarray(d_free_values, dtype=float).reshape(
            (self.free_ids.size, prescribed.shape[1])
        )
        free, factor, coupling, weights, areas = self.solve(
            points, prescribed, return_state=True
        )
        adjoint = factor.solve(d_free)
        d_prescribed = -coupling.T @ adjoint
        d_points = np.zeros_like(points)
        if self.stiffening_exponent == 0.0:
            return d_points, np.asarray(d_prescribed, dtype=float)

        edge_sensitivities = np.zeros(self.edge_vertices.shape[0], dtype=float)
        for local_edge, edge_id in enumerate(self.active_edges):
            free_a, free_b = self.endpoint_free_rows[local_edge]
            prescribed_a, prescribed_b = self.endpoint_prescribed_rows[local_edge]
            value_a = (
                free[free_a] if free_a >= 0 else prescribed[prescribed_a]
            )
            value_b = (
                free[free_b] if free_b >= 0 else prescribed[prescribed_b]
            )
            adjoint_a = (
                adjoint[free_a] if free_a >= 0 else np.zeros(prescribed.shape[1])
            )
            adjoint_b = (
                adjoint[free_b] if free_b >= 0 else np.zeros(prescribed.shape[1])
            )
            edge_sensitivities[edge_id] = -float(
                np.dot(adjoint_a - adjoint_b, value_a - value_b)
            )

        exponent = self.stiffening_exponent
        for cell_index, cell in enumerate(self.cells):
            area = float(areas[cell_index])
            if area <= self.area_floor:
                continue
            cell_edge_ids = self.cell_edges[cell_index]
            # w_ij = m_ij * A_f^{-chi}, so d w_ij / d A_f = m_ij * (-chi) *
            # A_f^{-chi-1}; the fixed multiplier scales each edge's contribution.
            area_sensitivity = (
                np.sum(
                    edge_sensitivities[cell_edge_ids]
                    * self.edge_distance_multiplier[cell_edge_ids]
                    * self.cell_edge_scales[cell_index]
                )
                * (-exponent)
                * area ** (-exponent - 1.0)
            )
            if area_sensitivity == 0.0:
                continue
            polygon = points[cell]
            area_vector = np.sum(
                np.cross(polygon, np.roll(polygon, -1, axis=0)),
                axis=0,
            )
            norm = float(np.linalg.norm(area_vector))
            if norm <= 1e-14:
                continue
            unit_normal = area_vector / norm
            previous = np.roll(polygon, 1, axis=0)
            following = np.roll(polygon, -1, axis=0)
            area_gradient = 0.5 * np.cross(
                following - previous,
                unit_normal.reshape((1, 3)),
            )
            np.add.at(
                d_points,
                cell,
                area_sensitivity * area_gradient,
            )
        return d_points, np.asarray(d_prescribed, dtype=float)

    def _validate_inputs(self, current_vertices, prescribed_values):
        points = np.asarray(current_vertices, dtype=float).reshape((-1, 3))
        if points.shape[0] != self.num_vertices:
            raise ValueError("current_vertices must contain the complete mesh.")
        prescribed = np.asarray(prescribed_values, dtype=float)
        if prescribed.ndim == 1:
            prescribed = prescribed.reshape((-1, 1))
        if prescribed.ndim != 2 or prescribed.shape[0] != self.prescribed_ids.size:
            raise ValueError(
                "prescribed_values rows must align with prescribed_ids."
            )
        return points, prescribed

    def _weights(self, points):
        areas = np.asarray(
            [_polygon_area(points[cell]) for cell in self.cells],
            dtype=float,
        )
        if self.stiffening_exponent == 0.0:
            # w_ij = m_ij * 1; the distance multiplier is still a fixed constant,
            # so the solve has no area dependence and the VJP w.r.t. points is 0.
            return (
                self.edge_distance_multiplier * self.edge_uniform_weights,
                areas,
            )
        contributions = np.maximum(areas, self.area_floor) ** (
            -self.stiffening_exponent
        )
        area_weights = np.asarray(
            [
                float(np.sum(contributions[faces] * scales))
                for faces, scales in zip(
                    self.edge_faces, self.edge_face_scales
                )
            ],
            dtype=float,
        )
        weights = self.edge_distance_multiplier * area_weights
        return weights, areas

    def assemble_matrices(self, current_vertices):
        """Return the unfactored current ``L_ff`` and ``L_fp`` blocks.

        The coupled quadratic-distortion extension uses the same scalar graph
        operator in ``kron(L, I3)`` form.  Keeping this assembly here ensures
        that the original and coupled paths use identical topology and weights.

        Parameters
        ----------
        current_vertices
            Complete current coordinate array used to compute area weights.

        Returns
        -------
        tuple
            Free-free matrix, free-prescribed matrix, edge weights, and cell
            areas.

        Raises
        ------
        ValueError
            If the coordinate array does not contain the complete mesh.
        """
        points = np.asarray(current_vertices, dtype=float).reshape((-1, 3))
        if points.shape[0] != self.num_vertices:
            raise ValueError("current_vertices must contain the complete mesh.")
        weights, areas = self._weights(points)
        free_matrix, coupling = self._assemble_matrices(weights)
        return free_matrix, coupling, weights, areas

    def _assemble_matrices(self, weights):
        num_free = self.free_ids.size
        num_prescribed = self.prescribed_ids.size
        diagonal = np.zeros(num_free, dtype=float)
        ff_rows = []
        ff_cols = []
        ff_values = []
        fp_rows = []
        fp_cols = []
        fp_values = []
        for local_edge, edge_id in enumerate(self.active_edges):
            weight = float(weights[edge_id])
            free_a, free_b = self.endpoint_free_rows[local_edge]
            prescribed_a, prescribed_b = self.endpoint_prescribed_rows[local_edge]
            if free_a >= 0:
                diagonal[free_a] += weight
                if free_b >= 0:
                    ff_rows.append(free_a)
                    ff_cols.append(free_b)
                    ff_values.append(-weight)
                else:
                    fp_rows.append(free_a)
                    fp_cols.append(prescribed_b)
                    fp_values.append(-weight)
            if free_b >= 0:
                diagonal[free_b] += weight
                if free_a >= 0:
                    ff_rows.append(free_b)
                    ff_cols.append(free_a)
                    ff_values.append(-weight)
                else:
                    fp_rows.append(free_b)
                    fp_cols.append(prescribed_a)
                    fp_values.append(-weight)
        ff_rows.extend(range(num_free))
        ff_cols.extend(range(num_free))
        ff_values.extend(diagonal.tolist())
        free_matrix = sp.csc_matrix(
            (ff_values, (ff_rows, ff_cols)),
            shape=(num_free, num_free),
        )
        coupling = sp.csr_matrix(
            (fp_values, (fp_rows, fp_cols)),
            shape=(num_free, num_prescribed),
        )
        return free_matrix, coupling

    def _assemble(self, weights):
        free_matrix, coupling = self._assemble_matrices(weights)
        return factorize_spd(free_matrix), coupling


class CurrentGraphSolveOperation(csdl.experimental.CustomExplicitOperationBeta):
    """Expose the current-area graph equilibrium as a CSDL operation.

    Parameters
    ----------
    model
        Fixed-topology current graph model.
    """

    def __init__(self, model: CurrentGraphModel):
        super().__init__()
        self.model = model

    def evaluate(self, current_vertices, prescribed_values):
        """Declare the differentiable current-graph solution.

        Parameters
        ----------
        current_vertices
            Complete CSDL coordinate array.
        prescribed_values
            CSDL Dirichlet values.

        Returns
        -------
        csdl.Variable
            Solved free values.
        """
        self.declare_input("current_vertices", current_vertices)
        self.declare_input("prescribed_values", prescribed_values)
        free_values = self.create_output(
            "free_values",
            (self.model.free_ids.size, prescribed_values.shape[1]),
        )
        self.declare_vjp_function(CurrentGraphSolveVJP, model=self.model)
        return free_values

    def compute(self, inputs, outputs):
        """Evaluate the numeric graph solve.

        Parameters
        ----------
        inputs
            Custom-operation inputs for coordinates and boundary values.
        outputs
            Mutable outputs receiving the free solution.
        """
        outputs["free_values"] = self.model.solve(
            inputs["current_vertices"],
            inputs["prescribed_values"],
        )


class CurrentGraphSolveVJP(csdl.experimental.CustomExplicitOperationBeta):
    """Apply an IFT adjoint including current-area derivatives.

    Parameters
    ----------
    model
        Fixed-topology current graph model.
    """

    def __init__(self, model: CurrentGraphModel):
        super().__init__()
        self.model = model

    def evaluate(self, inputs, d_outputs):
        """Declare coordinate and boundary-value cotangents.

        Parameters
        ----------
        inputs
            Forward operation inputs.
        d_outputs
            Cotangent of the free solution.

        Returns
        -------
        dict[str, csdl.Variable]
            Cotangent variables for coordinates and prescribed values.
        """
        current_vertices = inputs["current_vertices"]
        prescribed_values = inputs["prescribed_values"]
        d_free_values = d_outputs["free_values"]
        self.declare_input("current_vertices", current_vertices)
        self.declare_input("prescribed_values", prescribed_values)
        self.declare_input("d_free_values", d_free_values)
        d_current = self.create_output(
            "d_current_vertices", current_vertices.shape
        )
        d_prescribed = self.create_output(
            "d_prescribed_values", prescribed_values.shape
        )
        return {
            "current_vertices": d_current,
            "prescribed_values": d_prescribed,
        }

    def compute(self, inputs, outputs):
        """Evaluate the numeric implicit VJP.

        Parameters
        ----------
        inputs
            Reverse inputs including the free-solution cotangent.
        outputs
            Mutable outputs receiving input cotangents.
        """
        d_current, d_prescribed = self.model.compute_vjp(
            inputs["current_vertices"],
            inputs["prescribed_values"],
            inputs["d_free_values"],
        )
        outputs["d_current_vertices"] = d_current
        outputs["d_prescribed_values"] = d_prescribed


def _polygon_area(polygon):
    points = np.asarray(polygon, dtype=float).reshape((-1, 3))
    normal = np.sum(
        np.cross(points, np.roll(points, -1, axis=0)),
        axis=0,
    )
    return 0.5 * float(np.linalg.norm(normal))


__all__ = [
    "CurrentGraphModel",
    "CurrentGraphSolveOperation",
    "CurrentGraphSolveVJP",
]
