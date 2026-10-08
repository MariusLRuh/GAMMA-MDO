"""Element-local affine-residual regularization for polygon mesh motion.

For each baseline polygon with ``n >= 4`` vertices, this module removes the
best affine displacement in a fixed local tangent chart and penalizes only the
remaining ``n - 3`` nodal modes.  A regular quad therefore contributes the
classic hourglass projector while a linear triangle contributes no term.

The regularizer is scalar and isotropic.  Its matrix is assembled once and the
same sparse factorization serves every Cartesian/component column in the
current-area graph continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sp

from gamma_mdo.preprocessing.mesh_io import _as_mesh_data

from .current_graph_solve import CurrentGraphModel
from .spd_solve_custom_op import factorize_spd


@dataclass(frozen=True)
class NgonAffineConfig:
    """Configure the fixed element-local affine-residual regularizer.

    Parameters
    ----------
    lambda_ngon
        Finite nonnegative regularization strength.
    geometry_tolerance
        Positive tolerance for tangent-chart rank checks.
    """

    lambda_ngon: float = 0.0
    geometry_tolerance: float = 1e-11

    def __post_init__(self):
        """Validate the regularization strength and geometry tolerance.

        Raises
        ------
        ValueError
            If either value is non-finite or negative.
        """
        strength = float(self.lambda_ngon)
        tolerance = float(self.geometry_tolerance)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("lambda_ngon must be finite and nonnegative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("geometry_tolerance must be finite and positive.")
        object.__setattr__(self, "lambda_ngon", strength)
        object.__setattr__(self, "geometry_tolerance", tolerance)


@dataclass(frozen=True)
class NgonAffineSystem:
    """Store the partitioned scalar affine-residual matrix.

    Attributes
    ----------
    free_ids, prescribed_ids
        Vertex IDs defining the matrix partition order.
    free_matrix
        Free-free affine-residual block.
    coupling
        Free-prescribed affine-residual block.
    prescribed_matrix
        Prescribed-prescribed affine-residual block.
    num_regularized_elements
        Number of active polygons with at least four vertices.
    num_hourglass_modes
        Total retained non-affine modes, ``sum(n - 3)``.
    maximum_warp_ratio
        Largest baseline normal-to-tangent extent ratio.
    """

    free_ids: np.ndarray
    prescribed_ids: np.ndarray
    free_matrix: sp.csc_matrix
    coupling: sp.csr_matrix
    prescribed_matrix: sp.csc_matrix
    num_regularized_elements: int
    num_hourglass_modes: int
    maximum_warp_ratio: float

    def full_matrix(self) -> sp.csc_matrix:
        """Reconstruct the complete matrix in active partition order.

        Returns
        -------
        scipy.sparse.csc_matrix
            Symmetric block matrix over free then prescribed vertices.
        """
        return sp.bmat(
            [
                [self.free_matrix, self.coupling],
                [self.coupling.T, self.prescribed_matrix],
            ],
            format="csc",
        )


class NgonAffineAssembler:
    """Assemble affine-residual projectors for active polygons.

    Parameters
    ----------
    config
        Geometry tolerance and default regularization settings.

    Notes
    -----
    For an ``n``-vertex polygon, the projector annihilates
    ``span{1, u, v}`` in its tangent chart and retains exactly ``n - 3``
    non-affine modes.
    """

    def __init__(self, config: NgonAffineConfig | None = None):
        self.config = config or NgonAffineConfig()

    def assemble(self, mesh, *, free_ids, prescribed_ids) -> NgonAffineSystem:
        """Assemble the partitioned affine-residual matrix.

        Parameters
        ----------
        mesh
            Baseline polygonal surface mesh.
        free_ids
            Unknown graph vertex IDs.
        prescribed_ids
            Dirichlet IDs covering every active co-element neighbor.

        Returns
        -------
        NgonAffineSystem
            Fixed partitioned projector sum and geometry diagnostics.

        Raises
        ------
        ValueError
            If partitions or an active polygon chart are invalid.
        """
        mesh_data = _as_mesh_data(mesh)
        points = np.asarray(mesh_data.vertices, dtype=float).reshape((-1, 3))
        free_ids, prescribed_ids = _validate_partition(free_ids, prescribed_ids)
        active_ids = np.concatenate((free_ids, prescribed_ids))
        active_local = {int(vertex): row for row, vertex in enumerate(active_ids)}
        free_set = set(int(vertex) for vertex in free_ids)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        values: list[np.ndarray] = []
        regularized = 0
        modes = 0
        maximum_warp = 0.0
        for raw_block in mesh_data.cell_blocks.values():
            block = np.asarray(raw_block, dtype=np.int64)
            if block.ndim != 2 or block.shape[1] < 4:
                continue
            for cell in block:
                if not any(int(vertex) in free_set for vertex in cell):
                    continue
                missing = [
                    int(vertex)
                    for vertex in cell
                    if int(vertex) not in active_local
                ]
                if missing:
                    raise ValueError(
                        "An affine-regularized polygon touching the free set "
                        f"contains unclassified vertices {missing[:10]}; use "
                        "element_neighbors(mesh, free_ids) for prescribed_ids."
                    )
                projector, warp_ratio = _affine_residual_projector(
                    points[cell], tolerance=self.config.geometry_tolerance
                )
                local = np.asarray(
                    [active_local[int(vertex)] for vertex in cell],
                    dtype=np.int64,
                )
                nonzero_row, nonzero_col = np.nonzero(np.abs(projector) > 1e-14)
                rows.append(local[nonzero_row])
                cols.append(local[nonzero_col])
                values.append(projector[nonzero_row, nonzero_col])
                regularized += 1
                modes += int(cell.size - 3)
                maximum_warp = max(maximum_warp, warp_ratio)

        size = active_ids.size
        matrix = sp.coo_matrix(
            (
                np.concatenate(values) if values else np.empty(0),
                (
                    np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
                    np.concatenate(cols) if cols else np.empty(0, dtype=np.int64),
                ),
            ),
            shape=(size, size),
        ).tocsc()
        matrix.sum_duplicates()
        matrix = (0.5 * (matrix + matrix.T)).tocsc()
        split = free_ids.size
        return NgonAffineSystem(
            free_ids=free_ids,
            prescribed_ids=prescribed_ids,
            free_matrix=matrix[:split, :split].tocsc(),
            coupling=matrix[:split, split:].tocsr(),
            prescribed_matrix=matrix[split:, split:].tocsc(),
            num_regularized_elements=regularized,
            num_hourglass_modes=modes,
            maximum_warp_ratio=float(maximum_warp),
        )


class CurrentGraphNgonAffineModel:
    """Couple current-area graph increments to a fixed affine penalty.

    Parameters
    ----------
    graph_model
        Current-area scalar graph model.
    affine_system
        Fixed affine-residual matrix with matching partitions.
    lambda_ngon
        Positive regularization strength.
    baseline_vertices
        Baseline coordinates used to normalize graph and affine diagonals.

    Notes
    -----
    The fixed penalty acts on total correction while the graph term acts on
    the current increment.
    """

    def __init__(
        self,
        graph_model: CurrentGraphModel,
        affine_system: NgonAffineSystem,
        *,
        lambda_ngon: float,
        baseline_vertices,
    ):
        self.graph_model = graph_model
        self.affine_system = affine_system
        self.lambda_ngon = float(lambda_ngon)
        if self.lambda_ngon <= 0.0:
            raise ValueError(
                "CurrentGraphNgonAffineModel requires positive lambda_ngon."
            )
        if not np.array_equal(
            graph_model.free_ids, affine_system.free_ids
        ) or not np.array_equal(
            graph_model.prescribed_ids,
            affine_system.prescribed_ids,
        ):
            raise ValueError("Graph and affine-regularizer partitions must match.")

        baseline_lff, _, _, _ = graph_model.assemble_matrices(baseline_vertices)
        graph_diagonal = np.asarray(baseline_lff.diagonal(), dtype=float)
        affine_diagonal = np.asarray(
            affine_system.free_matrix.diagonal(), dtype=float
        )
        graph_positive = graph_diagonal[graph_diagonal > 0.0]
        affine_positive = affine_diagonal[affine_diagonal > 1e-14]
        if not graph_positive.size:
            raise ValueError("The graph free matrix has no positive diagonal.")
        if not affine_positive.size:
            raise ValueError("The active n-gon affine matrix is identically zero.")
        self.normalization_scale = float(
            np.median(graph_positive) / np.median(affine_positive)
        )
        self.k_ff = (
            self.normalization_scale * affine_system.free_matrix
        ).tocsc()
        self.k_fp = (
            self.normalization_scale * affine_system.coupling
        ).tocsr()

    def solve(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
        *,
        return_state: bool = False,
    ):
        """Solve one regularized graph increment.

        Parameters
        ----------
        current_vertices
            Complete current mesh coordinates.
        incremental_prescribed
            Current Dirichlet increment.
        current_free_correction
            Accumulated free-vertex correction.
        total_prescribed_correction
            Accumulated prescribed-vertex correction.
        return_state
            Whether to include factorization and graph assembly state.

        Returns
        -------
        numpy.ndarray or tuple
            Free increment, optionally followed by reusable solve state.
        """
        points, incremental, current_free, total_prescribed = self._validate_inputs(
            current_vertices,
            incremental_prescribed,
            current_free_correction,
            total_prescribed_correction,
        )
        lff, lfp, weights, areas = self.graph_model.assemble_matrices(points)
        strength = self.lambda_ngon
        matrix = (lff + strength * self.k_ff).tocsc()
        factor = factorize_spd(matrix)
        rhs = (
            -(lfp @ incremental)
            - strength * (self.k_ff @ current_free + self.k_fp @ total_prescribed)
        )
        output = np.asarray(factor.solve(np.asarray(rhs)), dtype=float)
        if return_state:
            return output, factor, lfp, weights, areas
        return output

    def compute_vjp(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
        d_free_increment,
    ):
        """Apply the implicit VJP of the regularized increment solve.

        Parameters
        ----------
        current_vertices
            Complete current mesh coordinates.
        incremental_prescribed
            Current Dirichlet increment.
        current_free_correction
            Accumulated free correction.
        total_prescribed_correction
            Accumulated prescribed correction.
        d_free_increment
            Cotangent of the solved free increment.

        Returns
        -------
        tuple[numpy.ndarray, ...]
            Cotangents for all four differentiable inputs.
        """
        points, incremental, current_free, total_prescribed = self._validate_inputs(
            current_vertices,
            incremental_prescribed,
            current_free_correction,
            total_prescribed_correction,
        )
        output, factor, graph_fp, _, areas = self.solve(
            points,
            incremental,
            current_free,
            total_prescribed,
            return_state=True,
        )
        d_output = np.asarray(d_free_increment, dtype=float).reshape(output.shape)
        adjoint = np.asarray(factor.solve(d_output), dtype=float)
        strength = self.lambda_ngon
        d_incremental = np.asarray(-(graph_fp.T @ adjoint), dtype=float)
        d_current_free = np.asarray(
            -strength * (self.k_ff.T @ adjoint), dtype=float
        )
        d_total_prescribed = np.asarray(
            -strength * (self.k_fp.T @ adjoint), dtype=float
        )
        d_points = self._graph_coordinate_vjp(
            points,
            incremental,
            output,
            adjoint,
            areas,
        )
        return d_points, d_incremental, d_current_free, d_total_prescribed

    def _graph_coordinate_vjp(self, points, prescribed, free, adjoint, areas):
        model = self.graph_model
        d_points = np.zeros_like(points)
        if model.stiffening_exponent == 0.0:
            return d_points
        edge_sensitivities = np.zeros(model.edge_vertices.shape[0], dtype=float)
        for local_edge, edge_id in enumerate(model.active_edges):
            free_a, free_b = model.endpoint_free_rows[local_edge]
            prescribed_a, prescribed_b = model.endpoint_prescribed_rows[local_edge]
            value_a = free[free_a] if free_a >= 0 else prescribed[prescribed_a]
            value_b = free[free_b] if free_b >= 0 else prescribed[prescribed_b]
            adjoint_a = (
                adjoint[free_a] if free_a >= 0 else np.zeros(free.shape[1])
            )
            adjoint_b = (
                adjoint[free_b] if free_b >= 0 else np.zeros(free.shape[1])
            )
            edge_sensitivities[edge_id] = -float(
                np.dot(adjoint_a - adjoint_b, value_a - value_b)
            )
        exponent = model.stiffening_exponent
        for cell_index, cell in enumerate(model.cells):
            area = float(areas[cell_index])
            if area <= model.area_floor:
                continue
            edge_ids = model.cell_edges[cell_index]
            area_sensitivity = (
                np.sum(
                    edge_sensitivities[edge_ids]
                    * model.edge_distance_multiplier[edge_ids]
                    * model.cell_edge_scales[cell_index]
                )
                * (-exponent)
                * area ** (-exponent - 1.0)
            )
            if area_sensitivity == 0.0:
                continue
            polygon = points[cell]
            area_vector = np.sum(
                np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0
            )
            norm = float(np.linalg.norm(area_vector))
            if norm <= 1e-14:
                continue
            unit_normal = area_vector / norm
            area_gradient = 0.5 * np.cross(
                np.roll(polygon, -1, axis=0) - np.roll(polygon, 1, axis=0),
                unit_normal.reshape((1, 3)),
            )
            np.add.at(d_points, cell, area_sensitivity * area_gradient)
        return d_points

    def _validate_inputs(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
    ):
        points = np.asarray(current_vertices, dtype=float).reshape((-1, 3))
        if points.shape[0] != self.graph_model.num_vertices:
            raise ValueError("current_vertices must contain the complete mesh.")
        incremental = np.asarray(incremental_prescribed, dtype=float)
        current_free = np.asarray(current_free_correction, dtype=float)
        total_prescribed = np.asarray(total_prescribed_correction, dtype=float)
        if (
            incremental.ndim != 2
            or incremental.shape[0] != self.graph_model.prescribed_ids.size
        ):
            raise ValueError("incremental_prescribed has an invalid shape.")
        expected_free = (self.graph_model.free_ids.size, incremental.shape[1])
        expected_prescribed = (
            self.graph_model.prescribed_ids.size,
            incremental.shape[1],
        )
        if current_free.shape != expected_free:
            raise ValueError("current_free_correction has an invalid shape.")
        if total_prescribed.shape != expected_prescribed:
            raise ValueError("total_prescribed_correction has an invalid shape.")
        return points, incremental, current_free, total_prescribed


class CurrentGraphNgonAffineSolveOperation(
    csdl.experimental.CustomExplicitOperationBeta
):
    """Expose the current-graph affine-regularized solve to CSDL.

    Parameters
    ----------
    model
        Configured scalar graph and affine-residual model.
    """

    def __init__(self, model: CurrentGraphNgonAffineModel):
        super().__init__()
        self.model = model

    def evaluate(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
    ):
        """Declare the differentiable regularized increment.

        Parameters
        ----------
        current_vertices
            Complete current coordinate variable.
        incremental_prescribed
            Current Dirichlet increment variable.
        current_free_correction
            Accumulated free correction variable.
        total_prescribed_correction
            Accumulated prescribed correction variable.

        Returns
        -------
        csdl.Variable
            Solved free increment.
        """
        self.declare_input("current_vertices", current_vertices)
        self.declare_input("incremental_prescribed", incremental_prescribed)
        self.declare_input("current_free_correction", current_free_correction)
        self.declare_input(
            "total_prescribed_correction", total_prescribed_correction
        )
        output = self.create_output("free_increment", current_free_correction.shape)
        self.declare_vjp_function(
            CurrentGraphNgonAffineSolveVJP, model=self.model
        )
        return output

    def compute(self, inputs, outputs):
        """Evaluate the numeric regularized solve.

        Parameters
        ----------
        inputs
            Forward custom-operation inputs.
        outputs
            Mutable outputs receiving ``free_increment``.
        """
        outputs["free_increment"] = self.model.solve(
            inputs["current_vertices"],
            inputs["incremental_prescribed"],
            inputs["current_free_correction"],
            inputs["total_prescribed_correction"],
        )


class CurrentGraphNgonAffineSolveVJP(
    csdl.experimental.CustomExplicitOperationBeta
):
    """Apply the implicit VJP of the graph/n-gon solve.

    Parameters
    ----------
    model
        Configured scalar graph and affine-residual model.
    """

    def __init__(self, model: CurrentGraphNgonAffineModel):
        super().__init__()
        self.model = model

    def evaluate(self, inputs, d_outputs):
        """Declare cotangents for every differentiable model input.

        Parameters
        ----------
        inputs
            Forward custom-operation inputs.
        d_outputs
            Cotangent of ``free_increment``.

        Returns
        -------
        dict[str, csdl.Variable]
            Input cotangent variables keyed by forward input name.
        """
        for name in (
            "current_vertices",
            "incremental_prescribed",
            "current_free_correction",
            "total_prescribed_correction",
        ):
            self.declare_input(name, inputs[name])
        self.declare_input("d_free_increment", d_outputs["free_increment"])
        return {
            "current_vertices": self.create_output(
                "d_current_vertices", inputs["current_vertices"].shape
            ),
            "incremental_prescribed": self.create_output(
                "d_incremental_prescribed", inputs["incremental_prescribed"].shape
            ),
            "current_free_correction": self.create_output(
                "d_current_free_correction", inputs["current_free_correction"].shape
            ),
            "total_prescribed_correction": self.create_output(
                "d_total_prescribed_correction",
                inputs["total_prescribed_correction"].shape,
            ),
        }

    def compute(self, inputs, outputs):
        """Evaluate the numeric implicit VJP.

        Parameters
        ----------
        inputs
            Reverse inputs including the free-increment cotangent.
        outputs
            Mutable outputs receiving all input cotangents.
        """
        derivatives = self.model.compute_vjp(
            inputs["current_vertices"],
            inputs["incremental_prescribed"],
            inputs["current_free_correction"],
            inputs["total_prescribed_correction"],
            inputs["d_free_increment"],
        )
        (
            outputs["d_current_vertices"],
            outputs["d_incremental_prescribed"],
            outputs["d_current_free_correction"],
            outputs["d_total_prescribed_correction"],
        ) = derivatives


def _affine_residual_projector(polygon, *, tolerance):
    """Project nodal values onto the non-affine tangent-chart complement.

    The affine design matrix has columns ``[1, u, v]``. Its range is the
    three-dimensional affine subspace, so ``I - Q Q.T`` annihilates every
    affine field and has rank ``n - 3`` for a valid ``n``-vertex polygon.

    Parameters
    ----------
    polygon
        Baseline polygon coordinates.
    tolerance
        Positive rank tolerance for the local tangent chart.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Symmetric residual projector and baseline warp ratio.

    Raises
    ------
    ValueError
        If the polygon has fewer than four vertices or a degenerate chart.
    """
    points = np.asarray(polygon, dtype=float).reshape((-1, 3))
    if points.shape[0] < 4:
        raise ValueError("An affine-residual polygon must have at least four vertices.")
    center = np.mean(points, axis=0)
    centered = points - center
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size < 2 or singular_values[1] <= tolerance:
        raise ValueError("An affine-residual polygon is baseline-degenerate.")
    coordinates = centered @ right[:2].T
    design = np.column_stack((np.ones(points.shape[0]), coordinates))
    basis, triangular = np.linalg.qr(design, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) <= tolerance:
        raise ValueError("An affine-residual polygon has a rank-deficient chart.")
    projector = np.eye(points.shape[0]) - basis @ basis.T
    projector = 0.5 * (projector + projector.T)
    projector[np.abs(projector) < 1e-14] = 0.0
    normal_extent = float(np.max(np.abs(centered @ right[2])))
    tangent_extent = max(float(singular_values[0]), tolerance)
    return projector, normal_extent / tangent_extent


def _validate_partition(free_ids, prescribed_ids):
    free_ids = np.asarray(free_ids, dtype=np.int64).reshape(-1)
    prescribed_ids = np.asarray(prescribed_ids, dtype=np.int64).reshape(-1)
    if np.intersect1d(free_ids, prescribed_ids).size:
        raise ValueError("free_ids and prescribed_ids must be disjoint.")
    if np.unique(free_ids).size != free_ids.size:
        raise ValueError("free_ids must not contain duplicates.")
    if np.unique(prescribed_ids).size != prescribed_ids.size:
        raise ValueError("prescribed_ids must not contain duplicates.")
    return free_ids, prescribed_ids


__all__ = [
    "CurrentGraphNgonAffineModel",
    "CurrentGraphNgonAffineSolveOperation",
    "CurrentGraphNgonAffineSolveVJP",
    "NgonAffineAssembler",
    "NgonAffineConfig",
    "NgonAffineSystem",
]
