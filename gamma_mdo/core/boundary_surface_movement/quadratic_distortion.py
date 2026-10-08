"""Fixed quadratic element-distortion regularization for graph mesh motion.

The regularizer is assembled once from a fixed virtual triangulation of the
baseline polygon mesh.  It augments the state-dependent graph continuation
without changing the delivered mesh topology:

    1/2 dw.T K_L(x_previous) dw
      + lambda/2 (w_current + dw).T K_dist (w_current + dw).

``K_dist`` is positive semidefinite.  The anchored graph term makes the active
free system positive definite.  All Cartesian DOFs use node-major ordering
``[x0, y0, z0, x1, ...]``.
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
class DistortionModeCoefficients:
    """Weight local linearized distortion modes.

    Parameters
    ----------
    area
        Isotropic in-plane area-change weight.
    deviatoric
        Deviatoric normal-strain weight.
    shear
        Symmetric in-plane shear weight.
    rotation
        In-plane antisymmetric rotation weight.
    normal
        Out-of-plane gradient weight.
    """

    area: float = 1.0
    deviatoric: float = 1.0
    shear: float = 1.0
    rotation: float = 0.0
    normal: float = 0.5

    def __post_init__(self):
        """Validate and normalize each distortion coefficient.

        Every coefficient is converted to ``float`` and stored back on the
        frozen instance.

        Raises
        ------
        ValueError
            If any coefficient is non-finite or negative.
        """
        for name in ("area", "deviatoric", "shear", "rotation", "normal"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} distortion coefficient must be nonnegative.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class QuadraticDistortionConfig:
    """Configure the fixed quadratic distortion regularizer.

    Parameters
    ----------
    lambda_dist
        Finite nonnegative global regularization strength.
    mode
        ``"full_gradient"`` or ``"strain_distortion"`` energy.
    coefficients
        Relative weights for individual linearized modes.
    area_tolerance
        Positive virtual-triangle degeneracy threshold.
    geometry_tolerance
        Positive polygon chart and topology tolerance.
    """

    lambda_dist: float = 0.0
    mode: str = "strain_distortion"
    coefficients: DistortionModeCoefficients = DistortionModeCoefficients()
    area_tolerance: float = 1e-14
    geometry_tolerance: float = 1e-10

    def __post_init__(self):
        """Validate the distortion strength, mode, and tolerances.

        The mode is lowercased and stripped before checking, and the
        normalized values are stored back on the frozen instance.

        Raises
        ------
        ValueError
            If ``lambda_dist`` is non-finite or negative, ``mode`` is not
            ``"full_gradient"`` or ``"strain_distortion"``, or a tolerance is
            outside its permitted range.
        """
        strength = float(self.lambda_dist)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("lambda_dist must be finite and nonnegative.")
        mode = str(self.mode).strip().lower()
        if mode not in ("full_gradient", "strain_distortion"):
            raise ValueError(
                "quadratic distortion mode must be 'full_gradient' or "
                "'strain_distortion'."
            )
        if float(self.area_tolerance) <= 0.0:
            raise ValueError("area_tolerance must be positive.")
        if float(self.geometry_tolerance) <= 0.0:
            raise ValueError("geometry_tolerance must be positive.")
        object.__setattr__(self, "lambda_dist", strength)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "area_tolerance", float(self.area_tolerance))
        object.__setattr__(
            self, "geometry_tolerance", float(self.geometry_tolerance)
        )


@dataclass(frozen=True)
class SubtriangleRecord:
    """Store baseline integration data for one virtual triangle.

    Attributes
    ----------
    coefficient_matrix
        Map from polygon nodes to the three virtual triangle nodes.
    local_frame
        Orthonormal tangent-normal basis.
    shape_gradient
        Linear triangle shape-function gradients.
    gradient_map
        Map from nodal displacement DOFs to displacement gradients.
    reference_area
        Positive baseline triangle area.
    normalized_weight
        Triangle area divided by total polygon integration area.
    """

    coefficient_matrix: np.ndarray
    local_frame: np.ndarray
    shape_gradient: np.ndarray
    gradient_map: np.ndarray
    reference_area: float
    normalized_weight: float


@dataclass(frozen=True)
class ElementRecord:
    """Store fixed preprocessing for one active polygon.

    Attributes
    ----------
    element_id
        Stable element index in mesh block order.
    node_ids
        Global polygon vertex IDs.
    cell_type
        Source mesh cell type.
    triangulation
        Native, centroid-fan, or ear-clipped integration scheme.
    is_convex
        Whether the projected baseline polygon is convex.
    warp_ratio
        Normal extent relative to tangent extent.
    subtriangles
        Virtual-triangle integration records.
    """

    element_id: int
    node_ids: np.ndarray
    cell_type: str
    triangulation: str
    is_convex: bool
    warp_ratio: float
    subtriangles: tuple[SubtriangleRecord, ...]


@dataclass(frozen=True)
class QuadraticDistortionSystem:
    """Store a partitioned fixed quadratic regularization matrix.

    Attributes
    ----------
    free_ids, prescribed_ids
        Vertex IDs defining the node-major matrix partition.
    free_matrix
        Free-free regularization block.
    coupling
        Free-prescribed regularization block.
    prescribed_matrix
        Prescribed-prescribed regularization block.
    records
        Optional retained per-element preprocessing records.
    num_centroid_fans, num_ear_clipped
        Counts of polygon integration strategies.
    maximum_warp_ratio
        Largest baseline polygon warp ratio.
    """

    free_ids: np.ndarray
    prescribed_ids: np.ndarray
    free_matrix: sp.csc_matrix
    coupling: sp.csr_matrix
    prescribed_matrix: sp.csc_matrix
    records: tuple[ElementRecord, ...]
    num_centroid_fans: int
    num_ear_clipped: int
    maximum_warp_ratio: float

    @property
    def num_free(self) -> int:
        """Return the number of free vertices.

        Returns
        -------
        int
            Number of free vertex IDs.
        """
        return int(self.free_ids.size)

    @property
    def num_prescribed(self) -> int:
        """Return the number of prescribed vertices.

        Returns
        -------
        int
            Number of prescribed vertex IDs.
        """
        return int(self.prescribed_ids.size)

    def full_matrix(self) -> sp.csc_matrix:
        """Reconstruct the complete active regularization matrix.

        Returns
        -------
        scipy.sparse.csc_matrix
            Symmetric block matrix in free-then-prescribed node-major order.
        """
        return sp.bmat(
            [
                [self.free_matrix, self.coupling],
                [self.coupling.T, self.prescribed_matrix],
            ],
            format="csc",
        )


class QuadraticDistortionAssembler:
    """Assemble virtual-triangle distortion energies on mixed polygons.

    Parameters
    ----------
    config
        Distortion mode, weights, and geometric tolerances.
    """

    def __init__(self, config: QuadraticDistortionConfig | None = None):
        self.config = config or QuadraticDistortionConfig()

    def assemble(
        self,
        mesh,
        *,
        free_ids: np.ndarray,
        prescribed_ids: np.ndarray,
        retain_records: bool = False,
    ) -> QuadraticDistortionSystem:
        """Assemble the partitioned fixed distortion matrix.

        Parameters
        ----------
        mesh
            Baseline mixed-polygon surface mesh.
        free_ids
            Unknown graph vertex IDs.
        prescribed_ids
            Dirichlet IDs covering active co-element neighbors.
        retain_records
            Whether to retain detailed virtual-triangle records.

        Returns
        -------
        QuadraticDistortionSystem
            Partitioned matrix and polygon integration diagnostics.

        Raises
        ------
        ValueError
            If partitions or active polygon geometry are invalid.
        """
        mesh_data = _as_mesh_data(mesh)
        points = np.asarray(mesh_data.vertices, dtype=float).reshape((-1, 3))
        free_ids, prescribed_ids = _validate_partition(free_ids, prescribed_ids)
        active_ids = np.concatenate((free_ids, prescribed_ids))
        active_local = {
            int(vertex): row for row, vertex in enumerate(active_ids)
        }
        free_set = set(int(item) for item in free_ids)

        row_blocks: list[np.ndarray] = []
        col_blocks: list[np.ndarray] = []
        value_blocks: list[np.ndarray] = []
        records: list[ElementRecord] = []
        element_id = 0
        fan_count = 0
        ear_count = 0
        maximum_warp = 0.0

        for cell_type, raw_block in mesh_data.cell_blocks.items():
            cells = np.asarray(raw_block, dtype=np.int64)
            if cells.ndim != 2 or cells.shape[1] < 3:
                element_id += int(cells.shape[0]) if cells.ndim else 0
                continue
            for cell in cells:
                current_id = element_id
                element_id += 1
                if not any(int(vertex) in free_set for vertex in cell):
                    continue
                missing = [
                    int(vertex)
                    for vertex in cell
                    if int(vertex) not in active_local
                ]
                if missing:
                    raise ValueError(
                        "A distortion element touching the free set contains "
                        f"unclassified vertices {missing[:10]}. Use "
                        "element_neighbors(mesh, free_ids) for prescribed_ids."
                    )

                record, local_matrix = self._element_matrix(
                    current_id,
                    str(cell_type),
                    np.asarray(cell, dtype=np.int64),
                    points[cell],
                )
                if record.triangulation == "centroid_fan":
                    fan_count += 1
                elif record.triangulation == "ear_clip":
                    ear_count += 1
                maximum_warp = max(maximum_warp, record.warp_ratio)
                if retain_records:
                    records.append(record)

                local_nodes = np.asarray(
                    [active_local[int(vertex)] for vertex in cell],
                    dtype=np.int64,
                )
                local_dofs = (
                    3 * local_nodes[:, None] + np.arange(3, dtype=np.int64)[None, :]
                ).reshape(-1)
                nonzero_rows, nonzero_cols = np.nonzero(
                    np.abs(local_matrix) > 1e-15
                )
                row_blocks.append(local_dofs[nonzero_rows])
                col_blocks.append(local_dofs[nonzero_cols])
                value_blocks.append(local_matrix[nonzero_rows, nonzero_cols])

        size = 3 * active_ids.size
        matrix = sp.coo_matrix(
            (
                np.concatenate(value_blocks) if value_blocks else np.empty(0),
                (
                    np.concatenate(row_blocks)
                    if row_blocks
                    else np.empty(0, dtype=np.int64),
                    np.concatenate(col_blocks)
                    if col_blocks
                    else np.empty(0, dtype=np.int64),
                ),
            ),
            shape=(size, size),
        ).tocsc()
        matrix.sum_duplicates()
        matrix = (0.5 * (matrix + matrix.T)).tocsc()

        split = 3 * free_ids.size
        return QuadraticDistortionSystem(
            free_ids=free_ids,
            prescribed_ids=prescribed_ids,
            free_matrix=matrix[:split, :split].tocsc(),
            coupling=matrix[:split, split:].tocsr(),
            prescribed_matrix=matrix[split:, split:].tocsc(),
            records=tuple(records),
            num_centroid_fans=fan_count,
            num_ear_clipped=ear_count,
            maximum_warp_ratio=float(maximum_warp),
        )

    def _element_matrix(
        self,
        element_id: int,
        cell_type: str,
        node_ids: np.ndarray,
        polygon: np.ndarray,
    ) -> tuple[ElementRecord, np.ndarray]:
        polygon = np.asarray(polygon, dtype=float).reshape((-1, 3))
        count = int(polygon.shape[0])
        if count < 3:
            raise ValueError(f"Element {element_id} has fewer than three vertices.")
        if np.unique(node_ids).size != node_ids.size:
            raise ValueError(f"Element {element_id} contains repeated node IDs.")

        projected, warp_ratio = _project_polygon(
            polygon,
            tolerance=self.config.geometry_tolerance,
            element_id=element_id,
        )
        _check_simple_polygon(
            projected,
            tolerance=self.config.geometry_tolerance,
            element_id=element_id,
        )
        convex = _is_convex(projected, self.config.geometry_tolerance)
        coefficient_matrices, triangulation = _fixed_subtriangulation(
            projected,
            convex=convex,
            tolerance=self.config.geometry_tolerance,
            element_id=element_id,
        )

        raw_subtriangles = []
        total_area = 0.0
        for coefficient_matrix in coefficient_matrices:
            triangle = polygon.T @ coefficient_matrix
            frame, shape_gradient, area = _triangle_operators(
                triangle,
                area_tolerance=self.config.area_tolerance,
                element_id=element_id,
            )
            gradient_map = coefficient_matrix @ shape_gradient
            raw_subtriangles.append(
                (coefficient_matrix, frame, shape_gradient, gradient_map, area)
            )
            total_area += area
        if total_area <= self.config.area_tolerance:
            raise ValueError(
                f"Element {element_id} has degenerate virtual-triangulation area."
            )

        local_matrix = np.zeros((3 * count, 3 * count), dtype=float)
        records = []
        for coefficient_matrix, frame, shape_gradient, gradient_map, area in (
            raw_subtriangles
        ):
            weight = area / total_area
            gradient_operator = np.kron(gradient_map.T, np.eye(3))
            q_matrix = _distortion_q(
                frame,
                mode=self.config.mode,
                coefficients=self.config.coefficients,
            )
            local_matrix += (
                weight
                * gradient_operator.T
                @ q_matrix
                @ gradient_operator
            )
            records.append(
                SubtriangleRecord(
                    coefficient_matrix=np.asarray(coefficient_matrix, dtype=float),
                    local_frame=frame,
                    shape_gradient=shape_gradient,
                    gradient_map=gradient_map,
                    reference_area=float(area),
                    normalized_weight=float(weight),
                )
            )
        local_matrix = 0.5 * (local_matrix + local_matrix.T)
        return (
            ElementRecord(
                element_id=int(element_id),
                node_ids=np.asarray(node_ids, dtype=np.int64),
                cell_type=cell_type,
                triangulation=triangulation,
                is_convex=bool(convex),
                warp_ratio=float(warp_ratio),
                subtriangles=tuple(records),
            ),
            local_matrix,
        )


class CurrentGraphDistortionModel:
    """Couple graph increments to a fixed total-distortion penalty.

    Parameters
    ----------
    graph_model
        Current-area scalar graph model.
    distortion_system
        Fixed three-dimensional distortion matrix.
    lambda_dist
        Positive regularization strength.
    constrained_free_dofs
        Optional node-major free DOFs fixed to zero.
    baseline_vertices
        Baseline coordinates used for diagonal normalization.
    """

    def __init__(
        self,
        graph_model: CurrentGraphModel,
        distortion_system: QuadraticDistortionSystem,
        *,
        lambda_dist: float,
        constrained_free_dofs: np.ndarray | None = None,
        baseline_vertices: np.ndarray,
    ):
        self.graph_model = graph_model
        self.distortion_system = distortion_system
        self.lambda_dist = float(lambda_dist)
        if self.lambda_dist <= 0.0:
            raise ValueError(
                "CurrentGraphDistortionModel requires a positive lambda_dist."
            )
        if not np.array_equal(
            graph_model.free_ids, distortion_system.free_ids
        ) or not np.array_equal(
            graph_model.prescribed_ids, distortion_system.prescribed_ids
        ):
            raise ValueError("Graph and distortion partitions must be identical.")

        self.num_free = int(graph_model.free_ids.size)
        self.num_prescribed = int(graph_model.prescribed_ids.size)
        total_free_dofs = 3 * self.num_free
        constrained = np.asarray(
            np.empty(0, dtype=np.int64)
            if constrained_free_dofs is None
            else constrained_free_dofs,
            dtype=np.int64,
        ).reshape(-1)
        if constrained.size:
            if np.any((constrained < 0) | (constrained >= total_free_dofs)):
                raise ValueError("constrained_free_dofs contains an invalid DOF.")
            constrained = np.unique(constrained)
        self.constrained_free_dofs = constrained
        self.active_free_dofs = np.setdiff1d(
            np.arange(total_free_dofs, dtype=np.int64),
            constrained,
            assume_unique=True,
        )

        baseline_lff, _, _, _ = graph_model.assemble_matrices(
            baseline_vertices
        )
        graph_diagonal = np.repeat(np.asarray(baseline_lff.diagonal()), 3)
        graph_diagonal = graph_diagonal[self.active_free_dofs]
        distortion_diagonal = np.asarray(
            distortion_system.free_matrix.diagonal()
        )[self.active_free_dofs]
        graph_positive = graph_diagonal[graph_diagonal > 0.0]
        distortion_positive = distortion_diagonal[distortion_diagonal > 0.0]
        if not graph_positive.size:
            raise ValueError("The graph free matrix has no positive diagonal.")
        if not distortion_positive.size:
            raise ValueError("The distortion free matrix is identically zero.")
        self.normalization_scale = float(
            np.median(graph_positive) / np.median(distortion_positive)
        )
        self.k_dist_ff = (
            self.normalization_scale * distortion_system.free_matrix
        ).tocsc()
        self.k_dist_fp = (
            self.normalization_scale * distortion_system.coupling
        ).tocsr()
        self.redundancy = _redundancy_measure(
            sp.kron(baseline_lff, sp.eye(3), format="csc"),
            self.k_dist_ff,
        )

    def solve(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
        *,
        return_state: bool = False,
    ):
        """Solve one coupled graph/distortion increment.

        Parameters
        ----------
        current_vertices
            Complete current mesh coordinates.
        incremental_prescribed
            Current Dirichlet increments in 3-column blocks.
        current_free_correction
            Accumulated free corrections.
        total_prescribed_correction
            Accumulated prescribed corrections.
        return_state
            Whether to return factorization and assembly state.

        Returns
        -------
        numpy.ndarray or tuple
            Free increment, optionally followed by reusable solve state.
        """
        (
            points,
            incremental,
            current_free,
            total_prescribed,
            num_blocks,
        ) = self._validate_inputs(
            current_vertices,
            incremental_prescribed,
            current_free_correction,
            total_prescribed_correction,
        )
        lff, lfp, weights, areas = self.graph_model.assemble_matrices(points)
        graph_ff = sp.kron(lff, sp.eye(3), format="csc")
        graph_fp = sp.kron(lfp, sp.eye(3), format="csr")
        active = self.active_free_dofs
        matrix = (
            graph_ff[active][:, active]
            + self.lambda_dist * self.k_dist_ff[active][:, active]
        ).tocsc()
        factor = factorize_spd(matrix)
        output = np.zeros_like(current_free)
        block_states = []
        for block in range(num_blocks):
            columns = slice(3 * block, 3 * (block + 1))
            inc_p = incremental[:, columns].reshape(-1)
            cur_f = current_free[:, columns].reshape(-1)
            total_p = total_prescribed[:, columns].reshape(-1)
            rhs_full = (
                -(graph_fp @ inc_p)
                - self.lambda_dist
                * (self.k_dist_ff @ cur_f + self.k_dist_fp @ total_p)
            )
            solved_active = factor.solve(np.asarray(rhs_full)[active])
            delta = np.zeros(3 * self.num_free, dtype=float)
            delta[active] = solved_active
            output[:, columns] = delta.reshape((self.num_free, 3))
            block_states.append((delta, inc_p, cur_f, total_p))
        if return_state:
            return (
                output,
                factor,
                graph_ff,
                graph_fp,
                weights,
                areas,
                tuple(block_states),
            )
        return output

    def compute_vjp(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
        d_free_increment,
    ):
        """Apply the implicit VJP of the coupled increment solve.

        Parameters
        ----------
        current_vertices
            Complete current mesh coordinates.
        incremental_prescribed
            Current Dirichlet increments.
        current_free_correction
            Accumulated free corrections.
        total_prescribed_correction
            Accumulated prescribed corrections.
        d_free_increment
            Cotangent of the solved free increment.

        Returns
        -------
        tuple[numpy.ndarray, ...]
            Cotangents for all four differentiable inputs.
        """
        points = np.asarray(current_vertices, dtype=float).reshape((-1, 3))
        (
            output,
            factor,
            graph_ff,
            graph_fp,
            _,
            areas,
            block_states,
        ) = self.solve(
            current_vertices,
            incremental_prescribed,
            current_free_correction,
            total_prescribed_correction,
            return_state=True,
        )
        d_output = np.asarray(d_free_increment, dtype=float).reshape(output.shape)
        d_points = np.zeros_like(points)
        d_incremental = np.zeros_like(np.asarray(incremental_prescribed, dtype=float))
        d_current_free = np.zeros_like(
            np.asarray(current_free_correction, dtype=float)
        )
        d_total_prescribed = np.zeros_like(
            np.asarray(total_prescribed_correction, dtype=float)
        )
        active = self.active_free_dofs
        edge_sensitivities = np.zeros(
            self.graph_model.edge_vertices.shape[0], dtype=float
        )

        for block, (delta, inc_p, _, _) in enumerate(block_states):
            columns = slice(3 * block, 3 * (block + 1))
            d_block = d_output[:, columns].reshape(-1)
            adjoint = np.zeros(3 * self.num_free, dtype=float)
            adjoint[active] = factor.solve(d_block[active])
            d_incremental[:, columns] = np.asarray(
                -(graph_fp.T @ adjoint)
            ).reshape((self.num_prescribed, 3))
            d_current_free[:, columns] = np.asarray(
                -self.lambda_dist * (self.k_dist_ff.T @ adjoint)
            ).reshape((self.num_free, 3))
            d_total_prescribed[:, columns] = np.asarray(
                -self.lambda_dist * (self.k_dist_fp.T @ adjoint)
            ).reshape((self.num_prescribed, 3))

            free_values = delta.reshape((self.num_free, 3))
            prescribed_values = inc_p.reshape((self.num_prescribed, 3))
            free_adjoint = adjoint.reshape((self.num_free, 3))
            for local_edge, edge_id in enumerate(self.graph_model.active_edges):
                free_a, free_b = self.graph_model.endpoint_free_rows[local_edge]
                prescribed_a, prescribed_b = (
                    self.graph_model.endpoint_prescribed_rows[local_edge]
                )
                value_a = (
                    free_values[free_a]
                    if free_a >= 0
                    else prescribed_values[prescribed_a]
                )
                value_b = (
                    free_values[free_b]
                    if free_b >= 0
                    else prescribed_values[prescribed_b]
                )
                adjoint_a = (
                    free_adjoint[free_a] if free_a >= 0 else np.zeros(3)
                )
                adjoint_b = (
                    free_adjoint[free_b] if free_b >= 0 else np.zeros(3)
                )
                edge_sensitivities[edge_id] += -float(
                    np.dot(adjoint_a - adjoint_b, value_a - value_b)
                )

        exponent = self.graph_model.stiffening_exponent
        if exponent:
            for cell_index, cell in enumerate(self.graph_model.cells):
                area = float(areas[cell_index])
                if area <= self.graph_model.area_floor:
                    continue
                area_sensitivity = (
                    np.sum(
                        edge_sensitivities[
                            self.graph_model.cell_edges[cell_index]
                        ]
                        * self.graph_model.edge_distance_multiplier[
                            self.graph_model.cell_edges[cell_index]
                        ]
                        * self.graph_model.cell_edge_scales[cell_index]
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
                area_gradient = 0.5 * np.cross(
                    np.roll(polygon, -1, axis=0)
                    - np.roll(polygon, 1, axis=0),
                    unit_normal.reshape((1, 3)),
                )
                np.add.at(
                    d_points,
                    cell,
                    area_sensitivity * area_gradient,
                )
        return (
            d_points,
            d_incremental,
            d_current_free,
            d_total_prescribed,
        )

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
        if incremental.ndim != 2 or incremental.shape[0] != self.num_prescribed:
            raise ValueError("incremental_prescribed has an invalid shape.")
        if incremental.shape[1] == 0 or incremental.shape[1] % 3:
            raise ValueError(
                "incremental_prescribed columns must contain 3D component blocks."
            )
        expected_free = (self.num_free, incremental.shape[1])
        expected_prescribed = (self.num_prescribed, incremental.shape[1])
        if current_free.shape != expected_free:
            raise ValueError("current_free_correction has an invalid shape.")
        if total_prescribed.shape != expected_prescribed:
            raise ValueError("total_prescribed_correction has an invalid shape.")
        return (
            points,
            incremental,
            current_free,
            total_prescribed,
            incremental.shape[1] // 3,
        )


class CurrentGraphDistortionSolveOperation(
    csdl.experimental.CustomExplicitOperationBeta
):
    """Expose the coupled graph/distortion solve to CSDL.

    Parameters
    ----------
    model
        Configured current-graph distortion model.
    """

    def __init__(self, model: CurrentGraphDistortionModel):
        super().__init__()
        self.model = model

    def evaluate(
        self,
        current_vertices,
        incremental_prescribed,
        current_free_correction,
        total_prescribed_correction,
    ):
        """Declare the differentiable coupled increment.

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
        output = self.create_output(
            "free_increment", current_free_correction.shape
        )
        self.declare_vjp_function(
            CurrentGraphDistortionSolveVJP,
            model=self.model,
        )
        return output

    def compute(self, inputs, outputs):
        """Evaluate the numeric coupled solve.

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


class CurrentGraphDistortionSolveVJP(
    csdl.experimental.CustomExplicitOperationBeta
):
    """Apply the coupled IFT VJP including current graph weights.

    Parameters
    ----------
    model
        Configured current-graph distortion model.
    """

    def __init__(self, model: CurrentGraphDistortionModel):
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
                "d_incremental_prescribed",
                inputs["incremental_prescribed"].shape,
            ),
            "current_free_correction": self.create_output(
                "d_current_free_correction",
                inputs["current_free_correction"].shape,
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


def _project_polygon(polygon, *, tolerance, element_id):
    center = np.mean(polygon, axis=0)
    centered = polygon - center
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size < 2 or singular_values[1] <= tolerance:
        raise ValueError(f"Element {element_id} is baseline-degenerate.")
    newell = np.sum(
        np.cross(polygon, np.roll(polygon, -1, axis=0)),
        axis=0,
    )
    normal_norm = float(np.linalg.norm(newell))
    normal = (
        newell / normal_norm
        if normal_norm > tolerance
        else np.asarray(right[-1], dtype=float)
    )
    tangent_1 = None
    for edge in np.roll(polygon, -1, axis=0) - polygon:
        projected_edge = edge - float(np.dot(edge, normal)) * normal
        norm = float(np.linalg.norm(projected_edge))
        if norm > tolerance:
            tangent_1 = projected_edge / norm
            break
    if tangent_1 is None:
        raise ValueError(f"Element {element_id} has no valid baseline edge.")
    tangent_2 = np.cross(normal, tangent_1)
    coordinates = np.column_stack(
        (centered @ tangent_1, centered @ tangent_2)
    )
    signed_area = _signed_area(coordinates)
    if signed_area < 0.0:
        tangent_2 *= -1.0
        coordinates[:, 1] *= -1.0
        signed_area *= -1.0
    if signed_area <= tolerance:
        raise ValueError(f"Element {element_id} has zero projected area.")
    edge_lengths = np.linalg.norm(
        np.roll(polygon, -1, axis=0) - polygon,
        axis=1,
    )
    scale = max(float(np.mean(edge_lengths)), tolerance)
    plane_distance = np.abs(centered @ normal)
    warp_ratio = float(np.max(plane_distance) / scale)
    return coordinates, warp_ratio


def _signed_area(points):
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])
    )


def _check_simple_polygon(points, *, tolerance, element_id):
    count = points.shape[0]
    for index in range(count):
        a = points[index]
        b = points[(index + 1) % count]
        if np.linalg.norm(b - a) <= tolerance:
            raise ValueError(
                f"Element {element_id} contains a zero-length boundary edge."
            )
        for other in range(index + 1, count):
            if other in (index, (index + 1) % count):
                continue
            if index == 0 and other == count - 1:
                continue
            c = points[other]
            d = points[(other + 1) % count]
            if _segments_intersect(a, b, c, d, tolerance):
                raise ValueError(
                    f"Element {element_id} is self-intersecting in its best-fit plane."
                )


def _segments_intersect(a, b, c, d, tolerance):
    def cross(left, right):
        """Return the scalar cross product of two planar vectors.

        Parameters
        ----------
        left, right
            Two-component planar vectors.

        Returns
        -------
        float
            Signed scalar cross product.
        """
        return float(left[0] * right[1] - left[1] * right[0])

    ab = b - a
    cd = d - c
    values = (
        cross(ab, c - a),
        cross(ab, d - a),
        cross(cd, a - c),
        cross(cd, b - c),
    )
    return bool(
        values[0] * values[1] < -(tolerance**2)
        and values[2] * values[3] < -(tolerance**2)
    )


def _is_convex(points, tolerance):
    signs = []
    count = points.shape[0]
    for index in range(count):
        first = points[(index + 1) % count] - points[index]
        second = points[(index + 2) % count] - points[(index + 1) % count]
        turn = first[0] * second[1] - first[1] * second[0]
        if abs(turn) > tolerance:
            signs.append(np.sign(turn))
    return bool(signs and (all(item > 0 for item in signs) or all(item < 0 for item in signs)))


def _fixed_subtriangulation(points, *, convex, tolerance, element_id):
    count = points.shape[0]
    if count == 3:
        return (np.eye(3),), "native_triangle"
    if convex:
        alpha = np.full(count, 1.0 / count)
        matrices = []
        for index in range(count):
            following = (index + 1) % count
            matrix = np.zeros((count, 3), dtype=float)
            matrix[:, 0] = alpha
            matrix[index, 1] = 1.0
            matrix[following, 2] = 1.0
            matrices.append(matrix)
        return tuple(matrices), "centroid_fan"

    triangles = _ear_clip(points, tolerance=tolerance, element_id=element_id)
    matrices = []
    for triangle in triangles:
        matrix = np.zeros((count, 3), dtype=float)
        for column, vertex in enumerate(triangle):
            matrix[int(vertex), column] = 1.0
        matrices.append(matrix)
    return tuple(matrices), "ear_clip"


def _ear_clip(points, *, tolerance, element_id):
    remaining = list(range(points.shape[0]))
    triangles = []
    guard = 0
    while len(remaining) > 3:
        clipped = False
        for local_index, current in enumerate(remaining):
            previous = remaining[local_index - 1]
            following = remaining[(local_index + 1) % len(remaining)]
            a, b, c = points[[previous, current, following]]
            first = b - a
            second = c - b
            turn = float(
                first[0] * second[1] - first[1] * second[0]
            )
            if turn <= tolerance:
                continue
            if any(
                _point_in_triangle(
                    points[candidate],
                    a,
                    b,
                    c,
                    tolerance=tolerance,
                )
                for candidate in remaining
                if candidate not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[local_index]
            clipped = True
            break
        guard += 1
        if not clipped or guard > points.shape[0] ** 2:
            raise ValueError(
                f"Fixed ear clipping failed for baseline element {element_id}."
            )
    triangles.append(tuple(remaining))
    return tuple(triangles)


def _point_in_triangle(point, a, b, c, *, tolerance):
    def orient(first, second, third):
        """Return the signed planar orientation of three points.

        Parameters
        ----------
        first, second, third
            Ordered planar points.

        Returns
        -------
        float
            Twice the signed triangle area.
        """
        left = second - first
        right = third - first
        return float(left[0] * right[1] - left[1] * right[0])

    values = (
        orient(a, b, point),
        orient(b, c, point),
        orient(c, a, point),
    )
    return all(value >= -tolerance for value in values)


def _triangle_operators(triangle, *, area_tolerance, element_id):
    first, second, third = triangle.T
    edge_1 = second - first
    edge_2 = third - first
    length = float(np.linalg.norm(edge_1))
    cross = np.cross(edge_1, edge_2)
    twice_area = float(np.linalg.norm(cross))
    if length <= area_tolerance or twice_area <= 2.0 * area_tolerance:
        raise ValueError(
            f"Element {element_id} contains a degenerate virtual subtriangle."
        )
    tangent_1 = edge_1 / length
    normal = cross / twice_area
    tangent_2 = np.cross(normal, tangent_1)
    frame = np.column_stack((tangent_1, tangent_2, normal))
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [length, 0.0],
            [float(np.dot(edge_2, tangent_1)), float(np.dot(edge_2, tangent_2))],
        ]
    )
    delta = (
        (coordinates[1, 0] - coordinates[0, 0])
        * (coordinates[2, 1] - coordinates[0, 1])
        - (coordinates[2, 0] - coordinates[0, 0])
        * (coordinates[1, 1] - coordinates[0, 1])
    )
    if delta <= 0.0:
        raise ValueError(
            f"Element {element_id} contains a reversed virtual subtriangle."
        )
    u = coordinates[:, 0]
    v = coordinates[:, 1]
    shape_gradient = np.asarray(
        [
            [v[1] - v[2], u[2] - u[1]],
            [v[2] - v[0], u[0] - u[2]],
            [v[0] - v[1], u[1] - u[0]],
        ],
        dtype=float,
    ) / delta
    return frame, shape_gradient, 0.5 * twice_area


def _distortion_q(frame, *, mode, coefficients):
    if mode == "full_gradient":
        return np.eye(6)
    transform = np.kron(np.eye(2), frame.T)
    c = np.zeros((6, 6), dtype=float)
    c[0, [0, 4]] = np.sqrt(coefficients.area)
    c[1, 0] = np.sqrt(coefficients.deviatoric)
    c[1, 4] = -np.sqrt(coefficients.deviatoric)
    c[2, [1, 3]] = np.sqrt(coefficients.shear)
    c[3, 1] = -np.sqrt(coefficients.rotation)
    c[3, 3] = np.sqrt(coefficients.rotation)
    c[4, 2] = np.sqrt(coefficients.normal)
    c[5, 5] = np.sqrt(coefficients.normal)
    q_matrix = transform.T @ c.T @ c @ transform
    return 0.5 * (q_matrix + q_matrix.T)


def _redundancy_measure(graph_matrix, distortion_matrix):
    graph = sp.csc_matrix(graph_matrix)
    distortion = sp.csc_matrix(distortion_matrix)
    graph_norm_squared = float(graph.multiply(graph).sum())
    distortion_norm_squared = float(distortion.multiply(distortion).sum())
    if graph_norm_squared == 0.0 or distortion_norm_squared == 0.0:
        return float("nan")
    alpha = float(distortion.multiply(graph).sum()) / graph_norm_squared
    residual = distortion - alpha * graph
    return float(
        np.sqrt(float(residual.multiply(residual).sum()) / distortion_norm_squared)
    )


__all__ = [
    "CurrentGraphDistortionModel",
    "CurrentGraphDistortionSolveOperation",
    "CurrentGraphDistortionSolveVJP",
    "DistortionModeCoefficients",
    "ElementRecord",
    "QuadraticDistortionAssembler",
    "QuadraticDistortionConfig",
    "QuadraticDistortionSystem",
    "SubtriangleRecord",
]
