"""Propagation strategies for boundary-surface mesh movement.

Stage 2 of the pipeline -- turning exact seam displacements into free-vertex
motion -- is expressed here as a swappable *strategy*.  The contract mirrors
the existing ``DisplacementInterpolator.train(...) -> DisplacementSurrogate``
shape so the example script selects a strategy without touching the geometry
front-end (``solve_intersection``) or the projection back-end
(``project_onto_oml`` / ``reevaluate_vertices`` / ``combine_vertices``).

Two strategies are provided:

* ``RBFMotionSolver`` -- a thin adapter over today's meshless RBF interpolator
  (behavior unchanged; the RBF stays selectable).
* ``ElasticityMotionSolver`` -- Milestone 1: a reference-config, decoupled
  graph-Laplacian / edge-spring solve on the fuselage free band, factored once
  and back-substituted per design iteration (see ``elasticity.py`` and
  ``spd_solve_custom_op.py``).

Both return a ``MeshMotionField`` whose ``evaluate(vertices, vertex_ids)``
yields preprojected positions for the requested mesh rows, exactly as the RBF
surrogate does, so the back-end never changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import csdl_alpha as csdl
import numpy as np
import scipy.sparse as sp

from gamma_mdo.core.projections.function_set_evaluation_custom_op import (
    FunctionSetEvaluationModel,
    FunctionSetEvaluationOperation,
)
from gamma_mdo.preprocessing.mesh_io import _as_mesh_data

from .elasticity import (
    GraphLaplacianAssembler,
    StiffnessAssembler,
    element_neighbors,
    graph_neighbors,
)
from .geometry import (
    stack_component_coefficients,
    stack_component_coefficients_numpy,
)
from .intersections import (
    IntersectionParameters,
    IntersectionSolution,
    solve_intersection,
)
from .rbf import (
    DisplacementInterpolationParameters,
    DisplacementInterpolator,
)
from .spd_solve_custom_op import SPDSolveOperation


class MeshMotionField(Protocol):
    """Evaluate preprojected free-vertex positions for mesh rows.

    Notes
    -----
    Implementations preserve the input row order and return three coordinates
    per requested vertex.
    """

    def evaluate(
        self, *, vertices: csdl.Variable, vertex_ids: np.ndarray
    ) -> csdl.Variable:
        """Evaluate the trained motion field.

        Parameters
        ----------
        vertices
            Query coordinates aligned with ``vertex_ids``.
        vertex_ids
            Global IDs of the query rows.

        Returns
        -------
        csdl.Variable
            Preprojected coordinates in query-row order.
        """
        ...


class MeshMotionSolver(Protocol):
    """Train a motion field from deformed component coefficients.

    Notes
    -----
    The returned field is differentiable with respect to supplied CSDL
    coefficients.
    """

    def train(
        self,
        *,
        component_coeffs,
        query_component_coeffs: Mapping[object, object] | None = None,
    ) -> MeshMotionField:
        """Build a differentiable field for one geometry state.

        Parameters
        ----------
        component_coeffs
            Deformed driving-component coefficients.
        query_component_coeffs
            Optional coefficient overrides for query components.

        Returns
        -------
        MeshMotionField
            Trained preprojection field.
        """
        ...


@dataclass(frozen=True)
class ComponentReevaluation:
    """Owned mesh rows that follow a component exactly by parametric reeval.

    These are the lifting-surface (wing/tail) rows the RBF used to move: their
    deformed position is the exact evaluation of their fixed baseline
    ``[patch, u, v]`` on the component's deformed coefficients, so they never
    enter the elastic solve.  Providing them here lets ``evaluate`` reproduce
    the RBF's non-fuselage behavior and keeps the projection back-end
    unchanged.

    Parameters
    ----------
    component
        Geometry component evaluated exactly.
    vertex_ids
        Global mesh rows owned by the component.
    parametric_coordinates
        Fixed ``[patch, u, v]`` coordinates aligned with ``vertex_ids``.
    """

    component: object
    vertex_ids: np.ndarray
    parametric_coordinates: np.ndarray

    def __post_init__(self):
        """Normalize the vertex ids and parametric coordinates and check alignment.

        Reshapes ``vertex_ids`` to ``(n,)`` and ``parametric_coordinates`` to
        ``(n, 3)``, storing both back on the instance.

        Raises
        ------
        ValueError
            If the two do not align.
        """
        ids = np.asarray(self.vertex_ids, dtype=np.int64).reshape(-1)
        coordinates = np.asarray(
            self.parametric_coordinates, dtype=float
        ).reshape((-1, 3))
        if ids.size != coordinates.shape[0]:
            raise ValueError("vertex_ids and parametric_coordinates must align.")
        object.__setattr__(self, "vertex_ids", ids)
        object.__setattr__(self, "parametric_coordinates", coordinates)


@dataclass(frozen=True)
class GraphLoadStepState:
    """Store geometry-side data for one incremental graph solve.

    Attributes
    ----------
    free_reference
        Absolute component-reference positions for free rows.
    prescribed_deviations
        Absolute boundary deviations for unconstrained coordinate blocks.
    prescribed_deviations_y
        Boundary deviations for the symmetry-aware y block.
    solutions
        Exact component-intersection solutions.
    """

    free_reference: csdl.Variable
    prescribed_deviations: tuple[csdl.Variable, ...]
    prescribed_deviations_y: tuple[csdl.Variable, ...]
    solutions: tuple[IntersectionSolution, ...]


# ---------------------------------------------------------------------------
# RBF strategy (adapter)
# ---------------------------------------------------------------------------


class RBFMotionSolver:
    """Present the meshless RBF interpolator as a motion strategy.

    Parameters
    ----------
    mesh
        Baseline surface mesh.
    interpolation_params
        RBF centers, basis, and component-intersection metadata.
    """

    def __init__(
        self,
        *,
        mesh,
        interpolation_params: DisplacementInterpolationParameters,
    ):
        self._interpolator = DisplacementInterpolator(
            mesh=mesh,
            interpolation_params=interpolation_params,
        )

    def train(
        self,
        *,
        component_coeffs,
        query_component_coeffs: Mapping[object, object] | None = None,
    ) -> MeshMotionField:
        """Train the underlying meshless displacement interpolator.

        Parameters
        ----------
        component_coeffs
            Deformed driving-component coefficients.
        query_component_coeffs
            Optional coefficient overrides for query components.

        Returns
        -------
        MeshMotionField
            RBF displacement surrogate satisfying the common field protocol.
        """
        # ``DisplacementSurrogate`` already satisfies the MeshMotionField
        # contract (``evaluate(vertices=..., vertex_ids=...)``).
        return self._interpolator.train(
            component_coeffs=component_coeffs,
            query_component_coeffs=query_component_coeffs,
        )


# ---------------------------------------------------------------------------
# Elasticity strategy (Milestone 1)
# ---------------------------------------------------------------------------


class ElasticityMotionSolver:
    """Reference-config graph-Laplacian / edge-spring mesh-motion propagator.

    The elastic solve governs the fuselage free band ``free_ids`` between the
    moving seam and the frozen far field.  Its Dirichlet data is the prescribed
    set ``P`` -- every graph neighbor of the free band that is not itself free
    -- split into seam rows (exact bisection displacement) and frozen rows
    (parametric reevaluation on the owner component, which is zero where the
    owner is stationary).  Wing/tail surface rows are handled by
    ``component_reevaluations`` and never enter the solve.

    Parameters
    ----------
    mesh
        Baseline surface mesh.
    free_ids
        Global vertex IDs solved by graph propagation.
    intersection_params
        Exact seam problems providing moving Dirichlet data.
    parametric_coordinates
        Fixed geometry coordinates for every mesh vertex.
    components
        Geometry components that own mesh rows.
    component_reevaluations
        Rows moved by exact parametric reevaluation rather than the graph.
    stiffening_exponent
        Reference-area exponent used by the default assembler.
    use_query_seam_reference
        Whether query-component motion contributes to the seam reference.
    symmetry_plane_ids
        Optional global IDs constrained to a symmetry plane.
    assembler
        Optional custom stiffness assembler.
    prescribed_ids
        Optional explicit Dirichlet partition.
    symmetry_use_element_neighbors
        Whether symmetry partitions use the full co-element halo.
    distance_weighting
        Optional fixed reference-geodesic edge weighting.
    quad_diagonal_weight
        Weight for auxiliary quadrilateral bracing.
    quad_bracing_mode
        Auxiliary bracing topology.
    """

    def __init__(
        self,
        *,
        mesh,
        free_ids: np.ndarray,
        intersection_params: Sequence[IntersectionParameters],
        parametric_coordinates: np.ndarray,
        components: Sequence[object],
        component_reevaluations: Sequence[ComponentReevaluation] = (),
        stiffening_exponent: float = 0.0,
        use_query_seam_reference: bool = True,
        symmetry_plane_ids: np.ndarray | None = None,
        assembler: StiffnessAssembler | None = None,
        prescribed_ids: np.ndarray | None = None,
        symmetry_use_element_neighbors: bool = False,
        distance_weighting=None,
        quad_diagonal_weight: float = 0.0,
        quad_bracing_mode: str = "both_diagonals",
    ):
        self.use_query_seam_reference = bool(use_query_seam_reference)
        self.mesh = _as_mesh_data(mesh)
        self.initial_vertices = np.asarray(self.mesh.vertices, dtype=float).reshape((-1, 3))
        self.intersection_params = tuple(intersection_params)
        if not self.intersection_params:
            raise ValueError("intersection_params must contain at least one intersection.")
        self.parametric_coordinates = np.asarray(
            parametric_coordinates, dtype=float
        ).reshape((-1, 3))
        if self.parametric_coordinates.shape[0] != self.initial_vertices.shape[0]:
            raise ValueError("parametric_coordinates must have one row per mesh vertex.")
        self.components = tuple(components)
        self.component_reevaluations = tuple(component_reevaluations)

        self.free_ids = np.unique(np.asarray(free_ids, dtype=np.int64).reshape(-1))
        self.distance_weighting = distance_weighting
        self.symmetry_use_element_neighbors = bool(
            symmetry_use_element_neighbors
        )
        assembler = assembler or GraphLaplacianAssembler(
            stiffening_exponent=float(stiffening_exponent),
            distance_weighting=distance_weighting,
            quad_diagonal_weight=quad_diagonal_weight,
            quad_bracing_mode=quad_bracing_mode,
        )
        self.assembler = assembler
        active_quad_bracing_mode = (
            str(getattr(assembler, "quad_bracing_mode", "both_diagonals"))
            if float(getattr(assembler, "quad_diagonal_weight", 0.0)) > 0.0
            else None
        )

        # Prescribed set = every graph neighbor of the free band (seam ring +
        # frozen ring).  Growing P from connectivity guarantees the free-free
        # block has no stiffness leak.
        self.prescribed_ids = (
            graph_neighbors(
                self.mesh,
                self.free_ids,
                quad_bracing_mode=active_quad_bracing_mode,
            )
            if prescribed_ids is None
            else np.unique(np.asarray(prescribed_ids, dtype=np.int64).reshape(-1))
        )
        self._prescribed_local = {
            int(vertex): row for row, vertex in enumerate(self.prescribed_ids)
        }

        # Seam prescribed rows: which intersection solves them and at what row.
        self._seam_row_source: dict[int, tuple[int, int]] = {}
        for solution_index, parameters in enumerate(self.intersection_params):
            if parameters.vertex_ids is None:
                continue
            for solution_row, vertex_id in enumerate(parameters.vertex_ids):
                vertex_id = int(vertex_id)
                if vertex_id in self._prescribed_local and vertex_id not in self._seam_row_source:
                    self._seam_row_source[vertex_id] = (solution_index, solution_row)

        self._patch_to_component = _patch_to_component_map(self.components)

        # Factor L_ff once on the reference mesh; L_fp is constant.  The optional
        # fixed reference-geodesic distance weighting multiplies every edge; it
        # is shared with the current-area load-stepping model (read back from
        # ``self.assembler.distance_weighting``).
        self.system = assembler.assemble(
            self.mesh,
            free_ids=self.free_ids,
            prescribed_ids=self.prescribed_ids,
        )
        self._free_local = {
            int(vertex): row for row, vertex in enumerate(self.free_ids)
        }

        # Owner reference: each free vertex follows its component's rigid
        # component's rigid motion, and the elastic solve carries only the
        # seam's *departure* from that rigid motion.  The departure is zero at
        # the rigid outboard boundary, so the free/rigid interface does not fold
        # -- unlike a harmonic interpolation of the absolute displacement, which
        # cannot reproduce a component's affine (planform-scaling) motion on a
        # non-uniform mesh and folds the boundary cells.
        self._free_reference_groups = self._build_free_reference_groups(self.free_ids)
        # Seam prescribed rows grouped by the intersection that solves them,
        # with the driving-side parametric coordinates (for the departure).
        self._seam_prescribed_by_solution = self._build_seam_prescribed_index(
            self._prescribed_local
        )
        # Query-side seam reference, so a MOVING query component (e.g. a fuselage
        # with a diameter design variable) contributes its own rigid motion to
        # that block's seam departure instead of being assumed stationary.
        self._seam_query_reference = self._build_seam_query_reference()

        # Symmetry: pin the out-of-plane (y) displacement of the plane vertices
        # to zero.  The decoupled solve uses the main system for x/z (plane free)
        # and a second factorization for the y component (plane prescribed to 0);
        # x/z keep the plane free (the natural / Neumann symmetry condition).
        self.symmetry_plane_ids = None
        if symmetry_plane_ids is not None:
            plane_ids = np.unique(
                np.asarray(symmetry_plane_ids, dtype=np.int64).reshape(-1)
            )
            plane_ids = np.intersect1d(plane_ids, self.free_ids)
            if plane_ids.size:
                self.symmetry_plane_ids = plane_ids
        if self.symmetry_plane_ids is not None:
            plane_set = set(int(v) for v in self.symmetry_plane_ids)
            self.free_ids_y = np.array(
                [int(v) for v in self.free_ids if int(v) not in plane_set],
                dtype=np.int64,
            )
            if self.symmetry_use_element_neighbors:
                self.prescribed_ids_y = element_neighbors(
                    self.mesh, self.free_ids_y
                )
            else:
                self.prescribed_ids_y = graph_neighbors(
                    self.mesh,
                    self.free_ids_y,
                    quad_bracing_mode=active_quad_bracing_mode,
                )
            self._prescribed_local_y = {
                int(v): r for r, v in enumerate(self.prescribed_ids_y)
            }
            self.system_y = assembler.assemble(
                self.mesh,
                free_ids=self.free_ids_y,
                prescribed_ids=self.prescribed_ids_y,
            )
            self._free_reference_groups_y = self._build_free_reference_groups(
                self.free_ids_y
            )
            self._seam_prescribed_by_solution_y = self._build_seam_prescribed_index(
                self._prescribed_local_y
            )
            # Scatter the (F \ plane) y-deviation back into the full free rows;
            # plane rows stay zero (their y is pinned to 0, reference y is 0).
            free_y_rows = np.array(
                [self._free_local[int(v)] for v in self.free_ids_y], dtype=np.int64
            )
            self._scatter_y = sp.csr_matrix(
                (
                    np.ones(free_y_rows.size, dtype=float),
                    (free_y_rows, np.arange(free_y_rows.size)),
                ),
                shape=(self.free_ids.size, self.free_ids_y.size),
            )

    def _build_free_reference_groups(
        self, free_ids: np.ndarray
    ) -> tuple["_FreeReferenceGroup", ...]:
        """Group free vertices by owner component with baseline reeval positions.

        ``free_rows`` index into the supplied ``free_ids`` (which may be the full
        free set or the y-symmetry subset).
        """
        component_by_id = {id(component): component for component in self.components}
        rows_by_component: dict[int, list[int]] = {}
        for local_row, vertex in enumerate(free_ids):
            patch_id = int(round(float(self.parametric_coordinates[int(vertex), 0])))
            component = self._patch_to_component.get(patch_id)
            if component is None:
                raise ValueError(
                    f"Free vertex {int(vertex)} has patch {patch_id} that belongs "
                    "to no supplied component."
                )
            rows_by_component.setdefault(id(component), []).append(local_row)

        groups = []
        for component_key, free_rows in rows_by_component.items():
            component = component_by_id[component_key]
            free_rows = np.asarray(free_rows, dtype=np.int64)
            coordinates = self.parametric_coordinates[free_ids[free_rows]]
            baseline_positions = np.asarray(
                FunctionSetEvaluationModel(component).evaluate(
                    stack_component_coefficients_numpy(component),
                    coordinates,
                ),
                dtype=float,
            ).reshape((-1, 3))
            groups.append(
                _FreeReferenceGroup(
                    component=component,
                    free_rows=free_rows,
                    parametric_coordinates=coordinates,
                    baseline_positions=baseline_positions,
                )
            )
        return tuple(groups)

    def _build_seam_prescribed_index(
        self, prescribed_local: Mapping[int, int]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        by_solution: dict[int, list[tuple[int, int]]] = {}
        for vertex_id, (solution_index, solution_row) in self._seam_row_source.items():
            if vertex_id not in prescribed_local:
                continue
            by_solution.setdefault(solution_index, []).append(
                (prescribed_local[vertex_id], solution_row)
            )
        index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for solution_index, pairs in by_solution.items():
            parameters = self.intersection_params[solution_index]
            prescribed_rows = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
            solution_rows = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
            driving_coordinates = np.asarray(
                parameters.parametric_coords, dtype=float
            )[solution_rows]
            index[solution_index] = (prescribed_rows, solution_rows, driving_coordinates)
        return index

    def _build_seam_query_reference(
        self,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Seam coordinates on each intersection's QUERY component.

        Returns ``solution_index -> (parametric_coords, baseline_positions)``
        aligned with that intersection's ``vertex_ids``.  Used to subtract the
        query component's own rigid motion from the seam departure seen by the
        query-side free block; with a stationary query component both terms
        vanish and this reduces to the plain seam displacement.
        """
        from gamma_mdo.preprocessing import project_mesh_onto_components

        reference: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if not self.use_query_seam_reference:
            return reference
        for solution_index, parameters in enumerate(self.intersection_params):
            query = parameters.sdf_query_component
            if parameters.vertex_ids is None or query is None:
                continue
            seam_points = self.initial_vertices[
                np.asarray(parameters.vertex_ids, dtype=np.int64)
            ]
            projection = project_mesh_onto_components(
                mesh_vertices=seam_points,
                components=[query],
                projection_options=dict(parameters.projection_options or {}),
            )
            coordinates = np.asarray(
                projection[0].parametric_coordinates, dtype=float
            ).reshape((-1, 3))
            baseline_positions = np.asarray(
                FunctionSetEvaluationModel(query).evaluate(
                    stack_component_coefficients_numpy(query), coordinates
                ),
                dtype=float,
            ).reshape((-1, 3))
            reference[solution_index] = (coordinates, baseline_positions)
        return reference


    def _solve_geometry(
        self,
        *,
        component_coeffs,
        query_component_coeffs: Mapping[object, object] | None = None,
    ):
        driving_coefficients = _resolve_driving_coefficients(
            self.intersection_params, component_coeffs
        )
        query_component_coeffs = dict(query_component_coeffs or {})
        driving_by_id = {
            id(parameters.driving_component): coefficients
            for parameters, coefficients in zip(
                self.intersection_params, driving_coefficients
            )
        }

        # Recompute exact intersections (the geometry front-end).
        solutions: list[IntersectionSolution] = []
        for parameters, coefficients in zip(
            self.intersection_params, driving_coefficients
        ):
            query_coefficients = _component_mapping_get(
                query_component_coeffs,
                parameters.sdf_query_component,
                stack_component_coefficients(parameters.sdf_query_component),
            )
            solutions.append(
                solve_intersection(
                    parameters,
                    driving_coefficients=coefficients,
                    query_coefficients=query_coefficients,
                )
            )
        return (
            driving_by_id,
            query_component_coeffs,
            tuple(solutions),
        )

    def build_load_step_state(
        self,
        *,
        component_coeffs,
        query_component_coeffs: Mapping[object, object] | None = None,
    ) -> GraphLoadStepState:
        """Evaluate exact seams and component-relative graph boundary data.

        The returned quantities are absolute with respect to the baseline
        geometry.  A continuation driver differences consecutive states and
        solves those increments using graph weights from the current projected
        mesh.

        Parameters
        ----------
        component_coeffs
            Deformed driving-component coefficients.
        query_component_coeffs
            Optional coefficient overrides for query components.

        Returns
        -------
        GraphLoadStepState
            Absolute free references, prescribed deviations, and exact seams.
        """
        driving_by_id, query_component_coeffs, solutions = self._solve_geometry(
            component_coeffs=component_coeffs,
            query_component_coeffs=query_component_coeffs,
        )
        reference = self._assemble_free_reference(
            driving_by_id=driving_by_id,
            query_component_coeffs=query_component_coeffs,
        )
        deviations = tuple(
            self._assemble_block_deviation(
                component=group.component,
                solutions=solutions,
                driving_by_id=driving_by_id,
                query_component_coeffs=query_component_coeffs,
                seam_index=self._seam_prescribed_by_solution,
                num_prescribed=int(self.prescribed_ids.size),
            )
            for group in self._free_reference_groups
        )
        if self.symmetry_plane_ids is None:
            deviations_y = ()
        else:
            deviations_y = tuple(
                self._assemble_block_deviation(
                    component=group.component,
                    solutions=solutions,
                    driving_by_id=driving_by_id,
                    query_component_coeffs=query_component_coeffs,
                    seam_index=self._seam_prescribed_by_solution_y,
                    num_prescribed=int(self.prescribed_ids_y.size),
                )
                for group in self._free_reference_groups_y
            )
        return GraphLoadStepState(
            free_reference=reference,
            prescribed_deviations=deviations,
            prescribed_deviations_y=deviations_y,
            solutions=solutions,
        )

    def train(
        self,
        *,
        component_coeffs,
        query_component_coeffs: Mapping[object, object] | None = None,
    ) -> "ElasticityMotionField":
        """Solve the reference-config graph motion for one geometry state.

        Parameters
        ----------
        component_coeffs
            Deformed driving-component coefficients.
        query_component_coeffs
            Optional coefficient overrides for query components.

        Returns
        -------
        ElasticityMotionField
            Field combining graph displacements, exact reevaluations, and seam
            overrides.
        """
        driving_by_id, query_component_coeffs, solutions = self._solve_geometry(
            component_coeffs=component_coeffs,
            query_component_coeffs=query_component_coeffs,
        )

        # Owner-reference displacement per free row plus a per-component
        # deviation solve. L_ff is block diagonal across component blocks.
        reference = self._assemble_free_reference(
            driving_by_id=driving_by_id,
            query_component_coeffs=query_component_coeffs,
        )
        if self.symmetry_plane_ids is None:
            deviation_rhs = self._build_deviation_rhs(
                system=self.system,
                free_reference_groups=self._free_reference_groups,
                seam_index=self._seam_prescribed_by_solution,
                num_prescribed=int(self.prescribed_ids.size),
                num_free=int(self.free_ids.size),
                solutions=solutions,
                driving_by_id=driving_by_id,
                query_component_coeffs=query_component_coeffs,
            )
            free_deviation = SPDSolveOperation(self.system.factor).evaluate(
                deviation_rhs
            )
            u_f = reference + free_deviation
        else:
            u_f = reference + self._solve_symmetric_deviation(
                solutions=solutions,
                driving_by_id=driving_by_id,
                query_component_coeffs=query_component_coeffs,
            )

        return ElasticityMotionField(
            free_ids=self.free_ids,
            free_local=self._free_local,
            free_reference=self.initial_vertices[self.free_ids],
            free_displacement=u_f,
            solutions=solutions,
            component_reevaluations=self.component_reevaluations,
            reevaluation_coefficients=self._resolve_reevaluation_coefficients(
                driving_by_id, query_component_coeffs
            ),
        )

    def _assemble_free_reference(
        self,
        *,
        driving_by_id: Mapping[int, object],
        query_component_coeffs: Mapping[object, object],
    ) -> csdl.Variable:
        """Owner-component rigid displacement at each free vertex (the reference)."""
        num_free = int(self.free_ids.size)
        reference = csdl.Variable(value=np.zeros((num_free, 3), dtype=float))
        for group in self._free_reference_groups:
            coefficients = _resolve_component_coefficients(
                group.component, driving_by_id, query_component_coeffs
            )
            moved = FunctionSetEvaluationOperation(
                FunctionSetEvaluationModel(group.component)
            ).evaluate(coefficients, group.parametric_coordinates)
            displacement = moved - group.baseline_positions
            reference = reference.set(_row_slice(group.free_rows), displacement)
        return reference

    def _assemble_block_deviation(
        self,
        *,
        component: object,
        solutions: Sequence[IntersectionSolution],
        driving_by_id: Mapping[int, object],
        query_component_coeffs: Mapping[object, object],
        seam_index: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
        num_prescribed: int,
    ) -> csdl.Variable:
        """Seam departure from ``component``'s rigid motion (zero elsewhere).

        For a seam driven by ``component`` the departure is the exact seam
        position minus where the component's rigid motion places that seam's
        parametric point -- so it vanishes at the rigid boundary.  For a seam
        bordering a non-driving (e.g. fuselage) block, the reference is the
        component's rigid motion at the seam, approximated by the baseline seam
        position (exact when the component is stationary), which recovers the
        full seam displacement.  ``seam_index`` / ``num_prescribed`` select the
        target system (full free set, or the y-symmetry subset).
        """
        deviation = csdl.Variable(value=np.zeros((num_prescribed, 3), dtype=float))
        for solution_index, (
            prescribed_rows,
            solution_rows,
            driving_coordinates,
        ) in seam_index.items():
            solution = solutions[solution_index]
            deformed_seam = solution.deformed_vertices[_row_slice(solution_rows)]
            parameters = self.intersection_params[solution_index]
            if parameters.driving_component is component:
                rigid = FunctionSetEvaluationOperation(
                    FunctionSetEvaluationModel(component)
                ).evaluate(driving_by_id[id(component)], driving_coordinates)
                departure = deformed_seam - rigid
            else:
                departure = deformed_seam - solution.initial_vertices[solution_rows]
                query_reference = self._seam_query_reference.get(solution_index)
                if (
                    query_reference is not None
                    and parameters.sdf_query_component is component
                ):
                    # Subtract this (query) component's own rigid motion at the
                    # seam, so a moving fuselage leaves only the true departure.
                    coordinates, baseline_positions = query_reference
                    coefficients = _resolve_component_coefficients(
                        component, driving_by_id, query_component_coeffs
                    )
                    moved = FunctionSetEvaluationOperation(
                        FunctionSetEvaluationModel(component)
                    ).evaluate(coefficients, coordinates[solution_rows])
                    departure = departure - (
                        moved - baseline_positions[solution_rows]
                    )
            deviation = deviation.set(_row_slice(prescribed_rows), departure)
        return deviation

    def _build_deviation_rhs(
        self,
        *,
        system,
        free_reference_groups: Sequence["_FreeReferenceGroup"],
        seam_index: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
        num_prescribed: int,
        num_free: int,
        solutions: Sequence[IntersectionSolution],
        driving_by_id: Mapping[int, object],
        query_component_coeffs: Mapping[object, object],
    ) -> csdl.Variable:
        """Per-block ``rhs = -L_fp w_p`` over one system's free set."""
        rhs = csdl.Variable(value=np.zeros((num_free, 3), dtype=float))
        for group in free_reference_groups:
            block_deviation = self._assemble_block_deviation(
                component=group.component,
                solutions=solutions,
                driving_by_id=driving_by_id,
                query_component_coeffs=query_component_coeffs,
                seam_index=seam_index,
                num_prescribed=num_prescribed,
            )
            block_rhs = -1.0 * csdl.sparse.matmat(system.coupling, block_deviation)
            rhs = rhs.set(
                _row_slice(group.free_rows),
                block_rhs[_row_slice(group.free_rows)],
            )
        return rhs

    def _solve_symmetric_deviation(
        self,
        *,
        solutions: Sequence[IntersectionSolution],
        driving_by_id: Mapping[int, object],
        query_component_coeffs: Mapping[object, object],
    ) -> csdl.Variable:
        """Deviation over the full free set with the y-symmetry plane condition.

        x/z solve on the main system (plane free); y solves on the second
        factorization (plane pinned to 0), then scatters back into the full free
        rows with the plane rows left at zero.
        """
        rhs_xz = self._build_deviation_rhs(
            system=self.system,
            free_reference_groups=self._free_reference_groups,
            seam_index=self._seam_prescribed_by_solution,
            num_prescribed=int(self.prescribed_ids.size),
            num_free=int(self.free_ids.size),
            solutions=solutions,
            driving_by_id=driving_by_id,
            query_component_coeffs=query_component_coeffs,
        )
        deviation_xz = SPDSolveOperation(self.system.factor).evaluate(rhs_xz)

        rhs_y = self._build_deviation_rhs(
            system=self.system_y,
            free_reference_groups=self._free_reference_groups_y,
            seam_index=self._seam_prescribed_by_solution_y,
            num_prescribed=int(self.prescribed_ids_y.size),
            num_free=int(self.free_ids_y.size),
            solutions=solutions,
            driving_by_id=driving_by_id,
            query_component_coeffs=query_component_coeffs,
        )
        deviation_y_free = SPDSolveOperation(self.system_y.factor).evaluate(rhs_y)

        # Assemble the full-free-set deviation: x/z from the main solve, y from
        # the plane-pinned solve scattered back (plane rows stay 0).
        deviation_y = csdl.sparse.matmat(
            self._scatter_y, deviation_y_free[csdl.slice[:, 1:2]]
        )
        return csdl.concatenate(
            (
                deviation_xz[csdl.slice[:, 0:1]],
                deviation_y,
                deviation_xz[csdl.slice[:, 2:3]],
            ),
            axis=1,
        )

    def _resolve_reevaluation_coefficients(
        self,
        driving_by_id: Mapping[int, object],
        query_component_coeffs: Mapping[object, object],
    ) -> dict[int, object]:
        resolved: dict[int, object] = {}
        for reevaluation in self.component_reevaluations:
            resolved[id(reevaluation.component)] = _resolve_component_coefficients(
                reevaluation.component,
                driving_by_id,
                query_component_coeffs,
            )
        return resolved



@dataclass(frozen=True)
class _FreeReferenceGroup:
    """Free vertices owned by one component, with their baseline reeval points."""

    component: object
    free_rows: np.ndarray  # local indices into free_ids
    parametric_coordinates: np.ndarray
    baseline_positions: np.ndarray


class ElasticityMotionField:
    """Preprojected positions from a solved elastic mesh-motion field.

    ``evaluate`` returns, for every requested mesh row: the elastic result on
    the free band (reference + ``u_f``), exact parametric reevaluation on
    wing/tail rows, and the exact bisection override on seam rows -- the same
    row structure the RBF surrogate produces, so the projection back-end is
    unchanged.

    Parameters
    ----------
    free_ids
        Global graph-solved vertex IDs.
    free_local
        Mapping from global free ID to solver-row index.
    free_reference
        Baseline reference positions for free rows.
    free_displacement
        Solved differentiable free-row displacement.
    solutions
        Exact seam solutions used as final overrides.
    component_reevaluations
        Groups evaluated at fixed component parametric coordinates.
    reevaluation_coefficients
        Deformed component coefficients keyed by component identity.
    """

    def __init__(
        self,
        *,
        free_ids: np.ndarray,
        free_local: Mapping[int, int],
        free_reference: np.ndarray,
        free_displacement: csdl.Variable,
        solutions: Sequence[IntersectionSolution],
        component_reevaluations: Sequence[ComponentReevaluation],
        reevaluation_coefficients: Mapping[int, object],
    ):
        self.free_ids = np.asarray(free_ids, dtype=np.int64)
        self._free_local = dict(free_local)
        self.free_reference = np.asarray(free_reference, dtype=float).reshape((-1, 3))
        self.free_displacement = free_displacement
        self.solutions = tuple(solutions)
        self.component_reevaluations = tuple(component_reevaluations)
        self.reevaluation_coefficients = dict(reevaluation_coefficients)

    def evaluate(self, *, vertices: csdl.Variable, vertex_ids: np.ndarray) -> csdl.Variable:
        """Evaluate graph motion, reevaluations, and seam overrides.

        Parameters
        ----------
        vertices
            Query coordinates aligned with ``vertex_ids``.
        vertex_ids
            Global IDs of the query rows.

        Returns
        -------
        csdl.Variable
            Preprojected coordinates in query-row order.

        Raises
        ------
        ValueError
            If IDs do not align with query rows or reevaluation metadata.
        """
        query_points = np.asarray(
            getattr(vertices, "value", vertices), dtype=float
        ).reshape((-1, 3))
        query_variable = (
            vertices if hasattr(vertices, "value") else csdl.Variable(value=query_points)
        )
        ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
        if ids.size != query_points.shape[0]:
            raise ValueError("vertex_ids must align with the query vertices.")
        row_by_id = {int(vertex): row for row, vertex in enumerate(ids)}

        deformed = query_variable * 1.0

        # Free (fuselage) rows: reference + elastic displacement.
        free_query_rows = []
        free_solve_rows = []
        for vertex_id, query_row in row_by_id.items():
            solve_row = self._free_local.get(vertex_id)
            if solve_row is not None:
                free_query_rows.append(query_row)
                free_solve_rows.append(solve_row)
        if free_query_rows:
            free_query_rows = np.asarray(free_query_rows, dtype=np.int64)
            free_solve_rows = np.asarray(free_solve_rows, dtype=np.int64)
            deformed = deformed.set(
                _row_slice(free_query_rows),
                query_variable[_row_slice(free_query_rows)]
                + self.free_displacement[_row_slice(free_solve_rows)],
            )

        # Wing/tail owned rows: exact parametric reevaluation (overrides).
        for reevaluation in self.component_reevaluations:
            coefficients = self.reevaluation_coefficients[id(reevaluation.component)]
            metadata_row_by_id = {
                int(vertex): row
                for row, vertex in enumerate(reevaluation.vertex_ids)
            }
            selected = [
                int(vertex) for vertex in ids if int(vertex) in metadata_row_by_id
            ]
            if not selected:
                continue
            query_rows = np.asarray(
                [row_by_id[vertex] for vertex in selected], dtype=np.int64
            )
            metadata_rows = np.asarray(
                [metadata_row_by_id[vertex] for vertex in selected], dtype=np.int64
            )
            moved = FunctionSetEvaluationOperation(
                FunctionSetEvaluationModel(reevaluation.component)
            ).evaluate(
                coefficients,
                reevaluation.parametric_coordinates[metadata_rows],
            )
            deformed = deformed.set(_row_slice(query_rows), moved)

        # Seam rows: exact bisection override (applied last, as the RBF does).
        for solution in self.solutions:
            if solution.vertex_ids is None:
                continue
            for solution_row, vertex_id in enumerate(solution.vertex_ids):
                query_row = row_by_id.get(int(vertex_id))
                if query_row is None:
                    continue
                deformed = deformed.set(
                    csdl.slice[query_row : query_row + 1, :],
                    solution.deformed_vertices[
                        csdl.slice[solution_row : solution_row + 1, :]
                    ],
                )
        return deformed


# ---------------------------------------------------------------------------
# Coefficient resolution helpers
# ---------------------------------------------------------------------------


def _resolve_driving_coefficients(parameters, component_coeffs):
    if isinstance(component_coeffs, Mapping):
        resolved = []
        for item in parameters:
            missing = object()
            value = _component_mapping_get(
                component_coeffs, item.driving_component, missing
            )
            if value is missing:
                raise KeyError(
                    "Missing coefficients for an intersection driving component."
                )
            resolved.append(value)
        return resolved
    resolved = list(component_coeffs)
    if len(resolved) != len(parameters):
        raise ValueError("component_coeffs must align with intersection_params.")
    return resolved


def _resolve_component_coefficients(component, driving_by_id, query_component_coeffs):
    """Deformed coefficients for a component (driving, then query, then base)."""
    if id(component) in driving_by_id:
        return driving_by_id[id(component)]
    coefficients = _component_mapping_get(query_component_coeffs, component, None)
    if coefficients is None:
        coefficients = stack_component_coefficients(component)
    return coefficients


def _patch_to_component_map(components) -> dict[int, object]:
    mapping: dict[int, object] = {}
    for component in components:
        for patch_id in component.functions:
            mapping[int(patch_id)] = component
    return mapping


def _component_mapping_get(mapping, component, default):
    try:
        return mapping.get(component, default)
    except TypeError:
        return mapping.get(id(component), default)


def _row_slice(rows):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, :]
    return csdl.slice[rows.tolist(), :]


__all__ = [
    "ComponentReevaluation",
    "ElasticityMotionField",
    "ElasticityMotionSolver",
    "GraphLoadStepState",
    "MeshMotionField",
    "MeshMotionSolver",
    "RBFMotionSolver",
]
