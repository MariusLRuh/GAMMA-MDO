"""Single-field RBF propagation of exact intersection displacements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetClosestDistanceOperation,
    FunctionSetProjectionModel,
)
from gamma_mdo.core.projections.function_set_evaluation_custom_op import (
    FunctionSetEvaluationModel,
    FunctionSetEvaluationOperation,
)
from gamma_mdo.preprocessing.mesh_io import _as_mesh_data

from .geometry import (
    stack_component_coefficients,
    stack_component_coefficients_numpy,
)
from .intersections import (
    IntersectionParameters,
    IntersectionSolution,
    solve_intersection,
)


@dataclass(frozen=True)
class ComponentDisplacementData:
    """Fixed component coordinates used as RBF displacement observations.

    ``vertex_ids`` lists the mesh vertices covered by this component's data
    and ``parametric_coordinates`` holds their closest-point coordinates on
    this component.  When every component additionally provides ``distances``
    (baseline unsigned projection distances), the trainer builds smoothly
    blended displacement targets from all components; otherwise the
    seam-influence fallback is used and this data only supplies exact
    component-parametric targets for the covered rows.

    ``hard_parametric_targets`` marks components whose owned vertices follow
    the component motion exactly (lifting surfaces).  It defaults to True for
    intersection driving components and False otherwise.

    Attributes
    ----------
    component
        Geometry component these observations belong to; it owns the
        parametric coordinates below.
    vertex_ids
        Mesh vertex indices covered by this component, flattened to shape
        ``(n,)``.
    parametric_coordinates
        Closest-point coordinates on ``component`` for each covered vertex,
        reshaped to ``(n, 3)``. Must align with ``vertex_ids``.
    distances
        Baseline unsigned projection distance per covered vertex, shape
        ``(n,)``; stored as absolute values. ``None`` opts this component out
        of the blended-target path, leaving the seam-influence fallback.
    hard_parametric_targets
        Whether the owned vertices follow the component motion exactly.
        ``None`` defers to the default: ``True`` for intersection driving
        components, ``False`` otherwise.

    Raises
    ------
    ValueError
        At construction, if ``vertex_ids`` and ``parametric_coordinates`` do
        not align, or ``distances`` is given with a different length.
    """

    component: object
    vertex_ids: np.ndarray
    parametric_coordinates: np.ndarray
    distances: np.ndarray | None = None
    hard_parametric_targets: bool | None = None

    def __post_init__(self):
        """Normalize the observation arrays and check that they align.

        Reshapes ``vertex_ids`` to ``(n,)`` and ``parametric_coordinates`` to
        ``(n, 3)``, and stores ``distances`` as absolute values.

        Raises
        ------
        ValueError
            If the ids and coordinates do not align, or ``distances`` is given
            with a different length.
        """
        ids = np.asarray(self.vertex_ids, dtype=np.int64).reshape(-1)
        coordinates = np.asarray(self.parametric_coordinates, dtype=float).reshape((-1, 3))
        if ids.size != coordinates.shape[0]:
            raise ValueError("vertex_ids and parametric_coordinates must align.")
        object.__setattr__(self, "vertex_ids", ids)
        object.__setattr__(self, "parametric_coordinates", coordinates)
        if self.distances is not None:
            distances = np.abs(
                np.asarray(self.distances, dtype=float).reshape(-1)
            )
            if distances.size != ids.size:
                raise ValueError("distances must align with vertex_ids.")
            object.__setattr__(self, "distances", distances)


@dataclass(frozen=True)
class DisplacementInterpolationParameters:
    """Configuration for the global displacement RBF.

    Attributes
    ----------
    intersection_params
        One entry per intersection driving the displacement field, stored as a
        tuple. Must be non-empty.
    rbf_kernel
        Radial basis function name, validated at construction against the
        supported kernels.
    fit_mode
        ``"exact_interpolation"`` solves the interpolation system, but the
        ridge term is added to it, so the fit is exact only when
        ``regularization`` is zero and is softened for any nonzero value
        (including the default). ``"normalized_smoothing"`` instead forms a
        normalized kernel average and currently requires the Gaussian kernel,
        raising :exc:`ValueError` for any other.
    rbf_kernel_scale
        Explicit kernel length scale. ``None`` derives one from the training
        geometry.
    regularization
        Ridge term added to the diagonal of the training kernel matrix in the
        ``"exact_interpolation"`` path, trading exactness for conditioning.
        Only ``0.0`` leaves that fit genuinely interpolating.
    num_interpolation_vertices
        Number of mesh vertices selected as RBF centers.
    interpolation_vertex_selection_method
        How those centers are chosen: ``"farthest_point"`` or
        ``"density_based"``.
    component_displacement_data
        Per-component observation sets, stored as a tuple. The blended-target
        path is used only when every entry supplies ``distances``.
    seam_neighbor_blend_radius
        Blend radius around each seam, as one value for all intersections or
        one per intersection. Every value must be non-negative.
    seam_neighbor_component_blend
        Component-motion weight within that radius, as one value for all
        intersections or one per intersection. Every value must lie in
        ``[0, 1]``.
    seam_neighbor_support_cutoff
        Support weight below which a seam neighbor is dropped. Must lie in
        ``[0, 1]``.
    seam_neighbor_training_drive
        Let seam neighbors drive the training targets as well as the
        evaluation blend.
    smoothing_iterations
        Number of post-fit smoothing passes over the displacement field. Zero
        disables smoothing.
    smoothing_relaxation
        Relaxation factor of each smoothing pass. Must lie in ``[0, 1]``.
    sigma_phi
        Width of the blended-target influence function. Must be positive.
    seam_support_sigma
        Absolute width of an extra SDF support band around the deformed seam.
        ``None`` disables the absolute form.
    seam_support_sigma_factor
        Same band expressed as a multiple of each seam's baseline bounding-box
        diagonal. Must be non-negative; ``0.0`` disables it. The band is
        isotropic, so it can leak support past tight influence boxes, and it is
        off by default.
    setup_weight_alpha
        Exponent applied to the setup-time observation weights.
    projection_options
        Extra options forwarded to the projection used to build parametric
        targets. ``None`` uses the projection defaults.

    Raises
    ------
    ValueError
        At construction, if ``intersection_params`` is empty; ``rbf_kernel``,
        ``fit_mode``, or ``interpolation_vertex_selection_method`` is
        unrecognized; ``num_interpolation_vertices`` or
        ``smoothing_iterations`` is negative; ``seam_neighbor_blend_radius`` is
        neither scalar nor one value per intersection, or any of its values is
        negative; ``seam_neighbor_component_blend`` is neither scalar nor one
        value per intersection, or any of its values falls outside ``[0, 1]``;
        ``seam_neighbor_support_cutoff`` or
        ``smoothing_relaxation`` falls outside ``[0, 1]``; ``sigma_phi`` is not
        positive; ``seam_support_sigma`` is supplied and not positive;
        ``seam_support_sigma_factor`` or ``setup_weight_alpha`` is negative; or
        ``component_displacement_data`` supplies ``distances`` for some but not
        all components.
    """

    intersection_params: Sequence[IntersectionParameters]
    rbf_kernel: str = "cubic"
    fit_mode: str = "exact_interpolation"
    rbf_kernel_scale: float | None = None
    regularization: float = 1e-10
    num_interpolation_vertices: int = 1000
    interpolation_vertex_selection_method: str = "farthest_point"
    component_displacement_data: Sequence[ComponentDisplacementData] = ()
    seam_neighbor_blend_radius: float | Sequence[float] = 0.0
    seam_neighbor_component_blend: float | Sequence[float] = 1.0
    seam_neighbor_support_cutoff: float = 1e-3
    seam_neighbor_training_drive: bool = True
    smoothing_iterations: int = 0
    smoothing_relaxation: float = 0.5
    # Soft-blend target construction (active when every component in
    # ``component_displacement_data`` provides baseline distances).  The seam
    # support is carried by the per-intersection influence extents; a positive
    # ``seam_support_sigma_factor`` (times each seam's baseline bounding-box
    # diagonal) or absolute ``seam_support_sigma`` adds an SDF band around the
    # deformed seam for designs whose intersection travels far from the
    # baseline seam.  The band is isotropic, so it leaks support past tight
    # influence boxes (e.g. aft of a tail trailing edge) — off by default.
    sigma_phi: float = 0.1
    seam_support_sigma: float | None = None
    seam_support_sigma_factor: float = 0.0
    setup_weight_alpha: float = 1.0
    projection_options: Mapping | None = None

    def __post_init__(self):
        """Freeze the sequence fields and validate the configuration.

        ``intersection_params`` and ``component_displacement_data`` are stored
        back as tuples.

        Raises
        ------
        ValueError
            If ``intersection_params`` is empty; ``rbf_kernel``, ``fit_mode``,
            or ``interpolation_vertex_selection_method`` is unrecognized;
            ``num_interpolation_vertices`` or ``smoothing_iterations`` is
            negative; ``seam_neighbor_blend_radius`` is neither scalar nor one
            value per intersection or holds a negative value;
            ``seam_neighbor_component_blend`` is neither scalar nor one value
            per intersection or holds a value outside ``[0, 1]``;
            ``seam_neighbor_support_cutoff`` or ``smoothing_relaxation`` falls
            outside ``[0, 1]``; ``sigma_phi`` is not positive;
            ``seam_support_sigma`` is supplied and not positive;
            ``seam_support_sigma_factor`` or ``setup_weight_alpha`` is
            negative; or ``distances`` is supplied for some but not all
            entries of ``component_displacement_data``.
        """
        object.__setattr__(self, "intersection_params", tuple(self.intersection_params))
        object.__setattr__(
            self,
            "component_displacement_data",
            tuple(self.component_displacement_data),
        )
        if not self.intersection_params:
            raise ValueError("intersection_params must contain at least one intersection.")
        if self.rbf_kernel not in {
            "cubic",
            "gaussian",
            "multiquadric",
            "wendland_c2",
        }:
            raise ValueError("Unsupported rbf_kernel.")
        if self.fit_mode not in {"exact_interpolation", "normalized_smoothing"}:
            raise ValueError(
                "fit_mode must be 'exact_interpolation' or 'normalized_smoothing'."
            )
        if int(self.num_interpolation_vertices) < 0:
            raise ValueError("num_interpolation_vertices must be non-negative.")
        if int(self.smoothing_iterations) < 0:
            raise ValueError("smoothing_iterations must be non-negative.")
        _resolve_seam_neighbor_radii(
            self.seam_neighbor_blend_radius,
            len(self.intersection_params),
        )
        _resolve_per_intersection_fractions(
            self.seam_neighbor_component_blend,
            len(self.intersection_params),
        )
        if not 0.0 <= float(self.seam_neighbor_support_cutoff) <= 1.0:
            raise ValueError("seam_neighbor_support_cutoff must lie in [0, 1].")
        if not 0.0 <= float(self.smoothing_relaxation) <= 1.0:
            raise ValueError("smoothing_relaxation must lie in [0, 1].")
        if self.interpolation_vertex_selection_method not in {
            "farthest_point",
            "density_based",
        }:
            raise ValueError(
                "interpolation_vertex_selection_method must be 'farthest_point' "
                "or 'density_based'."
            )
        if float(self.sigma_phi) <= 0.0:
            raise ValueError("sigma_phi must be positive.")
        if (
            self.seam_support_sigma is not None
            and float(self.seam_support_sigma) <= 0.0
        ):
            raise ValueError("seam_support_sigma must be positive.")
        if float(self.seam_support_sigma_factor) < 0.0:
            raise ValueError("seam_support_sigma_factor must be non-negative.")
        if float(self.setup_weight_alpha) < 0.0:
            raise ValueError("setup_weight_alpha must be non-negative.")
        with_distances = sum(
            1
            for item in self.component_displacement_data
            if item.distances is not None
        )
        if with_distances not in (0, len(self.component_displacement_data)):
            raise ValueError(
                "Either every component in component_displacement_data must "
                "provide baseline distances or none may."
            )


class DisplacementInterpolator:
    """Setup-time selector and differentiable RBF trainer.

    Holds the mesh and the interpolation parameters, selects the RBF centers,
    and fits the global displacement field. The most recent
    :class:`DisplacementSurrogate` is retained so the interpolation vertices can
    be plotted afterwards.
    """

    def __init__(self, *, mesh, interpolation_params: DisplacementInterpolationParameters):
        self.mesh = _as_mesh_data(mesh)
        self.parameters = interpolation_params
        self._last_surrogate: DisplacementSurrogate | None = None

    def train(
        self,
        *,
        component_coeffs,
        query_component_coeffs: Mapping[object, object] | None = None,
    ) -> "DisplacementSurrogate":
        """Build exact seam states and fit the global displacement field.

        ``component_coeffs`` may be a sequence aligned with
        ``intersection_params`` or a mapping keyed by driving-component object.
        Query-component coefficients default to their initial values.

        Parameters
        ----------
        component_coeffs
            Deformed coefficients of the driving components, either a sequence
            aligned with ``intersection_params`` or a mapping keyed by driving
            component.
        query_component_coeffs
            Deformed coefficients of the components being queried, keyed by
            component. ``None`` or an absent entry leaves that component at its
            initial coefficients.

        Returns
        -------
        DisplacementSurrogate
            The fitted field, also retained for
            :meth:`plot_interpolation_vertices`.
        """
        driving_coefficients = _resolve_driving_coefficients(
            self.parameters.intersection_params,
            component_coeffs,
        )
        query_component_coeffs = dict(query_component_coeffs or {})

        solutions: list[IntersectionSolution] = []
        resolved_parameters: list[IntersectionParameters] = []
        for index, (parameters, coefficients) in enumerate(
            zip(self.parameters.intersection_params, driving_coefficients)
        ):
            driving_component = parameters.driving_component
            if driving_component is None:
                if isinstance(component_coeffs, Mapping):
                    raise ValueError(
                        "driving_component is required when component_coeffs is a mapping."
                    )
                # Sequence alignment can identify coefficients but cannot infer
                # a FunctionSet.  Require the explicit component for model setup.
                raise ValueError(
                    f"intersection_params[{index}].driving_component is required."
                )
            resolved_parameters.append(parameters)
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

        seam_points = np.vstack(
            [
                solution.initial_vertices[solution.training_rows]
                for solution in solutions
                if solution.training_rows.size
            ]
        )
        seam_ids = _all_seam_ids(solutions)
        interpolation_ids = _select_interpolation_vertices(
            np.asarray(self.mesh.vertices, dtype=float),
            seam_ids,
            int(self.parameters.num_interpolation_vertices),
            method=self.parameters.interpolation_vertex_selection_method,
            intersection_parameters=resolved_parameters,
            intersection_solutions=solutions,
        )
        interpolation_points = np.asarray(self.mesh.vertices, dtype=float)[interpolation_ids]
        training_points = np.vstack((seam_points, interpolation_points))

        seam_displacements = [
            solution.deformed_vertices[_row_slice(solution.training_rows)]
            - solution.initial_vertices[solution.training_rows]
            for solution in solutions
            if solution.training_rows.size
        ]
        seam_displacement_variable = (
            seam_displacements[0]
            if len(seam_displacements) == 1
            else csdl.vstack(tuple(seam_displacements))
        )
        mesh_vertices = np.asarray(self.mesh.vertices, dtype=float)
        seam_neighbor_radii = _resolve_seam_neighbor_radii(
            self.parameters.seam_neighbor_blend_radius,
            len(solutions),
        )
        seam_neighbor_blends = _resolve_per_intersection_fractions(
            self.parameters.seam_neighbor_component_blend,
            len(solutions),
        )
        neighbor_maps = _build_seam_neighbor_maps(
            self.mesh,
            solutions,
            seam_neighbor_radii,
        )

        component_data = self.parameters.component_displacement_data
        use_soft_blend = bool(component_data) and all(
            item.distances is not None for item in component_data
        )
        if use_soft_blend:
            component_coefficients = _resolve_component_coefficients(
                component_data,
                resolved_parameters,
                driving_coefficients,
                query_component_coeffs,
            )
            ownership = _component_ownership(
                component_data,
                mesh_vertices.shape[0],
            )
            hard_flags = _resolve_hard_flags(component_data, resolved_parameters)
            if interpolation_ids.size:
                interpolation_targets = _soft_blended_interpolation_targets(
                    parameters=self.parameters,
                    interpolation_ids=interpolation_ids,
                    interpolation_points=interpolation_points,
                    component_data=component_data,
                    component_coefficients=component_coefficients,
                    intersection_parameters=resolved_parameters,
                    solutions=solutions,
                    ownership=ownership,
                    hard_flags=hard_flags,
                    neighbor_maps=neighbor_maps,
                    seam_neighbor_radii=seam_neighbor_radii,
                    seam_neighbor_component_blends=seam_neighbor_blends,
                )
            component_update_data = tuple(
                (
                    item.component,
                    coefficients,
                    item.vertex_ids[ownership[item.vertex_ids] == item_index],
                    item.parametric_coordinates[
                        ownership[item.vertex_ids] == item_index
                    ],
                )
                for item_index, (item, hard, coefficients) in enumerate(
                    zip(component_data, hard_flags, component_coefficients)
                )
                if hard
            )
        else:
            if interpolation_ids.size:
                influence_matrix = _build_interpolation_influence_matrix(
                    interpolation_points,
                    resolved_parameters,
                    solutions,
                )
                interpolation_targets = csdl.matmat(
                    influence_matrix,
                    seam_displacement_variable,
                )
                interpolation_targets = _apply_component_displacement_observations(
                    interpolation_targets,
                    interpolation_ids,
                    interpolation_points,
                    component_data,
                    self.parameters.intersection_params,
                    driving_coefficients,
                )
            component_update_data = tuple(
                (
                    item.component,
                    _coefficients_for_component(
                        item.component,
                        self.parameters.intersection_params,
                        driving_coefficients,
                    ),
                    item.vertex_ids,
                    item.parametric_coordinates,
                )
                for item in component_data
            )

        training_displacements = (
            csdl.vstack((seam_displacement_variable, interpolation_targets))
            if interpolation_ids.size
            else seam_displacement_variable
        )

        kernel_scale = self.parameters.rbf_kernel_scale
        if kernel_scale is None:
            extent = np.linalg.norm(np.ptp(training_points, axis=0))
            kernel_scale = max(extent / max(np.sqrt(training_points.shape[0]), 1.0), 1e-6)

        surrogate = DisplacementSurrogate(
            training_points=training_points,
            training_displacements=training_displacements,
            kernel=self.parameters.rbf_kernel,
            fit_mode=self.parameters.fit_mode,
            kernel_scale=float(kernel_scale),
            regularization=float(self.parameters.regularization),
            intersection_solutions=tuple(solutions),
            intersection_driving_components=tuple(
                parameters.driving_component
                for parameters in resolved_parameters
            ),
            interpolation_vertex_ids=interpolation_ids,
            component_update_data=component_update_data,
            mesh=self.mesh,
            seam_neighbor_maps=neighbor_maps,
            seam_neighbor_blend_radii=seam_neighbor_radii,
            seam_neighbor_component_blends=seam_neighbor_blends,
            seam_neighbor_support_cutoff=float(
                self.parameters.seam_neighbor_support_cutoff
            ),
            smoothing_iterations=int(self.parameters.smoothing_iterations),
            smoothing_relaxation=float(self.parameters.smoothing_relaxation),
        )
        self._last_surrogate = surrogate
        return surrogate

    def plot_interpolation_vertices(self, *, show=True, plot_influence_regions=False):
        """Plot selected interpolation vertices after :meth:`train`.

        Plotting is kept optional and imported lazily so headless test
        environments do not require PyVista.

        Parameters
        ----------
        show
            Open a blocking render window. ``False`` builds the plot without
            displaying it.
        plot_influence_regions
            Accepted and currently ignored; the influence regions are not
            drawn.

        Returns
        -------
        None
            Rendering is a side effect.

        Raises
        ------
        RuntimeError
            If :meth:`train` has not been called, so no surrogate exists.
        ImportError
            If PyVista is unavailable.
        """
        del plot_influence_regions
        if self._last_surrogate is None:
            raise RuntimeError("train must be called before plotting interpolation vertices.")
        import pyvista as pv

        cloud = pv.PolyData(self._last_surrogate.training_points)
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, color="red", point_size=8, render_points_as_spheres=True)
        if show:
            plotter.show()
        return plotter


class DisplacementSurrogate:
    """Differentiable RBF field with exact seam-row replacement."""

    def __init__(
        self,
        *,
        training_points: np.ndarray,
        training_displacements,
        kernel: str,
        fit_mode: str,
        kernel_scale: float,
        regularization: float,
        intersection_solutions: tuple[IntersectionSolution, ...],
        intersection_driving_components: tuple[object, ...],
        interpolation_vertex_ids: np.ndarray,
        component_update_data: tuple,
        mesh,
        seam_neighbor_maps: tuple,
        seam_neighbor_blend_radii: Sequence[float],
        seam_neighbor_component_blends: Sequence[float],
        seam_neighbor_support_cutoff: float,
        smoothing_iterations: int,
        smoothing_relaxation: float,
    ):
        self.training_points = np.asarray(training_points, dtype=float)
        self.training_displacements = training_displacements
        self.kernel = kernel
        self.fit_mode = fit_mode
        self.kernel_scale = float(kernel_scale)
        self.regularization = float(regularization)
        self.intersection_solutions = intersection_solutions
        self.intersection_driving_components = intersection_driving_components
        self.interpolation_vertex_ids = np.asarray(interpolation_vertex_ids, dtype=np.int64)
        # Entries of (component, deformed coefficients, owned mesh vertex ids,
        # aligned parametric coordinates) whose rows follow their component
        # exactly at evaluation time.
        self.component_update_data = tuple(component_update_data)
        self.mesh = mesh
        self.seam_neighbor_blend_radii = tuple(
            float(radius) for radius in seam_neighbor_blend_radii
        )
        self.seam_neighbor_component_blends = tuple(
            float(value) for value in seam_neighbor_component_blends
        )
        self.seam_neighbor_support_cutoff = float(seam_neighbor_support_cutoff)
        self.smoothing_iterations = int(smoothing_iterations)
        self.smoothing_relaxation = float(smoothing_relaxation)
        self._seam_neighbor_maps = tuple(seam_neighbor_maps)

    def evaluate(self, *, vertices, vertex_ids=None):
        """Evaluate the fitted displacement field at query vertices.

        Parameters
        ----------
        vertices
            Query points of shape ``(n, 3)``, as an array or a CSDL variable. A
            plain array is wrapped in a variable so the result stays
            differentiable.
        vertex_ids
            Mesh indices of those vertices, used to apply the seam-neighbor
            blend. ``None`` evaluates the raw field with no per-vertex blending.

        Returns
        -------
        csdl_alpha.Variable
            **Deformed positions** of shape ``(n, 3)``, aligned with
            ``vertices``: the query points plus the interpolated displacement,
            not the displacement alone.
        """
        query_points = np.asarray(getattr(vertices, "value", vertices), dtype=float).reshape((-1, 3))
        query_variable = (
            vertices
            if hasattr(vertices, "value")
            else csdl.Variable(value=query_points)
        )
        operator = _rbf_evaluation_operator(
            self.training_points,
            query_points,
            kernel=self.kernel,
            fit_mode=self.fit_mode,
            kernel_scale=self.kernel_scale,
            regularization=self.regularization,
        )
        deformed = query_variable + csdl.matmat(operator, self.training_displacements)

        if vertex_ids is not None:
            ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
            row_by_id = {int(vertex_id): row for row, vertex_id in enumerate(ids)}
            # Component-owned vertices have an exact, inexpensive parametric
            # update.  Use it as the projection initial guess; the RBF remains
            # responsible for fuselage/interface motion, and exact seam rows
            # below supersede the fixed-parametric component update.
            for component, coefficients, update_ids, update_coordinates in (
                self.component_update_data
            ):
                metadata_row_by_id = {
                    int(vertex_id): row
                    for row, vertex_id in enumerate(update_ids)
                }
                selected_ids = [
                    int(vertex_id)
                    for vertex_id in ids
                    if int(vertex_id) in metadata_row_by_id
                ]
                if not selected_ids:
                    continue
                query_rows = np.asarray(
                    [row_by_id[vertex_id] for vertex_id in selected_ids],
                    dtype=np.int64,
                )
                metadata_rows = np.asarray(
                    [metadata_row_by_id[vertex_id] for vertex_id in selected_ids],
                    dtype=np.int64,
                )
                model = FunctionSetEvaluationModel(component)
                moved_points = FunctionSetEvaluationOperation(model).evaluate(
                    coefficients,
                    update_coordinates[metadata_rows],
                )
                deformed = deformed.set(
                    _row_slice(query_rows),
                    moved_points,
                )
            # Apply the seam-neighbor drive after component-parametric
            # initialization.  Otherwise the component update above silently
            # overwrites the protection precisely on the lifting-surface side
            # of a mixed seam cell.
            deformed = self._drive_seam_neighbors(
                query_variable=query_variable,
                deformed=deformed,
                vertex_ids=ids,
            )
            for solution in self.intersection_solutions:
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
            if self.smoothing_iterations:
                fixed_ids = _fixed_smoothing_vertex_ids(
                    self.component_update_data,
                    self.intersection_solutions,
                )
                smoothing_matrix = _build_local_smoothing_matrix(
                    self.mesh,
                    ids,
                    fixed_ids=fixed_ids,
                    relaxation=self.smoothing_relaxation,
                )
                displacement = deformed - query_variable
                for _ in range(self.smoothing_iterations):
                    displacement = csdl.sparse.matmat(
                        smoothing_matrix,
                        displacement,
                    )
                deformed = query_variable + displacement
        return deformed

    def _drive_seam_neighbors(self, *, query_variable, deformed, vertex_ids):
        """Blend the first off-seam rows toward exact seam displacement.

        Sparse RBF observations can let a seam-adjacent mesh row lag behind a
        large interface motion even when the seam itself is replaced exactly.
        A fixed graph-distance support closes that gap without changing the
        RBF fit or introducing a geometry-dependent branch in the graph.
        """
        if not any(radius > 0.0 for radius in self.seam_neighbor_blend_radii):
            return deformed
        for (
            solution,
            driving_component,
            neighbor_map,
            radius,
            component_blend,
        ) in zip(
            self.intersection_solutions,
            self.intersection_driving_components,
            self._seam_neighbor_maps,
            self.seam_neighbor_blend_radii,
            self.seam_neighbor_component_blends,
        ):
            if radius <= 0.0 or neighbor_map is None or solution.vertex_ids is None:
                continue
            (graph_distance, _), seam_interpolation = neighbor_map
            distances = graph_distance[vertex_ids]
            support = np.exp(-((distances / radius) ** 2))
            selected_rows = np.where(
                support >= self.seam_neighbor_support_cutoff
            )[0].astype(np.int64)
            if selected_rows.size == 0:
                continue
            seam_displacements = (
                solution.deformed_vertices - solution.initial_vertices
            )
            current_displacements = (
                deformed[_row_slice(selected_rows)]
                - query_variable[_row_slice(selected_rows)]
            )
            interpolated_displacements = csdl.matmat(
                seam_interpolation[vertex_ids[selected_rows]],
                seam_displacements,
            )
            blend = np.repeat(
                support[selected_rows].reshape((-1, 1)),
                3,
                axis=1,
            )
            component_ids = _update_ids_for_component(
                self.component_update_data,
                driving_component,
            )
            if component_ids.size and component_blend < 1.0:
                component_rows = np.isin(vertex_ids[selected_rows], component_ids)
                blend[component_rows] *= component_blend
            driven_displacements = (
                (1.0 - blend) * current_displacements
                + blend * interpolated_displacements
            )
            deformed = deformed.set(
                _row_slice(selected_rows),
                query_variable[_row_slice(selected_rows)]
                + driven_displacements,
            )
        return deformed


def _resolve_driving_coefficients(parameters, component_coeffs):
    if isinstance(component_coeffs, Mapping):
        resolved = []
        for item in parameters:
            missing = object()
            value = _component_mapping_get(
                component_coeffs,
                item.driving_component,
                missing,
            )
            if value is missing:
                raise KeyError(
                    "Missing coefficients for an intersection driving component. "
                    "Unhashable FunctionSet objects may be keyed by id(component)."
                )
            resolved.append(value)
        return resolved
    resolved = list(component_coeffs)
    if len(resolved) != len(parameters):
        raise ValueError("component_coeffs must align with intersection_params.")
    return resolved


def _apply_component_displacement_observations(
    targets,
    interpolation_ids,
    interpolation_points,
    component_data,
    intersection_parameters,
    driving_coefficients,
):
    if not component_data:
        return targets
    coefficients_by_component_id = {
        id(parameters.driving_component): coefficients
        for parameters, coefficients in zip(
            intersection_parameters,
            driving_coefficients,
        )
    }
    interpolation_row_by_id = {
        int(vertex_id): row
        for row, vertex_id in enumerate(interpolation_ids)
    }
    for item in component_data:
        coefficients = coefficients_by_component_id.get(id(item.component))
        if coefficients is None:
            raise KeyError(
                "component_displacement_data contains a component without "
                "deformed coefficients."
            )
        metadata_row_by_id = {
            int(vertex_id): row
            for row, vertex_id in enumerate(item.vertex_ids)
        }
        selected_ids = [
            int(vertex_id)
            for vertex_id in interpolation_ids
            if int(vertex_id) in metadata_row_by_id
        ]
        if not selected_ids:
            continue
        interpolation_rows = np.asarray(
            [interpolation_row_by_id[vertex_id] for vertex_id in selected_ids],
            dtype=np.int64,
        )
        metadata_rows = np.asarray(
            [metadata_row_by_id[vertex_id] for vertex_id in selected_ids],
            dtype=np.int64,
        )
        model = FunctionSetEvaluationModel(item.component)
        moved_points = FunctionSetEvaluationOperation(model).evaluate(
            coefficients,
            item.parametric_coordinates[metadata_rows],
        )
        displacements = moved_points - interpolation_points[interpolation_rows]
        targets = targets.set(
            _row_slice(interpolation_rows),
            displacements,
        )
    return targets


def _component_mapping_get(mapping, component, default):
    try:
        return mapping.get(component, default)
    except TypeError:
        return mapping.get(id(component), default)


def _resolve_seam_neighbor_radii(value, intersection_count):
    radii = np.asarray(value, dtype=float).reshape(-1)
    if radii.size == 1:
        radii = np.repeat(radii, int(intersection_count))
    if radii.size != int(intersection_count):
        raise ValueError(
            "seam_neighbor_blend_radius must be a scalar or contain one "
            "value per intersection."
        )
    if np.any(radii < 0.0):
        raise ValueError("seam_neighbor_blend_radius must be non-negative.")
    return tuple(float(radius) for radius in radii)


def _resolve_per_intersection_fractions(value, intersection_count):
    fractions = np.asarray(value, dtype=float).reshape(-1)
    if fractions.size == 1:
        fractions = np.repeat(fractions, int(intersection_count))
    if fractions.size != int(intersection_count):
        raise ValueError(
            "seam_neighbor_component_blend must be a scalar or "
            "contain one value per intersection."
        )
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("seam_neighbor_component_blend must lie in [0, 1].")
    return tuple(float(fraction) for fraction in fractions)


def _update_ids_for_component(component_update_data, component):
    blocks = [
        update_ids
        for update_component, _, update_ids, _ in component_update_data
        if update_component is component and update_ids.size
    ]
    return (
        np.unique(np.concatenate(blocks))
        if blocks
        else np.empty((0,), dtype=np.int64)
    )


def _build_seam_neighbor_maps(mesh, solutions, radii):
    """Precompute graph distances and the smooth along-seam map per seam."""
    return tuple(
        (
            _mesh_graph_distance_to_sources(mesh, solution.vertex_ids),
            _smooth_seam_interpolation_matrix(
                np.asarray(mesh.vertices, dtype=float),
                solution.initial_vertices,
            ),
        )
        if (
            radius > 0.0
            and solution.vertex_ids is not None
            and solution.vertex_ids.size
        )
        else None
        for solution, radius in zip(solutions, radii)
    )


def _resolve_component_coefficients(
    component_data,
    intersection_parameters,
    driving_coefficients,
    query_component_coeffs,
):
    """Deformed coefficients per component-data item.

    Driving components use their intersection coefficients; every other
    component falls back to the query-coefficient mapping and finally to its
    undeformed coefficient stack.
    """
    resolved = []
    for item in component_data:
        coefficients = None
        for parameters, candidate in zip(
            intersection_parameters,
            driving_coefficients,
        ):
            if parameters.driving_component is item.component:
                coefficients = candidate
                break
        if coefficients is None:
            coefficients = _component_mapping_get(
                query_component_coeffs,
                item.component,
                None,
            )
        if coefficients is None:
            coefficients = stack_component_coefficients(item.component)
        resolved.append(coefficients)
    return resolved


def _component_ownership(component_data, num_vertices):
    """Index of the closest component per mesh vertex (-1 when uncovered)."""
    distance_table = np.full((num_vertices, len(component_data)), np.inf)
    for index, item in enumerate(component_data):
        distance_table[item.vertex_ids, index] = item.distances
    ownership = np.argmin(distance_table, axis=1)
    ownership[~np.isfinite(np.min(distance_table, axis=1))] = -1
    return ownership


def _resolve_hard_flags(component_data, intersection_parameters):
    driving_ids = {
        id(parameters.driving_component)
        for parameters in intersection_parameters
    }
    return tuple(
        bool(item.hard_parametric_targets)
        if item.hard_parametric_targets is not None
        else id(item.component) in driving_ids
        for item in component_data
    )


def _expand_columns(vector, num_columns=3):
    return csdl.matmat(
        csdl.reshape(vector, (-1, 1)),
        np.ones((1, num_columns), dtype=float),
    )


def _soft_blended_interpolation_targets(
    *,
    parameters,
    interpolation_ids,
    interpolation_points,
    component_data,
    component_coefficients,
    intersection_parameters,
    solutions,
    ownership,
    hard_flags,
    neighbor_maps,
    seam_neighbor_radii,
    seam_neighbor_component_blends,
):
    """Geometry-consistent displacement targets for interpolation vertices.

    Each vertex blends the parametric displacements of every component,
    weighted by SDF participation of the deformed geometry evaluated at a
    provisionally moved position.  Near an intersection the target is blended
    toward the sum of both intersecting components' motions, rows owned by a
    hard (lifting-surface) component follow that component exactly, and
    near-seam rows are finally driven toward the smoothly interpolated exact
    seam displacement.
    """
    sigma_phi = float(parameters.sigma_phi)

    # Per-component parametric displacement observed at each interpolation
    # vertex's own closest point on that component.
    count = interpolation_ids.size
    displacements = []
    setup_distances = np.empty((count, len(component_data)), dtype=float)
    for item_index, (item, coefficients) in enumerate(
        zip(component_data, component_coefficients)
    ):
        row_by_id = {
            int(vertex_id): row
            for row, vertex_id in enumerate(item.vertex_ids)
        }
        try:
            rows = np.asarray(
                [row_by_id[int(vertex_id)] for vertex_id in interpolation_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "component_displacement_data must cover every interpolation "
                "vertex when baseline distances are provided."
            ) from error
        coordinates = item.parametric_coordinates[rows]
        evaluation_model = FunctionSetEvaluationModel(item.component)
        baseline_points = np.asarray(
            evaluation_model.evaluate(
                stack_component_coefficients_numpy(item.component),
                coordinates,
            ),
            dtype=float,
        ).reshape((-1, 3))
        moved_points = FunctionSetEvaluationOperation(evaluation_model).evaluate(
            coefficients,
            coordinates,
        )
        displacements.append(moved_points - baseline_points)
        setup_distances[:, item_index] = item.distances[rows]

    # Fixed provisional weights carry each vertex along with its nearby
    # components so the deformed SDFs are probed at post-motion positions.
    setup_weights = np.exp(
        -float(parameters.setup_weight_alpha) * setup_distances
    )
    setup_weights /= np.maximum(
        np.sum(setup_weights, axis=1, keepdims=True),
        1e-300,
    )
    provisional_points = csdl.Variable(value=interpolation_points)
    for weight_column, displacement in zip(setup_weights.T, displacements):
        provisional_points = provisional_points + displacement * np.repeat(
            weight_column.reshape((-1, 1)),
            3,
            axis=1,
        )

    sdf_options = dict(parameters.projection_options or {})
    sdf_options.update(
        {
            "sdf": True,
            "sdf_sign_mode": "normal",
            "output_mode": "distance",
        }
    )
    sdf_values = []
    for item, coefficients in zip(component_data, component_coefficients):
        sdf_model = FunctionSetProjectionModel(item.component, **sdf_options)
        sdf_values.append(
            FunctionSetClosestDistanceOperation(sdf_model).evaluate(
                coefficients,
                provisional_points,
            )
        )

    participations = [
        csdl.exp(-(value**2) * (1.0 / sigma_phi**2)) for value in sdf_values
    ]
    participation_sum = participations[0]
    for participation in participations[1:]:
        participation_sum = participation_sum + participation
    participation_sum = participation_sum + 1e-30

    targets = None
    for participation, displacement in zip(participations, displacements):
        term = displacement * _expand_columns(participation / participation_sum)
        targets = term if targets is None else targets + term

    # Near a deformed intersection, the mesh must follow the seam, which moves
    # with the sum of both components' motions.  The support is the smooth
    # union of two terms: a narrow SDF band (bandwidth scaled by each seam's
    # own extent) that tracks the deformed seam exactly, and the fixed
    # per-intersection influence extents, which carry the seam motion
    # gradually through the deformation region and vanish at its boundary,
    # where the final mesh is reevaluated at fixed parametric coordinates.
    # An SDF band alone concentrates large translations into a few cell rows
    # ahead of the moving component; an aircraft-sized band instead drags
    # distant cells and crushes them against stationary regions (nose, tail
    # cone).
    row_by_component_id = {
        id(item.component): row for row, item in enumerate(component_data)
    }
    for item_parameters, solution in zip(intersection_parameters, solutions):
        driving_row = row_by_component_id.get(
            id(item_parameters.driving_component)
        )
        query_row = row_by_component_id.get(
            id(item_parameters.sdf_query_component)
        )
        if driving_row is None or query_row is None:
            continue
        source_points = solution.initial_vertices[solution.training_rows]
        if source_points.shape[0]:
            difference = (
                interpolation_points[:, None, :] - source_points[None, :, :]
            )
            normalized = _normalized_influence_distance(
                difference,
                item_parameters,
            )
            box_support = item_parameters.weighting_function(
                np.min(normalized, axis=1)
            )
        else:
            box_support = np.zeros(count, dtype=float)
        seam_sigma = item_parameters.seam_support_sigma
        if seam_sigma is None:
            seam_sigma = parameters.seam_support_sigma
        if seam_sigma is None:
            seam_extent = np.linalg.norm(np.ptp(source_points, axis=0))
            seam_sigma = float(parameters.seam_support_sigma_factor) * max(
                seam_extent,
                1e-12,
            )
        if float(seam_sigma) > 0.0:
            indicator = (
                sdf_values[driving_row] ** 2 + sdf_values[query_row] ** 2
            )
            sdf_support = csdl.exp(
                -indicator * (1.0 / float(seam_sigma) ** 2)
            )
            support = _expand_columns(
                sdf_support + (1.0 - sdf_support) * box_support
            )
        else:
            support = np.repeat(box_support.reshape((-1, 1)), 3, axis=1)
        targets = (
            targets
            + (
                displacements[driving_row]
                + displacements[query_row]
                - targets
            )
            * support
        )

    # Rows owned by a hard component follow that component exactly.  Fuselage
    # (soft) rows keep the blended target: pinning them would fight the moving
    # seam support.
    for item_index, hard in enumerate(hard_flags):
        if not hard:
            continue
        owned_rows = np.where(
            ownership[interpolation_ids] == item_index
        )[0].astype(np.int64)
        if owned_rows.size == 0:
            continue
        targets = targets.set(
            _row_slice(owned_rows),
            displacements[item_index][_row_slice(owned_rows)],
        )

    if parameters.seam_neighbor_training_drive:
        targets = _apply_training_seam_neighbor_drive(
            targets,
            interpolation_ids=interpolation_ids,
            component_data=component_data,
            intersection_parameters=intersection_parameters,
            solutions=solutions,
            ownership=ownership,
            hard_flags=hard_flags,
            neighbor_maps=neighbor_maps,
            radii=seam_neighbor_radii,
            component_blends=seam_neighbor_component_blends,
            support_cutoff=float(parameters.seam_neighbor_support_cutoff),
        )
    return targets


def _apply_training_seam_neighbor_drive(
    targets,
    *,
    interpolation_ids,
    component_data,
    intersection_parameters,
    solutions,
    ownership,
    hard_flags,
    neighbor_maps,
    radii,
    component_blends,
    support_cutoff,
):
    """Drive near-seam training targets toward the exact seam displacement."""
    row_by_component_id = {
        id(item.component): row for row, item in enumerate(component_data)
    }
    for item_parameters, solution, maps, radius, component_blend in zip(
        intersection_parameters,
        solutions,
        neighbor_maps,
        radii,
        component_blends,
    ):
        if radius <= 0.0 or maps is None or solution.vertex_ids is None:
            continue
        (graph_distance, _), seam_interpolation = maps
        support = np.exp(
            -((graph_distance[interpolation_ids] / radius) ** 2)
        )
        selected_rows = np.where(support >= support_cutoff)[0].astype(np.int64)
        if selected_rows.size == 0:
            continue
        seam_displacements = (
            solution.deformed_vertices - solution.initial_vertices
        )
        interpolated_displacements = csdl.matmat(
            seam_interpolation[interpolation_ids[selected_rows]],
            seam_displacements,
        )
        blend = np.repeat(
            support[selected_rows].reshape((-1, 1)),
            3,
            axis=1,
        )
        driving_row = row_by_component_id.get(
            id(item_parameters.driving_component)
        )
        if (
            driving_row is not None
            and hard_flags[driving_row]
            and component_blend < 1.0
        ):
            owned_rows = (
                ownership[interpolation_ids[selected_rows]] == driving_row
            )
            blend[owned_rows] *= component_blend
        current = targets[_row_slice(selected_rows)]
        targets = targets.set(
            _row_slice(selected_rows),
            current + blend * (interpolated_displacements - current),
        )
    return targets


def _smooth_seam_interpolation_matrix(vertices, seam_vertices):
    """Map seam motion smoothly along the intersection curve.

    The previous nearest-seam map was discontinuous at Voronoi boundaries:
    adjacent surface vertices could copy different exact seam rows and fold a
    quad even though the seam itself was valid.  A fixed normalized Gaussian
    map preserves differentiability while varying continuously along the seam.
    """
    vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
    seam_vertices = np.asarray(
        getattr(seam_vertices, "value", seam_vertices),
        dtype=float,
    ).reshape((-1, 3))
    pairwise = np.linalg.norm(
        seam_vertices[:, None, :] - seam_vertices[None, :, :],
        axis=2,
    )
    np.fill_diagonal(pairwise, np.inf)
    spacing = float(np.median(np.min(pairwise, axis=1)))
    bandwidth = max(3.0 * spacing, 1e-6)

    squared_distance = np.sum(
        (vertices[:, None, :] - seam_vertices[None, :, :]) ** 2,
        axis=2,
    )
    # Subtracting the row minimum prevents underflow without changing the
    # normalized weights.
    shifted = squared_distance - np.min(squared_distance, axis=1, keepdims=True)
    weights = np.exp(-shifted / bandwidth**2)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _gather_rows(variable, rows):
    """Gather rows with repetition through a constant selection matrix."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    selection = np.zeros((rows.size, variable.shape[0]), dtype=float)
    selection[np.arange(rows.size), rows] = 1.0
    return csdl.matmat(selection, variable)


def _fixed_smoothing_vertex_ids(component_update_data, intersection_solutions):
    blocks = [
        update_ids
        for _, _, update_ids, _ in component_update_data
        if update_ids.size
    ]
    blocks.extend(
        solution.vertex_ids
        for solution in intersection_solutions
        if solution.vertex_ids is not None and solution.vertex_ids.size
    )
    return np.unique(np.concatenate(blocks)) if blocks else np.empty((0,), dtype=np.int64)


def _build_local_smoothing_matrix(mesh, vertex_ids, *, fixed_ids, relaxation):
    """Build a local displacement smoother with a stationary exterior.

    Neighbors outside ``vertex_ids`` represent mesh vertices that are not part
    of the deformation solve, so their displacement is exactly zero.  They
    must still contribute to the averaging denominator.  Omitting them
    imposes a zero-normal-gradient boundary condition and can leave an
    order-one displacement immediately beside an untouched vertex.
    """
    from scipy import sparse

    vertex_ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    fixed = set(np.asarray(fixed_ids, dtype=np.int64).tolist())
    row_by_global_id = {
        int(vertex_id): row
        for row, vertex_id in enumerate(vertex_ids)
    }
    neighbors = [set() for _ in range(mesh.vertices.shape[0])]
    for block in mesh.cell_blocks.values():
        for cell in np.asarray(block, dtype=np.int64):
            for local_index, vertex_id in enumerate(cell):
                vertex_id = int(vertex_id)
                neighbors[vertex_id].update(
                    int(item)
                    for item in np.delete(cell, local_index)
                )

    rows = []
    columns = []
    values = []
    alpha = float(relaxation)
    for row, global_id in enumerate(vertex_ids):
        global_id = int(global_id)
        all_neighbors = neighbors[global_id]
        local_neighbors = [
            row_by_global_id[neighbor]
            for neighbor in all_neighbors
            if neighbor in row_by_global_id
        ]
        if global_id in fixed or not all_neighbors or alpha == 0.0:
            rows.append(row)
            columns.append(row)
            values.append(1.0)
            continue
        rows.append(row)
        columns.append(row)
        values.append(1.0 - alpha)
        # Missing matrix entries for exterior neighbors multiply their known
        # zero displacement.  Dividing by the full valence makes that
        # stationary Dirichlet boundary condition explicit.
        neighbor_weight = alpha / len(all_neighbors)
        rows.extend([row] * len(local_neighbors))
        columns.extend(local_neighbors)
        values.extend([neighbor_weight] * len(local_neighbors))
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(vertex_ids.size, vertex_ids.size),
    )


def _mesh_graph_distance_to_sources(mesh, source_vertex_ids):
    """Return nearest-source geodesic distance and source row on the mesh."""
    import heapq

    points = np.asarray(mesh.vertices, dtype=float)
    source_ids = np.asarray(source_vertex_ids, dtype=np.int64).reshape(-1)
    vertex_count = points.shape[0]
    adjacency: list[dict[int, float]] = [dict() for _ in range(vertex_count)]
    for block in mesh.cell_blocks.values():
        for cell in np.asarray(block, dtype=np.int64):
            if cell.size < 2:
                continue
            for vertex_a, vertex_b in zip(cell, np.roll(cell, -1)):
                vertex_a = int(vertex_a)
                vertex_b = int(vertex_b)
                if vertex_a == vertex_b:
                    continue
                length = float(np.linalg.norm(points[vertex_a] - points[vertex_b]))
                old_length = adjacency[vertex_a].get(vertex_b, np.inf)
                if length < old_length:
                    adjacency[vertex_a][vertex_b] = length
                    adjacency[vertex_b][vertex_a] = length

    distances = np.full(vertex_count, np.inf, dtype=float)
    nearest_source_rows = np.full(vertex_count, -1, dtype=np.int64)
    queue = []
    for source_row, source_id in enumerate(source_ids):
        source_id = int(source_id)
        distances[source_id] = 0.0
        nearest_source_rows[source_id] = source_row
        heapq.heappush(queue, (0.0, source_id, source_row))

    while queue:
        distance, vertex_id, source_row = heapq.heappop(queue)
        if distance > distances[vertex_id] + 1e-15:
            continue
        if (
            abs(distance - distances[vertex_id]) <= 1e-15
            and source_row != nearest_source_rows[vertex_id]
        ):
            continue
        for neighbor_id, edge_length in adjacency[vertex_id].items():
            candidate = distance + edge_length
            if candidate + 1e-15 < distances[neighbor_id]:
                distances[neighbor_id] = candidate
                nearest_source_rows[neighbor_id] = source_row
                heapq.heappush(
                    queue,
                    (candidate, neighbor_id, source_row),
                )
    return distances, nearest_source_rows


def _coefficients_for_component(
    component,
    intersection_parameters,
    driving_coefficients,
):
    for parameters, coefficients in zip(
        intersection_parameters,
        driving_coefficients,
    ):
        if parameters.driving_component is component:
            return coefficients
    raise KeyError("No deformed coefficients supplied for component displacement data.")


def _all_seam_ids(solutions: Sequence[IntersectionSolution]) -> np.ndarray:
    blocks = [
        solution.vertex_ids
        for solution in solutions
        if solution.vertex_ids is not None and solution.vertex_ids.size
    ]
    return np.unique(np.concatenate(blocks)) if blocks else np.empty((0,), dtype=np.int64)


def _select_interpolation_vertices(
    points,
    excluded_ids,
    count,
    *,
    method,
    intersection_parameters=(),
    intersection_solutions=(),
):
    available = np.setdiff1d(
        np.arange(points.shape[0], dtype=np.int64),
        np.asarray(excluded_ids, dtype=np.int64),
        assume_unique=False,
    )
    if count <= 0 or available.size == 0:
        return np.empty((0,), dtype=np.int64)
    count = min(int(count), available.size)
    candidates = points[available]
    if method == "density_based":
        rows = np.unique(
            np.rint(np.linspace(0, available.size - 1, count)).astype(np.int64)
        )
        return available[rows]

    # Reserve most samples for the actual component-interaction supports.
    # Pure global farthest-point sampling undersamples the small wing/fuselage
    # and tail/fuselage bands on a full-aircraft mesh.
    local_selected: list[np.ndarray] = []
    if intersection_solutions:
        local_budget = int(round(0.75 * count))
        per_intersection = max(1, local_budget // len(intersection_solutions))
        already_selected = np.empty((0,), dtype=np.int64)
        for parameters, solution in zip(
            intersection_parameters,
            intersection_solutions,
        ):
            source = solution.initial_vertices[solution.training_rows]
            difference = points[available, None, :] - source[None, :, :]
            normalized = _normalized_influence_distance(difference, parameters)
            local_available = available[np.min(normalized, axis=1) < 1.0]
            local_available = np.setdiff1d(
                local_available,
                already_selected,
                assume_unique=False,
            )
            if local_available.size == 0:
                continue
            local_rows = _farthest_point_rows(
                points[local_available],
                min(per_intersection, local_available.size),
            )
            chosen = local_available[local_rows]
            local_selected.append(chosen)
            already_selected = np.union1d(already_selected, chosen)

    selected_local_ids = (
        np.unique(np.concatenate(local_selected))
        if local_selected
        else np.empty((0,), dtype=np.int64)
    )
    remaining_count = count - selected_local_ids.size
    remaining_available = np.setdiff1d(
        available,
        selected_local_ids,
        assume_unique=False,
    )
    if remaining_count <= 0:
        return np.sort(selected_local_ids[:count])
    global_rows = _farthest_point_rows(
        points[remaining_available],
        min(remaining_count, remaining_available.size),
    )
    return np.sort(
        np.concatenate((selected_local_ids, remaining_available[global_rows]))
    )


def _farthest_point_rows(candidates, count):
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count >= candidates.shape[0]:
        return np.arange(candidates.shape[0], dtype=np.int64)
    centroid = np.mean(candidates, axis=0)
    first = int(np.argmax(np.linalg.norm(candidates - centroid, axis=1)))
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = first
    distance_squared = np.sum((candidates - candidates[first]) ** 2, axis=1)
    distance_squared[first] = -1.0
    for row in range(1, count):
        current = int(np.argmax(distance_squared))
        selected[row] = current
        distance_squared = np.minimum(
            distance_squared,
            np.sum((candidates - candidates[current]) ** 2, axis=1),
        )
        distance_squared[selected[: row + 1]] = -1.0
    return np.sort(selected)


def _build_interpolation_influence_matrix(
    interpolation_points: np.ndarray,
    parameters: Sequence[IntersectionParameters],
    solutions: Sequence[IntersectionSolution],
) -> np.ndarray:
    total_seam_rows = sum(solution.training_rows.size for solution in solutions)
    if interpolation_points.shape[0] == 0:
        return np.empty((0, total_seam_rows), dtype=float)

    matrix = np.zeros((interpolation_points.shape[0], total_seam_rows), dtype=float)
    group_weights = np.zeros((interpolation_points.shape[0], len(solutions)), dtype=float)
    column_offset = 0
    nearest_rows = []
    for group_index, (item, solution) in enumerate(zip(parameters, solutions)):
        source_points = solution.initial_vertices[solution.training_rows]
        difference = interpolation_points[:, None, :] - source_points[None, :, :]
        normalized = _normalized_influence_distance(difference, item)
        nearest = np.argmin(normalized, axis=1)
        nearest_rows.append(nearest)
        group_weights[:, group_index] = item.weighting_function(
            normalized[np.arange(interpolation_points.shape[0]), nearest]
        )
        column_offset += source_points.shape[0]

    normalization = np.sum(group_weights, axis=1)
    normalization = np.where(normalization > 1.0, normalization, 1.0)
    column_offset = 0
    for group_index, (solution, nearest) in enumerate(zip(solutions, nearest_rows)):
        count = solution.training_rows.size
        weights = group_weights[:, group_index] / normalization
        matrix[
            np.arange(interpolation_points.shape[0]),
            column_offset + nearest,
        ] = weights
        column_offset += count
    return matrix


def _normalized_influence_distance(difference, parameters):
    if parameters.influence_ellipsoid_radii is not None:
        radii = np.asarray(parameters.influence_ellipsoid_radii, dtype=float).reshape(3)
        if np.any(radii <= 0.0):
            raise ValueError("influence_ellipsoid_radii must be positive.")
        return np.sqrt(np.sum((difference / radii) ** 2, axis=2))

    signed_extents = [
        np.asarray(parameters.influence_x, dtype=float),
        np.asarray(parameters.influence_y, dtype=float),
        np.asarray(parameters.influence_z, dtype=float),
    ]
    normalized_axes = []
    for axis, extents in enumerate(signed_extents):
        if extents.size != 2 or np.any(np.abs(extents) <= 0.0):
            raise ValueError("Each directional influence must contain two non-zero extents.")
        positive = abs(float(extents[0]))
        negative = abs(float(extents[1]))
        radius = np.where(difference[:, :, axis] >= 0.0, positive, negative)
        normalized_axes.append(np.abs(difference[:, :, axis]) / radius)
    return np.max(np.stack(normalized_axes, axis=2), axis=2)


def _kernel_matrix(points_a, points_b, *, kernel, kernel_scale):
    distances = np.linalg.norm(
        np.asarray(points_a)[:, None, :] - np.asarray(points_b)[None, :, :],
        axis=2,
    )
    scaled = distances / max(float(kernel_scale), 1e-12)
    if kernel == "cubic":
        return distances**3
    if kernel == "gaussian":
        return np.exp(-(scaled**2))
    if kernel == "multiquadric":
        return np.sqrt(1.0 + scaled**2)
    if kernel == "wendland_c2":
        compact = np.clip(1.0 - scaled, 0.0, 1.0)
        return compact**4 * (4.0 * scaled + 1.0)
    raise ValueError(f"Unsupported RBF kernel {kernel!r}.")


def _rbf_evaluation_operator(
    training_points,
    query_points,
    *,
    kernel,
    fit_mode,
    kernel_scale,
    regularization,
):
    training_points = np.asarray(training_points, dtype=float)
    query_points = np.asarray(query_points, dtype=float)
    if fit_mode == "normalized_smoothing":
        if kernel != "gaussian":
            raise ValueError(
                "normalized_smoothing currently requires the non-negative "
                "decaying Gaussian kernel."
            )
        query_kernel = _kernel_matrix(
            query_points,
            training_points,
            kernel=kernel,
            kernel_scale=kernel_scale,
        )
        row_sum = np.sum(query_kernel, axis=1, keepdims=True)
        operator = query_kernel / np.maximum(row_sum, 1e-15)
        unsupported = row_sum[:, 0] <= 1e-15
        if np.any(unsupported):
            nearest = np.argmin(
                np.linalg.norm(
                    query_points[unsupported, None, :]
                    - training_points[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            operator[unsupported] = 0.0
            operator[np.where(unsupported)[0], nearest] = 1.0
        return operator
    if fit_mode != "exact_interpolation":
        raise ValueError(f"Unsupported RBF fit mode {fit_mode!r}.")
    count = training_points.shape[0]
    if kernel == "wendland_c2":
        training_kernel = _kernel_matrix(
            training_points,
            training_points,
            kernel=kernel,
            kernel_scale=kernel_scale,
        )
        coefficient_operator = np.linalg.solve(
            training_kernel + float(regularization) * np.eye(count),
            np.eye(count),
        )
        return _kernel_matrix(
            query_points,
            training_points,
            kernel=kernel,
            kernel_scale=kernel_scale,
        ) @ coefficient_operator
    full_polynomial = np.column_stack((np.ones(count), training_points))
    polynomial_columns = _independent_columns(full_polynomial)
    polynomial = full_polynomial[:, polynomial_columns]
    polynomial_size = polynomial.shape[1]
    kernel_matrix = _kernel_matrix(
        training_points,
        training_points,
        kernel=kernel,
        kernel_scale=kernel_scale,
    )
    system = np.block(
        [
            [kernel_matrix + float(regularization) * np.eye(count), polynomial],
            [
                polynomial.T,
                np.zeros((polynomial_size, polynomial_size), dtype=float),
            ],
        ]
    )
    right_inverse = np.linalg.solve(
        system,
        np.vstack(
            (
                np.eye(count),
                np.zeros((polynomial_size, count), dtype=float),
            )
        ),
    )
    query_matrix = np.hstack(
        (
            _kernel_matrix(
                query_points,
                training_points,
                kernel=kernel,
                kernel_scale=kernel_scale,
            ),
            np.column_stack((np.ones(query_points.shape[0]), query_points))[
                :, polynomial_columns
            ],
        )
    )
    return query_matrix @ right_inverse


def _independent_columns(matrix, *, tolerance=1e-12):
    selected = []
    rank = 0
    for column in range(matrix.shape[1]):
        candidate = matrix[:, selected + [column]]
        candidate_rank = np.linalg.matrix_rank(candidate, tol=tolerance)
        if candidate_rank > rank:
            selected.append(column)
            rank = candidate_rank
    return np.asarray(selected, dtype=np.int64)


def _row_slice(rows):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, :]
    return csdl.slice[rows.tolist(), :]


__all__ = [
    "ComponentDisplacementData",
    "DisplacementInterpolationParameters",
    "DisplacementInterpolator",
    "DisplacementSurrogate",
]
