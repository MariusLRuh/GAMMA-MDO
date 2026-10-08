"""Differentiable, bracketed component-intersection solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
from gamma_mdo.core.weighting_functions import InverseDistanceWeighting, WeightingFunction

from .geometry import (
    stack_component_coefficients,
    stack_component_coefficients_numpy,
)


@dataclass(frozen=True)
class IntersectionParameters:
    """Configure one driving/query component intersection.

    Parameters
    ----------
    parametric_coords
        Setup-time ``[patch, u, v]`` coordinates on the driving component.
    sdf_query_component
        Component supplying the signed-distance residual.
    driving_component
        Component evaluated at the parametric coordinates.
    vertex_ids
        Optional global mesh IDs aligned with the seam rows.
    bisection_search_direction
        Driving parametric coordinate varied by the solve.
    bisection_tolerance, bisection_max_iter
        Bracketed-search convergence controls.
    vertices_to_include
        Seam rows retained as propagation training data.
    influence_x, influence_y, influence_z
        Axis-aligned influence extents.
    influence_ellipsoid_radii
        Optional ellipsoidal influence radii.
    seam_support_sigma
        Optional nonnegative soft seam-support bandwidth.
    weighting_function
        Distance weighting used by propagation.
    projection_options
        Options forwarded to the signed-distance model.
    print_status
        Whether the nonlinear solver prints convergence status.
    name
        Stable diagnostic and implicit-state name.
    """

    parametric_coords: np.ndarray
    sdf_query_component: object
    driving_component: object | None = None
    vertex_ids: np.ndarray | None = None
    bisection_search_direction: Literal["u", "v"] = "v"
    bisection_tolerance: float = 1e-10
    bisection_max_iter: int = 80
    vertices_to_include: str | float | int | np.ndarray = "all"
    influence_x: tuple[float, float] | list[float] = (2.0, -2.0)
    influence_y: tuple[float, float] | list[float] = (1.0, -1.0)
    influence_z: tuple[float, float] | list[float] = (0.5, -0.5)
    influence_ellipsoid_radii: tuple[float, float, float] | None = None
    # Absolute bandwidth of the SDF seam-support band around the deformed
    # intersection (soft-blend targets only).  ``None`` falls back to the
    # interpolation parameters; ``0`` disables the band for this
    # intersection — required when a tight influence extent (e.g. aft of a
    # tail trailing edge) must not be leaked past by the isotropic band.
    seam_support_sigma: float | None = None
    weighting_function: WeightingFunction = InverseDistanceWeighting()
    projection_options: dict[str, Any] | None = None
    print_status: bool = False
    name: str = "component_intersection"

    def __post_init__(self):
        """Normalize the parametric coordinates and check the vertex ids align.

        Reshapes ``parametric_coords`` to ``(n, 3)`` and stores it back on the
        frozen instance.

        Raises
        ------
        ValueError
            If ``vertex_ids`` is given and does not align with the coordinates.
        """
        coordinates = np.asarray(self.parametric_coords, dtype=float).reshape((-1, 3))
        object.__setattr__(self, "parametric_coords", coordinates)
        if self.vertex_ids is not None:
            ids = np.asarray(self.vertex_ids, dtype=np.int64).reshape(-1)
            if ids.size != coordinates.shape[0]:
                raise ValueError("vertex_ids and parametric_coords must have the same length.")
            object.__setattr__(self, "vertex_ids", ids)
        if self.bisection_search_direction not in ("u", "v"):
            raise ValueError("bisection_search_direction must be 'u' or 'v'.")
        if (
            self.seam_support_sigma is not None
            and float(self.seam_support_sigma) < 0.0
        ):
            raise ValueError("seam_support_sigma must be non-negative.")
        if float(self.bisection_tolerance) <= 0.0:
            raise ValueError("bisection_tolerance must be positive.")
        if int(self.bisection_max_iter) <= 0:
            raise ValueError("bisection_max_iter must be positive.")


@dataclass(frozen=True)
class IntersectionSolution:
    """Store baseline and current exact intersection vertices.

    Attributes
    ----------
    initial_vertices
        Baseline NumPy intersection coordinates.
    deformed_vertices
        Differentiable current intersection coordinates.
    vertex_ids
        Optional aligned global mesh IDs.
    training_rows
        Local seam rows retained for motion propagation.
    """

    initial_vertices: np.ndarray
    deformed_vertices: csdl.Variable
    vertex_ids: np.ndarray | None
    training_rows: np.ndarray


def solve_intersection(
    parameters: IntersectionParameters,
    *,
    driving_coefficients,
    query_coefficients=None,
) -> IntersectionSolution:
    """Solve an intersection with CSDL's elementwise bracketed search.

    The fixed parametric coordinate and patch ID come from setup-time
    projection.  The other coordinate is an implicit state bracketed by the
    full ``[0, 1]`` parametric span of that patch.  The residual is the query
    component SDF in normal-sign mode.

    Parameters
    ----------
    parameters
        Fixed intersection geometry and bracket settings.
    driving_coefficients
        Current coefficients of the driving component.
    query_coefficients
        Optional current query-component coefficients.

    Returns
    -------
    IntersectionSolution
        Baseline and differentiable current seam coordinates.

    Raises
    ------
    ValueError
        If no driving component is configured.
    """
    driving_component = parameters.driving_component
    if driving_component is None:
        raise ValueError(
            "IntersectionParameters.driving_component must be set before solving."
        )
    query_coefficients = (
        stack_component_coefficients(parameters.sdf_query_component)
        if query_coefficients is None
        else query_coefficients
    )

    evaluation_model = FunctionSetEvaluationModel(driving_component)
    sdf_options = dict(parameters.projection_options or {})
    sdf_options.update(
        {
            "sdf": True,
            "sdf_sign_mode": "normal",
            "output_mode": "distance",
        }
    )
    sdf_model = FunctionSetProjectionModel(
        parameters.sdf_query_component,
        **sdf_options,
    )

    coordinates = parameters.parametric_coords
    seed_vertices = evaluation_model.evaluate(
        stack_component_coefficients_numpy(driving_component),
        coordinates,
    )
    if coordinates.shape[0] == 0:
        return IntersectionSolution(
            initial_vertices=seed_vertices,
            deformed_vertices=csdl.Variable(value=np.empty((0, 3), dtype=float)),
            vertex_ids=parameters.vertex_ids,
            training_rows=np.empty((0,), dtype=np.int64),
        )

    solve_axis = 0 if parameters.bisection_search_direction == "u" else 1
    solve_column = 1 + solve_axis
    fixed_column = 1 + (1 - solve_axis)
    num_vertices = coordinates.shape[0]

    state = csdl.ImplicitVariable(
        name=f"{parameters.name}_parametric_{parameters.bisection_search_direction}",
        value=np.clip(coordinates[:, solve_column], 1e-10, 1.0 - 1e-10),
    )
    state_column = csdl.reshape(state, (num_vertices, 1))
    patch_column = coordinates[:, 0:1]
    fixed_coordinate = coordinates[:, fixed_column : fixed_column + 1]
    if solve_axis == 0:
        current_coordinates = csdl.concatenate(
            (patch_column, state_column, fixed_coordinate),
            axis=1,
        )
    else:
        current_coordinates = csdl.concatenate(
            (patch_column, fixed_coordinate, state_column),
            axis=1,
        )

    evaluation_operation = FunctionSetEvaluationOperation(evaluation_model)
    sdf_operation = FunctionSetClosestDistanceOperation(sdf_model)
    current_vertices = evaluation_operation.evaluate(
        driving_coefficients,
        current_coordinates,
    )
    residual = sdf_operation.evaluate(query_coefficients, current_vertices)

    solver = csdl.nonlinear_solvers.BracketedSearch(
        name=f"{parameters.name}_bracketed_search",
        print_status=bool(parameters.print_status),
        tolerance=float(parameters.bisection_tolerance),
        max_iter=int(parameters.bisection_max_iter),
        residual_jac_kwargs={"elementwise": True},
    )
    solver.add_state(
        state,
        residual,
        bracket=(
            np.zeros((num_vertices,), dtype=float),
            np.ones((num_vertices,), dtype=float),
        ),
        tolerance=float(parameters.bisection_tolerance),
    )
    solver.run()

    final_state_column = csdl.reshape(state, (num_vertices, 1))
    if solve_axis == 0:
        final_coordinates = csdl.concatenate(
            (patch_column, final_state_column, fixed_coordinate),
            axis=1,
        )
    else:
        final_coordinates = csdl.concatenate(
            (patch_column, fixed_coordinate, final_state_column),
            axis=1,
        )
    final_vertices = FunctionSetEvaluationOperation(evaluation_model).evaluate(
        driving_coefficients,
        final_coordinates,
    )
    # Anchor the RBF to the exact baseline intersection, not to the nearby mesh
    # seed.  At the baseline design this makes every propagated displacement
    # exactly zero while exact seam rows can still replace their mesh seeds.
    # This also prevents O(1e-7) RBF roundoff from nudging vertices across
    # degenerate CAD-patch boundaries at fuselage and lifting-surface tips.
    #
    # The anchor must come from the *baseline* geometry, so it is bisected
    # here with the fixed numpy coefficients.  Snapshotting the live solve's
    # graph-construction value instead silently zeroes every seam displacement
    # whenever the graph is built at a non-baseline design.
    baseline_intersection_vertices = _solve_baseline_intersection(
        parameters,
        evaluation_model=evaluation_model,
        sdf_model=sdf_model,
        solve_column=solve_column,
    )
    return IntersectionSolution(
        initial_vertices=baseline_intersection_vertices,
        deformed_vertices=final_vertices,
        vertex_ids=parameters.vertex_ids,
        training_rows=_select_training_rows(parameters),
    )


def _solve_baseline_intersection(
    parameters: IntersectionParameters,
    *,
    evaluation_model,
    sdf_model,
    solve_column: int,
) -> np.ndarray:
    """Bisect the baseline intersection on the fixed numpy geometry.

    Mirrors the differentiable bracketed search: for each seam row the free
    parametric coordinate is bisected over ``[0, 1]`` while the query
    component's normal-mode SDF provides the residual sign.
    """
    driving_coefficients = stack_component_coefficients_numpy(
        parameters.driving_component
    )
    query_coefficients = stack_component_coefficients_numpy(
        parameters.sdf_query_component
    )
    coordinates = parameters.parametric_coords.copy()
    count = coordinates.shape[0]

    def residual(state: np.ndarray) -> np.ndarray:
        """Evaluate baseline signed distances at trial coordinates.

        Parameters
        ----------
        state
            Trial values of the free parametric coordinate.

        Returns
        -------
        numpy.ndarray
            Signed-distance residual for each seam row.
        """
        coordinates[:, solve_column] = state
        points = evaluation_model.evaluate(driving_coefficients, coordinates)
        distances, _ = sdf_model.project(query_coefficients, points)
        return np.asarray(distances, dtype=float).reshape(-1)

    lower = np.zeros(count, dtype=float)
    upper = np.ones(count, dtype=float)
    lower_residual = residual(lower)
    midpoint = 0.5 * (lower + upper)
    for _ in range(int(parameters.bisection_max_iter)):
        midpoint = 0.5 * (lower + upper)
        midpoint_residual = residual(midpoint)
        if float(np.max(np.abs(midpoint_residual))) <= float(
            parameters.bisection_tolerance
        ):
            break
        same_side = np.sign(midpoint_residual) == np.sign(lower_residual)
        lower = np.where(same_side, midpoint, lower)
        lower_residual = np.where(same_side, midpoint_residual, lower_residual)
        upper = np.where(same_side, upper, midpoint)

    coordinates[:, solve_column] = midpoint
    return np.asarray(
        evaluation_model.evaluate(driving_coefficients, coordinates),
        dtype=float,
    ).reshape((-1, 3))


def _select_training_rows(parameters: IntersectionParameters) -> np.ndarray:
    count = parameters.parametric_coords.shape[0]
    selection = parameters.vertices_to_include
    if isinstance(selection, str):
        if selection != "all":
            raise ValueError("vertices_to_include string value must be 'all'.")
        return np.arange(count, dtype=np.int64)
    if isinstance(selection, np.ndarray) or (
        not np.isscalar(selection) and selection is not None
    ):
        rows = np.asarray(selection, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= count):
            raise ValueError("vertices_to_include contains an invalid row.")
        return np.unique(rows)
    if isinstance(selection, (float, np.floating)):
        fraction = float(selection)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("vertices_to_include fraction must lie in (0, 1].")
        sample_count = max(1, int(np.ceil(fraction * count)))
    else:
        sample_count = int(selection)
        if sample_count <= 0:
            raise ValueError("vertices_to_include count must be positive.")
        sample_count = min(sample_count, count)
    return np.unique(np.rint(np.linspace(0, count - 1, sample_count)).astype(np.int64))


__all__ = [
    "IntersectionParameters",
    "IntersectionSolution",
    "solve_intersection",
]
