"""Observable load-step and mixed-mesh tests for N-gon regularization."""

from types import SimpleNamespace

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import pytest

from gamma_mdo.core.boundary_surface_movement import (
    CurrentGraphModel,
    CurrentGraphNgonAffineModel,
    ElasticityMotionSolver,
    GraphLaplacianAssembler,
    GraphLoadStepState,
    NgonAffineAssembler,
    NgonAffineConfig,
    check_element_inversion,
    evaluate_mesh_quality,
    run_graph_load_steps,
    stack_component_coefficients,
)
from gamma_mdo.preprocessing import MeshData, ProjectionMetadata


def _regular_polygon(num_vertices, *, center_x=0.0):
    """Return a counter-clockwise unit polygon in the xy plane."""
    angles = np.arange(num_vertices) * (2.0 * np.pi / num_vertices)
    return np.column_stack(
        (
            center_x + np.cos(angles),
            np.sin(angles),
            np.zeros(num_vertices),
        )
    )


def _single_hexagon_mesh():
    """Build the true polygon6 ring used by the load-step derivative test."""
    vertices = _regular_polygon(6)
    cell = np.arange(6, dtype=np.int64).reshape((1, 6))
    return MeshData(
        vertices=vertices,
        connectivity=cell,
        cell_types=np.asarray(["polygon6"], dtype=object),
        cell_blocks={"polygon6": cell},
    )


def _projection_plane():
    """Construct a broad bilinear xy plane for an identity reprojection."""
    knots = (np.asarray([0.0, 0.0, 1.0, 1.0]),) * 2
    space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),
        knots=knots,
    )
    u, v = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    coefficients = np.stack(
        (6.0 * u - 3.0, 6.0 * v - 3.0, np.zeros_like(u)),
        axis=-1,
    )
    return lfs.FunctionSet(
        functions={0: lfs.Function(space=space, coefficients=coefficients)}
    )


class _SyntheticHexagonMotion(ElasticityMotionSolver):
    """Supply non-affine boundary data to the production load-step driver."""

    def __init__(self, mesh):
        self.initial_vertices = np.asarray(mesh.vertices, dtype=float)
        self.free_ids = np.asarray([1, 3, 5], dtype=np.int64)
        self.prescribed_ids = np.asarray([0, 2, 4], dtype=np.int64)
        self.assembler = GraphLaplacianAssembler(stiffening_exponent=0.0)
        self.symmetry_plane_ids = None
        self._free_local = {
            int(vertex): row for row, vertex in enumerate(self.free_ids)
        }
        self._free_reference_groups = (
            SimpleNamespace(free_rows=np.arange(3, dtype=np.int64)),
        )

    def build_load_step_state(self, *, component_coeffs, query_component_coeffs=None):
        """Prescribe the analytically non-affine ``(+δ, -δ, +δ)`` ring data."""
        del query_component_coeffs
        amplitude = component_coeffs["hourglass_amplitude"]
        pattern = np.zeros((3, 3), dtype=float)
        pattern[:, 1] = np.asarray([1.0, -1.0, 1.0])
        return GraphLoadStepState(
            free_reference=csdl.Variable(value=np.zeros((3, 3))),
            prescribed_deviations=(amplitude * pattern,),
            prescribed_deviations_y=(),
            solutions=(),
        )


def _run_hexagon_load_step(mesh, motion, plane, amplitude, lambda_ngon):
    """Run one complete graph-solve and OML-reprojection step."""
    coefficient_state = {
        "hourglass_amplitude": amplitude,
        id(plane): stack_component_coefficients(plane),
    }
    return run_graph_load_steps(
        motion=motion,
        mesh=mesh,
        initial_deformation_vertices=mesh.vertices,
        deformation_vertex_ids=np.arange(6, dtype=np.int64),
        component_coefficient_steps=[coefficient_state],
        projection_metadata=[
            ProjectionMetadata(
                component=plane,
                vertex_ids=np.arange(6, dtype=np.int64),
                allowed_patch_ids=(0,),
                parent_patch_ids=np.zeros(6, dtype=np.int64),
            )
        ],
        reevaluation_metadata=[],
        projection_options={
            "warm_start_nu": 7,
            "warm_start_nv": 7,
            "edge_map_num_samples": 5,
        },
        ngon_affine_config=(
            None
            if lambda_ngon == 0.0
            else NgonAffineConfig(lambda_ngon=lambda_ngon)
        ),
    )


@pytest.mark.timeout(20)
def test_polygon6_load_step_ngon_vjp_matches_centered_finite_difference():
    """Differentiate an observable N-gon term through solve and reprojection."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    mesh = _single_hexagon_mesh()
    plane = _projection_plane()
    amplitude = csdl.Variable(name="hourglass_amplitude", value=0.2)
    amplitude.set_as_design_variable()

    regularized = _run_hexagon_load_step(
        mesh,
        _SyntheticHexagonMotion(mesh),
        plane,
        amplitude,
        lambda_ngon=0.3,
    )
    unregularized = _run_hexagon_load_step(
        mesh,
        _SyntheticHexagonMotion(mesh),
        plane,
        amplitude,
        lambda_ngon=0.0,
    )
    assert regularized.ngon_affine_num_elements == 1
    assert regularized.ngon_affine_num_modes == 3
    assert unregularized.ngon_affine_num_modes == 0
    weights = np.zeros_like(mesh.vertices)
    weights[[1, 3, 5], 1] = np.asarray([1.0, 2.0, 4.0])
    objective = csdl.sum(regularized.final_mesh_vertices * weights)
    objective.add_name("weighted_reprojected_polygon6_coordinates")
    objective.set_as_objective()
    control = csdl.sum(unregularized.final_mesh_vertices * weights)
    analytic = csdl.derivative(objective, amplitude)
    control_derivative = csdl.derivative(control, amplitude)
    recorder.stop()

    analytic_value = float(np.asarray(analytic.value).reshape(-1)[0])
    control_value = float(np.asarray(control_derivative.value).reshape(-1)[0])
    assert abs(analytic_value - control_value) >= 1.0e-6

    simulator = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    baseline = np.asarray(amplitude.value).copy()
    best_error_per_pair = {}
    pair = (objective, amplitude)
    for step_size in (1.0e-4, 1.0e-5, 1.0e-6):
        simulator[amplitude] = baseline + step_size
        simulator.run()
        plus = float(np.asarray(simulator[objective]).reshape(-1)[0])
        simulator[amplitude] = baseline - step_size
        simulator.run()
        minus = float(np.asarray(simulator[objective]).reshape(-1)[0])
        centered = (plus - minus) / (2.0 * step_size)
        relative_error = abs(analytic_value - centered) / max(
            abs(analytic_value), abs(centered), 1.0e-14
        )
        best_error_per_pair[pair] = min(
            best_error_per_pair.get(pair, relative_error), relative_error
        )
    simulator[amplitude] = baseline

    assert best_error_per_pair
    assert max(best_error_per_pair.values()) < 1.0e-5


def _mixed_polygon_mesh():
    """Build disconnected regular quad, pentagon, and hexagon cells."""
    polygons = (
        ("quad", _regular_polygon(4, center_x=0.0)),
        ("polygon5", _regular_polygon(5, center_x=4.0)),
        ("polygon6", _regular_polygon(6, center_x=8.0)),
    )
    vertices = []
    blocks = {}
    connectivity = []
    cell_types = []
    offset = 0
    for cell_type, polygon in polygons:
        cell = np.arange(offset, offset + len(polygon), dtype=np.int64)
        vertices.append(polygon)
        blocks[cell_type] = cell.reshape((1, -1))
        connectivity.append(cell)
        cell_types.append(cell_type)
        offset += len(polygon)
    return MeshData(
        vertices=np.vstack(vertices),
        connectivity=np.asarray(connectivity, dtype=object),
        cell_types=np.asarray(cell_types, dtype=object),
        cell_blocks=blocks,
    )


def test_mixed_polygons_regularize_without_folds_or_inversions():
    """Exercise all n-minus-three modes on a safely deformed mixed mesh."""
    mesh = _mixed_polygon_mesh()
    prescribed_ids = np.asarray([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64)
    free_ids = np.setdiff1d(
        np.arange(mesh.vertices.shape[0], dtype=np.int64), prescribed_ids
    )
    system = NgonAffineAssembler().assemble(
        mesh,
        free_ids=free_ids,
        prescribed_ids=prescribed_ids,
    )
    assert system.num_hourglass_modes == sum(width - 3 for width in (4, 5, 6))

    graph = CurrentGraphModel(
        mesh,
        free_ids=free_ids,
        prescribed_ids=prescribed_ids,
        stiffening_exponent=0.0,
    )
    model = CurrentGraphNgonAffineModel(
        graph,
        system,
        lambda_ngon=0.3,
        baseline_vertices=mesh.vertices,
    )
    prescribed = np.zeros((prescribed_ids.size, 1), dtype=float)
    prescribed[:, 0] = 0.03 * (-1.0) ** np.arange(prescribed_ids.size)
    free = model.solve(
        mesh.vertices,
        prescribed,
        np.zeros((free_ids.size, 1)),
        prescribed,
    )
    deformed = mesh.vertices.copy()
    deformed[prescribed_ids, 2] += prescribed[:, 0]
    deformed[free_ids, 2] += free[:, 0]

    inversion = check_element_inversion(mesh=mesh, final_mesh_vertices=deformed)
    quality = evaluate_mesh_quality(mesh=mesh, vertices=deformed)
    assert inversion.num_inverted == 0
    assert quality.inverted_elements == 0
    assert quality.inverted_corners == 0
    assert quality.degenerate_elements == 0
    assert quality.minimum_scaled_jacobian > 0.0
    assert quality.minimum_area_ratio > 0.0
    assert np.isfinite(quality.maximum_aspect_ratio)
