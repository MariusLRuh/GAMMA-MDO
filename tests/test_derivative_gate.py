"""Fast analytical-derivative gate for the supported surface-motion path."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import pytest

from gamma_mdo.component_parameters import ComponentParameters
from gamma_mdo.core.boundary_surface_movement import (
    ElasticityMotionSolver,
    IntersectionParameters,
    NgonAffineConfig,
    deform_geometry,
    element_neighbors,
    run_graph_load_steps,
    stack_component_coefficients,
)
from gamma_mdo.preprocessing import (
    MeshData,
    ProjectionMetadata,
    VertexEvaluationMetadata,
)


def _plane_function_set(kind: str, *, patch_id: int):
    """Construct a bilinear plane used by the derivative-gate geometry."""
    knots = (np.array([0.0, 0.0, 1.0, 1.0]),) * 2
    space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),
        knots=knots,
    )
    u, v = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    if kind == "xy":
        coefficients = np.stack(
            (2.0 * u - 1.0, 2.0 * v - 1.0, np.zeros_like(u)),
            axis=-1,
        )
    elif kind == "yz":
        coefficients = np.stack(
            (np.zeros_like(u), 2.0 * u - 1.0, 2.0 * v - 1.0),
            axis=-1,
        )
    else:
        raise ValueError(f"Unknown plane kind {kind!r}.")
    return lfs.FunctionSet(
        functions={
            patch_id: lfs.Function(space=space, coefficients=coefficients)
        }
    )


def _structured_quad_patch():
    """Return the small mixed-role quad patch used by the derivative gate."""
    coordinates = np.linspace(-1.0, 1.0, 5)
    num_z = coordinates.size
    vertices = np.asarray(
        [[0.0, y, z] for y in coordinates for z in coordinates],
        dtype=float,
    )

    def vertex_id(y_index, z_index):
        return y_index * num_z + z_index

    quads = np.asarray(
        [
            [
                vertex_id(y_index, z_index),
                vertex_id(y_index + 1, z_index),
                vertex_id(y_index + 1, z_index + 1),
                vertex_id(y_index, z_index + 1),
            ]
            for y_index in range(coordinates.size - 1)
            for z_index in range(coordinates.size - 1)
        ],
        dtype=np.int64,
    )
    mesh = MeshData(
        vertices=vertices,
        connectivity=quads,
        cell_types=np.full(quads.shape[0], "quad", dtype=object),
        cell_blocks={"quad": quads},
    )
    seam_ids = np.asarray(
        [vertex_id(y_index, 2) for y_index in range(coordinates.size)],
        dtype=np.int64,
    )
    free_ids = np.asarray(
        [
            vertex_id(y_index, z_index)
            for y_index in range(coordinates.size)
            for z_index in (1, 3)
        ],
        dtype=np.int64,
    )
    parametric = np.asarray(
        [
            [10.0, (y + 1.0) / 2.0, (z + 1.0) / 2.0]
            for y in coordinates
            for z in coordinates
        ]
    )
    seam_parametric = np.asarray(
        [[0.0, 0.5, (y + 1.0) / 2.0] for y in coordinates]
    )
    return mesh, parametric, seam_parametric, free_ids, seam_ids


@pytest.mark.timeout(60)
def test_final_reprojected_mesh_vjp_matches_finite_difference():
    """Check the affine-nullspace control through graph motion and OML.

    The N-gon term is expected to be unobservable here because this fixture's
    deformation is affine in each element chart. The observable hourglass case
    is covered by the polygon6 operator and load-step tests.
    """
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    wing = _plane_function_set("xy", patch_id=0)
    fuselage = _plane_function_set("yz", patch_id=10)
    mesh, parametric, seam_parametric, free_ids, seam_ids = (
        _structured_quad_patch()
    )
    deformation_ids = np.concatenate((free_ids, seam_ids))
    reevaluation_ids = np.setdiff1d(
        np.arange(mesh.vertices.shape[0]), deformation_ids
    )
    symmetry_ids = np.where(np.abs(mesh.vertices[:, 1]) <= 1.0e-12)[0]

    translation_z = csdl.Variable(name="gate_translation_z", value=0.2)
    translation_z.set_as_design_variable()
    load_fractions = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    coefficient_states = [
        {
            id(wing): deform_geometry(
                component=wing,
                parameters=ComponentParameters(
                    translation_z=fraction * translation_z,
                    pivot=np.zeros(3),
                ),
            ),
            id(fuselage): stack_component_coefficients(fuselage),
        }
        for fraction in load_fractions
    ]

    intersection = IntersectionParameters(
        name="gate_intersection",
        parametric_coords=seam_parametric,
        vertex_ids=seam_ids,
        driving_component=wing,
        sdf_query_component=fuselage,
        bisection_search_direction="u",
        bisection_tolerance=1.0e-11,
        projection_options={
            "warm_start_nu": 10,
            "warm_start_nv": 10,
            "edge_map_num_samples": 7,
        },
    )
    motion = ElasticityMotionSolver(
        mesh=mesh,
        free_ids=free_ids,
        intersection_params=[intersection],
        parametric_coordinates=parametric,
        components=[wing, fuselage],
        component_reevaluations=(),
        stiffening_exponent=0.9,
        prescribed_ids=element_neighbors(mesh, free_ids),
        symmetry_use_element_neighbors=True,
        symmetry_plane_ids=symmetry_ids,
    )
    result = run_graph_load_steps(
        motion=motion,
        mesh=mesh,
        initial_deformation_vertices=mesh.vertices[deformation_ids],
        deformation_vertex_ids=deformation_ids,
        component_coefficient_steps=coefficient_states,
        projection_metadata=[
            ProjectionMetadata(
                component=fuselage,
                vertex_ids=free_ids,
                allowed_patch_ids=(10,),
                parent_patch_ids=np.full(free_ids.size, 10),
            ),
            ProjectionMetadata(
                component=wing,
                vertex_ids=seam_ids,
                allowed_patch_ids=(0,),
                parent_patch_ids=np.zeros(seam_ids.size, dtype=np.int64),
            ),
        ],
        reevaluation_metadata=[
            VertexEvaluationMetadata(
                component=fuselage,
                vertex_ids=reevaluation_ids,
                parametric_coordinates=parametric[reevaluation_ids],
            )
        ],
        projection_options={
            "warm_start_nu": 10,
            "warm_start_nv": 10,
            "edge_map_num_samples": 7,
        },
        load_fractions=load_fractions,
        ngon_affine_config=NgonAffineConfig(lambda_ngon=0.3),
        symmetry_plane_vertex_ids=symmetry_ids,
    )

    row_weights = 1.0 + np.arange(mesh.vertices.shape[0]) / mesh.vertices.shape[0]
    weights = np.zeros_like(mesh.vertices)
    weights[:, 2] = row_weights
    objective = csdl.sum(result.final_mesh_vertices * weights)
    objective.add_name("weighted_final_surface_coordinates")
    objective.set_as_objective()
    analytic = csdl.derivative(objective, translation_z)
    recorder.stop()

    expected = np.sum(row_weights[seam_ids]) + 0.5 * np.sum(
        row_weights[free_ids]
    )
    np.testing.assert_allclose(
        np.asarray(analytic.value).reshape(-1)[0],
        expected,
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    simulator = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    # Finite-difference error is U-shaped in the step size, so each derivative
    # pair is scored at its own best step. Every pair must then pass. Taking a
    # single minimum across pairs would let one correct derivative mask a
    # broken one as soon as M1 introduces more design variables.
    best_error_per_pair = {}
    for step_size in (1.0e-4, 1.0e-5, 1.0e-6):
        checks = simulator.check_optimization_derivatives(
            step_size=step_size,
            print_results=False,
            raise_on_error=False,
        )
        for pair, entry in checks.items():
            error = float(entry["rel_error"])
            best_error_per_pair[pair] = min(
                best_error_per_pair.get(pair, error), error
            )
    assert best_error_per_pair
    assert max(best_error_per_pair.values()) < 1.0e-5
