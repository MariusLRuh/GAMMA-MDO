from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.core.boundary_surface_movement import (
    TetraVolumeMesh,
    assemble_elastic_volume_system,
    assemble_graph_volume_system,
    evaluate_volume_quality,
    read_wall_position_history,
    write_wall_position_history,
)
from gamma_mdo.core.boundary_surface_movement.volume_mesh_motion import (
    _assemble_tetrahedral_elasticity_bsr,
)


def _tiny_volume_mesh() -> TetraVolumeMesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 0.0, 0.5],
            [1.0, 0.3, 0.3],
        ],
        dtype=float,
    )
    tetrahedra = np.asarray(
        [
            [0, 1, 2, 8],
            [0, 2, 7, 8],
            [0, 7, 1, 8],
            [3, 4, 5, 8],
            [3, 5, 6, 8],
            [3, 7, 4, 8],
            [3, 5, 7, 8],
        ],
        dtype=np.int64,
    )
    # The reader requires consistent orientation; emulate that invariant here.
    xyz = vertices[tetrahedra]
    determinant = np.einsum(
        "ij,ij->i",
        xyz[:, 1] - xyz[:, 0],
        np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
    )
    negative = determinant < 0.0
    tetrahedra[negative, 1:3] = tetrahedra[negative, 2:0:-1]
    return TetraVolumeMesh(
        source_path=Path("tiny.msh"),
        node_ids=np.arange(1, vertices.shape[0] + 1, dtype=np.int64),
        vertices=vertices,
        triangles=np.asarray(
            [
                [0, 1, 2],
                [0, 3, 7],
                [3, 4, 5],
                [3, 5, 6],
                [4, 5, 6],
            ],
            dtype=np.int64,
        ),
        triangle_physical_ids=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
        tetrahedra=tetrahedra,
        tetrahedron_physical_ids=np.full(
            tetrahedra.shape[0], 100, dtype=np.int64
        ),
        physical_names={},
    )


def test_reference_volume_systems_match_numpy_and_preserve_constraints():
    mesh = _tiny_volume_mesh()
    wall_displacement = np.asarray(
        [[0.10, 0.00, -0.03], [0.08, 0.02, -0.02], [0.12, -0.01, -0.04]]
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    wall_positions = csdl.Variable(
        value=mesh.vertices[mesh.aircraft_nodes] + wall_displacement
    )
    for system in (
        assemble_graph_volume_system(mesh),
        assemble_elastic_volume_system(mesh),
    ):
        expected = mesh.vertices + system.solve_numpy(wall_displacement)
        actual = np.asarray(system.evaluate(wall_positions).value, dtype=float)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)
        np.testing.assert_allclose(
            actual[mesh.aircraft_nodes],
            wall_positions.value,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            actual[mesh.fixed_outer_nodes],
            mesh.vertices[mesh.fixed_outer_nodes],
            atol=1.0e-12,
        )
        assert actual[7, 1] == mesh.vertices[7, 1]
    recorder.stop()


def test_volume_quality_detects_a_tetrahedron_inversion():
    mesh = _tiny_volume_mesh()
    baseline = evaluate_volume_quality(
        mesh.vertices, mesh.vertices, mesh.tetrahedra
    )
    assert baseline.inverted_tetrahedra == 0
    assert baseline.minimum_relative_jacobian == 1.0

    deformed = mesh.vertices.copy()
    deformed[8] = np.asarray([-0.25, 0.3, 0.3])
    quality = evaluate_volume_quality(
        mesh.vertices, deformed, mesh.tetrahedra
    )
    assert quality.inverted_tetrahedra > 0
    assert quality.minimum_relative_jacobian < 0.0


def test_elasticity_element_matches_direct_strain_displacement_matrix():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 0.9, 0.1],
            [0.1, 0.2, 1.1],
        ]
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    nu = 0.3
    actual = _assemble_tetrahedral_elasticity_bsr(
        vertices,
        cells,
        poisson_ratio=nu,
        stiffening_exponent=0.0,
        volume_floor=1.0e-15,
    ).toarray()

    affine = np.ones((4, 4))
    affine[:, 1:] = vertices
    gradients = np.linalg.inv(affine)[1:, :].T
    strain_displacement = np.zeros((6, 12))
    for node, (gx, gy, gz) in enumerate(gradients):
        dof = 3 * node
        strain_displacement[0, dof] = gx
        strain_displacement[1, dof + 1] = gy
        strain_displacement[2, dof + 2] = gz
        strain_displacement[3, dof : dof + 2] = [gy, gx]
        strain_displacement[4, dof + 1 : dof + 3] = [gz, gy]
        strain_displacement[5, [dof, dof + 2]] = [gz, gx]
    lame_lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lame_mu = 1.0 / (2.0 * (1.0 + nu))
    constitutive = np.zeros((6, 6))
    constitutive[:3, :3] = lame_lambda
    constitutive[:3, :3] += 2.0 * lame_mu * np.eye(3)
    constitutive[3:, 3:] = lame_mu * np.eye(3)
    determinant = np.linalg.det(affine)
    expected = (
        abs(determinant)
        / 6.0
        * strain_displacement.T
        @ constitutive
        @ strain_displacement
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-15)


def test_wall_position_history_round_trip(tmp_path):
    mesh = _tiny_volume_mesh()
    baseline = mesh.vertices[mesh.aircraft_nodes]
    history = (
        baseline + np.asarray([0.02, 0.0, -0.01]),
        baseline + np.asarray([0.04, 0.0, -0.02]),
    )
    path = tmp_path / "wall_history.npz"
    write_wall_position_history(path, mesh, history, (0.5, 1.0))
    loaded = read_wall_position_history(path, mesh)

    assert loaded.fractions == (0.5, 1.0)
    np.testing.assert_array_equal(loaded.positions, history)
