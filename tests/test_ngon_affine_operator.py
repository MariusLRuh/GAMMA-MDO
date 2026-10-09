"""Operator-level checks for the element-local N-gon affine regularizer."""

import numpy as np

from gamma_mdo.core.boundary_surface_movement import (
    CurrentGraphModel,
    CurrentGraphNgonAffineModel,
    NgonAffineAssembler,
    NgonAffineConfig,
)
from gamma_mdo.core.boundary_surface_movement.ngon_affine import (
    _affine_residual_projector,
)
from gamma_mdo.preprocessing import MeshData


def _single_cell_mesh(vertices, cell_type):
    """Build a one-cell surface mesh without altering its polygon width."""
    cell = np.arange(len(vertices), dtype=np.int64).reshape((1, -1))
    return MeshData(
        vertices=np.asarray(vertices, dtype=float),
        connectivity=cell,
        cell_types=np.asarray([cell_type], dtype=object),
        cell_blocks={cell_type: cell},
    )


def _regular_polygon(num_vertices):
    """Return a unit regular polygon in the xy plane."""
    angles = np.arange(num_vertices) * (2.0 * np.pi / num_vertices)
    return np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros(num_vertices))
    )


def test_affine_residual_projector_has_the_expected_nullspace_and_rank():
    """Verify affine annihilation and full retention of alternating modes."""
    hexagon = _regular_polygon(6)
    projector, _ = _affine_residual_projector(hexagon, tolerance=1.0e-11)
    u = hexagon[:, 0]
    v = hexagon[:, 1]

    np.testing.assert_allclose(projector, projector.T, atol=1.0e-14)
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-14)
    assert np.linalg.matrix_rank(projector, tol=1.0e-10) == 3
    for affine_field in (
        np.ones(6),
        u,
        v,
        2.0 + 3.0 * u - 5.0 * v,
    ):
        assert np.linalg.norm(projector @ affine_field) < 1.0e-12

    alternating = (-1.0) ** np.arange(6)
    retained = np.linalg.norm(projector @ alternating) / np.linalg.norm(
        alternating
    )
    assert abs(retained - 1.0) < 1.0e-12

    quad = _regular_polygon(4)
    quad_projector, _ = _affine_residual_projector(quad, tolerance=1.0e-11)
    quad_alternating = (-1.0) ** np.arange(4)
    quad_retained = np.linalg.norm(
        quad_projector @ quad_alternating
    ) / np.linalg.norm(quad_alternating)
    assert np.linalg.matrix_rank(quad_projector, tol=1.0e-10) == 1
    assert abs(quad_retained - 1.0) < 1.0e-12


def test_assembler_counts_n_minus_three_modes_per_active_polygon():
    """Count three hex modes, one quad mode, and no triangle modes."""
    assembler = NgonAffineAssembler()
    cases = ((6, "polygon6", 3), (4, "quad", 1), (3, "triangle", 0))
    for width, cell_type, expected_modes in cases:
        mesh = _single_cell_mesh(_regular_polygon(width), cell_type)
        system = assembler.assemble(
            mesh,
            free_ids=np.arange(width, dtype=np.int64),
            prescribed_ids=np.empty(0, dtype=np.int64),
        )
        assert system.num_hourglass_modes == expected_modes


def test_non_affine_hexagon_boundary_data_changes_the_primal_solution():
    """Match the independently derived harmonic and affine-completion limits."""
    vertices = _regular_polygon(6)
    mesh = _single_cell_mesh(vertices, "polygon6")
    free_ids = np.asarray([1, 3, 5], dtype=np.int64)
    prescribed_ids = np.asarray([0, 2, 4], dtype=np.int64)
    delta = 0.7
    prescribed = delta * np.asarray([[1.0], [-1.0], [1.0]])

    graph = CurrentGraphModel(
        mesh,
        free_ids=free_ids,
        prescribed_ids=prescribed_ids,
        stiffening_exponent=0.0,
    )
    harmonic = graph.solve(vertices, prescribed).reshape(-1)
    harmonic_expected = delta * np.asarray([0.0, 0.0, 1.0])
    np.testing.assert_allclose(harmonic, harmonic_expected, atol=1.0e-12)

    # Independently derive the unique affine field a + b*x + c*y through the
    # three prescribed vertices, then evaluate it at the three free vertices.
    prescribed_chart = np.column_stack(
        (np.ones(3), vertices[prescribed_ids, :2])
    )
    coefficients = np.linalg.solve(prescribed_chart, prescribed.reshape(-1))
    free_chart = np.column_stack((np.ones(3), vertices[free_ids, :2]))
    affine_expected = free_chart @ coefficients
    np.testing.assert_allclose(
        affine_expected,
        delta * np.asarray([-1.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0]),
        atol=1.0e-12,
    )

    affine_system = NgonAffineAssembler().assemble(
        mesh,
        free_ids=free_ids,
        prescribed_ids=prescribed_ids,
    )

    def regularized_solution(strength):
        model = CurrentGraphNgonAffineModel(
            graph,
            affine_system,
            lambda_ngon=strength,
            baseline_vertices=vertices,
        )
        return model.solve(
            vertices,
            prescribed,
            np.zeros((3, 1)),
            prescribed,
        ).reshape(-1)

    moderate = regularized_solution(0.3)
    affine_limit = regularized_solution(1.0e6)
    np.testing.assert_allclose(affine_limit, affine_expected, atol=1.0e-5)
    assert np.linalg.norm(moderate - harmonic, ord=np.inf) >= 0.04 * delta
