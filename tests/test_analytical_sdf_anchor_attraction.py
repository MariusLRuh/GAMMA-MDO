import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gamma_mdo.core.boundary_surface_movement import analytical_SDF_anchor_attraction_noarg as sdf_mod


def _central_difference_gradient(func, points: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    gradient = np.zeros_like(points)
    for axis in range(3):
        step = np.zeros(3, dtype=float)
        step[axis] = epsilon
        gradient[:, axis] = (func(points + step) - func(points - step)) / (2.0 * epsilon)
    return gradient


def test_sdf_primitive_analytical_gradients_match_finite_difference():
    primitive_cases = [
        (
            "capped_cylinder",
            np.array(
                [
                    [0.0, 1.4, 0.3],
                    [4.6, 0.2, 0.1],
                    [4.5, 1.1, 0.0],
                    [0.2, 0.35, 0.2],
                    [4.0, 0.1, 0.1],
                ],
                dtype=float,
            ),
            lambda pts: sdf_mod.sdf_capped_cylinder(pts, center=(0.0, 0.0, 0.0), radius=0.7, half_length=4.2),
            lambda pts: sdf_mod._sdf_capped_cylinder_value_gradient_hessian(
                pts,
                center=(0.0, 0.0, 0.0),
                radius=0.7,
                half_length=4.2,
            ),
        ),
        (
            "box",
            np.array(
                [
                    [0.6, 6.0, 0.2],
                    [0.55, 4.1, 0.03],
                    [0.1, 0.0, 0.2],
                    [0.9, 5.3, 0.0],
                ],
                dtype=float,
            ),
            lambda pts: sdf_mod.sdf_box(
                pts,
                center=(0.5, 0.0, 0.0),
                half_size=(0.35, 5.5, 0.08),
                rotation=sdf_mod.rotation_matrix_y(np.radians(12.0)),
            ),
            lambda pts: sdf_mod._sdf_box_value_gradient_hessian(
                pts,
                center=(0.5, 0.0, 0.0),
                half_size=(0.35, 5.5, 0.08),
                rotation=sdf_mod.rotation_matrix_y(np.radians(12.0)),
            ),
        ),
        (
            "cone",
            np.array(
                [
                    [0.2, 0.1, 0.05],
                    [0.8, 0.7, 0.1],
                    [2.2, 0.2, 0.0],
                    [2.1, 1.4, 0.0],
                    [-0.3, 0.2, 0.1],
                    [1.7, 0.1, 0.0],
                ],
                dtype=float,
            ),
            lambda pts: sdf_mod.sdf_cone(pts, apex=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), height=2.0, base_radius=1.0),
            lambda pts: sdf_mod._sdf_cone_value_gradient_hessian(
                pts,
                apex=(0.0, 0.0, 0.0),
                axis=(1.0, 0.0, 0.0),
                height=2.0,
                base_radius=1.0,
            ),
        ),
    ]

    for case_name, points, value_func, derivative_func in primitive_cases:
        _, gradient_analytic, _ = derivative_func(points)
        gradient_fd = _central_difference_gradient(value_func, points)
        np.testing.assert_allclose(
            gradient_analytic,
            gradient_fd,
            rtol=2e-5,
            atol=5e-7,
            err_msg=f"Gradient mismatch for {case_name}.",
        )


def test_aircraft_sdf_analytical_gradient_matches_finite_difference():
    geometry = sdf_mod.AircraftGeometry()
    points = np.array(
        [
            [0.0, 1.3, 0.25],
            [-1.2, 0.35, 0.12],
            [4.7, 0.55, 0.18],
            [4.5, 0.08, 0.02],
            [0.6, 3.5, 0.35],
            [0.55, 4.2, 0.02],
            [-3.4, 1.6, 0.25],
            [-3.3, 0.05, 2.0],
            [0.6, 0.9, 0.08],
        ],
        dtype=float,
    )

    phi_analytic, gradient_analytic = sdf_mod.sdf_with_gradient(points, geometry=geometry, smooth_radius=0.18)
    phi_fd, gradient_fd = sdf_mod._sdf_with_gradient_finite_difference(
        points,
        geometry=geometry,
        smooth_radius=0.18,
        epsilon=1e-6,
    )

    np.testing.assert_allclose(phi_analytic, phi_fd, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gradient_analytic, gradient_fd, rtol=2e-5, atol=5e-7)


def test_aircraft_sdf_analytical_hessian_matches_finite_difference():
    geometry = sdf_mod.AircraftGeometry()
    points = np.array(
        [
            [0.0, 1.3, 0.25],
            [-1.2, 0.35, 0.12],
            [4.7, 0.55, 0.18],
            [0.6, 3.5, 0.35],
            [-3.4, 1.6, 0.25],
        ],
        dtype=float,
    )

    _, _, hessian_analytic = sdf_mod.sdf_gradient_hessian(points, geometry=geometry, smooth_radius=0.18)
    _, _, hessian_fd = sdf_mod._sdf_gradient_hessian_finite_difference(
        points,
        geometry=geometry,
        smooth_radius=0.18,
        epsilon=1e-5,
    )

    np.testing.assert_allclose(hessian_analytic, hessian_fd, rtol=1e-4, atol=5e-6)
