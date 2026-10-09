from __future__ import annotations

import numpy as np
import pytest

from gamma_mdo.core.projections import function_set_projection_numpy as projection_numpy


class _Coefficients:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)


class _Function:
    def __init__(self, coefficients):
        self.coefficients = _Coefficients(coefficients)


class _FunctionSet:
    def __init__(self):
        self.functions = {
            2: _Function([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
            7: _Function([[7.0, 0.0, 0.0]]),
        }

    def evaluate(self, parametric_coordinates):
        values = []
        for patch_id, uv in parametric_coordinates:
            assert isinstance(patch_id, int)
            assert np.asarray(uv).shape == (2,)
            values.append([patch_id, uv[0], uv[1]])
        return np.asarray(values, dtype=float)


class _ProjectionModel:
    patch_ids = (2, 7)

    def __init__(self, *, function_set, **options):
        self.function_set = function_set
        self.options = options
        self.calls = []

    def project(self, *, stacked_coefficients, points):
        self.calls.append(
            (
                np.asarray(stacked_coefficients, dtype=float).copy(),
                np.asarray(points, dtype=float).copy(),
            )
        )
        state = {
            "patch_id": np.asarray([7, 2], dtype=np.int64),
            "uv": np.asarray([[0.75, 0.25], [0.1, 0.9]], dtype=float),
        }
        return np.asarray([0.01, 0.02]), state


def test_project_returns_coordinates_accepted_by_function_set_evaluate(monkeypatch):
    monkeypatch.setattr(
        projection_numpy,
        "FunctionSetProjectionModel",
        _ProjectionModel,
    )
    function_set = _FunctionSet()
    projector = projection_numpy.FunctionSetProjector(
        function_set,
        warm_start_nu=25,
        warm_start_nv=17,
    )
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    coordinates = projector.project(points)

    assert projector.model.options == {"warm_start_nu": 25, "warm_start_nv": 17}
    assert [patch_id for patch_id, _ in coordinates] == [7, 2]
    np.testing.assert_allclose(coordinates[0][1], [0.75, 0.25])
    np.testing.assert_allclose(coordinates[1][1], [0.1, 0.9])
    np.testing.assert_allclose(
        function_set.evaluate(coordinates),
        [[7.0, 0.75, 0.25], [2.0, 0.1, 0.9]],
    )

    used_coefficients, used_points = projector.model.calls[0]
    np.testing.assert_allclose(
        used_coefficients,
        [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [7.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(used_points, points)


def test_project_uses_explicit_stacked_coefficients(monkeypatch):
    monkeypatch.setattr(
        projection_numpy,
        "FunctionSetProjectionModel",
        _ProjectionModel,
    )
    projector = projection_numpy.FunctionSetProjector(_FunctionSet())
    coefficients = np.arange(12, dtype=float).reshape(4, 3)

    projector.project(np.zeros((2, 3)), stacked_coefficients=coefficients)

    used_coefficients, _ = projector.model.calls[0]
    np.testing.assert_array_equal(used_coefficients, coefficients)


def test_projection_state_requires_aligned_patch_ids_and_uv():
    with pytest.raises(ValueError, match="same number of points"):
        projection_numpy._function_set_coordinates_from_state(
            {
                "patch_id": np.asarray([0, 1]),
                "uv": np.asarray([[0.25, 0.75]]),
            }
        )
