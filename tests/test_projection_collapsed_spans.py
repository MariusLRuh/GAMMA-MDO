"""Closest-point regressions for zero-length spline segments at sharp edges."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import pytest

import gamma_mdo.core.projections.warm_start_candidate_projection_numpy as wsc
from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
    stack_function_set_coefficients,
)
from gamma_mdo.core.projections.warm_start_projections import WarmStartResult

SEGMENT = 1.0 / 6.0


@pytest.fixture
def recorder():
    """Own CSDL state for each independent regression."""
    rec = csdl.Recorder(inline=True)
    rec.start()
    try:
        yield rec
    finally:
        rec.stop()


def _thin_airfoil_loop():
    """Build a chordwise loop with a sharp trailing edge, as OpenVSP exports it.

    Six cubic Bezier segments run trailing edge -> lower skin -> leading edge
    -> upper skin -> trailing edge. The first and last segments have zero length
    at the trailing edge, so ``v`` in ``[0, 1/6]`` and ``[5/6, 1]`` both map onto
    the trailing-edge line.
    """
    def half_thickness(x):
        return 0.06 * np.sin(np.pi * x)

    trailing_edge = [(1.0, 0.0)] * 3
    lower = [(x, -half_thickness(x)) for x in (1.0, 0.83, 0.67, 0.5, 0.33, 0.17)]
    upper = [(x, half_thickness(x)) for x in (0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0)]
    section = np.array(trailing_edge + lower + upper + trailing_edge)
    assert section.shape == (19, 2)
    net = np.stack([
        np.column_stack((section, np.full(len(section), z))) for z in (0.0, 1.0)
    ])
    v_knots = np.r_[np.zeros(4), np.repeat(np.arange(1, 6) * SEGMENT, 3), np.ones(4)]
    space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2, degree=(1, 3), coefficients_shape=net.shape[:2],
        knots=(np.array([0.0, 0.0, 1.0, 1.0]), v_knots),
    )
    return lfs.FunctionSet(functions={0: lfs.Function(space=space, coefficients=net)})


def _point_on_surface(function_set, u, v):
    return np.asarray(
        function_set.functions[0].evaluate(np.array([[u, v]])).value
    ).reshape(1, 3)


def _project_from_seed(monkeypatch, function_set, point, seed_uv, *, escape):
    """Project ``point`` with the warm start forced to ``seed_uv``."""
    def forced_warm_start(*, mesh, points):
        count = np.asarray(points).shape[0]
        return WarmStartResult(
            patch_id=np.zeros(count, dtype=int),
            uv0=np.tile(np.asarray(seed_uv, dtype=float), (count, 1)),
            closest_pts=np.asarray(points, dtype=float),
            cell_ids=np.zeros(count, dtype=int),
            # A large seed distance keeps the distance-outlier retry quiet, as
            # in the C208 trailing-edge cases this reproduces.
            dist2=np.full(count, 1.0),
        )

    with monkeypatch.context() as patch:
        patch.setattr(wsc, "warm_start_from_triangulation", forced_warm_start)
        if not escape:
            patch.setattr(wsc, "_build_collapsed_span_escape_specs", lambda *a: [])
        model = FunctionSetProjectionModel(function_set)
        coefficients = stack_function_set_coefficients(function_set, model.patch_ids)
        distance, state = model.project(coefficients, point)
    return float(np.ravel(distance)[0]), state


def test_trailing_edge_strips_are_detected_and_grouped(recorder):
    """Both zero-length trailing-edge segments share one image."""
    fs = _thin_airfoil_loop()
    function = fs.functions[0]
    groups = wsc._collapsed_span_groups(
        np.asarray(function.coefficients.value), function.space.degree, function.space.knots,
    )
    assert groups[0] == []
    assert len(groups[1]) == 1
    np.testing.assert_allclose(np.array(groups[1][0]), [[0.0, SEGMENT], [5 * SEGMENT, 1.0]])


def test_seed_inside_a_collapsed_strip_escapes_to_the_true_foot(monkeypatch, recorder):
    """A Newton seed inside the strip stops at once without the escape pass."""
    fs = _thin_airfoil_loop()
    point = _point_on_surface(fs, 0.5, 1.3 * SEGMENT)
    seed = (0.5, 0.5 * SEGMENT)
    trapped, _ = _project_from_seed(monkeypatch, fs, point, seed, escape=False)
    assert trapped > 1e-2
    distance, state = _project_from_seed(monkeypatch, fs, point, seed, escape=True)
    assert distance < 1e-9
    assert state["candidate_kind"][0] == "collapsed_span_escape"


def test_foot_beside_a_trailing_edge_strip_checks_the_other_skin(monkeypatch, recorder):
    """A point on the upper skin seeded on the lower skin finds the upper foot."""
    fs = _thin_airfoil_loop()
    point = _point_on_surface(fs, 0.5, 4.9 * SEGMENT)
    seed = (0.5, 1.05 * SEGMENT)
    wrong_skin, _ = _project_from_seed(monkeypatch, fs, point, seed, escape=False)
    assert wrong_skin > 1e-3
    distance, state = _project_from_seed(monkeypatch, fs, point, seed, escape=True)
    assert distance < 1e-9
    assert state["uv"][0][1] > 4 * SEGMENT


def test_feet_away_from_collapsed_strips_get_no_extra_candidates(recorder):
    """The escape pass only touches feet in or beside a collapsed span."""
    fs = _thin_airfoil_loop()
    function = fs.functions[0]
    groups = wsc._collapsed_span_groups(
        np.asarray(function.coefficients.value), function.space.degree, function.space.knots,
    )
    unique_knots = tuple(np.unique(k) for k in function.space.knots)
    specs = wsc._build_collapsed_span_escape_specs(
        np.array([0, 0]), np.array([[0.5, 2.5 * SEGMENT], [0.5, 3.5 * SEGMENT]]),
        {0: (groups, unique_knots)},
    )
    assert specs == []
