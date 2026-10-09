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


def test_adjacent_collapsed_spans_merge_only_when_their_images_match(recorder):
    """A discontinuity between two collapsed spans keeps them apart."""
    net = np.stack([
        np.array([[0., 0., z], [0., 0., z], [1., 0., z], [1., 0., z]]) for z in (0.0, 1.0)
    ])
    knots = (np.array([0., 0., 1., 1.]), np.array([0., 0., .5, .5, 1., 1.]))
    groups = wsc._collapsed_span_groups(net, (1, 1), knots)
    assert groups[1] == [[(0.0, 0.5)], [(0.5, 1.0)]]
    # The tolerance is relative: the same net scaled by 1e-6 gives the same answer.
    assert wsc._collapsed_span_groups(net * 1e-6, (1, 1), knots)[1] == groups[1]


def _derivative_check(monkeypatch, function_set, point, seed_uv, *, free_seed_only=False):
    """Compare projection Jacobians with finite differences on one smooth branch."""
    def forced_warm_start(*, mesh, points):
        count = np.asarray(points).shape[0]
        return WarmStartResult(
            patch_id=np.zeros(count, dtype=int),
            uv0=np.tile(np.asarray(seed_uv, dtype=float), (count, 1)),
            closest_pts=np.asarray(points, dtype=float), cell_ids=np.zeros(count, dtype=int),
            dist2=np.full(count, 1.0),
        )

    def free_seed_specs(*, patch_id, uv0, **_):
        return [
            wsc._CandidateSpec(point_index=i, patch_id=int(pid), uv0=tuple(map(float, uv)),
                               fixed_axis=-1, fixed_value=0.0, kind="warm_start_patch")
            for i, (pid, uv) in enumerate(zip(patch_id, uv0))
        ]

    from gamma_mdo.core.projections.function_set_projection_custom_op import (
        FunctionSetProjectionOperation,
    )

    with monkeypatch.context() as patch:
        patch.setattr(wsc, "warm_start_from_triangulation", forced_warm_start)
        if free_seed_only:
            patch.setattr(wsc, "_build_candidate_specs", free_seed_specs)
        model = FunctionSetProjectionModel(function_set)
        coefficients = stack_function_set_coefficients(function_set, model.patch_ids)
        _, base = model.project(coefficients, point[None])
        c = csdl.Variable(value=coefficients)
        p = csdl.Variable(value=point[None])
        projected = FunctionSetProjectionOperation(model, return_parametric=False).evaluate(c, p)
        jac_p = np.asarray(csdl.derivative(projected, p).value)
        # An affine change of every control point keeps coincident control
        # points coincident, so collapsed spans stay collapsed, as under the
        # rigid strut motion of the C208 example. A generic perturbation would
        # open the strip, where the projection is not differentiable.
        rng = np.random.default_rng(208)
        direction = coefficients @ rng.normal(size=(3, 3)).T + rng.normal(size=3)
        jac_c = np.asarray(csdl.derivative(projected, c).value) @ direction.ravel()

        def forward(cp, pp):
            _, state = model.project(cp, pp[None])
            # Same smooth branch: the foot moves continuously with the input.
            assert np.linalg.norm(state["projected_points"][0] - base["projected_points"][0]) < 1e-4
            return state["projected_points"][0]

        h = 1e-6
        fd_p = np.column_stack([
            (forward(coefficients, point + h * e) - forward(coefficients, point - h * e)) / (2 * h)
            for e in np.eye(3)
        ])
        fd_c = (forward(coefficients + h * direction, point)
                - forward(coefficients - h * direction, point)) / (2 * h)
    np.testing.assert_allclose(jac_p, fd_p, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(jac_c, fd_c, atol=1e-7, rtol=1e-5)
    return base["candidate_kind"][0], jac_p


def test_escape_foot_derivatives_match_finite_differences(monkeypatch, recorder):
    """An escape solution is an ordinary smooth foot on the skin."""
    fs = _thin_airfoil_loop()
    point = _point_on_surface(fs, 0.5, 1.3 * SEGMENT)[0] + np.array([0.0, -0.01, 0.0])
    kind, _ = _derivative_check(monkeypatch, fs, point, (0.5, 0.5 * SEGMENT))
    assert kind == "collapsed_span_escape"


def test_foot_inside_a_strip_is_an_edge_line_foot(monkeypatch, recorder):
    """Past a sharp trailing edge a foot left inside the strip moves along the edge."""
    fs = _thin_airfoil_loop()
    point = np.array([1.05, 0.0, 0.4])
    kind, jac_p = _derivative_check(
        monkeypatch, fs, point, (0.4, 0.5 * SEGMENT), free_seed_only=True,
    )
    # No escape is strictly closer than the edge itself, so the strip foot stays
    # and is labeled as a fixed-v line foot.
    assert kind == "collapsed_v_c0_line"
    # The edge runs along z: only the z offset of the query moves the foot.
    np.testing.assert_allclose(jac_p, np.diag([0.0, 0.0, 1.0]), atol=1e-9)
