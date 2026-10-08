"""Closest-point and derivative regressions for interior spline creases."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel, FunctionSetClosestDistanceOperation,
    stack_function_set_coefficients,
)
from gamma_mdo.core.projections.function_set_projection_custom_op import FunctionSetProjectionOperation


@pytest.fixture
def recorder():
    """Own CSDL state for each independent regression."""
    rec = csdl.Recorder(inline=True)
    rec.start()
    try:
        yield rec
    finally:
        rec.stop()


def _fin():
    """Build a closed fin-like loft without external geometry assets."""
    rows = []
    for chord, thickness, leading_x, z in (
        (9., 0., 14.2, 0.), (9., .8, 14.2, 0.), (7., .6, 16.5, 1.2),
        (3., .3, 21.5, 4.8), (3., 0., 21.5, 4.8),
    ):
        t = np.linspace(0., np.pi, 21)
        x = leading_x + chord * .5 * (1 + np.cos(t))
        y = thickness * .5 * np.sin(t)
        section = np.vstack((np.column_stack((x, y)), np.column_stack((x[::-1][1:], -y[::-1][1:]))))
        rows.append(np.column_stack((section, np.full(41, z))))
    return _surface(np.stack(rows), (1, 3),
                    (np.array([0, 0, .25, .5, .75, 1, 1.]),
                     np.r_[np.zeros(3), np.linspace(0, 1, 39), np.ones(3)]))


def _surface(net, degrees, knots):
    """Create a one-patch spline fixture."""
    space = lfs.BSplineSpaceNew(num_parametric_dimensions=2, degree=degrees,
                               coefficients_shape=net.shape[:2], knots=knots)
    return lfs.FunctionSet(functions={0: lfs.Function(space=space, coefficients=net)})


def _roof(*, corner=False):
    """Make a sharp valley with known exterior and an optional crease crossing."""
    x = np.array([-1., 0., 1.])
    y = x if corner else np.array([-1., 1.])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = 4 * abs(xx) + (4 * abs(yy) if corner else 0.)
    knots = np.array([0., 0., .5, 1., 1.])
    return _surface(np.stack((xx, yy, zz), axis=-1), (1, 1),
                    (knots, knots if corner else np.array([0., 0., 1., 1.])))


def _model(function_set, **kwargs):
    """Return the projection model and its original coefficient stack."""
    model = FunctionSetProjectionModel(function_set, **kwargs)
    return model, stack_function_set_coefficients(function_set, model.patch_ids)


def test_fin_crease_is_closer_than_unconstrained_newton(recorder):
    """Compare against an independent one-dimensional spline minimization."""
    fs = _fin()
    model, coeff = _model(fs)
    p0 = np.array([6.449014311457227, -5.957186430946168, -.5843645648992676])
    offsets = np.array([[0, 0, 0], [.01, 0, 0], [-.01, 0, 0], [0, .01, 0],
                        [0, -.01, 0], [0, 0, .01], [0, 0, -.01], [.01, -.01, .01]])
    points = p0 + offsets
    distance, state = model.project(coeff, points)
    curve = BSpline(fs.functions[0].space.knots[1], coeff.reshape(5, 41, 3)[1], 3)
    spans = np.unique(curve.t)
    references = []
    for point in points:
        objective = lambda v: np.sum((curve(v) - point)**2)
        candidates = [objective(v) for v in spans]
        candidates += [minimize_scalar(objective, bounds=(a, b), method='bounded',
                                       options={'xatol': 1e-15}).fun
                       for a, b in zip(spans[:-1], spans[1:])]
        references.append(np.sqrt(min(candidates)))
    assert np.all(state['converged'])
    np.testing.assert_array_equal(state['uv'][:, 0], .25)
    np.testing.assert_allclose(distance, references, atol=1e-9, rtol=0)


def _derivative_check(recorder, fs, point, kind, fixed_axes):
    """Differentiate actual public operations on a verified stable feature."""
    model, coeff = _model(fs)
    direction = np.random.default_rng(182).normal(size=coeff.shape)
    c = csdl.Variable(value=coeff)
    p = csdl.Variable(value=np.array([point]))
    outputs = [FunctionSetProjectionOperation(model, return_parametric=uv).evaluate(c, p)
               for uv in (True, False)]
    outputs.append(FunctionSetClosestDistanceOperation(model).evaluate(c, p))
    jac_p = [csdl.derivative(o, p).value.copy() for o in outputs]
    jac_c = [csdl.derivative(o, c).value.copy() @ direction.ravel() for o in outputs]
    _, base = model.project(coeff, np.array([point]))
    assert base['candidate_kind'] == [kind]

    def forward(cp, pp):
        """Require every perturbed solve to select the same surface feature."""
        distance, state = model.project(cp, pp[None])
        assert np.all(state['converged'])
        assert state['candidate_kind'] == [kind]
        np.testing.assert_array_equal(state['uv'][0, fixed_axes], base['uv'][0, fixed_axes])
        return [np.r_[0., state['uv'][0]], state['projected_points'][0], np.ravel(distance)]

    h = 1e-6
    fd_p = [np.zeros_like(j) for j in jac_p]
    for axis in range(3):
        shift = np.eye(3)[axis]*h
        plus, minus = forward(coeff, point+shift), forward(coeff, point-shift)
        for index in range(3):
            fd_p[index][:, axis] = (plus[index]-minus[index])/(2*h)
    plus, minus = forward(coeff+h*direction, point), forward(coeff-h*direction, point)
    for index in range(3):
        np.testing.assert_allclose(jac_p[index], fd_p[index], atol=2e-8, rtol=1e-5)
        np.testing.assert_allclose(jac_c[index], (plus[index]-minus[index])/(2*h), atol=2e-8, rtol=1e-5)
    np.testing.assert_allclose(jac_p[0][1+np.asarray(fixed_axes)], 0, atol=1e-12)
    np.testing.assert_allclose(jac_c[0][1+np.asarray(fixed_axes)], 0, atol=1e-12)
    if len(fixed_axes) == 2:
        np.testing.assert_allclose(jac_p[1], 0, atol=1e-12)


@pytest.mark.parametrize('axis', [0, 1])
def test_crease_uv_point_and_distance_derivatives(recorder, axis):
    """Catch a missing fixed-axis mask in either parametric direction."""
    fs = _fin()
    if axis == 1:
        fun = fs.functions[0]
        fs = _surface(fun.coefficients.value.transpose(1, 0, 2)[::-1], (3, 1),
                      (1-fun.space.knots[1][::-1], fun.space.knots[0]))
    _derivative_check(recorder, fs,
                      np.array([6.449014311457227, -5.957186430946168, -.5843645648992676]),
                      f"current_{'uv'[axis]}_c0_line", [axis])


def test_crease_corner_selection_and_derivatives(recorder):
    """A crease crossing fixes both coordinates, but remains coefficient-dependent."""
    fs = _roof(corner=True)
    model, coeff = _model(fs)
    distance, state = model.project(coeff, np.array([[0., 0., -1.]]))
    np.testing.assert_array_equal(state['uv'], [[.5, .5]])
    np.testing.assert_allclose(distance, [1.], atol=1e-14)
    _derivative_check(recorder, fs, np.array([0., 0., -1.]), 'current_c0_corner_point', [0, 1])


def test_sharp_crease_normal_sign_uses_both_faces(recorder):
    """An exterior query can lie behind one incident face's normal half-space."""
    model, coeff = _model(_roof(), sdf=True, sdf_sign_mode='normal')
    distance, state = model.project(coeff, np.array([[-1., 0., -1.]]))
    assert state['candidate_kind'] == ['current_u_c0_line']
    np.testing.assert_array_equal(state['uv'], [[.5, .5]])
    np.testing.assert_allclose(distance, [np.sqrt(2)], atol=1e-12)


def test_v_crease_normal_sign(recorder):
    """Exercise the second parametric direction with preserved orientation."""
    original = _roof().functions[0]
    net = original.coefficients.value.transpose(1, 0, 2)[::-1]
    fs = _surface(net, (1, 1), (original.space.knots[1], original.space.knots[0]))
    model, coeff = _model(fs, sdf=True, sdf_sign_mode='normal')
    distance, state = model.project(coeff, np.array([[-1., 0., -1.]]))
    assert state['candidate_kind'] == ['current_v_c0_line']
    np.testing.assert_allclose(distance, [np.sqrt(2)], atol=1e-12)


def test_corner_normal_sign(recorder):
    """Score all four incident faces of an exterior corner projection."""
    model, coeff = _model(_roof(corner=True), sdf=True, sdf_sign_mode='normal')
    distance, state = model.project(coeff, np.array([[-1., -1., -1.]]))
    assert state['candidate_kind'] == ['current_c0_corner_point']
    np.testing.assert_allclose(distance, [np.sqrt(3)], atol=1e-12)


def test_repeated_cubic_v_knot_sign(recorder):
    """Resolve one-sided normals at a cubic knot with multiplicity three."""
    along, across = np.meshgrid([-1., 1.], np.linspace(-1., 1., 7), indexing='ij')
    net = np.stack((across, -along, 4*abs(across)), axis=-1)
    fs = _surface(net, (1, 3), (np.array([0., 0., 1., 1.]),
                               np.r_[np.zeros(4), [.5]*3, np.ones(4)]))
    model, coeff = _model(fs, sdf=True, sdf_sign_mode='normal')
    distance, state = model.project(coeff, np.array([[-1., 0., -1.]]))
    assert state['candidate_kind'] == ['current_v_c0_line']
    np.testing.assert_array_equal(state['uv'][:, 1], [.5])
    np.testing.assert_allclose(distance, [np.sqrt(2)], atol=1e-12)
