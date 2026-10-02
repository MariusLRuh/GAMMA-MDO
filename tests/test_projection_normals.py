"""CAD normal-sign regressions without triangulated inside/outside classification."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import pytest

from bsm3.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel, stack_function_set_coefficients,
)
from bsm3.core.projections.surface_normals_numpy import _PatchNormals


@pytest.fixture
def recorder():
    """Own the CSDL graph used to construct each synthetic surface."""
    rec = csdl.Recorder(inline=True)
    rec.start()
    try:
        yield rec
    finally:
        rec.stop()


def _surface(net, degrees, knots):
    """Build one oriented tensor-product patch."""
    space = lfs.BSplineSpace(num_parametric_dimensions=2, degree=degrees,
                            coefficients_shape=net.shape[:2], knots=knots)
    return lfs.FunctionSet({0: lfs.Function(space=space, coefficients=net)})


def _model(fs):
    """Forbid the enclosed-sign path even if a future default changes."""
    model = FunctionSetProjectionModel(fs, sdf=True, sdf_sign_mode='normal')
    def forbidden(*args, **kwargs):
        """Make any triangulated sign query fail explicitly."""
        raise AssertionError('enclosed sign invoked')
    model._compute_sdf_metadata = forbidden
    return model, stack_function_set_coefficients(fs, model.patch_ids)


@pytest.mark.parametrize('slope', [1., 4.])
@pytest.mark.parametrize('concave', [False, True])
def test_incident_normal_sign_for_sharp_edges(recorder, slope, concave):
    """Check both material sides of sharp folds, including a 90 degree fold."""
    x, y = np.meshgrid([-4., 0., 4.], [-4., 4.], indexing='ij')
    direction = -1 if concave else 1
    fs = _surface(np.stack((x, y, direction*slope*abs(x)), axis=-1), (1, 1),
                  (np.array([0., 0., .5, 1., 1.]), np.array([0., 0., 1., 1.])))
    model, coeff = _model(fs)
    # The material is above the graph (Su cross Sv points into it).
    point = np.array([[.2, 0., -direction]])
    distance, state = model.project(coeff, point)
    assert state['candidate_kind'] == ['current_u_c0_line']
    assert not state['sign_ambiguous'].any()
    np.testing.assert_allclose(state['projected_points'], 0, atol=1e-12)
    np.testing.assert_allclose(distance, direction*np.sqrt(1.04), atol=1e-12)
    assert state['normal_support_count'][0] == 2
    # For the sharper concave fold the maximum individual face score is
    # positive, although this query is inside: averaging fixes that case.
    if concave and slope == 4:
        p = np.array([[1., 0., 1.]])
        value, _ = model.project(coeff, p)
        assert value[0] < 0


def _collapsed(*, axis=1, degree=3, boundary=False):
    """Make two oriented planar faces separated by a constant parameter strip."""
    # Three Bezier spans; the first/last approach the common edge to order p.
    left = np.array([[-2., 2.]] + [[0., 0.]]*degree)
    middle = np.zeros((degree+1, 2))
    right = np.array([[0., 0.]]*degree + [[2., 2.]])
    section = np.vstack((left, middle[1:], right[1:]))
    knots = np.r_[np.zeros(degree+1), [.3]*degree, [.7]*degree, np.ones(degree+1)]
    if boundary:
        section = np.vstack((middle, right[1:]))
        knots = np.r_[np.zeros(degree+1), [.5]*degree, np.ones(degree+1)]
    net = np.array([np.column_stack((section[:, 0], np.full(len(section), y),
                                    section[:, 1])) for y in (2., -2.)])
    linear = np.array([0., 0., 1., 1.])
    if axis == 0:
        net = net.transpose(1, 0, 2)[:, ::-1]
        return _surface(net, (degree, 1), (knots, linear))
    return _surface(net, (1, degree), (linear, knots))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('degree', [1, 3])
def test_collapsed_strip_analytic_limits_and_secants(recorder, axis, degree):
    """Recover first- or third-order limits in either parameter direction."""
    model, coeff = _model(_collapsed(axis=axis, degree=degree))
    info = model.patch_infos[0]
    patch = _PatchNormals(info, coeff.reshape(info.coefficient_shape))
    uv = np.array([.5, .5])
    interval = patch.interval(uv, axis)
    assert interval == (.3, .7)
    point, _ = patch.partial(uv, (0, 0))
    faces = []
    for boundary, side in zip(interval, (-1, 1)):
        face = patch.limit(uv, axis, boundary, side, point)
        assert face is not None
        faces.append(face[0])
        # Independent finite differences at a resolvable interior offset.
        regular = uv.copy()
        regular[axis] = boundary+side*.01
        h = 1e-4
        tangents = []
        for a in (0, 1):
            offset = np.eye(2)[a]*h
            plus, _ = patch.partial(regular+offset, (0, 0))
            minus, _ = patch.partial(regular-offset, (0, 0))
            tangents.append((plus-minus)/(2*h))
        normal = np.cross(*tangents)
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(face[0], normal, atol=1e-10)
    np.testing.assert_allclose(np.sum(faces, axis=0)/np.linalg.norm(np.sum(faces, axis=0)),
                               [0., 0., 1.], atol=1e-12)
    diagnostics = {}
    sign, _, normal = model._compute_normal_sign_metadata(
        coeff, np.array([[0., 0., -1.]]), point[None], np.array([0]), uv[None],
        ['current_interior'], np.array([False]), diagnostics=diagnostics)
    np.testing.assert_array_equal(sign, [1.])
    assert diagnostics['collapsed_parametric_axes'][0, axis]
    assert not diagnostics['sign_ambiguous'][0]
    np.testing.assert_allclose(normal, [[0., 0., 1.]], atol=1e-12)


def test_missing_incident_face_is_ambiguous(recorder):
    """A boundary strip without a neighbor must not invent an outside sign."""
    model, coeff = _model(_collapsed(boundary=True))
    diagnostics = {}
    sign, _, _ = model._compute_normal_sign_metadata(
        coeff, np.array([[0., 0., -1.]]), np.zeros((1, 3)),
        np.array([0]), np.array([[.5, .2]]), ['current_interior'], np.array([False]),
        diagnostics=diagnostics)
    assert np.isnan(sign[0])
    assert diagnostics['sign_ambiguous'][0]


def test_translated_collapsed_airfoil_sign_and_distance(recorder):
    """Regress roundoff normals on a translated cubic leading-edge strip."""
    t = np.linspace(0., 1., 12)
    x = 190.+4*(1-t)
    z = .25*np.sin(np.pi*(1-t))**.8
    section = np.vstack((np.column_stack((x, z))[:-1],
                         np.repeat([[190., 0.]], 4, axis=0),
                         np.column_stack((x[::-1], -z[::-1]))[1:]))
    n = len(section)
    net = np.array([np.column_stack((section[:, 0], np.full(n, y), section[:, 1]))
                    for y in (-1., -12.)])
    fs = _surface(net, (1, 3), (np.array([0., 0., 1., 1.]),
                               np.r_[np.zeros(3), np.linspace(0, 1, n-2), np.ones(3)]))
    model, coeff = _model(fs)
    rng = np.random.default_rng(11)
    points = np.column_stack((190.-rng.uniform(4, 15, 200),
                              rng.uniform(-11, -2, 200), rng.uniform(-.2, .2, 200)))
    values, state = model.project(coeff, points)
    strip = state['collapsed_parametric_axes'][:, 1]
    assert strip.sum() > 50
    assert not state['sign_ambiguous'][strip].any()
    assert np.all(values[strip] > 0)
    exact = np.hypot(points[:, 0]-190., points[:, 2])
    np.testing.assert_allclose(values[strip], exact[strip], atol=1e-11, rtol=0)
    # Non-strip candidates exercise a separate, known closest-point defect;
    # this regression specifically covers normal recovery at the correct edge.
    unsigned = FunctionSetProjectionModel(fs, sdf=False)
    plain, reference = unsigned.project(coeff, points)
    np.testing.assert_array_equal(state['uv'], reference['uv'])
    np.testing.assert_array_equal(state['projected_points'], reference['projected_points'])
    np.testing.assert_array_equal(np.abs(values[strip]), plain[strip])


def test_collapsed_boundary_uses_incident_neighbor(recorder):
    """Recover the missing side from CAD adjacency instead of a seed triangle."""
    right = _collapsed(boundary=True).functions[0]
    section = np.array([[-2., 2.], [0., 0.]])
    net = np.array([np.column_stack((section[:, 0], np.full(2, y), section[:, 1]))
                    for y in (2., -2.)])
    left = _surface(net, (1, 1), (np.array([0., 0., 1., 1.]),)*2).functions[0]
    model, coeff = _model(lfs.FunctionSet({0: right, 1: left}))
    assert (0, 'v0') in model.edge_map
    diagnostics = {}
    sign, _, normals = model._compute_normal_sign_metadata(
        coeff, np.array([[0., 0., -1.]]), np.zeros((1, 3)), np.array([0]),
        np.array([[.5, .2]]), ['current_interior'], np.array([False]),
        diagnostics=diagnostics)
    assert not diagnostics['sign_ambiguous'][0]
    assert diagnostics['normal_support_count'][0] == 2
    np.testing.assert_array_equal(sign, [1.])
    np.testing.assert_allclose(normals, [[0., 0., 1.]], atol=1e-12)


def test_cancelled_incident_normals_are_ambiguous(recorder):
    """A folded-back zero-thickness sheet has no usable combined direction."""
    fs = _collapsed(degree=1)
    net = fs.functions[0].coefficients.value.copy()
    net[..., 0] = abs(net[..., 0])
    net[..., 2] = 0.
    fs = _surface(net, (1, 1), fs.functions[0].space.knots)
    model, coeff = _model(fs)
    diagnostics = {}
    sign, _, _ = model._compute_normal_sign_metadata(
        coeff, np.array([[0., 0., -1.]]), np.zeros((1, 3)), np.array([0]),
        np.array([[.5, .5]]), ['current_interior'], np.array([False]),
        diagnostics=diagnostics)
    assert np.isnan(sign[0])
    assert diagnostics['sign_ambiguous'][0]


def test_two_patch_right_angle_sign(recorder):
    """Combine CAD normals at a 90 degree patch join for an exterior query."""
    patches = {}
    for pid, span in enumerate(([-4., 0.], [0., 4.])):
        x, y = np.meshgrid(span, [-4., 4.], indexing='ij')
        patches[pid] = _surface(np.stack((x, y, abs(x)), axis=-1), (1, 1),
                               (np.array([0., 0., 1., 1.]),)*2).functions[0]
    model, coeff = _model(lfs.FunctionSet(patches))
    values, state = model.project(coeff, np.array([[.2, 0., -1.]]))
    assert not state['sign_ambiguous'].any()
    assert state['normal_support_count'][0] == 2
    np.testing.assert_allclose(state['reference_normals'], [[0., 0., 1.]], atol=1e-12)
    np.testing.assert_allclose(values, [np.sqrt(1.04)], atol=1e-12)


def _bipyramid(*, order=1, axis=0, sectors=False):
    """Build an oriented octahedron with collapsed pole rows and exact distances."""
    ring = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.],
                     [0., -1., 0.], [1., 0., 0.]])
    lower = np.tile([0., 0., -1.], (5, 1))
    upper = np.tile([0., 0., 1.], (5, 1))
    net = np.array([lower]*order + [ring] + [upper]*order)
    radial = np.r_[np.zeros(order+1), [.5]*order, np.ones(order+1)]
    circumferential = np.r_[0., np.linspace(0, 1, 5), 1.]
    patches = {}
    for pid in range(4 if sectors else 1):
        c = net[:, pid:pid+2] if sectors else net
        k = np.array([0., 0., 1., 1.]) if sectors else circumferential
        degrees, knots = (order, 1), (radial, k)
        if axis == 1:
            c = c.transpose(1, 0, 2)[::-1]
            degrees, knots = degrees[::-1], knots[::-1]
        patches[pid] = _surface(c, degrees, knots).functions[0]
    return lfs.FunctionSet(patches)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('order', [1, 2])
def test_pole_sdf_is_finite_and_matches_exact_geometry(recorder, axis, order):
    """Recover both poles, including vanishing first radial derivatives."""
    model, coeff = _model(_bipyramid(axis=axis, order=order))
    points = np.array([[0., 0., -1.2], [.02, -.01, -1.2],
                       [0., 0., 1.2], [-.02, .01, 1.2], [0., 0., -1.], [0., 0., 1.],
                       [0., 0., -.8], [0., 0., .8]])
    values, state = model.project(coeff, points)
    expected = np.array([.2, np.sqrt(.0405), .2, np.sqrt(.0405),
                         0., 0., -.2/np.sqrt(3), -.2/np.sqrt(3)])
    assert np.isfinite(values).all()
    assert not state['sign_ambiguous'].any()
    assert state['converged'].all()
    assert state['pole_normal_recovered'][:6].all()
    np.testing.assert_allclose(values, expected, atol=1e-10)
    unsigned = FunctionSetProjectionModel(model.function_set_wrapper, sdf=False)
    plain, reference = unsigned.project(coeff, points)
    np.testing.assert_array_equal(state['projected_points'], reference['projected_points'])
    np.testing.assert_array_equal(state['uv'], reference['uv'])
    np.testing.assert_array_equal(abs(values), plain)


def test_pole_normal_independent_of_arbitrary_uv_and_seed_resolution(recorder):
    """A pole's sign and combined normal must not depend on its redundant label."""
    fs = _bipyramid(order=2)
    for resolution in (11, 23):
        model = FunctionSetProjectionModel(fs, sdf=True, sdf_sign_mode='normal',
                                          warm_start_nu=resolution, warm_start_nv=resolution)
        coeff = stack_function_set_coefficients(fs, model.patch_ids)
        _, state = model.project(coeff, np.array([[.02, .01, 1.2]]))
        assert state['pole_normal_recovered'][0]
        np.testing.assert_allclose(state['reference_normals'], [[0., 0., -1.]], atol=1e-12)
        uv = np.column_stack((np.ones(9), np.linspace(0, 1, 9)))
        diagnostics = {}
        signs, _, normals = model._compute_normal_sign_metadata(
            coeff, np.tile([.02, .01, 1.2], (9, 1)), np.tile([0., 0., 1.], (9, 1)),
            np.zeros(9, dtype=int), uv, ['current_degenerate_point']*9,
            np.zeros(9, dtype=bool), diagnostics=diagnostics)
        np.testing.assert_array_equal(signs, 1.)
        np.testing.assert_allclose(normals, np.tile([0., 0., -1.], (9, 1)), atol=1e-12)


def test_pole_normal_combines_all_patch_sectors(recorder):
    """Four independent CAD sectors must give the same pole normal as one patch."""
    model, coeff = _model(_bipyramid(sectors=True))
    values, state = model.project(coeff, np.array([[.02, .01, 1.2], [.02, .01, -1.2]]))
    assert state['normal_support_count'].tolist() == [4, 4]
    np.testing.assert_allclose(state['reference_normals'], [[0., 0., -1.], [0., 0., 1.]], atol=1e-12)
    np.testing.assert_allclose(values, np.sqrt(.0405), atol=1e-12)


@pytest.mark.parametrize('curvature', [1., -1.])
def test_smooth_pole_recovers_mixed_derivative_normal(recorder, curvature):
    """A bowl's collapsed row has a unique limiting normal despite Sv being zero."""
    ring = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.],
                     [0., -1., 0.], [1., 0., 0.]])
    net = np.array([np.zeros_like(ring), ring*.5, ring+[0, 0, curvature]])
    fs = _surface(net, (2, 1), (np.r_[np.zeros(3), np.ones(3)],
                               np.r_[0., np.linspace(0, 1, 5), 1.]))
    model, coeff = _model(fs)
    values, state = model.project(coeff, np.array([[0., 0., -.25*curvature], [0., 0., 0.]]))
    assert state['pole_normal_recovered'].all()
    np.testing.assert_allclose(values, [.25*curvature, 0.], atol=1e-12)
    np.testing.assert_allclose(state['reference_normals'], [[0, 0, 1]]*2, atol=1e-12)


def test_pole_signed_distance_query_derivative(recorder):
    """The SDF remains differentiable while the closest pole remains fixed."""
    fs = _bipyramid(order=2)
    model, coeff = _model(fs)
    point = np.array([[.02, .01, 1.2]])
    value, state = model.project(coeff, point)
    expected = (point-[0., 0., 1.])/value[:, None]
    analytic, coefficient_vjp = model.compute_vjp(coeff, point, np.ones(1), state)
    fd = np.zeros((1, 3))
    for j in range(3):
        offset = np.eye(3)[j]*1e-6
        plus, sp = model.project(coeff, point+offset)
        minus, sm = model.project(coeff, point-offset)
        assert sp['pole_normal_recovered'].all() and sm['pole_normal_recovered'].all()
        fd[0, j] = (plus[0]-minus[0])/2e-6
    np.testing.assert_allclose(analytic, expected, atol=1e-10)
    np.testing.assert_allclose(analytic, fd, atol=1e-9)
    # Translate the whole control net: this preserves the pole geometry,
    # unlike perturbing one of its redundant end-row controls independently.
    direction = np.broadcast_to([.2, -.1, .3], coeff.shape)
    plus, sp = model.project(coeff+1e-6*direction, point)
    minus, sm = model.project(coeff-1e-6*direction, point)
    assert sp['pole_normal_recovered'].all() and sm['pole_normal_recovered'].all()
    np.testing.assert_allclose(np.sum(coefficient_vjp*direction),
                               (plus[0]-minus[0])/2e-6, atol=1e-9)


def test_zero_area_pole_fails_explicitly_instead_of_nan(recorder):
    """An entirely collapsed patch cannot supply a normal or a meaningful SDF."""
    fs = _surface(np.zeros((2, 2, 3)), (1, 1), (np.array([0., 0., 1., 1.]),)*2)
    model, coeff = _model(fs)
    with pytest.raises(ValueError, match='CAD normal fan'):
        model._compute_normal_sign_metadata(
            coeff, np.array([[0., 0., -1.]]), np.zeros((1, 3)), np.array([0]),
            np.array([[0., .5]]), ['current_degenerate_point'], np.array([False]))


def test_pole_normal_survives_rotation_translation_and_scaling(recorder):
    """The pole fallback must not assume an aircraft axis or an absolute size."""
    fs = _bipyramid(order=2)
    fun = fs.functions[0]
    rotation, _ = np.linalg.qr(np.array([[1., 2., -3.], [-2., 4., 1.], [3., 1., 2.]]))
    assert np.linalg.det(rotation) > 0
    scale = .003
    shift = np.array([190., -23., 81.])
    net = scale*fun.coefficients.value @ rotation.T + shift
    moved = _surface(net, fun.space.degree, fun.space.knots)
    model, coeff = _model(moved)
    query = scale*np.array([[.02, .01, 1.2]]) @ rotation.T + shift
    values, state = model.project(coeff, query)
    assert state['pole_normal_recovered'][0]
    assert not state['sign_ambiguous'][0]
    np.testing.assert_allclose(values, scale*np.sqrt(.0405), atol=1e-11)
    np.testing.assert_allclose(state['reference_normals'], np.array([[0., 0., -1.]]) @ rotation.T, atol=1e-9)
