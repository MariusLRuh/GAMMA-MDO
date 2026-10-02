"""Derivative checks for baseline chordwise-coordinate restoration."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np

from bsm3.core.boundary_surface_movement.projection import project_onto_oml
from bsm3.preprocessing import get_projection_metadata


def _plane():
    """Create a single bilinear patch with physical x/y equal to u/v."""
    knots = (np.array([0., 0., 1., 1.]),) * 2
    space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2, degree=(1, 1),
        coefficients_shape=(2, 2), knots=knots,
    )
    u, v = np.meshgrid([0., 1.], [0., 1.], indexing="ij")
    return lfs.FunctionSet(functions={0: lfs.Function(
        space=space,
        coefficients=np.stack((u, v, np.zeros_like(u)), axis=-1),
    )})


def test_chordwise_restore_is_differentiable():
    """A half restore retains half of the projected v sensitivity."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        component = _plane()
        point = csdl.Variable(value=np.array([[0.4, 0.7, 0.1]]))
        metadata = get_projection_metadata(
            component=component,
            vertices=point.value,
            vertex_ids=np.array([0]),
            para_coords=np.array([[0., 0.4, 0.3]]),
            chordwise_reference=np.array([0.3]),
            chordwise_restore_weight=np.array([0.5]),
        )
        output = project_onto_oml(
            deformed_mesh_vertices=point,
            deformed_mesh_vertex_ids=np.array([0]),
            projection_metadata=[metadata],
            projection_options={"warm_start_nu": 11, "warm_start_nv": 11},
        ).values
        analytic = csdl.derivative(output, point)
        np.testing.assert_allclose(output.value[0], [0.4, 0.5, 0.], atol=1e-12)
        np.testing.assert_allclose(analytic.value[1, 1], 0.5, atol=1e-8)
        baseline = point.value.copy()
        h = 1e-5
        point.value = baseline + np.array([[0., h, 0.]])
        recorder.execute()
        plus = output.value[0, 1]
        point.value = baseline - np.array([[0., h, 0.]])
        recorder.execute()
        minus = output.value[0, 1]
        np.testing.assert_allclose((plus - minus) / (2 * h), 0.5, rtol=1e-5)
    finally:
        recorder.stop()
