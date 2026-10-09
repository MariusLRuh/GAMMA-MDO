"""Regression tests for the warm-start projection robustness fixes.

These pin the two mechanisms that removed the resolution-dependent junction-cell
inversions in the mesh-movement pipeline:

* Fix 1 - the orthogonality Newton now reports ``boundary_clamped``: a point that
  "converged" only because the active set masked an outward residual at a patch
  bound. The driver routes those points into the retry path.
* Fix 4 - the local retry re-seeds the Newton at *several* length scales around
  the current guess (not a single one-cell jitter, and no redundant zero
  offset), so a point parked on a spurious boundary/interior minimum can reach
  the true closest point deeper inside the patch.
"""

from __future__ import annotations

import numpy as np

from gamma_mdo.core.projections.orthogonality_projection_numpy import (
    project_points_orthogonality_newton_numpy,
)
from gamma_mdo.core.projections.warm_start_candidate_projection_numpy import (
    _build_local_retry_candidate_specs,
)


def _flat_xy_patch():
    """A single bilinear patch S(u, v) = (2u - 1, 2v - 1, 0) spanning [-1, 1]^2,
    as the (coeffs, degrees, knot_vectors) the numpy Newton consumes directly."""
    grid_u, grid_v = np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")
    coeffs = np.stack(
        (2.0 * grid_u - 1.0, 2.0 * grid_v - 1.0, np.zeros_like(grid_u)), axis=-1
    )
    degrees = (1, 1)
    knot_vectors = (np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0]))
    return coeffs, degrees, knot_vectors


def test_boundary_clamped_flag_detects_outward_masked_convergence():
    """A point whose closest surface point lies past a patch edge parks on the
    edge with a masked residual and reports converged; ``boundary_clamped`` must
    flag exactly that case and leave genuine interior projections alone."""
    coeffs, degrees, knots = _flat_xy_patch()

    # Point above the interior -> closest point is interior (0, 0, 0).
    # Point beyond the u=1 edge -> unconstrained closest is at u=1.5, which the
    # Newton clips to u=1 while its raw residual still points outward (u>1).
    points = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=float)
    u0s = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    result = project_points_orthogonality_newton_numpy(points, u0s, coeffs, degrees, knots)

    assert result.boundary_clamped is not None
    assert result.converged.all()  # both report converged (the subtle part)
    # interior point: not clamped; beyond-edge point: clamped at u=1.
    assert not bool(result.boundary_clamped[0])
    assert bool(result.boundary_clamped[1])
    assert result.uv[1, 0] == 1.0  # parked on the u=1 bound


def test_local_retry_is_multiscale_and_drops_the_zero_offset():
    """The local retry must probe several scales (so it can escape a boundary
    minimum well inside the patch) and must not re-run the exact failing seed."""
    seed = np.array([0.5, 1.0], dtype=float)  # clamped on the v=1 edge
    specs = _build_local_retry_candidate_specs(
        selected_patch_id=np.array([7], dtype=int),
        selected_uv=seed[None, :],
        point_indices=np.array([0], dtype=int),
        uv_step=0.02,
        scale_factors=(1.0, 4.0, 16.0),
    )

    assert specs, "expected retry candidate specs"
    uvs = np.array([spec.uv0 for spec in specs], dtype=float)

    # All candidates stay on the seed's patch.
    assert all(spec.patch_id == 7 for spec in specs)
    # No candidate re-runs the exact failing seed (the old [0, 0] offset).
    assert not np.any(np.all(np.isclose(uvs, seed[None, :]), axis=1))
    # The coarsest scale reaches deep into the interior: 1 - 16*0.02 = 0.68.
    assert uvs[:, 1].min() <= 0.7
    # Several distinct scales are present, not a single one-cell stencil.
    v_offsets = np.abs(np.round(uvs[:, 1] - 1.0, 6))
    assert np.count_nonzero(np.unique(v_offsets)) >= 3
