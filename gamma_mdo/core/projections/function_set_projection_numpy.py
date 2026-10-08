"""NumPy-facing projection of points onto an LFS FunctionSet.

This module is an additive adapter around GAMMA's existing triangulation-
accelerated projection model.  It does not create a CSDL operation or add
anything to a CSDL graph.  The returned coordinates use the native
``FunctionSet.evaluate`` convention: one ``(patch_id, uv)`` tuple per point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
    stack_function_set_coefficients,
)


FunctionSetCoordinates = list[tuple[int, np.ndarray]]


class FunctionSetProjector:
    """Project points onto a FunctionSet without creating a CSDL operation.

    Parameters
    ----------
    function_set : object
        FunctionSet-like object exposing a ``functions`` mapping.
    **projection_options
        Options forwarded unchanged to GAMMA's
        :class:`FunctionSetProjectionModel`, including ``warm_start_nu`` and
        ``warm_start_nv``.

    Notes
    -----
    Constructing a projector caches the patch sampling topology, basis
    stencils, and patch-edge relationships.  Reusing the instance avoids
    repeating that setup.  The physical warm-start mesh is still rebuilt from
    the coefficients supplied to each :meth:`project` call.
    """

    def __init__(self, function_set: Any, **projection_options: Any) -> None:
        self.function_set = function_set
        self.model = FunctionSetProjectionModel(
            function_set=function_set,
            **projection_options,
        )

    def project(
        self,
        points: np.ndarray,
        *,
        stacked_coefficients: np.ndarray | None = None,
    ) -> FunctionSetCoordinates:
        """Return closest parametric coordinates for ``points``.

        Parameters
        ----------
        points : ndarray of shape (n, physical_dimension)
            Physical points to project.
        stacked_coefficients : ndarray, optional
            Coefficients to use for the projection.  When omitted, the current
            coefficient values are read from ``function_set`` in the same
            patch order used by the underlying projection model.

        Returns
        -------
        list[tuple[int, ndarray]]
            Coordinates in the native LFS representation
            ``[(patch_id, array([u, v])), ...]``.  The result can be passed
            directly to ``function_set.evaluate(parametric_coordinates=...)``.
        """

        if stacked_coefficients is None:
            stacked_coefficients = stack_function_set_coefficients(
                self.function_set,
                self.model.patch_ids,
            )

        _, state = self.model.project(
            stacked_coefficients=stacked_coefficients,
            points=points,
        )
        return _function_set_coordinates_from_state(state)


def _function_set_coordinates_from_state(state: dict[str, object]) -> FunctionSetCoordinates:
    """Convert projection state to the native LFS coordinate representation."""

    try:
        patch_ids = np.asarray(state["patch_id"], dtype=np.int64).reshape(-1)
        uv = np.asarray(state["uv"], dtype=float).reshape(-1, 2)
    except KeyError as exc:
        raise ValueError(
            "Projection state must contain 'patch_id' and 'uv' entries."
        ) from exc

    if patch_ids.shape[0] != uv.shape[0]:
        raise ValueError(
            "Projection state entries 'patch_id' and 'uv' must contain the "
            "same number of points."
        )

    return [
        (int(patch_id), coordinates.copy())
        for patch_id, coordinates in zip(patch_ids, uv)
    ]


__all__ = ["FunctionSetCoordinates", "FunctionSetProjector"]
