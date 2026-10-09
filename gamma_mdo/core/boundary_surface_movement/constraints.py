"""Fixed geometric constraints for boundary-surface mesh motion."""

from __future__ import annotations

import csdl_alpha as csdl
import numpy as np


def identify_symmetry_plane_vertices(
    vertices,
    *,
    axis: int = 1,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return baseline vertex IDs on ``coordinate[axis] = 0``.

    Identification is a setup-time operation.  The returned set therefore
    remains fixed throughout design evaluation and differentiation.

    Parameters
    ----------
    vertices
        Baseline coordinate array with three columns.
    axis
        Coordinate axis normal to the symmetry plane.
    tolerance
        Nonnegative absolute coordinate tolerance.

    Returns
    -------
    numpy.ndarray
        Sorted integer indices lying on the symmetry plane.

    Raises
    ------
    ValueError
        If ``axis`` or ``tolerance`` is invalid.
    """
    points = np.asarray(
        getattr(vertices, "value", vertices),
        dtype=float,
    ).reshape((-1, 3))
    coordinate = int(axis)
    if coordinate not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    return np.where(np.abs(points[:, coordinate]) <= tol)[0].astype(np.int64)


def enforce_symmetry_plane(
    mesh_vertices,
    *,
    vertex_ids: np.ndarray,
    axis: int = 1,
    coordinate: float = 0.0,
) -> csdl.Variable:
    """Set one coordinate of a fixed vertex set to an exact constant.

    The overwritten coordinate has an exact zero derivative, which is the
    appropriate essential condition for a fixed symmetry plane.  The other two
    coordinates remain fully differentiable.

    Parameters
    ----------
    mesh_vertices
        CSDL or NumPy coordinate array with shape ``(num_vertices, 3)``.
    vertex_ids
        Fixed vertex indices whose coordinate is overwritten.
    axis
        Coordinate axis normal to the symmetry plane.
    coordinate
        Finite constant assigned on the selected axis.

    Returns
    -------
    csdl.Variable
        Coordinates with the essential symmetry condition applied.

    Raises
    ------
    ValueError
        If shapes, indices, the axis, or the coordinate are invalid.
    """
    values = (
        mesh_vertices
        if hasattr(mesh_vertices, "value")
        else csdl.Variable(value=np.asarray(mesh_vertices, dtype=float))
    )
    if len(values.shape) != 2 or values.shape[1] != 3:
        raise ValueError("mesh_vertices must have shape (num_vertices, 3).")
    ids = np.unique(np.asarray(vertex_ids, dtype=np.int64).reshape(-1))
    if ids.size == 0:
        return values * 1.0
    if np.any(ids < 0) or np.any(ids >= values.shape[0]):
        raise ValueError("vertex_ids contains an out-of-range vertex.")
    coordinate_axis = int(axis)
    if coordinate_axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    fixed_value = float(coordinate)
    if not np.isfinite(fixed_value):
        raise ValueError("coordinate must be finite.")
    return values.set(
        _coordinate_slice(ids, coordinate_axis),
        csdl.Variable(
            value=np.full((ids.size, 1), fixed_value, dtype=float)
        ),
    )


def _coordinate_slice(rows, column):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, column : column + 1]
    return csdl.slice[rows.tolist(), column : column + 1]


__all__ = [
    "enforce_symmetry_plane",
    "identify_symmetry_plane_vertices",
]
