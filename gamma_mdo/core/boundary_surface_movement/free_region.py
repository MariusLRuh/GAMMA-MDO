"""Per-component free-vertex selection for the elasticity propagator.

The elastic solve needs a *free set* ``F`` (unknown, elastically solved
vertices) and a *prescribed set* (Dirichlet: seam + frozen).  Rather than the
legacy influence-box / length-fraction machinery, the free set is specified
declaratively per component as an intersection of normalized-coordinate slabs:

    wing:     y in [0.0, 0.5)  (|y| / semi-span)          -> inboard band free
    fuselage: x in [0.1, 0.95) ((x - x_min)/(x_max-x_min)) -> mid-body free

A vertex is free iff it satisfies the x AND y AND z criteria of its
component's region (an unset axis imposes no constraint).  Free vertices are
unknowns in all three Cartesian components; everything else on the component is
prescribed (moves rigidly with the component by parametric reevaluation, or
holds the exact seam displacement).  The free set may span several components
(e.g. an inboard-wing band plus a fuselage band); the elastic solve blends the
seam motion through it and leaves the rigid far field as Dirichlet data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AxisRange:
    """A normalized free interval on one axis.

    ``mode`` selects the normalization of the raw coordinate ``c`` over the
    component's vertices:

    * ``"extent"`` -- ``t = (c - c_min) / (c_max - c_min)`` in ``[0, 1]``
      (fraction from the low end, e.g. fuselage station from the nose).
    * ``"abs"`` -- ``t = |c| / max(|c|)`` in ``[0, 1]`` (symmetric fraction from
      zero, e.g. spanwise fraction ``|y| / semi-span``).

    A vertex is free on this axis iff ``lower <= t < upper`` (``None`` bounds
    are unbounded).  The lower bound is inclusive and the upper bound exclusive,
    matching the ``0.1 <= x < 0.95`` convention.

    Parameters
    ----------
    lower
        Optional inclusive normalized lower bound.
    upper
        Optional exclusive normalized upper bound.
    mode
        Normalization mode, ``"extent"`` or ``"abs"``.
    """

    lower: float | None = None
    upper: float | None = None
    mode: str = "extent"

    def __post_init__(self):
        """Validate the mode and the bound ordering.

        Raises
        ------
        ValueError
            If ``mode`` is neither ``"extent"`` nor ``"abs"``, or the bounds
            are inconsistent.
        """
        if self.mode not in ("extent", "abs"):
            raise ValueError("AxisRange.mode must be 'extent' or 'abs'.")
        if (
            self.lower is not None
            and self.upper is not None
            and float(self.lower) > float(self.upper)
        ):
            raise ValueError("AxisRange.lower must not exceed AxisRange.upper.")

    def mask(self, coordinate: np.ndarray, *, reference: np.ndarray) -> np.ndarray:
        """Select coordinates inside this normalized interval.

        Parameters
        ----------
        coordinate
            Values to classify.
        reference
            Component values defining the normalization extent.

        Returns
        -------
        numpy.ndarray
            Boolean mask aligned with ``coordinate``.
        """
        coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
        reference = np.asarray(reference, dtype=float).reshape(-1)
        if self.mode == "extent":
            lower_ref = float(np.min(reference))
            span = float(np.max(reference)) - lower_ref
            normalized = (
                np.zeros_like(coordinate)
                if span <= 0.0
                else (coordinate - lower_ref) / span
            )
        else:  # "abs"
            scale = float(np.max(np.abs(reference))) if reference.size else 0.0
            normalized = (
                np.zeros_like(coordinate)
                if scale <= 0.0
                else np.abs(coordinate) / scale
            )
        mask = np.ones_like(coordinate, dtype=bool)
        if self.lower is not None:
            mask &= normalized >= float(self.lower)
        if self.upper is not None:
            mask &= normalized < float(self.upper)
        return mask


@dataclass(frozen=True)
class ComponentFreeRegion:
    """Free-vertex criteria for one component (AND of per-axis slabs).

    An unset axis (``None``) imposes no constraint on that axis.  By default the
    normalization reference is the component's own owned vertices; supply
    ``reference_vertices`` to normalize against a different extent.

    Parameters
    ----------
    component
        Geometry component owning the candidate vertices.
    x, y, z
        Optional normalized interval on each coordinate axis.
    reference_vertices
        Optional coordinates defining normalization extents.
    """

    component: object
    x: AxisRange | None = None
    y: AxisRange | None = None
    z: AxisRange | None = None
    reference_vertices: np.ndarray | None = None

    def free_mask(self, vertices: np.ndarray) -> np.ndarray:
        """Select vertices satisfying every configured axis interval.

        Parameters
        ----------
        vertices
            Candidate coordinates with shape ``(num_vertices, 3)``.

        Returns
        -------
        numpy.ndarray
            Boolean free-vertex mask.
        """
        vertices = np.asarray(vertices, dtype=float).reshape((-1, 3))
        reference = (
            vertices
            if self.reference_vertices is None
            else np.asarray(self.reference_vertices, dtype=float).reshape((-1, 3))
        )
        mask = np.ones(vertices.shape[0], dtype=bool)
        for axis, criterion in enumerate((self.x, self.y, self.z)):
            if criterion is None:
                continue
            mask &= criterion.mask(
                vertices[:, axis], reference=reference[:, axis]
            )
        return mask


def select_free_vertices(
    *,
    free_regions: Sequence[ComponentFreeRegion],
    component_vertex_ids: Mapping[object, np.ndarray],
    mesh_vertices: np.ndarray,
    exclude_ids: np.ndarray | Sequence[int] = (),
) -> np.ndarray:
    """Return the sorted union of free vertex ids across all components.

    ``component_vertex_ids`` maps each region's component (or ``id(component)``)
    to the global mesh vertex ids owned by it.  ``exclude_ids`` (e.g. seam
    vertices) are always prescribed and removed from the result.

    Parameters
    ----------
    free_regions
        Per-component normalized free-region declarations.
    component_vertex_ids
        Component-to-global-vertex ownership mapping.
    mesh_vertices
        Complete baseline coordinate array.
    exclude_ids
        Global IDs forced to remain prescribed.

    Returns
    -------
    numpy.ndarray
        Sorted unique global IDs selected as free.

    Raises
    ------
    KeyError
        If a region's component is absent from ``component_vertex_ids``.
    """
    mesh_vertices = np.asarray(mesh_vertices, dtype=float).reshape((-1, 3))
    excluded = set(int(vertex) for vertex in np.asarray(exclude_ids, dtype=np.int64).reshape(-1))

    free_blocks: list[np.ndarray] = []
    for region in free_regions:
        owned_ids = _component_mapping_get(
            component_vertex_ids, region.component, None
        )
        if owned_ids is None:
            raise KeyError(
                "component_vertex_ids is missing an entry for a free region's "
                "component."
            )
        owned_ids = np.asarray(owned_ids, dtype=np.int64).reshape(-1)
        if owned_ids.size == 0:
            continue
        mask = region.free_mask(mesh_vertices[owned_ids])
        free_blocks.append(owned_ids[mask])

    if not free_blocks:
        return np.empty((0,), dtype=np.int64)
    free_ids = np.unique(np.concatenate(free_blocks))
    if excluded:
        free_ids = np.array(
            [vertex for vertex in free_ids if int(vertex) not in excluded],
            dtype=np.int64,
        )
    return free_ids


def _component_mapping_get(mapping, component, default):
    # Accept mappings keyed by the component object or by ``id(component)``.
    # FunctionSet is unhashable, so ``mapping.get(component)`` raises; a hashable
    # component instead misses silently, so fall through to the id key in both.
    sentinel = object()
    try:
        value = mapping.get(component, sentinel)
    except TypeError:
        value = sentinel
    if value is sentinel:
        return mapping.get(id(component), default)
    return value


__all__ = [
    "AxisRange",
    "ComponentFreeRegion",
    "select_free_vertices",
]
