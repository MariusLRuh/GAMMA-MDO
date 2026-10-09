"""Geometry parameter containers used by boundary-surface movement.

The classes in this module describe simple, differentiable component
transformations.  They deliberately do not contain aircraft-specific geometry
logic: callers can provide an explicit pivot and reference planform values, or
let :func:`gamma_mdo.core.boundary_surface_movement.deform_geometry` infer a useful
default from the component control points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComponentParameters:
    """Rigid-body parameters shared by all component types.

    Parameters are allowed to be Python scalars, NumPy arrays, or CSDL
    variables.  Rotations use degrees and are applied in x-y-z order about
    ``pivot``.

    Attributes
    ----------
    translation_x, translation_y, translation_z
        Rigid translation along each global axis, in the model's length units.
        Applied after the rotations.
    rotation_x_degrees, rotation_y_degrees, rotation_z_degrees
        Rotation about each axis through ``pivot``, in **degrees**, composed in
        x-y-z order.
    pivot
        Point the rotations act about, as three coordinates. ``None`` lets
        :func:`gamma_mdo.core.boundary_surface_movement.deform_geometry` infer a
        default from the component's control points.
    """

    translation_x: Any = 0.0
    translation_y: Any = 0.0
    translation_z: Any = 0.0
    rotation_x_degrees: Any = 0.0
    rotation_y_degrees: Any = 0.0
    rotation_z_degrees: Any = 0.0
    pivot: Any | None = None


@dataclass(frozen=True)
class WingParameters(ComponentParameters):
    """Planform and rigid-body parameters for a lifting surface.

    ``area`` and ``aspect_ratio`` are absolute target values.  When both are
    ``None`` the planform scaling is skipped entirely and only the rigid-body
    transform applies.  Supplying just one leaves the other at its resolved
    reference value, so that quantity is held rather than left free.  Supplying
    both scales the span and chord while preserving the requested planform
    values:

    ``span_scale = sqrt(area_ratio * aspect_ratio_ratio)``

    ``chord_scale = sqrt(area_ratio / aspect_ratio_ratio)``

    Thickness is unchanged.  A uniform incidence change is represented by
    ``rotation_y_degrees``.  If ``spanwise_scaling_root`` is supplied, span
    scaling is anchored at that distance from the pivot's spanwise station and
    applied only outboard of it; the target tip span is unchanged.

    Inherits the rigid-body fields of :class:`ComponentParameters`.

    Attributes
    ----------
    area
        Absolute target planform area. ``None`` holds the resolved
        ``reference_area``; if ``aspect_ratio`` is also ``None`` the scaling is
        skipped.
    aspect_ratio
        Absolute target aspect ratio. ``None`` holds the resolved
        ``reference_aspect_ratio``; if ``area`` is also ``None`` the scaling is
        skipped.
    reference_area
        Area of the undeformed planform, forming the denominator of
        ``area_ratio``. ``None`` infers it from the component's control-point
        extents as chord extent times span extent.
    reference_aspect_ratio
        Aspect ratio of the undeformed planform, forming the denominator of
        ``aspect_ratio_ratio``. ``None`` infers it from the same extents as
        span extent squared over the inferred area.
    chord_axis
        Index of the global axis along which chord is measured and scaled.
    span_axis
        Index of the global axis along which span is measured and scaled.
    spanwise_scaling_root
        Distance **from the pivot's spanwise station** at which span scaling is
        anchored; only stations farther outboard than it move, and the tip span
        target is preserved. It is measured about the pivot, which coincides
        with the symmetry plane only when the pivot's spanwise coordinate is
        zero. ``None`` scales the whole span uniformly. Must be non-negative
        and strictly inboard of the tip, or the deformation raises
        :exc:`ValueError`.
    """

    area: Any | None = None
    aspect_ratio: Any | None = None
    reference_area: float | None = None
    reference_aspect_ratio: float | None = None
    chord_axis: int = 0
    span_axis: int = 1
    spanwise_scaling_root: float | None = None


@dataclass(frozen=True)
class TailParameters(ComponentParameters):
    """Rigid-body parameters for a tail surface.

    Carries no fields of its own; it inherits the translation, rotation, and
    ``pivot`` fields of :class:`ComponentParameters` unchanged.

    The separate type expresses intent, letting a tail be declared distinctly
    from a wing, but it currently has **no distinct dispatch**: the deformation
    path special-cases only :class:`WingParameters` and
    :class:`FuselageParameters`, so a tail is transformed by the generic
    :class:`ComponentParameters` route and behaves identically to one.
    """


@dataclass(frozen=True)
class FuselageParameters(ComponentParameters):
    """Cross-section (diameter) scaling plus rigid-body parameters for a body.

    ``diameter_scale`` multiplies the two cross-section axes about ``pivot``,
    leaving the longitudinal axis unchanged; ``1.0`` is the reference body.
    Because the scaling is applied about the pivot, a pivot on the symmetry
    plane keeps ``y = 0`` mapped to ``y = 0``, so a mirror-symmetric mesh stays
    symmetric (and the symmetry-plane condition in the elastic solve stays
    exact).

    Inherits the rigid-body fields of :class:`ComponentParameters`.

    Attributes
    ----------
    diameter_scale
        Multiplier applied to the two cross-section axes about ``pivot``;
        ``1.0`` reproduces the reference body and ``None`` disables the
        scaling.
    diameter_scale_axes
        The two global axis indices treated as the cross-section, leaving the
        remaining longitudinal axis unscaled.
    """

    diameter_scale: Any | None = None
    diameter_scale_axes: tuple[int, int] = (1, 2)


__all__ = [
    "ComponentParameters",
    "FuselageParameters",
    "TailParameters",
    "WingParameters",
]
