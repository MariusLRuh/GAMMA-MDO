"""Differentiable component-coefficient transformations."""

from __future__ import annotations

from typing import Any

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.component_parameters import (
    ComponentParameters,
    FuselageParameters,
    WingParameters,
)


def component_patch_ids(component) -> tuple[int, ...]:
    """Return component patch IDs in canonical coefficient-stack order.

    Parameters
    ----------
    component
        Function set exposing a nonempty ``functions`` mapping.

    Returns
    -------
    tuple[int, ...]
        Sorted integer patch identifiers.

    Raises
    ------
    TypeError
        If ``component`` does not expose any functions.
    """
    functions = getattr(component, "functions", None)
    if functions is None or not functions:
        raise TypeError("component must expose a non-empty .functions mapping.")
    return tuple(sorted(int(patch_id) for patch_id in functions))


def classify_patch_sides(
    component,
    *,
    axis: int = 2,
    span_axis: int = 1,
    cap_span_fraction: float = 0.05,
) -> dict[int, int]:
    """Split a lifting surface's patches into upper (+1) and lower (-1) skins.

    The two skins carry opposite surface normals along ``axis`` (``z`` by
    default), so the sign of the control-net normal separates them.

    End caps (e.g. wing tip caps) are returned as ``0``: a patch whose extent
    along ``span_axis`` is under ``cap_span_fraction`` of the component's own
    extent is a closing cap, not a skin, and is excluded so it never becomes a
    projection target.  Degenerate patches are also ``0``.

    This supports restricting projection to the *same skin*: a free node may
    then slide spanwise across a patch boundary (which a parent-patch
    restriction clamps) but can never jump to the opposite skin (which an
    unrestricted any-patch projection allows on a thin surface).

    Parameters
    ----------
    component
        Lifting-surface function set.
    axis
        Coordinate axis used to classify the normal sign.
    span_axis
        Coordinate axis used to recognize end caps.
    cap_span_fraction
        Maximum relative span extent classified as a cap.

    Returns
    -------
    dict[int, int]
        Patch IDs mapped to ``+1`` (upper), ``-1`` (lower), or ``0`` (cap or
        degenerate patch).
    """
    axis = _validate_axis(axis, "axis")
    span_axis = _validate_axis(span_axis, "span_axis")

    control_points = [
        np.asarray(
            component.functions[patch_id].coefficients.value, dtype=float
        ).reshape((-1, 3))
        for patch_id in component_patch_ids(component)
    ]
    all_points = np.vstack(control_points)
    component_span = float(np.ptp(all_points[:, span_axis]))

    sides: dict[int, int] = {}
    for patch_id, flat in zip(component_patch_ids(component), control_points):
        coefficients = np.asarray(
            component.functions[patch_id].coefficients.value, dtype=float
        )
        if coefficients.ndim < 3 or min(coefficients.shape[:2]) < 2:
            sides[patch_id] = 0
            continue
        patch_span = float(np.ptp(flat[:, span_axis]))
        if component_span > 0.0 and patch_span < cap_span_fraction * component_span:
            sides[patch_id] = 0  # end cap, not a skin
            continue
        grid = coefficients.reshape(
            coefficients.shape[0], coefficients.shape[1], coefficients.shape[-1]
        )
        mid_u = grid.shape[0] // 2
        mid_v = grid.shape[1] // 2
        normal = np.cross(
            grid[-1, mid_v] - grid[0, mid_v],
            grid[mid_u, -1] - grid[mid_u, 0],
        )
        magnitude = float(np.linalg.norm(normal))
        sides[patch_id] = (
            int(np.sign(normal[axis])) if magnitude > 1e-12 else 0
        )
    return sides


def stack_component_coefficients(component) -> csdl.Variable:
    """Stack component coefficients in sorted patch-ID order.

    Parameters
    ----------
    component
        Function set containing CSDL coefficient arrays.

    Returns
    -------
    csdl.Variable
        Two-dimensional coefficient array with patches concatenated by row.
    """
    blocks = []
    for patch_id in component_patch_ids(component):
        coefficients = component.functions[patch_id].coefficients
        blocks.append(csdl.reshape(coefficients, (-1, coefficients.shape[-1])))
    if len(blocks) == 1:
        return blocks[0]
    return csdl.vstack(tuple(blocks))


def stack_component_coefficients_numpy(component) -> np.ndarray:
    """Stack numeric component coefficients in sorted patch-ID order.

    Parameters
    ----------
    component
        Function set containing coefficient arrays with numeric values.

    Returns
    -------
    numpy.ndarray
        Two-dimensional numeric coefficient array.
    """
    blocks = []
    for patch_id in component_patch_ids(component):
        value = np.asarray(component.functions[patch_id].coefficients.value, dtype=float)
        blocks.append(value.reshape((-1, value.shape[-1])))
    return np.vstack(blocks)


def deform_geometry(
    *,
    component,
    parameters: ComponentParameters,
) -> csdl.Variable:
    """Return differentiably transformed component coefficients.

    The function is intentionally side-effect free: the input FunctionSet
    remains the immutable geometric template used to construct projection and
    evaluation models, while the returned coefficient variable represents its
    current deformed state.

    Parameters
    ----------
    component
        Immutable component geometry template.
    parameters
        Translation, rotation, and component-specific scaling controls.

    Returns
    -------
    csdl.Variable
        Stacked differentiably transformed coefficients.

    Raises
    ------
    TypeError
        If ``parameters`` is not a supported component-parameter object.
    ValueError
        If the component is not three-dimensional or a scaling control is
        invalid.
    """
    if not isinstance(parameters, ComponentParameters):
        raise TypeError(
            "parameters must be ComponentParameters, WingParameters, or a "
            "subclass thereof."
        )

    coefficients = stack_component_coefficients(component)
    baseline = stack_component_coefficients_numpy(component)
    if baseline.shape[1] != 3:
        raise ValueError(
            "Boundary-surface movement currently requires 3D component coefficients."
        )

    pivot = _resolve_pivot(parameters.pivot, baseline, parameters)
    pivot_rows = csdl.matmat(np.ones((coefficients.shape[0], 1)), pivot)
    points = coefficients - pivot_rows

    if isinstance(parameters, WingParameters):
        points = _apply_planform_scaling(points, baseline, parameters)
    elif isinstance(parameters, FuselageParameters):
        points = _apply_diameter_scaling(points, parameters)

    points = _rotate_x(points, parameters.rotation_x_degrees)
    points = _rotate_y(points, parameters.rotation_y_degrees)
    points = _rotate_z(points, parameters.rotation_z_degrees)
    translation_rows = csdl.matmat(
        np.ones((coefficients.shape[0], 1)),
        _translation_row(parameters),
    )
    return points + pivot_rows + translation_rows


def _resolve_pivot(
    requested_pivot: Any | None,
    baseline: np.ndarray,
    parameters: ComponentParameters,
):
    if requested_pivot is not None:
        pivot = requested_pivot
        if hasattr(pivot, "shape"):
            return csdl.reshape(pivot, (1, 3))
        return np.asarray(pivot, dtype=float).reshape((1, 3))

    lower = np.min(baseline, axis=0)
    upper = np.max(baseline, axis=0)
    pivot = 0.5 * (lower + upper)
    if isinstance(parameters, WingParameters):
        chord_axis = _validate_axis(parameters.chord_axis, "chord_axis")
        span_axis = _validate_axis(parameters.span_axis, "span_axis")
        if chord_axis == span_axis:
            raise ValueError("chord_axis and span_axis must be different.")
        pivot[chord_axis] = lower[chord_axis] + 0.25 * (
            upper[chord_axis] - lower[chord_axis]
        )
        # The root is the point on the span axis nearest zero for conventional
        # full or half aircraft geometry.
        pivot[span_axis] = np.clip(0.0, lower[span_axis], upper[span_axis])
    return pivot.reshape((1, 3))


def _apply_planform_scaling(
    centered_points: csdl.Variable,
    baseline: np.ndarray,
    parameters: WingParameters,
) -> csdl.Variable:
    if parameters.area is None and parameters.aspect_ratio is None:
        return centered_points

    reference_area, reference_aspect_ratio = _resolve_reference_planform(
        baseline,
        parameters,
    )
    area = reference_area if parameters.area is None else parameters.area
    aspect_ratio = (
        reference_aspect_ratio
        if parameters.aspect_ratio is None
        else parameters.aspect_ratio
    )
    area_ratio = area / reference_area
    aspect_ratio_ratio = aspect_ratio / reference_aspect_ratio
    chord_scale = csdl.sqrt(area_ratio / aspect_ratio_ratio)
    span_scale = csdl.sqrt(area_ratio * aspect_ratio_ratio)

    scales: list[Any] = [1.0, 1.0, 1.0]
    chord_axis = _validate_axis(parameters.chord_axis, "chord_axis")
    span_axis = _validate_axis(parameters.span_axis, "span_axis")
    scales[chord_axis] = chord_scale
    scales[span_axis] = span_scale
    columns = [
        _scale_planform_column(
            centered_points,
            baseline,
            parameters,
            axis=axis,
            scale=scales[axis],
            span_axis=span_axis,
        )
        for axis in range(3)
    ]
    return csdl.concatenate(tuple(columns), axis=1)


def _apply_diameter_scaling(centered_points, parameters: FuselageParameters):
    """Scale the cross-section axes of a body about the pivot."""
    if parameters.diameter_scale is None:
        return centered_points
    axes = {
        _validate_axis(axis, "diameter_scale_axes")
        for axis in parameters.diameter_scale_axes
    }
    if len(axes) != 2:
        raise ValueError("diameter_scale_axes must name two distinct axes.")
    columns = []
    for axis in range(3):
        column = centered_points[csdl.slice[:, axis : axis + 1]]
        if axis in axes:
            column = column * _expand_scalar(
                parameters.diameter_scale, centered_points.shape[0]
            )
        columns.append(column)
    return csdl.concatenate(tuple(columns), axis=1)


def _scale_planform_column(
    centered_points,
    baseline,
    parameters,
    *,
    axis,
    scale,
    span_axis,
):
    column = centered_points[csdl.slice[:, axis : axis + 1]]
    if axis != span_axis or parameters.spanwise_scaling_root is None:
        return column * _expand_scalar(scale, centered_points.shape[0])

    root = float(parameters.spanwise_scaling_root)
    if root < 0.0:
        raise ValueError("spanwise_scaling_root must be non-negative.")
    pivot = (
        np.zeros(3)
        if parameters.pivot is None
        else np.asarray(
            getattr(parameters.pivot, "value", parameters.pivot),
            dtype=float,
        ).reshape(3)
    )
    baseline_span = baseline[:, span_axis] - pivot[span_axis]
    absolute_span = np.abs(baseline_span)
    tip = float(np.max(absolute_span))
    if root >= tip - 1e-12:
        raise ValueError("spanwise_scaling_root must lie inboard of the wing tip.")

    outboard_scale = (scale * tip - root) / (tip - root)
    outboard_mask = absolute_span > root
    sign = np.sign(baseline_span).reshape((-1, 1))
    root_rows = root * sign
    outboard_offset = column - root_rows
    scaled_outboard = root_rows + outboard_offset * _expand_scalar(
        outboard_scale,
        centered_points.shape[0],
    )
    return np.where(outboard_mask.reshape((-1, 1)), 1.0, 0.0) * scaled_outboard + np.where(
        outboard_mask.reshape((-1, 1)),
        0.0,
        1.0,
    ) * column


def _resolve_reference_planform(
    baseline: np.ndarray,
    parameters: WingParameters,
) -> tuple[float, float]:
    chord_extent = float(np.ptp(baseline[:, parameters.chord_axis]))
    span_extent = float(np.ptp(baseline[:, parameters.span_axis]))
    if chord_extent <= 0.0 or span_extent <= 0.0:
        raise ValueError(
            "Cannot infer planform references from degenerate control-point extents."
        )

    inferred_area = chord_extent * span_extent
    inferred_aspect_ratio = span_extent**2 / inferred_area
    reference_area = (
        inferred_area
        if parameters.reference_area is None
        else float(parameters.reference_area)
    )
    reference_aspect_ratio = (
        inferred_aspect_ratio
        if parameters.reference_aspect_ratio is None
        else float(parameters.reference_aspect_ratio)
    )
    if reference_area <= 0.0 or reference_aspect_ratio <= 0.0:
        raise ValueError(
            "reference_area and reference_aspect_ratio must be positive."
        )
    return reference_area, reference_aspect_ratio


def _translation_row(parameters: ComponentParameters):
    return csdl.concatenate(
        (
            _as_scalar_matrix(parameters.translation_x),
            _as_scalar_matrix(parameters.translation_y),
            _as_scalar_matrix(parameters.translation_z),
        ),
        axis=1,
    )


def _rotate_x(points, angle_degrees):
    angle = angle_degrees * (np.pi / 180.0)
    cosine = _expand_scalar(csdl.cos(angle), points.shape[0])
    sine = _expand_scalar(csdl.sin(angle), points.shape[0])
    x, y, z = _coordinate_columns(points)
    return csdl.concatenate((x, cosine * y - sine * z, sine * y + cosine * z), axis=1)


def _rotate_y(points, angle_degrees):
    angle = angle_degrees * (np.pi / 180.0)
    cosine = _expand_scalar(csdl.cos(angle), points.shape[0])
    sine = _expand_scalar(csdl.sin(angle), points.shape[0])
    x, y, z = _coordinate_columns(points)
    return csdl.concatenate((cosine * x + sine * z, y, -sine * x + cosine * z), axis=1)


def _rotate_z(points, angle_degrees):
    angle = angle_degrees * (np.pi / 180.0)
    cosine = _expand_scalar(csdl.cos(angle), points.shape[0])
    sine = _expand_scalar(csdl.sin(angle), points.shape[0])
    x, y, z = _coordinate_columns(points)
    return csdl.concatenate((cosine * x - sine * y, sine * x + cosine * y, z), axis=1)


def _coordinate_columns(points):
    return tuple(points[csdl.slice[:, axis : axis + 1]] for axis in range(3))


def _expand_scalar(value, rows: int):
    if np.isscalar(value):
        return np.full((rows, 1), float(value))
    return csdl.matmat(
        np.ones((rows, 1), dtype=float),
        csdl.reshape(value, (1, 1)),
    )


def _as_scalar_matrix(value):
    if np.isscalar(value):
        return np.asarray([[float(value)]])
    return csdl.reshape(value, (1, 1))


def _validate_axis(axis: int, name: str) -> int:
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"{name} must be 0, 1, or 2. Got {axis}.")
    return axis


__all__ = [
    "component_patch_ids",
    "deform_geometry",
    "stack_component_coefficients",
    "stack_component_coefficients_numpy",
]
