"""Neutral binding container for externally parameterized geometry.

:class:`GeometryModel` declares *what moves* and binds it to GAMMA's downstream
mesh behavior. It does **not** have to own the parameterization. The public
contract is::

    any differentiable CSDL/LFS-compatible parameterization
        -> deformed component coefficients
        -> GeometryModel.add_component(...)
        -> GAMMA intersections, graph motion, reprojection, diagnostics

Another package may own every design variable and every deformation operation.
It need not subclass GAMMA or call
:meth:`GeometryModel.design_variable`. What it hands over is the deformed
coefficients themselves. A component may additionally supply an optional
setup-time projection metadata builder.

The generic path
----------------

:meth:`GeometryModel.add_component` accepts a component's deformed
coefficients as either

* one stacked ``(N, 3)`` CSDL variable or array in sorted patch-ID order, the
  existing GAMMA stacking convention; or
* a mapping from patch ID to coefficient block, matching the imported LFS
  component's patch shapes.

GAMMA imports the baseline STEP body, locates the component by ``search_name``,
and validates patch IDs and block shapes *after* that import, when the
canonical values are known. The external target must keep a compatible
component topology and coefficient layout; changing the patch layout or
topology itself is outside this contract.

If derivatives are required, the external expression must belong to the same
caller-owned active recorder that is later passed to
:func:`gamma_mdo.mesh_motion.run`. GAMMA never creates or owns recorder state.

``free_region=None`` leaves the entire component free. The optional plain axis
mapping restricts it.

Optional conveniences
---------------------

:meth:`add_lifting_surface`, :meth:`add_body`, and :meth:`design_variable` are
conveniences for callers that would rather describe a rigid/planform motion or
a cross-section scaling than build coefficients themselves. They are neither
required nor the universal boundary, and they embed no aircraft- or
configuration-specific assumption. Motion targets they build are interpolated
over the load fraction, so each load step is incremental without the caller
writing that out.
"""

from __future__ import annotations

from typing import Any, Mapping

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.component_parameters import FuselageParameters, WingParameters

from .free_region import AxisRange, ComponentFreeRegion
from .geometry import (
    component_patch_ids,
    deform_geometry,
    stack_component_coefficients,
    stack_component_coefficients_numpy,
)
from .mesh_motion_config import _ComponentRecord, _IntersectionRecord

__all__ = ["GeometryModel"]


def _blend(fraction: float, target: Any, reference: float) -> Any:
    """Interpolate a motion target from its reference value.

    Parameters
    ----------
    fraction
        Load fraction in ``(0, 1]``.
    target
        Final value, either a CSDL variable or a float.
    reference
        Value the component holds at zero load.

    Returns
    -------
    Any
        ``reference + fraction * (target - reference)``.
    """
    return reference + fraction * (target - reference)


def _chord_pivot(vertices: np.ndarray, chord_fraction: float) -> np.ndarray:
    """Locate a chordwise pivot on an intersection curve.

    Parameters
    ----------
    vertices
        Intersection-curve vertices with shape ``(num_vertices, 3)``.
    chord_fraction
        Fraction of the leading-to-trailing chord at which to place the pivot.

    Returns
    -------
    numpy.ndarray
        Pivot with shape ``(1, 3)``, pinned to the symmetry plane.
    """
    leading = vertices[int(np.argmin(vertices[:, 0]))]
    trailing = vertices[int(np.argmax(vertices[:, 0]))]
    pivot = leading + chord_fraction * (trailing - leading)
    pivot[1] = 0.0
    return pivot.reshape((1, 3))


def _bounding_box_pivot(component) -> np.ndarray:
    """Locate a body pivot at its control-point bounding-box center.

    Parameters
    ----------
    component
        Imported geometry component exposing a ``functions`` mapping.

    Returns
    -------
    numpy.ndarray
        Pivot with shape ``(1, 3)``, pinned to the symmetry plane.
    """
    control_points = np.vstack(
        [
            np.asarray(
                component.functions[key].coefficients.value, dtype=float
            ).reshape((-1, 3))
            for key in sorted(component.functions)
        ]
    )
    pivot = 0.5 * (control_points.min(axis=0) + control_points.max(axis=0))
    pivot[1] = 0.0
    return pivot.reshape((1, 3))


def _free_region(component, region: Mapping[str, Any] | None):
    """Build a component free region from a high-level axis mapping.

    Parameters
    ----------
    component
        Imported geometry component.
    region
        ``None`` leaves the whole component free. Otherwise a mapping whose
        keys are ``"x"``, ``"y"``, or ``"z"`` and whose values are
        ``(lower, upper, mode)`` tuples. Bounds may be ``None``.

    Returns
    -------
    ComponentFreeRegion
        Region restricted to the requested axis ranges.

    Raises
    ------
    ValueError
        If an axis key or a range tuple is malformed.
    """
    if not region:
        return ComponentFreeRegion(component=component)
    axes: dict[str, AxisRange] = {}
    for axis, bounds in dict(region).items():
        if axis not in ("x", "y", "z"):
            raise ValueError(
                f"Free-region axis must be x, y, or z; got {axis!r}."
            )
        try:
            lower, upper, mode = bounds
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Free-region entry {axis!r} must be a "
                "(lower, upper, mode) tuple."
            ) from error
        axes[axis] = AxisRange(lower=lower, upper=upper, mode=mode)
    return ComponentFreeRegion(component=component, **axes)


def _resolve_external_coefficients(component, value, name: str):
    """Validate external coefficients against the imported component patches.

    Parameters
    ----------
    component
        Imported geometry component, whose patch IDs and block shapes are
        canonical.
    value
        Either one stacked ``(N, 3)`` CSDL variable or array in sorted
        patch-ID order, or a mapping from patch ID to coefficient block.
    name
        Component name, used in error messages.

    Returns
    -------
    Any
        A stacked ``(N, 3)`` value in the pipeline's convention. CSDL
        expressions are preserved, never converted to NumPy.

    Raises
    ------
    ValueError
        If patch IDs are missing or unexpected, a block shape disagrees with
        the imported patch, the stacked row count is wrong, or the trailing
        dimension is not 3.
    """
    patch_ids = list(component_patch_ids(component))
    expected_blocks = {
        patch_id: np.asarray(
            component.functions[patch_id].coefficients.value, dtype=float
        ).reshape((-1, 3)).shape[0]
        for patch_id in patch_ids
    }
    total_rows = sum(expected_blocks.values())

    if isinstance(value, Mapping):
        supplied = set(value)
        expected = set(patch_ids)
        missing = expected - supplied
        extra = supplied - expected
        if missing or extra:
            raise ValueError(
                f"Component {name!r} external coefficients have wrong patch "
                f"IDs; missing={sorted(missing)} unexpected={sorted(extra)}."
            )
        blocks = []
        for patch_id in patch_ids:
            block = value[patch_id]
            shape = tuple(getattr(block, "shape", ()))
            if len(shape) < 2 or shape[-1] != 3:
                raise ValueError(
                    f"Component {name!r} patch {patch_id} coefficients must "
                    f"have a trailing dimension of 3; got shape {shape}."
                )
            rows = int(np.prod(shape[:-1]))
            if rows != expected_blocks[patch_id]:
                raise ValueError(
                    f"Component {name!r} patch {patch_id} expects "
                    f"{expected_blocks[patch_id]} coefficient rows; got "
                    f"{rows}."
                )
            blocks.append(
                csdl.reshape(block, (rows, 3))
                if isinstance(block, csdl.Variable)
                else np.asarray(block, dtype=float).reshape((rows, 3))
            )
        if any(isinstance(block, csdl.Variable) for block in blocks):
            return (
                blocks[0]
                if len(blocks) == 1
                else csdl.concatenate(tuple(blocks), axis=0)
            )
        return np.vstack(blocks)

    shape = tuple(getattr(value, "shape", ()))
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(
            f"Component {name!r} stacked coefficients must have shape "
            f"(N, 3); got {shape}."
        )
    if shape[0] != total_rows:
        raise ValueError(
            f"Component {name!r} stacked coefficients must have "
            f"{total_rows} rows in sorted patch-ID order; got {shape[0]}."
        )
    return value


class GeometryModel:
    """Bind component coefficients and mesh behavior for one configuration.

    This is a declaration and binding envelope, not a parameterization. Add
    each component, say how its coefficients are produced, and name the
    independent closed intersection curves between components. The compiled
    records are private; callers use only the methods below.

    Examples
    --------
    The generic path. ``external_coefficients`` is produced by another
    package and may depend entirely on that package's own CSDL variables;
    GAMMA supplies no transformation here.

    >>> geometry = GeometryModel()  # doctest: +SKIP
    >>> geometry.add_component(  # doctest: +SKIP
    ...     name="wing",
    ...     search_name="wing",
    ...     deformed_coefficients=external_coefficients,
    ... )

    The optional convenience path, for callers who would rather describe a
    motion than build coefficients:

    >>> geometry = GeometryModel()  # doctest: +SKIP
    >>> sweep = geometry.design_variable("sweep", 0.0, lower=-5.0, upper=5.0)
    >>> geometry.add_lifting_surface(  # doctest: +SKIP
    ...     name="wing",
    ...     search_name="wing",
    ...     pivot_intersection="wing_root",
    ...     rotation_y_degrees=sweep,
    ... )
    """

    def __init__(self) -> None:
        """Create an empty geometry model.

        Constructing a model never touches global CSDL state. The caller owns
        the recorder and must have started it before registering any design
        variable through :meth:`design_variable`.
        """
        self._design_variables: dict[str, csdl.Variable] = {}
        self._components: list[_ComponentRecord] = []
        self._intersections: list[_IntersectionRecord] = []
        self._names: set[str] = set()
        self._pivot_refs: dict[str, str] = {}

    @property
    def design_variables(self) -> Mapping[str, csdl.Variable]:
        """Mapping of registered design-variable names to CSDL variables."""
        return dict(self._design_variables)

    @property
    def _component_records(self) -> list[_ComponentRecord]:
        """Private compiled component records; pipeline boundary only."""
        return list(self._components)

    @property
    def _intersection_records(self) -> list[_IntersectionRecord]:
        """Private compiled intersection records; pipeline boundary only."""
        return list(self._intersections)

    def design_variable(
        self,
        name: str,
        value: float,
        *,
        lower: float | None = None,
        upper: float | None = None,
        scaler: float | None = None,
    ) -> csdl.Variable:
        """Register one differentiable geometry control.

        Parameters
        ----------
        name
            Unique identifier; must be a valid Python identifier.
        value
            Initial value.
        lower, upper
            Optional optimization bounds.
        scaler
            Optional optimizer scaling factor.

        Returns
        -------
        csdl_alpha.Variable
            The registered variable, for use in component motion arguments.

        Raises
        ------
        ValueError
            If the name is invalid or already registered.
        RuntimeError
            If no CSDL recorder is active.
        """
        if not name or not name.isidentifier():
            raise ValueError("A design-variable name must be an identifier.")
        if name in self._design_variables:
            raise ValueError(f"Design variable {name!r} is already defined.")
        try:
            csdl.get_current_recorder()
        except Exception as error:
            raise RuntimeError(
                "A CSDL recorder must be active before registering a design "
                "variable. Start one with csdl.Recorder(inline=True).start() "
                "and stop it yourself; GeometryModel never owns recorder "
                "state."
            ) from error
        variable = csdl.Variable(name=name, value=float(value))
        # Always register: the method is named design_variable, so an
        # unbounded control must still reach the optimizer.
        variable.set_as_design_variable(
            lower=lower, upper=upper, scaler=scaler
        )
        self._design_variables[name] = variable
        return variable

    def add_component(
        self,
        *,
        name: str,
        search_name: str,
        deformed_coefficients: Any,
        free_region: Mapping[str, Any] | None = None,
        projection_name: str | None = None,
        projection_mode: str = "all",
        projection_metadata_builder=None,
    ) -> None:
        """Add a component driven by externally produced coefficients.

        This is the general entry point and the universal boundary. The caller
        supplies the deformed coefficients from any differentiable
        parameterization, owned entirely outside GAMMA if desired.
        :meth:`add_lifting_surface` and :meth:`add_body` are separate optional
        conveniences implemented alongside this method: each builds its own
        private record rather than calling through here.

        Parameters
        ----------
        name
            Unique component identifier.
        search_name
            Name passed to the STEP importer's component search.
        deformed_coefficients
            Target coefficients at full load, either one stacked ``(N, 3)``
            CSDL variable or array in sorted patch-ID order, or a mapping from
            patch ID to coefficient block. CSDL expressions are preserved, so
            derivatives through the final reprojected mesh stay analytic.
        free_region
            ``None`` leaves the whole component free. Otherwise a mapping of
            ``"x"``/``"y"``/``"z"`` to ``(lower, upper, mode)``.
        projection_name
            Diagnostic projection name; defaults to ``name``.
        projection_mode
            ``"all"`` or ``"lifting_surface"``.
        projection_metadata_builder
            Optional setup-time callback for a component-specific projection
            strategy. It receives the imported component, candidate vertex IDs,
            baseline parametric coordinates and baseline mesh vertices, and
            returns projection metadata groups. The returned groups are checked
            by the projection routine against the actual reprojected IDs.

        Raises
        ------
        ValueError
            If the name collides. Coefficient shapes and patch IDs are
            validated later, once the STEP component is imported and the
            canonical patch IDs are known.
        """
        self._check_name(name)

        def build(component, fraction, intersections):
            del intersections
            target = _resolve_external_coefficients(
                component, deformed_coefficients, name
            )
            baseline = stack_component_coefficients_numpy(component)
            if fraction == 1.0:
                # Hand the exact external target to the solve chain.
                return target
            return baseline + fraction * (target - baseline)

        self._components.append(
            _ComponentRecord(
                name=name,
                search_name=search_name,
                coefficient_builder=build,
                free_region_factory=(
                    lambda component: _free_region(component, free_region)
                ),
                projection_name=projection_name,
                projection_mode=projection_mode,
                projection_metadata_builder=projection_metadata_builder,
            )
        )

    def add_lifting_surface(
        self,
        *,
        name: str,
        search_name: str,
        pivot_intersection: str,
        translation_x: Any = 0.0,
        rotation_y_degrees: Any = 0.0,
        area: Any = None,
        aspect_ratio: Any = None,
        reference_area: float | None = None,
        reference_aspect_ratio: float | None = None,
        free_span_fraction: float = 0.3,
        projection_name: str | None = None,
        pivot_chord_fraction: float = 0.25,
    ) -> None:
        """Add a lifting surface driven by a named intersection curve.

        Parameters
        ----------
        name
            Unique component identifier.
        search_name
            Name passed to the geometry importer's component search.
        pivot_intersection
            Name of the intersection curve supplying the pivot and the span
            scaling root.
        translation_x
            Chordwise rigid translation target.
        rotation_y_degrees
            Incidence target in degrees.
        area, aspect_ratio
            Absolute planform targets. Supplying either requires both
            reference values.
        reference_area, reference_aspect_ratio
            Planform values the baseline geometry already has.
        free_span_fraction
            Fraction of the component semispan, in ``(0, 1]``, included in the
            graph-free root region. Vertices farther outboard are updated by
            fixed-parametric reevaluation.
        projection_name
            Diagnostic projection name; defaults to ``name``.
        pivot_chord_fraction
            Fraction of the intersection chord at which the pivot sits.

        Raises
        ------
        ValueError
            If names collide, fractions are out of range, or a planform target
            is supplied without its reference value.
        """
        self._check_name(name)
        if not 0.0 <= pivot_chord_fraction <= 1.0:
            raise ValueError("pivot_chord_fraction must lie in [0, 1].")
        if not 0.0 < free_span_fraction <= 1.0:
            raise ValueError("free_span_fraction must lie in (0, 1].")
        if area is not None and reference_area is None:
            raise ValueError("Supplying area also requires reference_area.")
        if aspect_ratio is not None and reference_aspect_ratio is None:
            raise ValueError(
                "Supplying aspect_ratio also requires reference_aspect_ratio."
            )

        def build(component, fraction, intersections):
            vertices = intersections[pivot_intersection]
            return deform_geometry(
                component=component,
                parameters=WingParameters(
                    translation_x=fraction * translation_x,
                    rotation_y_degrees=fraction * rotation_y_degrees,
                    area=(
                        None
                        if area is None
                        else _blend(fraction, area, reference_area)
                    ),
                    aspect_ratio=(
                        None
                        if aspect_ratio is None
                        else _blend(
                            fraction, aspect_ratio, reference_aspect_ratio
                        )
                    ),
                    reference_area=reference_area,
                    reference_aspect_ratio=reference_aspect_ratio,
                    pivot=_chord_pivot(vertices, pivot_chord_fraction),
                    spanwise_scaling_root=float(
                        np.median(np.abs(vertices[:, 1]))
                    ),
                ),
            )

        self._pivot_refs[name] = pivot_intersection
        self._components.append(
            _ComponentRecord(
                name=name,
                search_name=search_name,
                coefficient_builder=build,
                free_region_factory=(
                    lambda component: _free_region(
                        component, {"y": (None, free_span_fraction, "abs")}
                    )
                ),
                projection_name=projection_name,
                projection_mode="lifting_surface",
            )
        )

    def add_body(
        self,
        *,
        name: str,
        search_name: str,
        diameter_scale: Any = 1.0,
        free_axial_fraction: tuple[float, float] = (0.05, 0.97),
        projection_name: str | None = None,
    ) -> None:
        """Add a body whose cross-section scales about its bounding-box center.

        Parameters
        ----------
        name
            Unique component identifier.
        search_name
            Name passed to the geometry importer's component search.
        diameter_scale
            Cross-section scale target; ``1.0`` leaves the body unchanged.
        free_axial_fraction
            Lower and upper longitudinal extent fractions of the graph-free
            middle region. Nose and tail vertices outside that interval are
            updated by fixed-parametric reevaluation.
        projection_name
            Diagnostic projection name; defaults to ``name``.

        Raises
        ------
        ValueError
            If the name collides or the axial fractions are not an increasing
            pair inside ``[0, 1]``.
        """
        self._check_name(name)
        lower, upper = (float(v) for v in free_axial_fraction)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "free_axial_fraction must be increasing and within [0, 1]."
            )

        def build(component, fraction, intersections):
            del intersections
            return deform_geometry(
                component=component,
                parameters=FuselageParameters(
                    diameter_scale=_blend(fraction, diameter_scale, 1.0),
                    pivot=_bounding_box_pivot(component),
                ),
            )

        self._components.append(
            _ComponentRecord(
                name=name,
                search_name=search_name,
                coefficient_builder=build,
                free_region_factory=(
                    lambda component: _free_region(
                        component, {"x": (lower, upper, "extent")}
                    )
                ),
                projection_name=projection_name,
                projection_mode="all",
            )
        )

    def connect(
        self,
        *,
        name: str,
        driving_component: str,
        query_component: str,
        search_direction: str = "u",
        solver_name: str | None = None,
    ) -> None:
        """Declare one independent closed intersection curve.

        Parameters
        ----------
        name
            Unique intersection identifier, referenced by
            ``add_lifting_surface(pivot_intersection=...)``.
        driving_component
            Component whose parametric line drives the bracketed solve.
        query_component
            Component supplying the signed-distance residual.
        search_direction
            Parametric coordinate varied by the solve.
        solver_name
            Optional diagnostic name; defaults to ``name``.

        Raises
        ------
        ValueError
            If the name collides or either component is unknown.
        """
        if name in {record.name for record in self._intersections}:
            raise ValueError(f"Intersection {name!r} is already defined.")
        known = {record.name for record in self._components}
        missing = {driving_component, query_component}.difference(known)
        if missing:
            raise ValueError(
                f"Intersection {name!r} references unknown components: "
                + ", ".join(sorted(missing))
                + "."
            )
        self._intersections.append(
            _IntersectionRecord(
                name=name,
                driving_component=driving_component,
                query_component=query_component,
                bisection_search_direction=search_direction,
                solver_name=solver_name,
            )
        )

    def validate(self) -> None:
        """Check that the model is complete enough to run.

        Raises
        ------
        ValueError
            If no component has been registered, or if a lifting surface
            references an undefined intersection. Design variables may be
            created and owned entirely by another package, so none is
            required here.
        """
        if not self._components:
            raise ValueError("At least one component is required.")
        defined = {record.name for record in self._intersections}
        for component, intersection in self._pivot_refs.items():
            if intersection not in defined:
                raise ValueError(
                    f"Lifting surface {component!r} pivots on undefined "
                    f"intersection {intersection!r}."
                )

    def _check_name(self, name: str) -> None:
        """Reject empty, non-identifier, or duplicate component names."""
        if not name or not name.isidentifier():
            raise ValueError("A component name must be an identifier.")
        if name in self._names:
            raise ValueError(f"Component {name!r} is already defined.")
        self._names.add(name)
