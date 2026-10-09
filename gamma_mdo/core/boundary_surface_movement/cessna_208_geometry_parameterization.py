"""Private, geometry-only C208 parameterization for the example script.

The four planform targets are solved by lsdo_geo. Attachment positions are
hard material-coordinate constraints: the wing-side strut row and its interior
markers use the same wing FFD map; the fuselage-side row is unchanged.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import csdl_alpha as csdl
import numpy as np
from scipy.interpolate import BSpline, PPoly

from gamma_mdo.core.boundary_surface_movement import stack_component_coefficients_numpy
from gamma_mdo.core.boundary_surface_movement.cessna_208_example import (
    ASSET_HASHES,
    _section_area_centroid,
)
from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
)
from gamma_mdo.preprocessing import create_components
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_ffd_block_around_entities,
)
from lsdo_geo.core.parameterization.parameterization_solver import (
    GeometricVariables,
    ParameterizationSolver,
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs,
)


@dataclass
class C208GeometryGraph:
    """CSDL outputs and immutable reference data for geometry validation."""

    components: tuple
    coefficients: dict[str, csdl.Variable]
    controls: dict[str, csdl.Variable]
    metrics: dict[str, csdl.Variable]
    baseline_metrics: dict[str, float]
    states: dict[str, csdl.Variable]
    anchors: dict[str, csdl.Variable]
    baseline_anchors: dict[str, np.ndarray]
    wing_ffd_anchor_coordinates: np.ndarray
    fuselage_end_centroid: np.ndarray
    wing_end_centroid: csdl.Variable
    strut_length: csdl.Variable
    root_trailing_edge: csdl.Variable
    baseline_root_trailing_edge: np.ndarray
    stab_pre_rotation_root_chord: csdl.Variable


def _leading_edge_parameters(function) -> np.ndarray:
    """Freeze each baseline row's minimum-x spline location, away from seams."""
    coefficients = np.asarray(function.coefficients.value, dtype=float)
    v_knots = function.space.knots[1]
    v_degree = function.space.degree[1]
    values = []
    for row in coefficients:
        x_curve = BSpline(v_knots, row[:, 0], v_degree)
        roots = PPoly.from_spline(x_curve).derivative().roots(extrapolate=False)
        candidates = np.r_[
            0.0, 1.0, roots[np.isfinite(roots) & (roots > 0.0) & (roots < 1.0)]
        ]
        values.append(float(candidates[np.argmin(x_curve(candidates))]))
    return np.asarray(values)


def _planform(function, coefficients, leading_v):
    """Full symmetric three-station trapezoid area and span-squared/area."""
    u = np.asarray(function.space.knots[0][function.space.degree[0] : -function.space.degree[0]])
    # C208's degree-one span spline has row locations 0, 0.5, 1.
    if u.shape != (3,) or not np.allclose(u, [0.0, 0.5, 1.0]):
        raise ValueError("C208 wing/stabilizer span stations changed.")
    leading = function.evaluate(np.column_stack((u, leading_v)), coefficients=coefficients)
    trailing = function.evaluate(np.column_stack((u, np.zeros(3))), coefficients=coefficients)
    chord = [csdl.norm(trailing[i, :] - leading[i, :]) for i in range(3)]
    area = sum(
        (chord[i] + chord[i + 1]) * (leading[i + 1, 1] - leading[i, 1])
        for i in range(2)
    )
    span = 2.0 * (leading[2, 1] - leading[0, 1])
    return area, span * span / area, leading, trailing


def _interior_section_weights(function, count=64):
    """Three separated convex interior markers, linear in section coefficients."""
    knots = function.space.knots[1]
    degree = function.space.degree[1]
    v_center = np.linspace(0.0, 1.0, count, endpoint=False)
    center_weights = np.asarray(BSpline.design_matrix(v_center, knots, degree).mean(axis=0)).ravel()
    v_markers = np.array([0.10, 0.35, 0.70])
    boundary_weights = BSpline.design_matrix(v_markers, knots, degree).toarray()
    return 0.8 * center_weights[None, :] + 0.2 * boundary_weights


def _mirror(flat_coefficients, coefficient_shape):
    """C208 port patch equals reversed-v starboard with negated y."""
    rows, cols = coefficient_shape[:2]
    reverse_v = np.arange(rows * cols).reshape(rows, cols)[:, ::-1].ravel().tolist()
    return flat_coefficients[reverse_v, :] * np.tile([1.0, -1.0, 1.0], (rows * cols, 1))


def _rotate_about_span_axis(points, angle_degrees, pivot):
    """Rotate flattened XYZ rows about +y through the current quarter chord."""
    pivot = csdl.reshape(pivot, (1, 3))
    offset = points - csdl.matmat(np.ones((points.shape[0], 1)), pivot)
    angle = angle_degrees * np.pi / 180.0
    cosine, sine = csdl.cos(angle), csdl.sin(angle)
    x = offset[csdl.slice[:, 0:1]]
    y = offset[csdl.slice[:, 1:2]]
    z = offset[csdl.slice[:, 2:3]]
    rotated = csdl.concatenate((cosine * x + sine * z, y, -sine * x + cosine * z), axis=1)
    return rotated + csdl.matmat(np.ones((points.shape[0], 1)), pivot)


def _ffd_planform(component, patch_id, name, cache_directory, extra_entities=()):
    """Create a two-section host FFD and its two inner-solver states."""
    starboard = create_components(keys=[[patch_id]], geometry=component)
    previous_directory = Path.cwd()
    try:
        os.chdir(cache_directory)
        block = construct_ffd_block_around_entities(
            entities=[starboard, *extra_entities],
            num_coefficients=(2, 2, 2),
            degree=(1, 1, 1),
            name=f"c208_{name}_ffd",
        )
    finally:
        os.chdir(previous_directory)
    sections = VolumeSectionalParameterization(
        parameterized_points=block.coefficients,
        principal_parametric_dimension=1,
        name=f"c208_{name}_sections",
    )
    chord_state = csdl.Variable(value=np.array([0.0]), name=f"{name}_chord_stretch_m")
    span_state = csdl.Variable(value=np.array([0.0]), name=f"{name}_tip_span_shift_m")
    inputs = VolumeSectionalParameterizationInputs()
    inputs.add_sectional_stretch(axis=0, stretch=csdl.concatenate((chord_state, chord_state)))
    inputs.add_sectional_translation(
        axis=1, translation=csdl.concatenate((np.zeros(1), span_state))
    )
    evaluated = block.evaluate_ffd(sections.evaluate(inputs, plot=False), plot=False)
    if not isinstance(evaluated, list):
        evaluated = [evaluated]
    shape = starboard.functions[patch_id].coefficients.shape
    return block, evaluated[0].reshape(shape), evaluated[1:], chord_state, span_state


def parameterize_c208_geometry(*, step_file: Path, cache_directory: Path, controls: dict[str, csdl.Variable]):
    """Build and solve the five-control, geometry-only C208 CSDL graph.

    The caller owns the active CSDL recorder. `cache_directory` receives only
    the STEP import cache. No mesh or intersection operation is constructed.
    """
    step_file = Path(step_file).resolve()
    if hashlib.sha256(step_file.read_bytes()).hexdigest() != ASSET_HASHES["cessna_208.stp"]:
        raise ValueError("C208 STEP asset hash changed.")
    expected_controls = {
        "wing_area_m2", "wing_aspect_ratio", "stab_area_m2",
        "stab_aspect_ratio", "stab_rotation_deg",
    }
    if set(controls) != expected_controls:
        raise ValueError(f"Expected controls {sorted(expected_controls)}")
    cache_directory = Path(cache_directory).resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(cache_directory)
        components = create_components(
            search_names=["Struts", "MainWing", "FuselageGeom", "Stab"],
            step_file=step_file,
        )
    finally:
        os.chdir(previous_directory)
    struts, wing, fuselage, stab = components
    strut_shape = struts.functions[4].coefficients.shape
    baseline_strut = np.asarray(struts.functions[4].coefficients.value, dtype=float)
    weights = _interior_section_weights(struts.functions[4])
    fuselage_anchors = weights @ baseline_strut[0]
    wing_anchors = weights @ baseline_strut[1]
    fuselage_centroid = _section_area_centroid(struts.functions[4], 0)
    wing_centroid = _section_area_centroid(struts.functions[4], 1)

    wing_block, wing_outputs, wing_extra, wing_chord, wing_span_shift = _ffd_planform(
        wing, 0, "wing", cache_directory,
        (baseline_strut[1], wing_anchors, wing_centroid[None, :]),
    )
    wing_end_row, wing_material_anchors, wing_end_centroid = wing_extra
    wing_end_centroid = wing_end_centroid.reshape((3,))
    _, stab_outputs, _, stab_chord, stab_span_shift = _ffd_planform(
        stab, 2, "stab", cache_directory
    )

    wing_v = _leading_edge_parameters(wing.functions[0])
    stab_v = _leading_edge_parameters(stab.functions[2])
    wing_area, wing_ar, _, _ = _planform(wing.functions[0], wing_outputs, wing_v)
    stab_area, stab_ar, stab_le, stab_te = _planform(stab.functions[2], stab_outputs, stab_v)
    baseline_wing_area, baseline_wing_ar, _, _ = _planform(
        wing.functions[0], wing.functions[0].coefficients, wing_v
    )
    baseline_stab_area, baseline_stab_ar, _, baseline_stab_te = _planform(
        stab.functions[2], stab.functions[2].coefficients, stab_v
    )
    baseline_metrics = {
        "wing_area_m2": float(baseline_wing_area.value.item()),
        "wing_aspect_ratio": float(baseline_wing_ar.value.item()),
        "stab_area_m2": float(baseline_stab_area.value.item()),
        "stab_aspect_ratio": float(baseline_stab_ar.value.item()),
    }

    solver = ParameterizationSolver()
    for state in (wing_chord, wing_span_shift, stab_chord, stab_span_shift):
        solver.add_state(state)
    geometric_variables = GeometricVariables()
    metrics = {
        "wing_area_m2": wing_area,
        "wing_aspect_ratio": wing_ar,
        "stab_area_m2": stab_area,
        "stab_aspect_ratio": stab_ar,
    }
    for name, value in metrics.items():
        geometric_variables.add_variable(value, controls[name])
    solver.evaluate(geometric_variables)

    # Rotate about the quarter chord of the resized root section. Translating
    # the rotated surface restores the *baseline* fuselage-frame root TE.
    quarter_chord = stab_le[0, :] + 0.25 * (stab_te[0, :] - stab_le[0, :])
    stab_flat = stab_outputs.reshape((-1, 3))
    rotated = _rotate_about_span_axis(stab_flat, controls["stab_rotation_deg"], quarter_chord)
    baseline_root_te = np.asarray(baseline_stab_te.value[0, :], dtype=float)
    rotated_root_te = _rotate_about_span_axis(
        stab_te[0:1, :], controls["stab_rotation_deg"], quarter_chord
    )
    compensation = baseline_root_te.reshape((1, 3)) - rotated_root_te
    stab_flat = rotated + csdl.matmat(np.ones((rotated.shape[0], 1)), compensation)
    root_trailing_edge = rotated_root_te + compensation

    wing_flat = wing_outputs.reshape((-1, 3))
    wing_port = _mirror(wing_flat, wing.functions[0].coefficients.shape)
    stab_port = _mirror(stab_flat, stab.functions[2].coefficients.shape)
    strut_starboard = csdl.concatenate((baseline_strut[0], wing_end_row), axis=0)
    strut_port = _mirror(strut_starboard, strut_shape)
    coefficients = {
        "strut": csdl.concatenate((strut_starboard, strut_port), axis=0),
        "wing": csdl.concatenate((wing_flat, wing_port), axis=0),
        "fuselage": csdl.Variable(value=stack_component_coefficients_numpy(fuselage)),
        "stab": csdl.concatenate((stab_flat, stab_port), axis=0),
    }
    actual_wing_anchors = csdl.matmat(weights, wing_end_row)
    anchors = {
        "wing_actual": actual_wing_anchors,
        "wing_material_target": wing_material_anchors,
        "fuselage_actual": csdl.matmat(weights, baseline_strut[0]),
    }
    return C208GeometryGraph(
        components=components,
        coefficients=coefficients,
        controls=controls,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        states={
            "wing_chord_stretch_m": wing_chord,
            "wing_tip_span_shift_m": wing_span_shift,
            "stab_chord_stretch_m": stab_chord,
            "stab_tip_span_shift_m": stab_span_shift,
        },
        anchors=anchors,
        baseline_anchors={"wing": wing_anchors, "fuselage": fuselage_anchors},
        wing_ffd_anchor_coordinates=np.asarray(
            wing_block.embedded_entity_parametric_coordinates[2], dtype=float
        ),
        fuselage_end_centroid=fuselage_centroid,
        wing_end_centroid=wing_end_centroid,
        strut_length=csdl.norm(wing_end_centroid - fuselage_centroid),
        root_trailing_edge=root_trailing_edge,
        baseline_root_trailing_edge=baseline_root_te,
        stab_pre_rotation_root_chord=stab_te[0, :] - stab_le[0, :],
    )


def validate_c208_geometry(graph: C208GeometryGraph, *, tolerance=1.0e-8) -> dict:
    """Gate metrics, hard anchors, embedding, symmetry, and projection branches."""
    struts, wing, fuselage, stab = graph.components
    residuals = {
        name: float((value.value - graph.controls[name].value).item())
        for name, value in graph.metrics.items()
    }
    anchor_residuals = {
        "wing_material_m": float(np.max(np.abs(
            graph.anchors["wing_actual"].value - graph.anchors["wing_material_target"].value
        ))),
        "fuselage_fixed_m": float(np.max(np.abs(
            graph.anchors["fuselage_actual"].value - graph.baseline_anchors["fuselage"]
        ))),
        "root_trailing_edge_m": float(np.max(np.abs(
            graph.root_trailing_edge.value - graph.baseline_root_trailing_edge
        ))),
        "root_trailing_edge_coefficient_m": float(np.max(np.abs(
            graph.coefficients["stab"].value[0] - graph.baseline_root_trailing_edge
        ))),
    }
    symmetry = {}
    for name, component, first_id in (
        ("strut", struts, 4), ("wing", wing, 0), ("stab", stab, 2)
    ):
        shape = component.functions[first_id].coefficients.shape
        count = shape[0] * shape[1]
        value = np.asarray(graph.coefficients[name].value)
        starboard = value[:count].reshape(shape)
        port = value[count:].reshape(shape)
        symmetry[name] = float(np.max(np.abs(
            port - starboard[:, ::-1, :] * np.array([1.0, -1.0, 1.0])
        )))

    marker_separation = {}
    for name, points in (
        ("wing", graph.anchors["wing_actual"].value),
        ("fuselage", graph.anchors["fuselage_actual"].value),
    ):
        marker_separation[name] = float(min(
            np.linalg.norm(points[i] - points[j])
            for i, j in ((0, 1), (0, 2), (1, 2))
        ))

    projection = {}
    for name, host, baseline, current_coefficients, current_points in (
        ("wing", wing, graph.baseline_anchors["wing"], graph.coefficients["wing"].value,
         graph.anchors["wing_actual"].value),
        ("fuselage", fuselage, graph.baseline_anchors["fuselage"],
         graph.coefficients["fuselage"].value, graph.anchors["fuselage_actual"].value),
    ):
        model = FunctionSetProjectionModel(
            host, sdf=True, sdf_sign_mode="normal", output_mode="distance"
        )
        initial_distance, initial = model.project(stack_component_coefficients_numpy(host), baseline)
        final_distance, final = model.project(current_coefficients, current_points)
        projection[name] = {
            "baseline_signed_distance_m": initial_distance.tolist(),
            "final_signed_distance_m": final_distance.tolist(),
            "max_signed_distance_change_m": float(np.max(np.abs(final_distance - initial_distance))),
            "baseline_patch_ids": initial["patch_id"].tolist(),
            "final_patch_ids": final["patch_id"].tolist(),
            "baseline_uv": initial["uv"].tolist(),
            "final_uv": final["uv"].tolist(),
            "baseline_candidate_kinds": list(initial["candidate_kind"]),
            "final_candidate_kinds": list(final["candidate_kind"]),
            "max_uv_change": float(np.max(np.abs(final["uv"] - initial["uv"]))),
            "nonconverged": int(np.count_nonzero(~initial["converged"]) + np.count_nonzero(~final["converged"])),
            "sign_ambiguous": int(np.count_nonzero(initial["sign_ambiguous"]) + np.count_nonzero(final["sign_ambiguous"])),
            "branch_changed": bool(
                np.any(initial["patch_id"] != final["patch_id"])
                or list(initial["candidate_kind"]) != list(final["candidate_kind"])
            ),
            "all_interior": bool(np.all(initial["inside_mask"]) and np.all(final["inside_mask"])),
        }
    report = {
        "inner_solver_converged_by_residual": all(abs(x) < tolerance for x in residuals.values()),
        "inner_solver_residuals": residuals,
        "anchor_residuals": anchor_residuals,
        "minimum_marker_separation_m": marker_separation,
        "symmetry_max_abs_m": symmetry,
        "projection": projection,
        "attachment_positions_m": {
            "fuselage_end_centroid": graph.fuselage_end_centroid.tolist(),
            "wing_end_centroid": graph.wing_end_centroid.value.tolist(),
            "fuselage_interior_markers": graph.anchors["fuselage_actual"].value.tolist(),
            "wing_interior_markers": graph.anchors["wing_actual"].value.tolist(),
            "stabilizer_root_trailing_edge": graph.coefficients["stab"].value[0].tolist(),
        },
        "strut_centroid_length_m": float(graph.strut_length.value.item()),
        "coefficients": {
            name: {"shape": list(value.shape), "patch_ids": ids}
            for name, value, ids in (
                ("strut", graph.coefficients["strut"], [4, 5]),
                ("wing", graph.coefficients["wing"], [0, 1]),
                ("fuselage", graph.coefficients["fuselage"], [6]),
                ("stab", graph.coefficients["stab"], [2, 3]),
            )
        },
    }
    if not report["inner_solver_converged_by_residual"]:
        raise ValueError(f"C208 inner solver missed targets: {residuals}")
    if any(value > tolerance for value in anchor_residuals.values()):
        raise ValueError(f"C208 attachment constraint failed: {anchor_residuals}")
    if any(value > tolerance for value in symmetry.values()):
        raise ValueError(f"C208 symmetry constraint failed: {symmetry}")
    if any(value < 0.01 for value in marker_separation.values()):
        raise ValueError(f"C208 attachment markers are not separated: {marker_separation}")
    if any(
        value["nonconverged"] or value["sign_ambiguous"] or value["branch_changed"]
        or not value["all_interior"] for value in projection.values()
    ):
        raise ValueError(f"C208 attachment embedding or closest-point branch changed: {projection}")
    return report
