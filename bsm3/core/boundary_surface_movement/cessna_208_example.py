"""Private CAD, projection, and reporting helpers for the C208 example.

This module supports the example script and is not a public GAMMA API.
"""

import hashlib
import json
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import csdl_alpha as csdl
import numpy as np
from scipy.interpolate import BSpline, PPoly

from bsm3.core.boundary_surface_movement import stack_component_coefficients_numpy
from bsm3.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
)
from bsm3.preprocessing import create_components, get_projection_metadata


ASSET_HASHES = {
    "cessna208_no_elevator_3.stp": "c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9",
    "cessna208_3_recombine_new.msh": "321b493a18ba0016274c0797a8754613388f09ade58beef0014480a92c1cbeb8",
}

def _section_area_centroid(function, row_index):
    """Calculate a C208 strut end-section centroid from its CAD spline."""
    coefficients = np.asarray(function.coefficients.value, dtype=float)
    curve = BSpline(
        function.space.knots[1], coefficients[row_index], function.space.degree[1]
    )
    points = np.asarray(curve(np.linspace(0.0, 1.0, 4097)), dtype=float)
    if np.linalg.norm(points[0] - points[-1]) > 1.0e-10:
        raise ValueError("C208 strut end section is not closed.")
    origin = points.mean(axis=0)
    relative = points - origin
    _, singular_values, _ = np.linalg.svd(relative, full_matrices=False)
    if singular_values[-1] / np.sqrt(len(points)) > 1.0e-10:
        raise ValueError("C208 strut end section is not planar.")
    cross = np.cross(relative[:-1], relative[1:])
    vector_area = cross.sum(axis=0)
    area_norm = np.linalg.norm(vector_area)
    if area_norm <= 1.0e-15:
        raise ValueError("C208 strut end section has zero area.")
    signed_twice_area = cross @ (vector_area / area_norm)
    return origin + np.sum(
        (relative[:-1] + relative[1:]) * signed_twice_area[:, None], axis=0
    ) / (3.0 * signed_twice_area.sum())


def _wing_full_span(wing):
    """Evaluate each degree-one-u CAD row's exact spline y extrema."""
    extrema = []
    for function in wing.functions.values():
        for row in np.asarray(function.coefficients.value, dtype=float):
            curve = BSpline(function.space.knots[1], row[:, 1], function.space.degree[1])
            roots = PPoly.from_spline(curve).derivative().roots(extrapolate=False)
            interior = roots[np.isfinite(roots) & (roots >= 0.0) & (roots <= 1.0)]
            extrema.extend(np.asarray(curve(np.r_[0.0, 1.0, interior])).tolist())
    return float(max(extrema) - min(extrema))


def _two_row_strut_map(*, fractions, baseline, fuselage_centroid,
                       wing_centroid, span, fuselage_length, nose_x):
    """Rigidly move the two C208 end rows and mirror the starboard patch."""
    baseline = np.asarray(baseline, dtype=float)
    if baseline.ndim != 3 or baseline.shape[0] != 2 or baseline.shape[2] != 3:
        raise ValueError("C208 strut patch must have two section rows.")
    gf = np.asarray(fuselage_centroid, dtype=float)
    gw = np.asarray(wing_centroid, dtype=float)
    dw = (fractions[0] - gw[1] / span) * span
    df = (fractions[1] - (gf[0] - nose_x) / fuselage_length) * fuselage_length
    moved_f = gf + csdl.expand(df, (3,)) * np.array([1.0, 0.0, 0.0])
    moved_w = gw + csdl.expand(dw, (3,)) * np.array([0.0, 1.0, 0.0])
    old_axis = csdl.Variable(value=gw - gf)
    old_axis = old_axis / csdl.norm(old_axis)
    new_axis = moved_w - moved_f
    new_axis = new_axis / csdl.norm(new_axis)
    cross = csdl.cross(old_axis, new_axis)
    cosine = csdl.sum(old_axis * new_axis)
    skew = csdl.Variable(value=np.zeros((3, 3)))
    for i, j, k, sign in (
        (0, 1, 2, -1), (0, 2, 1, 1), (1, 0, 2, 1),
        (1, 2, 0, -1), (2, 0, 1, -1), (2, 1, 0, 1),
    ):
        skew = skew.set(csdl.slice[i, j], sign * cross[k])
    rotation = np.eye(3) + skew + csdl.matmat(skew, skew) / (1 + cosine)
    rows = []
    for original, center, moved in (
        (baseline[0], gf, moved_f), (baseline[1], gw, moved_w)
    ):
        rows.append(
            csdl.expand(moved, original.shape, "j->ij")
            + csdl.matmat(original - center, rotation.T())
        )
    starboard = csdl.concatenate(rows, axis=0)
    reverse_v = np.arange(starboard.shape[0]).reshape(baseline.shape[:2])[:, ::-1]
    port = starboard[reverse_v.ravel().tolist(), :] * np.tile(
        [1.0, -1.0, 1.0], (starboard.shape[0], 1)
    )
    return csdl.concatenate((starboard, port), axis=0)


def _wing_projection_metadata(*, component, vertex_ids,
                              initial_parametric_coordinates, initial_vertices,
                              nose_core_m, nose_fade_m):
    """Restore baseline chordwise position near the nose with a smooth taper."""
    ids = np.asarray(vertex_ids, dtype=np.int64)
    xyz = np.asarray(initial_vertices, dtype=float)
    parent = np.asarray(initial_parametric_coordinates, dtype=float)
    patch_ids = tuple(sorted(component.functions))
    wing_ids = np.flatnonzero(np.isin(np.rint(parent[:, 0]).astype(int), patch_ids))
    max_span = float(np.max(np.abs(xyz[wing_ids, 1])))
    distance = np.full(len(xyz), np.inf)
    for y_center in np.arange(-max_span, max_span + 0.05, 0.05):
        strip = wing_ids[np.abs(xyz[wing_ids, 1] - y_center) < 0.025]
        if strip.size:
            distance[strip] = np.minimum(
                distance[strip], xyz[strip, 0] - np.min(xyz[strip, 0])
            )
    weight = np.clip((nose_fade_m - distance) / (nose_fade_m - nose_core_m), 0., 1.)
    band = ids[weight[ids] > 0.]
    other = ids[weight[ids] == 0.]
    groups = []
    if band.size:
        groups.append(get_projection_metadata(
            component=component, vertices=xyz[band], vertex_ids=band,
            para_coords=parent, allowed_patch_ids=patch_ids,
            chordwise_reference=parent[band, 2],
            chordwise_restore_weight=weight[band], name="wing_nose_taper",
        ))
    if other.size:
        groups.append(get_projection_metadata(
            component=component, vertices=xyz[other], vertex_ids=other,
            para_coords=None, allowed_patch_ids=patch_ids, name="wing_other",
        ))
    return groups


def _seam_residual_summary(result, components):
    """Measure every final seam against its intended fixed host CAD surface."""
    _, wing, fuselage, _ = components
    hosts = {
        "strut_wing": wing,
        "strut_fuselage": fuselage,
        "wing_fuselage": fuselage,
        "stab_fuselage": fuselage,
    }
    final = np.asarray(result.surface_coordinates.value, dtype=float)
    groups = result.surface_vertex_classification.intersection_vertex_ids
    summaries = {}
    for name, host in hosts.items():
        ids = np.asarray(groups[name], dtype=np.int64)
        model = FunctionSetProjectionModel(
            host, sdf=True, sdf_sign_mode="normal", output_mode="distance",
        )
        distance, status = model.project(
            stack_component_coefficients_numpy(host), final[ids],
        )
        values = np.asarray(distance, dtype=float).reshape(-1)
        failed = int(np.count_nonzero(~np.asarray(status["converged"], dtype=bool)))
        ambiguous = int(np.count_nonzero(np.asarray(
            status["sign_ambiguous"], dtype=bool,
        )))
        summaries[name] = {
            "vertices": int(ids.size),
            "max_abs_host_sdf_m": float(np.max(np.abs(values))),
            "nonfinite": int(np.count_nonzero(~np.isfinite(values))),
            "nonconverged": failed,
            "sign_ambiguous": ambiguous,
        }
    return summaries


def _check_final_seams(summaries, *, failure_file):
    """Reject invalid final intersections before writing a normal result."""
    invalid = {
        name: values for name, values in summaries.items()
        if (
            values["vertices"] <= 0
            or values["nonfinite"]
            or values["nonconverged"]
            or values["sign_ambiguous"]
            or not np.isfinite(values["max_abs_host_sdf_m"])
            or values["max_abs_host_sdf_m"] > 1.0e-8
        )
    }
    if invalid:
        failure_file.write_text(json.dumps({
            "intersections_valid": False,
            "final_seams": summaries,
            "invalid_final_seams": invalid,
        }, indent=2) + "\n")
        raise ValueError(f"Invalid final intersections: {invalid}")


def _validate_load_steps(load_steps):
    """Limit load steps and flag the known two-step derivative discrepancy."""
    if load_steps not in (1, 2):
        raise ValueError("The C208 example permits only one or two load steps.")
    if load_steps == 2:
        warnings.warn(
            "Two-step mesh gradients disagree with finite differences and must "
            "not be used for optimization. Only final seams are checked; "
            "intermediate-step intersections are not verified.",
            RuntimeWarning,
            stacklevel=2,
        )


def _sensitivity_report(result, controls, load_steps, vertex_ids):
    """Differentiate two fixed wing-vertex x coordinates at one load step."""
    if load_steps != 1:
        return {
            "status": "two-step mesh gradients disagree with finite differences; do not use for optimization",
        }
    gradients = {}
    for vertex_id in vertex_ids:
        weights = np.zeros_like(result.initial_surface_coordinates)
        weights[vertex_id, 0] = 1.0
        objective = csdl.sum(result.surface_coordinates * weights)
        gradients[str(vertex_id)] = {
            name: float(np.asarray(csdl.derivative(objective, control).value).reshape(-1)[0])
            for name, control in controls.items()
        }
    return {
        "status": "analytic one-step mesh-coordinate sensitivities",
        "units": "metres per unit attachment fraction",
        "x_coordinate_gradients": gradients,
    }



@dataclass(frozen=True)
class _PreparedCase:
    """Verified C208 inputs and baseline geometry for one run."""

    components: tuple
    strut_baseline: np.ndarray
    fuselage_centroid: np.ndarray
    wing_centroid: np.ndarray
    wing_span: float
    fuselage_nose_x: float
    fuselage_length: float
    wing_fraction: float
    fuselage_fraction: float
    output_directory: Path
    cache_directory: Path
    stem: str
    settings: dict
    recorder: object
    started_at: float


@contextmanager
def _prepare_case(*, step_file, mesh_file, output_directory, settings,
                  connection_drivers):
    """Verify assets, start CSDL, and measure the baseline attachments."""
    _validate_load_steps(settings["load_steps"])
    if settings["ngon_regularization_weight"] <= 0.0:
        raise ValueError(
            "The C208 mixed-cell example requires positive n-gon regularization."
        )
    for asset in (step_file, mesh_file):
        if not asset.is_file():
            raise FileNotFoundError(asset)
        if hashlib.sha256(asset.read_bytes()).hexdigest() != ASSET_HASHES[asset.name]:
            raise ValueError(f"C208 input asset changed: {asset.name}")
    connections = settings["additional_connections"]
    if len(set(connections)) != len(connections):
        raise ValueError("Additional connection names must be unique.")
    if unknown := set(connections) - connection_drivers.keys():
        raise ValueError(f"Unknown additional connections: {sorted(unknown)}")

    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    stem = (
        f"cessna_208_w{settings['wing_attachment_delta_fraction']:+.3f}"
        f"_f{settings['fuselage_attachment_delta_fraction']:+.3f}"
        f"_s{settings['load_steps']}_{settings_hash}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    cache = output_directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    started = time.monotonic()
    try:
        previous_dir = Path.cwd()
        try:
            os.chdir(cache)
            components = create_components(
                search_names=["Struts", "MainWing", "FuselageGeom", "Stab"],
                step_file=step_file,
            )
        finally:
            os.chdir(previous_dir)
        struts, wing, fuselage, _ = components
        if sorted(struts.functions) != [4, 5]:
            raise ValueError("C208 strut patch keys changed.")
        baseline = np.asarray(struts.functions[4].coefficients.value).copy()
        port_baseline = np.asarray(struts.functions[5].coefficients.value)
        if not np.array_equal(
            baseline[:, ::-1, :] * np.array([1.0, -1.0, 1.0]), port_baseline
        ):
            raise ValueError("C208 strut patches are not exactly mirrored.")
        gf = _section_area_centroid(struts.functions[4], 0)
        gw = _section_area_centroid(struts.functions[4], 1)
        span = _wing_full_span(wing)
        fuselage_x = np.asarray(fuselage.functions[6].coefficients.value)[..., 0]
        nose_x = float(fuselage_x.min())
        fuselage_length = float(fuselage_x.max() - nose_x)
        yield _PreparedCase(
            components=components,
            strut_baseline=baseline,
            fuselage_centroid=gf,
            wing_centroid=gw,
            wing_span=span,
            fuselage_nose_x=nose_x,
            fuselage_length=fuselage_length,
            wing_fraction=float(gw[1] / span),
            fuselage_fraction=float((gf[0] - nose_x) / fuselage_length),
            output_directory=output_directory,
            cache_directory=cache,
            stem=stem,
            settings=settings,
            recorder=recorder,
            started_at=started,
        )
    finally:
        recorder.stop()


def _finish_case(case, result, *, controls, sensitivity_vertex_ids):
    """Validate final seams and save the same C208 report as the driver."""
    seam_summary = _seam_residual_summary(result, case.components)
    _check_final_seams(
        seam_summary,
        failure_file=case.output_directory / f"{case.stem}.invalid.json",
    )
    sensitivity = _sensitivity_report(
        result, controls, case.settings["load_steps"], sensitivity_vertex_ids,
    )
    report = {
        "intersections_valid": True,
        "sensitivity": sensitivity,
        "settings": case.settings,
        "final_seams": seam_summary,
        "elapsed_seconds": time.monotonic() - case.started_at,
        "baseline_fractions": [case.wing_fraction, case.fuselage_fraction],
        "wing_full_span_m": case.wing_span,
        "fuselage_length_m": case.fuselage_length,
        "fuselage_section_centroid_m": case.fuselage_centroid.tolist(),
        "wing_section_centroid_m": case.wing_centroid.tolist(),
        "surface_cell_counts": {
            kind: int(len(cells))
            for kind, cells in result.surface_mesh.cell_blocks.items()
        },
        "surface_vertices": len(result.initial_surface_coordinates),
        "folds": result.surface_fold_count,
        "inverted": result.surface_inversion_report.num_inverted,
        "degenerate": result.surface_quality_report.degenerate_elements,
        "min_scaled_jacobian": result.surface_quality_report.minimum_scaled_jacobian,
        "projection_failures": result.surface_projection_status.num_nonconverged,
        "max_coordinate_change_m": float(np.max(abs(
            result.surface_coordinates.value - result.initial_surface_coordinates
        ))),
        "graph_free_vertices": int(
            result.surface_vertex_classification.graph_free_vertex_ids.size
        ),
        "parametrically_prescribed_vertices": int(
            result.surface_vertex_classification.parametrically_prescribed_vertex_ids.size
        ),
    }
    (case.output_directory / f"{case.stem}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report
