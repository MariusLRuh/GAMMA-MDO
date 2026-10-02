"""Move both Cessna 208 struts on a mixed triangle/quad CFD surface mesh.

Edit the settings below and run this file directly. The starboard wing end
moves spanwise, the fuselage end moves fore/aft, and the port strut mirrors it.
The selected default is a sampled-valid inboard/aft motion; no broader design
rectangle is claimed. All geometry and projection operations remain in CSDL,
including the tapered leading-edge chordwise-coordinate restoration.
"""

import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import csdl_alpha as csdl
import numpy as np
from scipy.interpolate import BSpline, PPoly

import bsm3.mesh_motion as mm
from bsm3.core.boundary_surface_movement import stack_component_coefficients_numpy
from bsm3.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
)
from bsm3.preprocessing import create_components, get_projection_metadata


# SETTINGS — edit these values; run this script without command-line arguments.
# Positive wing delta moves the starboard strut end outboard; port mirrors it.
# Positive fuselage delta moves the other end toward increasing x.
# Fractions use the C208 full CAD wing span and fuselage-body x length.
# These exploratory offsets do not define a validated design rectangle.
WING_ATTACHMENT_DELTA_FRACTION = -0.05
FUSELAGE_ATTACHMENT_DELTA_FRACTION = +0.05
LOAD_STEPS = 1  # Two-step gradients disagree with finite differences; view only.

# Free vertices move through the graph solve. Others follow the CAD parameter
# positions. None frees the entire component. "abs" means |y|/max(|y|);
# "extent" means (x-min x)/(max x-min x). Lower bound included, upper excluded.
WING_FREE_REGION = {"y": (0.10, 0.90, "abs")}
FUSELAGE_FREE_REGION = {"x": (0.05, 0.95, "extent")}
# True lets non-seam strut vertices redistribute through the graph solve.
STRUT_VERTICES_FREE = True

# The two strut seams always run. C208 has no Verts-Full component.
ADDITIONAL_CONNECTIONS = ("wing_fuselage", "stab_fuselage")

# Inverse-cell-area graph weight exponent; larger makes small cells stiffer.
STIFFENING_EXPONENT = 1.5
# Penalizes non-affine quad motion (including hourglass modes), not a guarantee
# that every recombined quad has positive oriented corners.
NGON_REGULARIZATION_WEIGHT = 3.0
# Distance weighting multiplies graph-edge weights by distance from seams.
# The beta, length and decay choices have no effect while disabled.
DISTANCE_WEIGHTING_ENABLED = False
DISTANCE_BETA = 0.5
DISTANCE_LENGTH_SCALE_M = 0.5  # Metres; 10 ft would be 3.048 m.
DISTANCE_DECAY = "exp"
VISUALIZE = True  # Built-in viewer shows flagged elements in red.
# Restore each nose vertex's original chordwise CAD coordinate over the first
# 8 cm behind the leading edge, then fade the restore to zero by 16 cm.
NOSE_CORE_M = 0.08
NOSE_FADE_M = 0.16
# END SETTINGS


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "bsm3/core/boundary_surface_movement"
OUT = Path.home() / ".cache/gamma/cessna_208"
ASSET_HASHES = {
    "cessna208_no_elevator_3.stp": "c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9",
    "cessna208_3_recombine_new.msh": "321b493a18ba0016274c0797a8754613388f09ade58beef0014480a92c1cbeb8",
}
STEP_FILE = ASSETS / "cessna208_no_elevator_3.stp"
MESH_FILE = ASSETS / "cessna208_3_recombine_new.msh"
EMPTY_FREE_REGION = {"x": (0.0, 0.0, "extent")}
CONNECTION_HOSTS = {"wing_fuselage": "wing", "stab_fuselage": "stab"}
SENSITIVITY_VERTEX_IDS = (8115, 5000)


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
                              initial_parametric_coordinates, initial_vertices):
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
    weight = np.clip((NOSE_FADE_M - distance) / (NOSE_FADE_M - NOSE_CORE_M), 0., 1.)
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


def _validate_load_steps():
    """Limit load steps and flag the known two-step derivative discrepancy."""
    if LOAD_STEPS not in (1, 2):
        raise ValueError("The C208 example permits only one or two load steps.")
    if LOAD_STEPS == 2:
        warnings.warn(
            "Two-step mesh gradients disagree with finite differences and must "
            "not be used for optimization. Only final seams are checked; "
            "intermediate-step intersections are not verified.",
            RuntimeWarning,
            stacklevel=2,
        )


def _sensitivity_report(result, controls):
    """Differentiate two fixed wing-vertex x coordinates at one load step."""
    if LOAD_STEPS != 1:
        return {
            "status": "two-step mesh gradients disagree with finite differences; do not use for optimization",
        }
    gradients = {}
    for vertex_id in SENSITIVITY_VERTEX_IDS:
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


def main():
    """Run the two-sided strut-motion example with editable settings above.

    The viewer may open inside the mesh-motion call before final seam validity
    is checked. An invalid run raises and writes an explicitly invalid
    diagnostic report, not a normal result report.
    """
    _validate_load_steps()
    if NGON_REGULARIZATION_WEIGHT <= 0.0:
        raise ValueError("The C208 mixed-cell example requires positive n-gon regularization.")
    for asset in (STEP_FILE, MESH_FILE):
        if not asset.is_file():
            raise FileNotFoundError(asset)
        if hashlib.sha256(asset.read_bytes()).hexdigest() != ASSET_HASHES[asset.name]:
            raise ValueError(f"C208 input asset changed: {asset.name}")
    if len(set(ADDITIONAL_CONNECTIONS)) != len(ADDITIONAL_CONNECTIONS):
        raise ValueError("Additional connection names must be unique.")
    if unknown := set(ADDITIONAL_CONNECTIONS) - CONNECTION_HOSTS.keys():
        raise ValueError(f"Unknown additional connections: {sorted(unknown)}")

    settings = {
        "geometry_file": STEP_FILE.name,
        "surface_mesh_file": MESH_FILE.name,
        "units": "metres",
        "wing_attachment_delta_fraction": WING_ATTACHMENT_DELTA_FRACTION,
        "fuselage_attachment_delta_fraction": FUSELAGE_ATTACHMENT_DELTA_FRACTION,
        "load_steps": LOAD_STEPS,
        "wing_free_region": WING_FREE_REGION,
        "fuselage_free_region": FUSELAGE_FREE_REGION,
        "strut_vertices_free": STRUT_VERTICES_FREE,
        "additional_connections": ADDITIONAL_CONNECTIONS,
        "stiffening_exponent": STIFFENING_EXPONENT,
        "ngon_regularization_weight": NGON_REGULARIZATION_WEIGHT,
        "distance_weighting_enabled": DISTANCE_WEIGHTING_ENABLED,
        "distance_beta": DISTANCE_BETA,
        "distance_length_scale_m": DISTANCE_LENGTH_SCALE_M,
        "symmetry_half_mesh": False,
        "distance_decay": DISTANCE_DECAY,
        "nose_core_m": NOSE_CORE_M,
        "nose_fade_m": NOSE_FADE_M,
        "visualize": VISUALIZE,
    }
    settings_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    stem = (
        f"cessna_208_w{WING_ATTACHMENT_DELTA_FRACTION:+.3f}"
        f"_f{FUSELAGE_ATTACHMENT_DELTA_FRACTION:+.3f}"
        f"_s{LOAD_STEPS}_{settings_hash}"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache"
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
                step_file=STEP_FILE,
            )
        finally:
            os.chdir(previous_dir)
        struts, wing, fuselage, stab = components
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
        wing_fraction = float(gw[1] / span)
        fuselage_fraction = float((gf[0] - nose_x) / fuselage_length)

        geometry = mm.GeometryModel()
        wing_control = geometry.design_variable(
            "wing_attachment_span_fraction",
            wing_fraction + WING_ATTACHMENT_DELTA_FRACTION,
        )
        fuselage_control = geometry.design_variable(
            "fuselage_attachment_x_fraction",
            fuselage_fraction + FUSELAGE_ATTACHMENT_DELTA_FRACTION,
        )
        fractions = csdl.concatenate(
            (wing_control.reshape((1,)), fuselage_control.reshape((1,)))
        )
        target = _two_row_strut_map(
            fractions=fractions,
            baseline=baseline,
            fuselage_centroid=gf,
            wing_centroid=gw,
            span=span,
            fuselage_length=fuselage_length,
            nose_x=nose_x,
        )
        geometry.add_component(
            name="strut", search_name="Struts", deformed_coefficients=target,
            free_region=None if STRUT_VERTICES_FREE else EMPTY_FREE_REGION,
        )
        for name, search_name, component, region in (
            ("wing", "MainWing", wing, WING_FREE_REGION),
            ("fuselage", "FuselageGeom", fuselage, FUSELAGE_FREE_REGION),
            ("stab", "Stab", stab, EMPTY_FREE_REGION),
        ):
            geometry.add_component(
                name=name,
                search_name=search_name,
                deformed_coefficients=stack_component_coefficients_numpy(component),
                free_region=region,
                projection_metadata_builder=(
                    _wing_projection_metadata if name == "wing" else None
                ),
            )
        geometry.connect(
            name="strut_wing", driving_component="strut",
            query_component="wing", search_direction="u",
        )
        geometry.connect(
            name="strut_fuselage", driving_component="strut",
            query_component="fuselage", search_direction="u",
        )
        for connection in ADDITIONAL_CONNECTIONS:
            geometry.connect(
                name=connection, driving_component=CONNECTION_HOSTS[connection],
                query_component="fuselage", search_direction="u",
            )

        motion = mm.MeshMotion(
            surface=mm.SurfaceMotion(
                load_steps=LOAD_STEPS,
                stiffening_exponent=STIFFENING_EXPONENT,
                polygon_regularization=mm.PolygonRegularization(
                    weight=NGON_REGULARIZATION_WEIGHT
                ),
                distance_weighting=mm.DistanceWeighting(
                    enabled=DISTANCE_WEIGHTING_ENABLED,
                    beta=DISTANCE_BETA,
                    length_scale=DISTANCE_LENGTH_SCALE_M,
                    decay=DISTANCE_DECAY,
                ),
            ),
            # Keep going when cells invert so the viewer can show the failure.
            quality=mm.QualityChecks(surface=True, fail_on_surface_inversion=False),
            visualization=mm.Visualization(enabled=VISUALIZE),
            # C208 mesh is asymmetric and some quads cross y=0.
            symmetry=False,
            diagnostic_dump=OUT / f"{stem}.npz",
        )
        result = mm.run(
            inputs=mm.InputFiles(
                geometry_file=STEP_FILE,
                surface_mesh_file=MESH_FILE,
                cache_directory=cache,
            ),
            geometry=geometry,
            motion=motion,
            recorder=recorder,
        )
        seam_summary = _seam_residual_summary(result, components)
        _check_final_seams(seam_summary, failure_file=OUT / f"{stem}.invalid.json")
        sensitivity = _sensitivity_report(result, {
            "wing_attachment_span_fraction": wing_control,
            "fuselage_attachment_x_fraction": fuselage_control,
        })
        report = {
            "intersections_valid": True,
            "sensitivity": sensitivity,
            "settings": settings,
            "final_seams": seam_summary,
            "elapsed_seconds": time.monotonic() - started,
            "baseline_fractions": [wing_fraction, fuselage_fraction],
            "wing_full_span_m": span,
            "fuselage_length_m": fuselage_length,
            "fuselage_section_centroid_m": gf.tolist(),
            "wing_section_centroid_m": gw.tolist(),
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
        (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return result
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
