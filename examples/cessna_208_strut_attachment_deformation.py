"""Move Cessna 208 strut attachments on the mixed triangle/quad CFD mesh.

Edit the settings below and run this file directly. The example shows the
geometry design variables, component connections, mesh-motion settings, and
final checks in the same order as the E175 examples. Numerical CAD utilities
live in a private helper module.
"""

import hashlib
import json
import os
from functools import partial
from pathlib import Path
import time

import csdl_alpha as csdl
import numpy as np

import bsm3.mesh_motion as mm
from bsm3.core.boundary_surface_movement import stack_component_coefficients_numpy
from bsm3.core.boundary_surface_movement.cessna_208_example import (
    _check_final_seams,
    _section_area_centroid,
    _seam_residual_summary,
    _sensitivity_report,
    _two_row_strut_map,
    _validate_load_steps,
    _wing_full_span,
    _wing_projection_metadata,
)
from bsm3.preprocessing import create_components


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "bsm3/core/boundary_surface_movement"

# SETTINGS — edit these values; run this script without command-line arguments.
# 1. Geometry, surface mesh, and output. Input hashes are pinned below.
STEP_FILE = ASSETS / "cessna208_no_elevator_3.stp"
MESH_FILE = ASSETS / "cessna208_3_recombine_new.msh"
OUT = Path.home() / ".cache/gamma/cessna_208"

# 2. Geometry design point.
# Positive wing delta moves the starboard strut end outboard; port mirrors it.
# Positive fuselage delta moves the other end toward increasing x.
# Set it to 0.0 to hold that attachment fixed; its seam still follows the strut.
# Fractions use the C208 full CAD wing span and fuselage-body x length.
# These exploratory offsets do not define a validated design rectangle.
WING_ATTACHMENT_DELTA_FRACTION = -0.05
FUSELAGE_ATTACHMENT_DELTA_FRACTION = +0.05
LOAD_STEPS = 1  # Two-step gradients disagree with finite differences; view only.

# 3. Mesh regions and component intersections.
# Free vertices move through the graph solve. Others follow the CAD parameter
# positions. None frees the entire component. "abs" means |y|/max(|y|);
# "extent" means (x-min x)/(max x-min x). Lower bound included, upper excluded.
WING_FREE_REGION = {"y": (0.10, 0.90, "abs")}
FUSELAGE_FREE_REGION = {"x": (0.05, 0.95, "extent")}
# True lets non-seam strut vertices redistribute through the graph solve.
STRUT_VERTICES_FREE = True

# The two strut seams always run. C208 has no Verts-Full component.
ADDITIONAL_CONNECTIONS = ("wing_fuselage", "stab_fuselage")

# 4. Mesh-motion and projection settings.
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

# 5. Report two representative wing-vertex sensitivities at one load step.
SENSITIVITY_VERTEX_IDS = (8115, 5000)
# END SETTINGS

ASSET_HASHES = {
    "cessna208_no_elevator_3.stp": "c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9",
    "cessna208_3_recombine_new.msh": "321b493a18ba0016274c0797a8754613388f09ade58beef0014480a92c1cbeb8",
}
EMPTY_FREE_REGION = {"x": (0.0, 0.0, "extent")}
ADDITIONAL_CONNECTION_DRIVERS = {
    "wing_fuselage": "wing", "stab_fuselage": "stab",
}


def main():
    """Run the two-sided strut-motion example with editable settings above.

    The viewer may open inside the mesh-motion call before final seam validity
    is checked. An invalid run raises and writes an explicitly invalid
    diagnostic report, not a normal result report.
    """
    _validate_load_steps(LOAD_STEPS)
    if NGON_REGULARIZATION_WEIGHT <= 0.0:
        raise ValueError("The C208 mixed-cell example requires positive n-gon regularization.")
    for asset in (STEP_FILE, MESH_FILE):
        if not asset.is_file():
            raise FileNotFoundError(asset)
        if hashlib.sha256(asset.read_bytes()).hexdigest() != ASSET_HASHES[asset.name]:
            raise ValueError(f"C208 input asset changed: {asset.name}")
    if len(set(ADDITIONAL_CONNECTIONS)) != len(ADDITIONAL_CONNECTIONS):
        raise ValueError("Additional connection names must be unique.")
    if unknown := set(ADDITIONAL_CONNECTIONS) - ADDITIONAL_CONNECTION_DRIVERS.keys():
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
        # 1. Import CAD components and measure the two attachment stations.
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

        # 2. Declare both design variables and the moving strut.
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
                    partial(
                        _wing_projection_metadata,
                        nose_core_m=NOSE_CORE_M,
                        nose_fade_m=NOSE_FADE_M,
                    ) if name == "wing" else None
                ),
            )
        # 3. Solve the four component intersections.
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
                name=connection, driving_component=ADDITIONAL_CONNECTION_DRIVERS[connection],
                query_component="fuselage", search_direction="u",
            )

        # 4. Move the triangle/quad surface mesh on the fixed host CAD.
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
        # 5. Check final seams, then write quality and sensitivity results.
        seam_summary = _seam_residual_summary(result, components)
        _check_final_seams(seam_summary, failure_file=OUT / f"{stem}.invalid.json")
        sensitivity = _sensitivity_report(result, {
            "wing_attachment_span_fraction": wing_control,
            "fuselage_attachment_x_fraction": fuselage_control,
        }, LOAD_STEPS, SENSITIVITY_VERTEX_IDS)
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
