"""Move Cessna 208 strut attachments on the mixed triangle/quad CFD mesh.

Edit the settings below and run this file directly. The example shows the
geometry design variables, component connections, mesh-motion settings, and
final checks in the same order as the E175 examples. Numerical CAD utilities
live in a private helper module.
"""

from functools import partial
from pathlib import Path

import csdl_alpha as csdl

import bsm3.mesh_motion as mm
from bsm3.core.boundary_surface_movement import stack_component_coefficients_numpy
from bsm3.core.boundary_surface_movement.cessna_208_example import (
    _finish_case,
    _prepare_case,
    _two_row_strut_map,
    _wing_projection_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "bsm3/core/boundary_surface_movement"

# SETTINGS — edit these values; run this script without command-line arguments.
# 1. Geometry, surface mesh, and output. The helper verifies both input hashes.
STEP_FILE = ASSETS / "cessna_208.stp"
MESH_FILE = ASSETS / "cessna_208.msh"
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

# 3. Mesh regions.
# Free vertices move through the graph solve. Others follow the CAD parameter
# positions. None frees the entire component. "abs" means |y|/max(|y|);
# "extent" means (x-min x)/(max x-min x). Lower bound included, upper excluded.
WING_FREE_REGION = {"y": (0.10, 0.90, "abs")}
FUSELAGE_FREE_REGION = {"x": (0.05, 0.95, "extent")}
# True lets non-seam strut vertices redistribute through the graph solve.
STRUT_VERTICES_FREE = True

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
# Wing vertices within the hold distance of the local leading edge keep their
# original chordwise CAD position (they still slide spanwise). The hold fades
# linearly to zero at the fade distance. Without the hold, leading-edge cells
# can be squeezed chordwise.
WING_LEADING_EDGE_HOLD_M = 0.08
WING_LEADING_EDGE_FADE_M = 0.16

# 5. Derivative check (one load step only).
# The script always prints the analytic sensitivity of the mean final mesh node
# (x, y, z) to each design variable. True also re-runs the complete pipeline
# at design +/- h and compares those vectors with centred finite differences.
# Each step size adds four pipeline re-executions.
CHECK_DERIVATIVES = False
DERIVATIVE_CHECK_STEP_SIZES = (1.0e-4, 1.0e-5)
# END SETTINGS

# A zero-width range selects no vertices: the component follows its CAD
# parameter positions. (None or {} would instead free the whole component.)
NO_FREE_VERTICES = {"x": (0.0, 0.0, "extent")}


def main():
    """Run the two-sided strut-motion example with editable settings above.

    The viewer may open inside the mesh-motion call before final seam validity
    is checked. An invalid run raises and writes an explicitly invalid
    diagnostic report, not a normal result report.
    """
    # Save the chosen settings with the output so different runs are traceable.
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
        "stiffening_exponent": STIFFENING_EXPONENT,
        "ngon_regularization_weight": NGON_REGULARIZATION_WEIGHT,
        "distance_weighting_enabled": DISTANCE_WEIGHTING_ENABLED,
        "distance_beta": DISTANCE_BETA,
        "distance_length_scale_m": DISTANCE_LENGTH_SCALE_M,
        "symmetry_half_mesh": False,
        "distance_decay": DISTANCE_DECAY,
        "wing_leading_edge_hold_m": WING_LEADING_EDGE_HOLD_M,
        "wing_leading_edge_fade_m": WING_LEADING_EDGE_FADE_M,
        "visualize": VISUALIZE,
        "check_derivatives": CHECK_DERIVATIVES,
        "derivative_check_step_sizes": DERIVATIVE_CHECK_STEP_SIZES,
    }
    # Prepare the verified C208 assets and baseline attachment geometry.
    with _prepare_case(
        step_file=STEP_FILE,
        mesh_file=MESH_FILE,
        output_directory=OUT,
        settings=settings,
    ) as case:
        _, wing, fuselage, stab = case.components

        # Build the moving strut and register the fixed host components.
        geometry = mm.GeometryModel()
        wing_control = geometry.design_variable(
            "wing_attachment_span_fraction",
            case.wing_fraction + WING_ATTACHMENT_DELTA_FRACTION,
        )
        fuselage_control = geometry.design_variable(
            "fuselage_attachment_x_fraction",
            case.fuselage_fraction + FUSELAGE_ATTACHMENT_DELTA_FRACTION,
        )
        fractions = csdl.concatenate(
            (wing_control.reshape((1,)), fuselage_control.reshape((1,)))
        )
        target = _two_row_strut_map(
            fractions=fractions,
            baseline=case.strut_baseline,
            fuselage_centroid=case.fuselage_centroid,
            wing_centroid=case.wing_centroid,
            span=case.wing_span,
            fuselage_length=case.fuselage_length,
            nose_x=case.fuselage_nose_x,
        )
        geometry.add_component(
            name="strut", search_name="Struts", deformed_coefficients=target,
            free_region=None if STRUT_VERTICES_FREE else NO_FREE_VERTICES,
        )
        for name, search_name, component, region in (
            ("wing", "MainWing", wing, WING_FREE_REGION),
            ("fuselage", "FuselageGeom", fuselage, FUSELAGE_FREE_REGION),
            ("stab", "Stab", stab, NO_FREE_VERTICES),
        ):
            geometry.add_component(
                name=name,
                search_name=search_name,
                deformed_coefficients=stack_component_coefficients_numpy(component),
                free_region=region,
                projection_metadata_builder=(
                    partial(
                        _wing_projection_metadata,
                        leading_edge_hold_m=WING_LEADING_EDGE_HOLD_M,
                        leading_edge_fade_m=WING_LEADING_EDGE_FADE_M,
                    ) if name == "wing" else None
                ),
            )
        # Keep every seam vertex on its intersection curve. The strut seams
        # move with the strut; the wing-root and stab-root seams stay on the
        # fixed junctions while the surrounding fuselage mesh slides.
        geometry.connect(
            name="strut_wing", driving_component="strut",
            query_component="wing", search_direction="u",
        )
        geometry.connect(
            name="strut_fuselage", driving_component="strut",
            query_component="fuselage", search_direction="u",
        )
        geometry.connect(
            name="wing_fuselage", driving_component="wing",
            query_component="fuselage", search_direction="u",
        )
        geometry.connect(
            name="stab_fuselage", driving_component="stab",
            query_component="fuselage", search_direction="u",
        )

        # Move the triangle/quad mesh while keeping the host CAD fixed.
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
            diagnostic_dump=OUT / f"{case.stem}.npz",
        )
        result = mm.run(
            inputs=mm.InputFiles(
                geometry_file=STEP_FILE,
                surface_mesh_file=MESH_FILE,
                cache_directory=case.cache_directory,
            ),
            geometry=geometry,
            motion=motion,
            recorder=case.recorder,
        )
        # Validate final seams; report mesh quality and the mean-node
        # sensitivity to each design variable (one load step only), plus the
        # optional finite-difference derivative check.
        _finish_case(
            case, result,
            controls={
                "wing_attachment_span_fraction": wing_control,
                "fuselage_attachment_x_fraction": fuselage_control,
            },
        )
        return result


if __name__ == "__main__":
    main()
