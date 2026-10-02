"""Move the Cessna 208 strut attachments on its mixed triangle/quad surface.

Edit the settings below, then run this file directly. Positive wing motion is
outboard on starboard and mirrored on port; positive fuselage motion is toward
increasing x. The CAD and source mesh remain fixed.
"""

from pathlib import Path

from bsm3.core.boundary_surface_movement.cessna_208_example import (
    C208Settings,
    run_c208_example,
)


# SETTINGS — edit these values; run this script without command-line arguments.
# Positive wing delta moves the starboard strut end outboard; port mirrors it.
# Positive fuselage delta moves the other end toward increasing x.
# Set it to 0.0 to hold that attachment fixed; its seam still follows the strut.
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

OUT = Path.home() / ".cache/gamma/cessna_208"


def main():
    """Build geometry, move the mixed-cell mesh, and report seam quality."""
    settings = C208Settings(
        wing_delta=WING_ATTACHMENT_DELTA_FRACTION,
        fuselage_delta=FUSELAGE_ATTACHMENT_DELTA_FRACTION,
        load_steps=LOAD_STEPS,
        wing_free_region=WING_FREE_REGION,
        fuselage_free_region=FUSELAGE_FREE_REGION,
        strut_vertices_free=STRUT_VERTICES_FREE,
        additional_connections=ADDITIONAL_CONNECTIONS,
        stiffening_exponent=STIFFENING_EXPONENT,
        ngon_weight=NGON_REGULARIZATION_WEIGHT,
        distance_enabled=DISTANCE_WEIGHTING_ENABLED,
        distance_beta=DISTANCE_BETA,
        distance_length_scale_m=DISTANCE_LENGTH_SCALE_M,
        distance_decay=DISTANCE_DECAY,
        visualize=VISUALIZE,
        nose_core_m=NOSE_CORE_M,
        nose_fade_m=NOSE_FADE_M,
    )
    return run_c208_example(config=settings, output_directory=OUT)


if __name__ == "__main__":
    main()
