"""Deform an E175 CFD surface mesh with the differentiable mesh-motion pipeline.

The script walks the five stages the pipeline performs, in order:

1. **Choose geometry and mesh files.** A STEP body supplies the outer mold
   line; a surface mesh supplies the nodes that must follow it.
2. **Define design variables and component motion.** Each design variable is a
   differentiable control. Components declare how they respond, and
   ``connect`` names the closed intersection curves between them.
3. **Choose mesh-motion and quality settings.** Load stepping and seam-distance
   weighting for the graph-Laplacian solve.
4. **Run the differentiable mesh-motion model.** Intersections are recomputed,
   the interior is propagated, graph-moved nodes are reprojected onto the
   deformed outer mold line, and exact intersections are retained.
5. **Inspect the result.** Fold, inversion, and quality diagnostics come back
   on the result object.

This script deliberately demonstrates the **optional** built-in
lifting-surface and body helpers, because they keep stage 2 short and
readable. They are not the required entry point. An external parameterization
replaces stage 2 with :meth:`GeometryModel.add_component`, handing over its own
deformed coefficients; stages 3-5 and the entire downstream GAMMA pipeline —
intersections, graph motion, reprojection, and diagnostics — are identical
either way.

Run it directly::

    python examples/e175_surface_deformation.py

The advanced ``e175_quad_panel_calibration.py`` example uses the curated
quad-dominant panel and reports vertex classifications and projection status.
"""

from pathlib import Path
import tempfile

import csdl_alpha as csdl

import gamma_mdo.mesh_motion as mm

ASSETS = (
    Path(__file__).resolve().parents[1]
    / "gamma_mdo"
    / "core"
    / "boundary_surface_movement"
)
STEP_FILE = ASSETS / "e175.stp"
SURFACE_MESH_FILE = ASSETS / "e175_r1_wall.msh"
CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "gamma_mdo_e175_example_cache"

# Full-size deformation targets, and the neutral value each one moves from.
# ``deformation_scale`` interpolates between the two, so one number controls
# the whole design point coherently.
FULL_DEFORMATION = {
    "wing_shift": (0.0, 0.35),
    "wing_incidence": (0.0, 0.75),
    "wing_area": (70.0, 71.5),
    "tail_incidence": (0.0, 1.2),
    "fuselage_width": (1.0, 1.02),
}
# The tracked triangle wall is covered end to end at this full design point.
DEFAULT_DEFORMATION_SCALE = 1.0


def main(
    *,
    geometry_file: Path = STEP_FILE,
    surface_mesh_file: Path = SURFACE_MESH_FILE,
    cache_directory: Path = CACHE_DIRECTORY,
    deformation_scale: float = DEFAULT_DEFORMATION_SCALE,
    polygon_regularization_weight: float = 0.0,
    check_derivatives: bool = False,
    visualize: bool = False,
    diagnostic_dump: Path | None = None,
) -> mm.MeshMotionResult:
    """Deform the E175 surface mesh and report its quality.

    Parameters
    ----------
    geometry_file
        STEP body defining the outer mold line.
    surface_mesh_file
        Surface mesh whose nodes follow the deformed geometry.
    cache_directory
        Directory for reusable setup data. Reused runs are much faster.
    deformation_scale
        One control for the whole design point. ``0.0`` leaves the geometry
        neutral and ``1.0`` applies the full targets in
        :data:`FULL_DEFORMATION`.
    polygon_regularization_weight
        N-gon affine-residual weight. It is inactive on the default
        triangle-only wall and intended for panel-mesh calibration.
    check_derivatives
        Run the configured finite-difference convergence sweep after building
        the analytic derivative graph.
    visualize
        Open an interactive view of the final deformed mesh.
    diagnostic_dump
        Optional NPZ path for vertex classifications and mesh diagnostics.

    Returns
    -------
    mm.MeshMotionResult
        Differentiable coordinates plus forward diagnostics.
    """
    # 1. Choose geometry and mesh files
    inputs = mm.InputFiles(
        geometry_file=geometry_file,
        surface_mesh_file=surface_mesh_file,
        cache_directory=cache_directory,
    )

    # 2. Define design variables and component motion
    # Built-in helpers are used here for readability. An external
    # parameterization would instead call geometry.add_component(...) with its
    # own deformed coefficients; nothing below this stage changes.
    # The recorder is created, started, and stopped here: GAMMA never owns
    # global CSDL state. The try begins immediately so a failure in any stage
    # below still stops it.
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        geometry = mm.GeometryModel()
        values = {
            name: neutral + deformation_scale * (target - neutral)
            for name, (neutral, target) in FULL_DEFORMATION.items()
        }
        wing_shift = geometry.design_variable(
            "wing_shift", values["wing_shift"]
        )
        wing_incidence = geometry.design_variable(
            "wing_incidence", values["wing_incidence"]
        )
        wing_area = geometry.design_variable("wing_area", values["wing_area"])
        tail_incidence = geometry.design_variable(
            "tail_incidence", values["tail_incidence"]
        )
        fuselage_width = geometry.design_variable(
            "fuselage_width", values["fuselage_width"]
        )

        geometry.add_lifting_surface(
            name="wing",
            search_name="wing",
            pivot_intersection="wing_root",
            translation_x=wing_shift,
            rotation_y_degrees=wing_incidence,
            area=wing_area,
            reference_area=70.0,
            reference_aspect_ratio=8.4,
        )
        geometry.add_lifting_surface(
            name="tail",
            search_name="HT",
            pivot_intersection="tail_root",
            rotation_y_degrees=tail_incidence,
            projection_name="horizontal_tail",
        )
        geometry.add_body(
            name="fuselage",
            search_name="fuselage",
            diameter_scale=fuselage_width,
        )
        geometry.connect(
            name="wing_root",
            driving_component="wing",
            query_component="fuselage",
        )
        geometry.connect(
            name="tail_root",
            driving_component="tail",
            query_component="fuselage",
        )

        # 3. Choose mesh-motion and quality settings
        motion = mm.MeshMotion(
            surface=mm.SurfaceMotion(
                load_steps=2,
                stiffening_exponent=1.5,
                distance_weighting=mm.DistanceWeighting(
                    enabled=True,
                    beta=5.0,
                    length_scale=10.0,
                    decay="exp",
                ),
                polygon_regularization=mm.PolygonRegularization(
                    weight=polygon_regularization_weight
                ),
            ),
            quality=mm.QualityChecks(surface=True),
            visualization=mm.Visualization(enabled=visualize),
            derivative_check=mm.DerivativeCheck(enabled=check_derivatives),
            symmetry=True,
            diagnostic_dump=diagnostic_dump,
        )

        # 4. Run the differentiable mesh-motion model
        result = mm.run(
            inputs=inputs,
            geometry=geometry,
            motion=motion,
            recorder=recorder,
        )
    finally:
        recorder.stop()

    # 5. Inspect the result
    result.print_summary()
    if motion.derivative_check.enabled:
        mm.run_fd_sweep(recorder, motion.derivative_check.step_sizes)
    return result


if __name__ == "__main__":
    main()
