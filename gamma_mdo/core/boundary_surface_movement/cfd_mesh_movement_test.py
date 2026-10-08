"""Quality-first E175 surface and Euler volume-mesh deformation pipeline.

Edit the three configuration sections below, then run this module directly.
There is deliberately no command-line or environment-variable configuration.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Support both:
#   python -m gamma_mdo.core.boundary_surface_movement.cfd_mesh_movement_test
# and:
#   python /absolute/path/to/cfd_mesh_movement_test.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import csdl_alpha as csdl

from gamma_mdo.assets import asset_path, require
import numpy as np

from gamma_mdo.core.boundary_surface_movement.geometry_model import GeometryModel
from gamma_mdo.core.boundary_surface_movement.mesh_motion_config import (
    DerivativeCheck,
    DistanceWeighting,
    QualityChecks,
    InputFiles,
    PolygonRegularization,
    MeshMotion,
    SurfaceMotion,
    Visualization,
    VolumeMotion,
)
from gamma_mdo.core.boundary_surface_movement.mesh_motion_pipeline import (
    run_mesh_motion,
    run_fd_sweep,
    select_fd_objective,
)


# ---------------------------------------------------------------------------
# 1. Geometry and matching mesh files
# ---------------------------------------------------------------------------
ASSET_DIRECTORY = Path(__file__).resolve().parent
# R5 Fluent tetrahedral Euler mesh: about 2.2 M tetrahedra inside a 400 m
# half-sphere, converted from millimeters to meters, with its 90 387-triangle
# aircraft wall. Download it once with ``python -m gamma_mdo.assets download e175-r5``.
R5_ASSETS = ("e175-r5-wall", "e175-r5-wall-map", "e175-r5-volume")

MODEL_FILES = InputFiles(
    geometry_file=asset_path("e175-geometry"),
    surface_mesh_file=asset_path("e175-r5-wall"),
    volume_mesh_file=asset_path("e175-r5-volume"),
    volume_wall_map_file=asset_path("e175-r5-wall-map"),
    cache_directory=ASSET_DIRECTORY,
)


# ---------------------------------------------------------------------------
# 2. Geometry design point
# ---------------------------------------------------------------------------
# Deliberately extreme deformation used by the mesh-validity regression.
GEOMETRY_VALUES = {
    "wing_translation_x": 0.0, # 4.51,       # m
    "wing_rotation_degrees": 0.0, #5.01,    # deg
    "tail_rotation_degrees": 0.0, #8.01,    # deg
    "wing_area": 70.01 * 1.001, #1.25,        # m^2 (reference: 70.0)
    "wing_aspect_ratio": 8.41 * 1.0, # 0.75,  # reference: 8.4
    "fuselage_diameter_scale": 1.001 * 1., # 1.25,
}


def create_geometry_model() -> GeometryModel:
    """Build the E175 geometry model with its registered design variables.

    Returns
    -------
    GeometryModel
        Driver-owned design variables, components, and intersections. The
        motion is identical to the previous hand-written declaration; the
        general mechanism now lives in :class:`GeometryModel`.
    """
    geometry = GeometryModel()
    wing_translation_x = geometry.design_variable(
        "wing_translation_x",
        GEOMETRY_VALUES["wing_translation_x"],
        lower=-4.0,
        upper=4.0,
        scaler=1.0 / 3.0,
    )
    wing_rotation_degrees = geometry.design_variable(
        "wing_rotation_degrees",
        GEOMETRY_VALUES["wing_rotation_degrees"],
        lower=-5.0,
        upper=5.0,
        scaler=1.0 / 5.0,
    )
    tail_rotation_degrees = geometry.design_variable(
        "tail_rotation_degrees",
        GEOMETRY_VALUES["tail_rotation_degrees"],
        lower=-8.0,
        upper=8.0,
        scaler=1.0 / 8.0,
    )
    wing_area = geometry.design_variable(
        "wing_area",
        GEOMETRY_VALUES["wing_area"],
        lower=56.0,
        upper=84.0,
        scaler=1.0 / 70.0,
    )
    wing_aspect_ratio = geometry.design_variable(
        "wing_aspect_ratio",
        GEOMETRY_VALUES["wing_aspect_ratio"],
        lower=6.3,
        upper=10.08,
        scaler=1.0 / 8.4,
    )
    fuselage_diameter_scale = geometry.design_variable(
        "fuselage_diameter_scale",
        GEOMETRY_VALUES["fuselage_diameter_scale"],
        lower=0.8,
        upper=1.25,
        scaler=1.0,
    )

    geometry.add_lifting_surface(
        name="wing",
        search_name="wing",
        pivot_intersection="wing_fuse",
        translation_x=wing_translation_x,
        rotation_y_degrees=wing_rotation_degrees,
        area=wing_area,
        aspect_ratio=wing_aspect_ratio,
        reference_area=70.0,
        reference_aspect_ratio=8.4,
    )
    geometry.add_lifting_surface(
        name="tail",
        search_name="HT",
        pivot_intersection="tail_fuse",
        rotation_y_degrees=tail_rotation_degrees,
        projection_name="horizontal_tail",
    )
    geometry.add_body(
        name="fuselage",
        search_name="fuselage",
        diameter_scale=fuselage_diameter_scale,
    )
    geometry.connect(
        name="wing_fuse",
        driving_component="wing",
        query_component="fuselage",
        solver_name="wing_fuselage",
    )
    geometry.connect(
        name="tail_fuse",
        driving_component="tail",
        query_component="fuselage",
        solver_name="tail_fuselage",
    )
    return geometry


# ---------------------------------------------------------------------------
# 3. Surface and volume deformation settings
# ---------------------------------------------------------------------------
OUTPUT_DIRECTORY = Path.home() / ".cache" / "gamma" / "e175_r5_deformation_results"

MESH_MOTION = MeshMotion(
    surface=SurfaceMotion(
        load_steps=2,
        stiffening_exponent=1.5,
        distance_weighting=DistanceWeighting(
            enabled=True,
            beta=5.0,
            length_scale=10.0,
            cap=float("inf"),
            decay="exp",
            power=1.0,
        ),
        # The production wall is triangle-only, so there are no n-gon
        # hourglass modes for the affine regularizer to constrain.
        polygon_regularization=PolygonRegularization(weight=0.0),
    ),
    volume=VolumeMotion(
        mode="off",
        load_mode="synchronized",
        # Follow the two nonlinear surface states exactly.
        synchronized_load_steps=None,
        elasticity_poisson_ratio=0.3,
        elasticity_stiffening_exponent=0.75,
        output_directory=OUTPUT_DIRECTORY,
        write_meshes=True,
    ),
    quality=QualityChecks(
        surface=True,
        volume=True,
        gmsh_volume_metrics=True,
    ),
    visualization=Visualization(enabled=True, opacity=1.0),
    derivative_check=DerivativeCheck(
        enabled=False,
        objective="surface_coordinates",
        step_sizes=(1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6),
    ),
    # The extracted wall is already the y>=0 half mesh.
    symmetry=True,
)


def run_deformation_test():
    """Run the configured E175 surface and volume deformation once.

    A concrete E175 driver, not a generic entry point: the model files,
    geometry parameterization, and motion settings are the module-level
    constants edited in place above, so the function takes no arguments.

    Starts its own inline recorder, builds the geometry model, runs the
    mesh-motion pipeline, and stops the recorder. When the configured
    derivative check is enabled, an objective is selected before the recorder
    stops and the finite-difference sweep runs afterwards. Calling this function
    therefore has conditional side effects: with the configured settings it
    writes deformed meshes to the output directory, opens visualization
    windows, and runs the finite-difference sweep. Which of those occur is
    governed by the module-level configuration, but they are side effects of
    this call.

    Returns
    -------
    MeshMotionResult
        The pipeline result, including surface and volume coordinates and the
        quality diagnostics gathered during the solve.
    """
    for name in R5_ASSETS:
        require(name)
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    geometry = create_geometry_model()
    result = run_mesh_motion(
        recorder=recorder,
        input_files=MODEL_FILES,
        geometry=geometry,
        config=MESH_MOTION,
    )

    if MESH_MOTION.derivative_check.enabled:
        select_fd_objective(
            result,
            MESH_MOTION.derivative_check.objective,
        )
    recorder.stop()

    if MESH_MOTION.derivative_check.enabled:
        run_fd_sweep(
            recorder,
            MESH_MOTION.derivative_check.step_sizes,
        )
    return result


if __name__ == "__main__":
    run_deformation_test()
