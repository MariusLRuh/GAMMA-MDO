"""Build an E175 geometry-to-panel-to-fuel-burn optimization graph.

This advanced example uses the public GAMMA API and the optional VortexAD
adapter. It registers geometry design variables, a cruise lift constraint, and
a fuel-burn objective. VortexAD is imported only in stage 4; when it is absent,
the adapter raises an installation message containing the exact supported pin.

The mission and parasite-drag constants below are illustrative analysis inputs,
not validated E175 performance data. Replace them with sourced values before
using the model for design decisions.
"""

from pathlib import Path
import tempfile

import csdl_alpha as csdl

import bsm3.mesh_motion as mm


ASSETS = (
    Path(__file__).resolve().parents[1]
    / "bsm3"
    / "core"
    / "boundary_surface_movement"
)
STEP_FILE = ASSETS / "e175.stp"
PANEL_MESH_FILE = ASSETS / "e175_quad_panel.msh"
CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "gamma_e175_panel_cache"

REFERENCE_WING_AREA_M2 = 70.0
REFERENCE_WING_ASPECT_RATIO = 8.4
REFERENCE_MEAN_AERODYNAMIC_CHORD_M = 3.5
ILLUSTRATIVE_PARASITE_DRAG_COEFFICIENT = 0.020
ILLUSTRATIVE_INITIAL_WEIGHT_NEWTON = 4.0e5

# Zero-based node pairs for the tracked panel mesh. They prevent sharp
# wing/body and tail/body root edges from being mistaken for trailing edges.
TRAILING_EDGE_EDGES_TO_IGNORE = (
    (883, 1956),
    (9038, 8044),
    (12192, 8228),
    (5110, 1067),
)


def main(
    *,
    geometry_file: Path = STEP_FILE,
    panel_mesh_file: Path = PANEL_MESH_FILE,
    cache_directory: Path = CACHE_DIRECTORY,
    check_derivatives: bool = False,
) -> tuple[mm.MeshMotionResult, mm.PanelAerodynamicOutputs, csdl.Variable]:
    """Construct and evaluate the E175 fuel-burn optimization model.

    Parameters
    ----------
    geometry_file
        STEP outer mould line supplied to GAMMA.
    panel_mesh_file
        Full-aircraft triangle/quad surface mesh supplied to VortexAD.
    cache_directory
        Writable directory for reusable GAMMA setup data.
    check_derivatives
        Whether to run the public centered finite-difference sweep for the
        registered fuel-burn objective after graph construction.

    Returns
    -------
    tuple
        Mesh-motion result, named panel outputs, and cruise fuel-burn objective.
    """
    # 1. Choose geometry, panel mesh, and mission files/settings.
    inputs = mm.InputFiles(
        geometry_file=geometry_file,
        surface_mesh_file=panel_mesh_file,
        cache_directory=cache_directory,
    )
    mission = mm.FuelBurnParameters(
        design_range_m=2.0e6,
        thrust_specific_fuel_consumption_kg_per_newton_second=1.7e-5,
        cruise_speed_m_s=230.0,
        initial_weight_newton=ILLUSTRATIVE_INITIAL_WEIGHT_NEWTON,
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        # 2. Define optimization variables and component motion.
        geometry = mm.GeometryModel()
        wing_area = geometry.design_variable(
            "wing_area_m2", 70.0, lower=60.0, upper=82.0, scaler=1.0 / 70.0
        )
        wing_aspect_ratio = geometry.design_variable(
            "wing_aspect_ratio", 8.4, lower=7.0, upper=10.0, scaler=1.0 / 8.4
        )
        wing_incidence = geometry.design_variable(
            "wing_incidence_degrees", 0.0, lower=-3.0, upper=4.0, scaler=0.25
        )
        tail_incidence = geometry.design_variable(
            "tail_incidence_degrees", 0.0, lower=-5.0, upper=5.0, scaler=0.2
        )
        fuselage_width = geometry.design_variable(
            "fuselage_width_scale", 1.0, lower=0.9, upper=1.1
        )
        geometry.add_lifting_surface(
            name="wing",
            search_name="wing",
            pivot_intersection="wing_root",
            rotation_y_degrees=wing_incidence,
            area=wing_area,
            aspect_ratio=wing_aspect_ratio,
            reference_area=REFERENCE_WING_AREA_M2,
            reference_aspect_ratio=REFERENCE_WING_ASPECT_RATIO,
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
            name="wing_root", driving_component="wing", query_component="fuselage"
        )
        geometry.connect(
            name="tail_root", driving_component="tail", query_component="fuselage"
        )

        # 3. Choose surface motion. The optimization manages its own objective,
        # so MeshMotion.derivative_check stays disabled and cannot replace it.
        motion = mm.MeshMotion(
            surface=mm.SurfaceMotion(
                load_steps=2,
                stiffening_exponent=1.5,
                distance_weighting=mm.DistanceWeighting(
                    enabled=True, beta=5.0, length_scale=10.0
                ),
                polygon_regularization=mm.PolygonRegularization(weight=0.3),
            ),
            quality=mm.QualityChecks(surface=True),
            symmetry=False,
        )
        mesh_motion = mm.run(
            inputs=inputs,
            geometry=geometry,
            motion=motion,
            recorder=recorder,
        )

        # 4. Append the optional panel solve to the same differentiable graph.
        recorder.inline = False
        reference_chord = REFERENCE_MEAN_AERODYNAMIC_CHORD_M * csdl.sqrt(
            (wing_area / REFERENCE_WING_AREA_M2)
            / (wing_aspect_ratio / REFERENCE_WING_ASPECT_RATIO)
        )
        aerodynamics = mm.build_panel_aerodynamics(
            mesh_motion,
            mm.PanelCondition(
                reference_area_m2=wing_area,
                reference_chord_m=reference_chord,
                trailing_edge_edges_to_ignore=TRAILING_EDGE_EDGES_TO_IGNORE,
            ),
        )

        # 5. Register the lift constraint and fuel-burn objective explicitly.
        total_drag_coefficient = (
            aerodynamics.induced_drag_coefficient
            + ILLUSTRATIVE_PARASITE_DRAG_COEFFICIENT
        )
        fuel_burn = mm.compute_fuel_burn(
            aerodynamics.lift_coefficient,
            total_drag_coefficient,
            mission,
        )
        aerodynamics.lift_newton.set_as_constraint(
            equals=ILLUSTRATIVE_INITIAL_WEIGHT_NEWTON,
            scaler=1.0 / ILLUSTRATIVE_INITIAL_WEIGHT_NEWTON,
        )
        fuel_burn.set_as_objective(scaler=1.0 / 1.0e5)
    finally:
        recorder.stop()

    simulator = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    simulator.run()
    mesh_motion.print_summary()
    print(f"CL: {simulator[aerodynamics.lift_coefficient]}")
    print(f"CDi: {simulator[aerodynamics.induced_drag_coefficient]}")
    print(f"cruise fuel burn [N]: {simulator[fuel_burn]}")
    if check_derivatives:
        mm.run_fd_sweep(recorder, (1.0e-4, 1.0e-5, 1.0e-6))
    return mesh_motion, aerodynamics, fuel_burn


if __name__ == "__main__":
    main()
