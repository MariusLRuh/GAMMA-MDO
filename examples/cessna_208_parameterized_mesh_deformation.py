"""Move the C208 CFD surface mesh with the lsdo_geo parameterized CAD.

Edit the five targets in ``cessna_208_geometry_parameterization.py``, then
run this module. The four solved component coefficient arrays feed the same
GAMMA mesh-motion path used by the earlier strut-only C208 example.
"""

import hashlib
import json
from functools import partial
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

import gamma_mdo.mesh_motion as mm
from examples import cessna_208_geometry_parameterization as design
from gamma_mdo.core.boundary_surface_movement.cessna_208_example import (
    ASSET_HASHES,
    _check_final_seams,
    _wing_projection_metadata,
)
from gamma_mdo.core.boundary_surface_movement.cessna_208_geometry_parameterization import (
    parameterize_c208_geometry,
    validate_c208_geometry,
)
from gamma_mdo.core.projections.function_set_closest_distance_custom_op import (
    FunctionSetProjectionModel,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "gamma_mdo/core/boundary_surface_movement"
STEP_FILE = ASSETS / "cessna_208.stp"
MESH_FILE = ASSETS / "cessna_208.msh"
OUT = ROOT / "c208_parameterized_mesh_results"

# The design targets come from the geometry-only example, so both scripts use
# the same five editable values. Leave visualization off in automated runs.
VISUALIZE = True
LOAD_STEPS = 1
WING_FREE_REGION = {"y": (0.10, 0.90, "abs")}
FUSELAGE_FREE_REGION = {"x": (0.05, 0.95, "extent")}
STRUT_VERTICES_FREE = True
STIFFENING_EXPONENT = 1.5
NGON_REGULARIZATION_WEIGHT = 3.0
WING_LEADING_EDGE_HOLD_M = 0.08
WING_LEADING_EDGE_FADE_M = 0.16
NO_FREE_VERTICES = {"x": (0.0, 0.0, "extent")}
COEFFICIENT_ORDER = ("strut", "wing", "fuselage", "stab")


def _targets_from_design_example():
    return {
        "wing_area_m2": design.WING_AREA_M2,
        "wing_aspect_ratio": design.WING_ASPECT_RATIO,
        "stab_area_m2": design.STAB_AREA_M2,
        "stab_aspect_ratio": design.STAB_ASPECT_RATIO,
        "stab_rotation_deg": design.STAB_ROTATION_DEG,
    }


def _verify_assets():
    for path in (STEP_FILE, MESH_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != ASSET_HASHES[path.name]:
            raise ValueError(f"C208 input asset changed: {path.name}")


def _deformed_seam_summary(result, graph):
    """Check each final intersection against its deformed host surface."""
    hosts = {
        "strut_wing": (graph.components[1], graph.coefficients["wing"].value),
        "strut_fuselage": (graph.components[2], graph.coefficients["fuselage"].value),
        "wing_fuselage": (graph.components[2], graph.coefficients["fuselage"].value),
        "stab_fuselage": (graph.components[2], graph.coefficients["fuselage"].value),
    }
    final = np.asarray(result.surface_coordinates.value, dtype=float)
    groups = result.surface_vertex_classification.intersection_vertex_ids
    summaries = {}
    for name, (host, coefficients) in hosts.items():
        ids = np.asarray(groups[name], dtype=np.int64)
        model = FunctionSetProjectionModel(
            host, sdf=True, sdf_sign_mode="normal", output_mode="distance"
        )
        distance, status = model.project(coefficients, final[ids])
        values = np.asarray(distance, dtype=float).reshape(-1)
        summaries[name] = {
            "vertices": int(ids.size),
            "max_abs_host_sdf_m": float(np.max(np.abs(values))),
            "nonfinite": int(np.count_nonzero(~np.isfinite(values))),
            "nonconverged": int(np.count_nonzero(~np.asarray(status["converged"], dtype=bool))),
            "sign_ambiguous": int(np.count_nonzero(status["sign_ambiguous"])),
        }
    return summaries


def main(*, output_directory=None, visualize=None, targets=None):
    """Solve CAD controls, deform the surface mesh, and report seam/quality checks."""
    output_directory = Path(OUT if output_directory is None else output_directory)
    visualize = VISUALIZE if visualize is None else visualize
    targets = _targets_from_design_example() if targets is None else dict(targets)
    _verify_assets()
    output_directory.mkdir(parents=True, exist_ok=True)
    cache = output_directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        geometry = mm.GeometryModel()
        controls = {
            name: geometry.design_variable(name, float(value))
            for name, value in targets.items()
        }
        graph = parameterize_c208_geometry(
            step_file=STEP_FILE, cache_directory=cache, controls=controls
        )
        geometry_validation = validate_c208_geometry(graph)

        for name, search_name, region in (
            ("strut", "Struts", None if STRUT_VERTICES_FREE else NO_FREE_VERTICES),
            ("wing", "MainWing", WING_FREE_REGION),
            ("fuselage", "FuselageGeom", FUSELAGE_FREE_REGION),
            ("stab", "Stab", NO_FREE_VERTICES),
        ):
            geometry.add_component(
                name=name,
                search_name=search_name,
                deformed_coefficients=graph.coefficients[name],
                free_region=region,
                projection_metadata_builder=(
                    partial(
                        _wing_projection_metadata,
                        leading_edge_hold_m=WING_LEADING_EDGE_HOLD_M,
                        leading_edge_fade_m=WING_LEADING_EDGE_FADE_M,
                    ) if name == "wing" else None
                ),
            )
        for name, driving, query in (
            ("strut_wing", "strut", "wing"),
            ("strut_fuselage", "strut", "fuselage"),
            ("wing_fuselage", "wing", "fuselage"),
            ("stab_fuselage", "stab", "fuselage"),
        ):
            geometry.connect(
                name=name, driving_component=driving,
                query_component=query, search_direction="u",
            )

        motion = mm.MeshMotion(
            surface=mm.SurfaceMotion(
                load_steps=LOAD_STEPS,
                stiffening_exponent=STIFFENING_EXPONENT,
                polygon_regularization=mm.PolygonRegularization(
                    weight=NGON_REGULARIZATION_WEIGHT
                ),
                distance_weighting=mm.DistanceWeighting(enabled=False),
            ),
            quality=mm.QualityChecks(surface=True, fail_on_surface_inversion=False),
            visualization=mm.Visualization(enabled=visualize),
            symmetry=False,
            diagnostic_dump=output_directory / "c208_parameterized_surface.npz",
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
        seams = _deformed_seam_summary(result, graph)
        _check_final_seams(
            seams,
            failure_file=output_directory / "c208_parameterized_mesh.invalid.json",
        )
        initial_inverted = set(result.initial_inversion_report.inverted_element_ids.tolist())
        final_inverted = set(result.surface_inversion_report.inverted_element_ids.tolist())
        report = {
            "targets": {name: float(value.value.item()) for name, value in controls.items()},
            "geometry_validation": geometry_validation,
            "final_seams": seams,
            "surface_vertices": int(result.initial_surface_coordinates.shape[0]),
            "surface_cell_counts": {
                kind: int(len(cells)) for kind, cells in result.surface_mesh.cell_blocks.items()
            },
            "folds": result.surface_fold_count,
            "initial_inverted": len(initial_inverted),
            "final_inverted": len(final_inverted),
            "new_inverted": len(final_inverted - initial_inverted),
            "degenerate": result.surface_quality_report.degenerate_elements,
            "min_scaled_jacobian": result.surface_quality_report.minimum_scaled_jacobian,
            "projection_failures": result.surface_projection_status.num_nonconverged,
            "max_coordinate_change_m": float(np.max(np.abs(
                result.surface_coordinates.value - result.initial_surface_coordinates
            ))),
            "coefficient_order": COEFFICIENT_ORDER,
            "coefficient_shapes": {
                name: list(graph.coefficients[name].shape) for name in COEFFICIENT_ORDER
            },
            "surface_dump": "c208_parameterized_surface.npz",
        }
        (output_directory / "c208_parameterized_mesh_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return result, graph, report
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
