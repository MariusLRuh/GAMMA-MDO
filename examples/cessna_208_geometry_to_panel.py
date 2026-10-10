"""Cessna 208: CAD parameterization, surface-mesh deformation, panel aerodynamics.

The example runs three stages in one differentiable CSDL graph:

1. Parameterize the C208 CAD with lsdo_geo. Five design targets (wing area and
   aspect ratio, stabilizer area and aspect ratio, stabilizer pitch) are met
   by an inner solve that keeps both struts attached to the wing and fuselage.
2. Deform the C208 CFD surface mesh with the solved CAD. The four
   intersections (strut-wing, strut-fuselage, wing-fuselage, stab-fuselage)
   are recomputed exactly and checked against the deformed CAD.
3. Optionally solve VortexAD's steady panel method on the deformed surface and
   report CL and the induced drag. This stage is dense over all 30,627 panels
   and needs about 48 GB of memory and 15-20 minutes, so it is off by default
   (``RUN_PANEL_ANALYSIS``).

Edit the settings below, then run from the repository root::

    python examples/cessna_208_geometry_to_panel.py

The design targets default to the baseline CAD. Coordinates and areas are SI;
the stabilizer pitch is in degrees. The flight condition is an illustrative
cruise point, not validated C208 performance data.
"""

import hashlib
import json
import time
from functools import partial
from pathlib import Path

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo
import numpy as np

import gamma_mdo.mesh_motion as mm
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
OUT = ROOT / "c208_results"

# Stage 1: design targets. Each multiplier is 1 at the baseline CAD.
WING_AREA_M2 = 25.985382406299443 * 1.0
WING_ASPECT_RATIO = 9.543398722310108 * 1.0
STAB_AREA_M2 = 6.507076017257924 * 1.0
STAB_ASPECT_RATIO = 6.000007139374481 * 1.0
STAB_ROTATION_DEG = 0.0  # Positive rotates the stabilizer nose up about +y.

# Stage 2: mesh motion. A free region lists the vertices solved by the graph
# propagation as normalized coordinate ranges of the component: "abs" uses
# |y| / semispan, "extent" uses (x - x_min) / length. All other vertices follow
# the CAD at their baseline parametric coordinates.
WING_FREE_REGION = {"y": (0.0, 0.8, "abs")}
FUSELAGE_FREE_REGION = {"x": (0.1, 0.97, "extent")}
STAB_FREE_REGION = {"y": (0.0, 0.5, "abs")}
STRUT_VERTICES_FREE = True
NO_FREE_VERTICES = {"x": (0.0, 0.0, "extent")}
LOAD_STEPS = 1
STIFFENING_EXPONENT = 1.5
NGON_REGULARIZATION_WEIGHT = 100.0
DISTANCE_WEIGHTING = False
# Hold the baseline chordwise CAD coordinate near the wing leading edge.
WING_LEADING_EDGE_HOLD_M = 0.08
WING_LEADING_EDGE_FADE_M = 0.16
# The mesh's wing-root trailing edge sits up to 3.4 mm off the fuselage CAD,
# beyond the default 2 mm seam tolerance. 4 mm selects exactly its 158
# wing-fuselage junction vertices; the other seams keep the default.
WING_ROOT_SEAM_TOLERANCE_M = 4e-3

# Stage 3: panel aerodynamics, an illustrative cruise at 10,000 ft (ISA).
RUN_PANEL_ANALYSIS = False
MACH = 0.27
ALPHA_DEG = 2.0
DENSITY_KG_M3 = 0.9046
SPEED_OF_SOUND_M_S = 328.4
# The fixed wake leaves each trailing edge straight along +x. The strut
# trailing edge climbs from the fuselage to the wing, so its wake would cut
# through the stabilizer; with False the struts shed no wake.
STRUT_WAKE = False
# Trailing-edge edges dropped at each wing and stabilizer root so the inboard
# wake edge does not run along the fuselage surface.
JUNCTION_WAKE_GAP_EDGES = 1
# One starboard strut-fuselage junction edge passes the trailing-edge angle
# test; it is excluded even when the struts shed a wake.
TRAILING_EDGE_EDGES_TO_IGNORE = ((17534, 17950),)

# Open the updated CAD, the seam view, and the panel solution after running.
VISUALIZE = True

COEFFICIENT_ORDER = ("strut", "wing", "fuselage", "stab")
JUNCTION_SEAMS = ("wing_fuselage", "stab_fuselage")
SEAM_COLORS = {
    "strut_wing": "orange",
    "strut_fuselage": "gold",
    "wing_fuselage": "limegreen",
    "stab_fuselage": "magenta",
}


def default_targets():
    """Return the five design targets set at the top of this file."""
    return {
        "wing_area_m2": WING_AREA_M2,
        "wing_aspect_ratio": WING_ASPECT_RATIO,
        "stab_area_m2": STAB_AREA_M2,
        "stab_aspect_ratio": STAB_ASPECT_RATIO,
        "stab_rotation_deg": STAB_ROTATION_DEG,
    }


def _verify_assets():
    for path in (STEP_FILE, MESH_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != ASSET_HASHES[path.name]:
            raise ValueError(f"C208 input asset changed: {path.name}")


def updated_geometry(graph):
    """Reassemble the solved component coefficients as one plottable CAD geometry."""
    functions = {}
    for name, component in zip(COEFFICIENT_ORDER, graph.components):
        coefficients = graph.coefficients[name]
        offset = 0
        for patch_id in sorted(component.functions):
            original = component.functions[patch_id]
            shape = original.coefficients.shape
            count = int(np.prod(shape[:-1]))
            functions[patch_id] = lfs.Function(
                space=original.space,
                coefficients=coefficients[csdl.slice[offset : offset + count, :]].reshape(shape),
                name=original.name,
            )
            offset += count
        if offset != coefficients.shape[0]:
            raise ValueError(f"Coefficient row count changed for {name}.")
    return lsdo_geo.Geometry(functions=functions, name="Updated Cessna 208 CAD")


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


def _cell_blocks(result):
    """Return the triangle and quad blocks in the order panels are numbered."""
    blocks = result.surface_mesh.cell_blocks
    return [
        np.asarray(blocks[name], dtype=np.int64)
        for name in blocks
        if name in ("triangle", "quad") and len(blocks[name])
    ]


def _panel_faces(result):
    """Return PyVista faces in panel order: triangles, then quads."""
    return np.hstack([
        np.column_stack([np.full(len(cells), cells.shape[1]), cells]).ravel()
        for cells in _cell_blocks(result)
    ])


def _mesh_edges(result):
    """Return the unique surface edges as an (n, 2) array of sorted node pairs."""
    edges = [
        np.sort(np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
        for cells in _cell_blocks(result)
    ]
    return np.unique(np.concatenate(edges), axis=0)


def _component_edges(result, component, seams):
    """Return every mesh edge whose two nodes lie on ``component`` or its ``seams``."""
    classification = result.surface_vertex_classification
    owned = np.zeros(len(result.initial_surface_coordinates), dtype=bool)
    owned[classification.component_vertex_ids[component]] = True
    for seam in seams:
        owned[classification.intersection_vertex_ids[seam]] = True
    edges = _mesh_edges(result)
    return tuple(map(tuple, edges[owned[edges[:, 0]] & owned[edges[:, 1]]].tolist()))


def _junction_edges(result, seams, rings):
    """Return every edge with a node within ``rings - 1`` edges of the ``seams``."""
    edges = _mesh_edges(result)
    near = np.zeros(len(result.initial_surface_coordinates), dtype=bool)
    for seam in seams:
        near[result.surface_vertex_classification.intersection_vertex_ids[seam]] = True
    for _ in range(rings - 1):
        grown = near.copy()
        grown[edges[near[edges[:, 0]], 1]] = True
        grown[edges[near[edges[:, 1]], 0]] = True
        near = grown
    return tuple(map(tuple, edges[near[edges[:, 0]] | near[edges[:, 1]]].tolist()))


def _ignored_trailing_edges(result):
    """Collect the node pairs excluded from VortexAD's trailing-edge detection."""
    ignored = TRAILING_EDGE_EDGES_TO_IGNORE
    if not STRUT_WAKE:
        ignored += _component_edges(result, "strut", ("strut_wing", "strut_fuselage"))
    if JUNCTION_WAKE_GAP_EDGES:
        ignored += _junction_edges(result, JUNCTION_SEAMS, JUNCTION_WAKE_GAP_EDGES)
    return ignored


def _quads_folded_across_seams(triangles, quads, groups):
    """Return cell ids of quads whose diagonal, not an edge, lies on a seam."""
    folded = []
    for ids in groups.values():
        on_seam = np.isin(quads, np.asarray(ids, dtype=np.int64))
        for corner in (0, 1):
            diagonal = on_seam[:, corner] & on_seam[:, corner + 2]
            off_diagonal = ~on_seam[:, corner + 1] & ~on_seam[:, (corner + 3) % 4]
            folded.append(np.flatnonzero(diagonal & off_diagonal))
    return len(triangles) + np.unique(np.concatenate(folded))


def _plot_seam_vertices(result):
    """Show every seam's vertices on the baseline (left) and deformed (right) surface.

    Quads folded across a seam are cyan: the input mesh has no edge along the
    junction there, so consecutive seam vertices are opposite quad corners.
    """
    import pyvista as pv

    blocks = result.surface_mesh.cell_blocks
    triangles = np.asarray(blocks.get("triangle", np.empty((0, 3))), dtype=np.int64)
    quads = np.asarray(blocks.get("quad", np.empty((0, 4))), dtype=np.int64)
    faces = _panel_faces(result)
    groups = result.surface_vertex_classification.intersection_vertex_ids
    folded = _quads_folded_across_seams(triangles, quads, groups)
    inverted = np.asarray(result.surface_inversion_report.inverted_element_ids, dtype=np.int64)
    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 900))
    for column, (title, points) in enumerate((
        ("baseline", result.initial_surface_coordinates),
        ("deformed", result.surface_coordinates.value),
    )):
        plotter.subplot(0, column)
        surface = pv.PolyData(np.asarray(points, dtype=float), faces)
        plotter.add_mesh(surface, color="lightgray", show_edges=True, edge_color="black", line_width=0.3)
        if folded.size:
            plotter.add_mesh(
                surface.extract_cells(folded), color="cyan", show_edges=True,
                edge_color="black", label=f"quads folded across a seam ({folded.size})",
            )
        if title == "deformed" and inverted.size:
            plotter.add_mesh(surface.extract_cells(inverted), color="red", label="inverted cells")
        for name, ids in groups.items():
            plotter.add_points(
                surface.points[np.asarray(ids, dtype=np.int64)],
                color=SEAM_COLORS.get(name, "blue"), point_size=10,
                render_points_as_spheres=True, label=f"{name} ({len(ids)})",
            )
        plotter.add_legend()
        plotter.add_text(f"{title} surface: seam vertices", font_size=10)
    plotter.link_views()
    plotter.show()


def _plot_panel_solution(points, faces, doublet, pressure, screenshot, interactive):
    """Show doublet strength (left) and pressure coefficient (right) side by side."""
    import pyvista as pv

    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 900), off_screen=not interactive)
    for column, (title, values, cmap, clim) in enumerate((
        ("doublet strength", doublet, "coolwarm",
         np.percentile(np.abs(doublet), 99) * np.array([-1.0, 1.0])),
        ("pressure coefficient", pressure, "jet", (-1.5, 1.0)),
    )):
        plotter.subplot(0, column)
        surface = pv.PolyData(points, faces)
        surface.cell_data[title] = values
        plotter.add_mesh(surface, scalars=title, cmap=cmap, clim=clim, show_edges=False)
        plotter.add_text(title, font_size=10)
        plotter.view_isometric()
    plotter.link_views()
    plotter.show(screenshot=str(screenshot))


def _panel_condition(result, targets):
    """Return the freestream and wake settings for the deformed design."""
    area = float(targets["wing_area_m2"])
    span = float(np.sqrt(area * targets["wing_aspect_ratio"]))
    return mm.PanelCondition(
        reference_area_m2=area,
        reference_chord_m=area / span,
        mach=MACH,
        alpha_degrees=ALPHA_DEG,
        density_kg_m3=DENSITY_KG_M3,
        speed_of_sound_m_s=SPEED_OF_SOUND_M_S,
        trailing_edge_edges_to_ignore=_ignored_trailing_edges(result),
        drag_type="Trefftz",
    )


def main(*, output_directory=None, targets=None, run_panel_analysis=None, visualize=None):
    """Run the three stages and write ``c208_report.json``.

    Returns the solved geometry graph, the mesh-motion result, the panel
    outputs (``None`` when the panel stage is off), and the report.
    """
    output_directory = Path(OUT if output_directory is None else output_directory)
    targets = default_targets() if targets is None else dict(targets)
    run_panel_analysis = RUN_PANEL_ANALYSIS if run_panel_analysis is None else run_panel_analysis
    visualize = VISUALIZE if visualize is None else visualize
    _verify_assets()
    cache = output_directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        # 1. Parameterize the CAD: meet the five targets with the struts attached.
        geometry = mm.GeometryModel()
        controls = {
            name: geometry.design_variable(name, float(value))
            for name, value in targets.items()
        }
        graph = parameterize_c208_geometry(
            step_file=STEP_FILE, cache_directory=cache, controls=controls
        )
        geometry_validation = validate_c208_geometry(graph)
        updated_cad = updated_geometry(graph) if visualize else None

        # 2. Deform the surface mesh with the solved CAD coefficients.
        for name, search_name, region in (
            ("strut", "Struts", None if STRUT_VERTICES_FREE else NO_FREE_VERTICES),
            ("wing", "MainWing", WING_FREE_REGION),
            ("fuselage", "FuselageGeom", FUSELAGE_FREE_REGION),
            ("stab", "Stab", STAB_FREE_REGION),
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
        for name, driving, query, tolerance in (
            ("strut_wing", "strut", "wing", None),
            ("strut_fuselage", "strut", "fuselage", None),
            ("wing_fuselage", "wing", "fuselage", WING_ROOT_SEAM_TOLERANCE_M),
            ("stab_fuselage", "stab", "fuselage", None),
        ):
            geometry.connect(
                name=name, driving_component=driving,
                query_component=query, search_direction="u",
                intersection_tolerance=tolerance,
            )
        motion = mm.MeshMotion(
            surface=mm.SurfaceMotion(
                load_steps=LOAD_STEPS,
                stiffening_exponent=STIFFENING_EXPONENT,
                polygon_regularization=mm.PolygonRegularization(
                    weight=NGON_REGULARIZATION_WEIGHT
                ),
                distance_weighting=mm.DistanceWeighting(enabled=DISTANCE_WEIGHTING),
            ),
            quality=mm.QualityChecks(surface=True, fail_on_surface_inversion=False),
            symmetry=False,
            diagnostic_dump=output_directory / "c208_surface.npz",
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
        _check_final_seams(seams, failure_file=output_directory / "c208_mesh.invalid.json")

        # 3. Append the panel solve; JAX evaluates it, not the inline backend.
        aerodynamics = None
        if run_panel_analysis:
            recorder.inline = False
            aerodynamics = mm.build_panel_aerodynamics(result, _panel_condition(result, targets))
    finally:
        recorder.stop()

    np.savez(
        output_directory / "c208_geometry_coefficients.npz",
        **{name: graph.coefficients[name].value for name in COEFFICIENT_ORDER},
    )
    initial_inverted = set(result.initial_inversion_report.inverted_element_ids.tolist())
    final_inverted = set(result.surface_inversion_report.inverted_element_ids.tolist())
    report = {
        "units": {"length": "m", "area": "m^2", "angle": "deg", "force": "N"},
        "targets": {name: float(value.value.item()) for name, value in controls.items()},
        "geometry": {
            "baseline_metrics": graph.baseline_metrics,
            "achieved_metrics": {
                name: float(value.value.item()) for name, value in graph.metrics.items()
            },
            "validation": geometry_validation,
            "coefficient_file": "c208_geometry_coefficients.npz",
            "coefficient_order": COEFFICIENT_ORDER,
            "coefficient_shapes": {
                name: list(graph.coefficients[name].shape) for name in COEFFICIENT_ORDER
            },
        },
        "mesh_motion": {
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
            "surface_dump": "c208_surface.npz",
        },
        "aerodynamics": None,
    }

    if aerodynamics is not None:
        started = time.perf_counter()
        simulator = csdl.experimental.JaxSimulator(
            recorder=recorder,
            additional_outputs=[
                aerodynamics.lift_coefficient, aerodynamics.induced_drag_coefficient,
                aerodynamics.lift_newton, aerodynamics.induced_drag_newton,
                aerodynamics.trefftz_induced_drag_coefficient,
                aerodynamics.trefftz_induced_drag_newton,
                aerodynamics.pressure_coefficient, aerodynamics.doublet_strength,
            ],
            gpu=False,
        )
        simulator.run()

        def scalar(variable):
            return float(np.asarray(simulator[variable]).reshape(-1)[0])

        condition = _panel_condition(result, targets)
        lift_coefficient = scalar(aerodynamics.lift_coefficient)
        trefftz_drag_coefficient = scalar(aerodynamics.trefftz_induced_drag_coefficient)
        aspect_ratio = float(targets["wing_aspect_ratio"])
        doublet = np.asarray(simulator[aerodynamics.doublet_strength], dtype=float).reshape(-1)
        pressure = np.asarray(simulator[aerodynamics.pressure_coefficient], dtype=float).reshape(-1)
        report["aerodynamics"] = {
            "mach": MACH,
            "alpha_deg": ALPHA_DEG,
            "density_kg_m3": DENSITY_KG_M3,
            "speed_of_sound_m_s": SPEED_OF_SOUND_M_S,
            "reference_area_m2": float(condition.reference_area_m2),
            "reference_chord_m": float(condition.reference_chord_m),
            "strut_wake": STRUT_WAKE,
            "junction_wake_gap_edges": JUNCTION_WAKE_GAP_EDGES,
            "panels": int(pressure.size),
            "CL": lift_coefficient,
            "CDi_pressure": scalar(aerodynamics.induced_drag_coefficient),
            "CDi_Trefftz": trefftz_drag_coefficient,
            "L_N": scalar(aerodynamics.lift_newton),
            "Di_pressure_N": scalar(aerodynamics.induced_drag_newton),
            "Di_Trefftz_N": scalar(aerodynamics.trefftz_induced_drag_newton),
            # Span efficiency e = CL^2 / (pi AR CDi), from the Trefftz-plane drag.
            "span_efficiency_Trefftz": lift_coefficient**2
            / (np.pi * aspect_ratio * trefftz_drag_coefficient),
            "solve_seconds": time.perf_counter() - started,
            "distributions": "c208_panel_distributions.npz",
        }
        np.savez_compressed(
            output_directory / "c208_panel_distributions.npz",
            points=np.asarray(result.surface_coordinates.value, dtype=float),
            faces=_panel_faces(result),
            doublet_strength=doublet,
            pressure_coefficient=pressure,
        )

    (output_directory / "c208_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(
        {key: report[key] for key in ("targets", "mesh_motion", "aerodynamics")}, indent=2
    ), flush=True)

    if visualize:
        updated_cad.plot()
        _plot_seam_vertices(result)
    if aerodynamics is not None:
        _plot_panel_solution(
            np.asarray(result.surface_coordinates.value, dtype=float),
            _panel_faces(result), doublet, pressure,
            screenshot=output_directory / "c208_panel_distributions.png",
            interactive=visualize,
        )
    return graph, result, aerodynamics, report


if __name__ == "__main__":
    main()
