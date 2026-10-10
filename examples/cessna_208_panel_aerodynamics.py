"""Solve VortexAD's steady panel method on the deformed C208 surface mesh.

Edit the five targets in ``cessna_208_geometry_parameterization.py``, then run
this module. It deforms the surface mesh with
``cessna_208_parameterized_mesh_deformation.py`` and appends the panel solve to
the same CSDL graph. It reports CL and the induced drag from surface pressure
and from the Trefftz plane, and plots the doublet strength and pressure
coefficient on the deformed surface.

Mesh motion is evaluated inline; the panel graph is then built without inline
evaluation and run by the JAX simulator, which needs far less memory. VortexAD
is optional; install the pinned revision listed in ``docs/src/integrations.md``.
The solve is dense over all 30,627 panels and needs roughly 40 GB of memory,
so expect heavy swapping on smaller machines.
The flight condition below is an illustrative cruise point, not validated
C208 performance data.
"""

import json
import time
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

import gamma_mdo.mesh_motion as mm
from examples import cessna_208_parameterized_mesh_deformation as deformation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "c208_panel_results"

VISUALIZE = True
# Illustrative cruise at 10,000 ft in the standard atmosphere.
MACH = 0.27
ALPHA_DEG = 2.0
DENSITY_KG_M3 = 0.9046
SPEED_OF_SOUND_M_S = 328.4
# Zero-based node pair on the starboard strut-fuselage junction. Its sharp edge
# passes the trailing-edge angle test and would shed a spurious wake from the
# same cell as the strut trailing edge; the 590 remaining edges are the wing,
# stab, and strut trailing edges.
TRAILING_EDGE_EDGES_TO_IGNORE = ((17534, 17950),)
# The fixed wake leaves each trailing edge straight along +x. The strut
# trailing edge climbs from the fuselage to the wing, so its wake sheet cuts
# through the horizontal stabilizer. With False the struts shed no wake and
# carry no circulation; the wing and stab trailing edges are unaffected.
STRUT_WAKE = False


def _panel_faces(result):
    """Return PyVista faces in the adapter's panel order: triangles, then quads."""
    return np.hstack([
        np.column_stack([np.full(len(cells), cells.shape[1]), cells]).ravel()
        for name, cells in result.surface_mesh.cell_blocks.items()
        if name in ("triangle", "quad") and len(cells)
    ])


def _component_edges(result, component, seams):
    """Return every mesh edge whose two nodes lie on ``component`` or its ``seams``."""
    classification = result.surface_vertex_classification
    owned = np.zeros(len(result.initial_surface_coordinates), dtype=bool)
    owned[classification.component_vertex_ids[component]] = True
    for seam in seams:
        owned[classification.intersection_vertex_ids[seam]] = True
    edges = set()
    for name, cells in result.surface_mesh.cell_blocks.items():
        if name not in ("triangle", "quad"):
            continue
        cells = np.asarray(cells, dtype=np.int64)
        for start, end in zip(cells.T, np.roll(cells, -1, axis=1).T):
            inside = owned[start] & owned[end]
            edges.update(zip(
                np.minimum(start, end)[inside].tolist(),
                np.maximum(start, end)[inside].tolist(),
            ))
    return tuple(sorted(edges))


def _plot_distributions(points, faces, doublet, pressure, screenshot, interactive):
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


def main(*, output_directory=None, visualize=None, targets=None):
    """Deform the C208 surface, solve the panel method, and report CL and CDi."""
    output_directory = Path(OUT if output_directory is None else output_directory)
    visualize = VISUALIZE if visualize is None else visualize
    output_directory.mkdir(parents=True, exist_ok=True)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        result, _, motion_report = deformation.main(
            output_directory=output_directory / "mesh_motion",
            visualize=False,
            targets=targets,
            recorder=recorder,
        )
        # Reference geometry from the design targets, held fixed in the solve.
        area = motion_report["targets"]["wing_area_m2"]
        span = float(np.sqrt(area * motion_report["targets"]["wing_aspect_ratio"]))
        ignored_edges = TRAILING_EDGE_EDGES_TO_IGNORE
        if not STRUT_WAKE:
            ignored_edges += _component_edges(
                result, "strut", ("strut_wing", "strut_fuselage")
            )
        recorder.inline = False
        outputs = mm.build_panel_aerodynamics(
            result,
            mm.PanelCondition(
                reference_area_m2=area,
                reference_chord_m=area / span,
                mach=MACH,
                alpha_degrees=ALPHA_DEG,
                density_kg_m3=DENSITY_KG_M3,
                speed_of_sound_m_s=SPEED_OF_SOUND_M_S,
                trailing_edge_edges_to_ignore=ignored_edges,
                drag_type="Trefftz",
            ),
        )
    finally:
        recorder.stop()

    panel_variables = [
        outputs.lift_coefficient, outputs.induced_drag_coefficient,
        outputs.lift_newton, outputs.induced_drag_newton,
        outputs.trefftz_induced_drag_coefficient, outputs.trefftz_induced_drag_newton,
        outputs.pressure_coefficient, outputs.doublet_strength,
    ]
    started = time.perf_counter()
    simulator = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_outputs=panel_variables, gpu=False
    )
    simulator.run()
    solve_seconds = time.perf_counter() - started

    def value(variable):
        return np.asarray(simulator[variable], dtype=float)

    def scalar(variable):
        return float(value(variable).reshape(-1)[0])

    lift_coefficient = scalar(outputs.lift_coefficient)
    trefftz_drag_coefficient = scalar(outputs.trefftz_induced_drag_coefficient)
    aspect_ratio = span**2 / area
    doublet = value(outputs.doublet_strength).reshape(-1)
    pressure = value(outputs.pressure_coefficient).reshape(-1)
    points = np.asarray(result.surface_coordinates.value, dtype=float)
    report = {
        "condition": {
            "mach": MACH,
            "alpha_deg": ALPHA_DEG,
            "density_kg_m3": DENSITY_KG_M3,
            "speed_of_sound_m_s": SPEED_OF_SOUND_M_S,
            "reference_area_m2": area,
            "reference_chord_m": area / span,
            "span_m": span,
            "strut_wake": STRUT_WAKE,
        },
        "CL": lift_coefficient,
        "CDi_pressure": scalar(outputs.induced_drag_coefficient),
        "CDi_Trefftz": trefftz_drag_coefficient,
        "L_N": scalar(outputs.lift_newton),
        "Di_pressure_N": scalar(outputs.induced_drag_newton),
        "Di_Trefftz_N": scalar(outputs.trefftz_induced_drag_newton),
        # Span efficiency e = CL^2 / (pi AR CDi), from the Trefftz-plane drag.
        "span_efficiency_Trefftz": lift_coefficient**2
        / (np.pi * aspect_ratio * trefftz_drag_coefficient),
        "panels": int(pressure.size),
        "panel_solve_seconds": solve_seconds,
        "mesh_motion": motion_report,
    }
    (output_directory / "c208_panel_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(
        output_directory / "c208_panel_distributions.npz",
        points=points,
        faces=_panel_faces(result),
        doublet_strength=doublet,
        pressure_coefficient=pressure,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "mesh_motion"}, indent=2))
    _plot_distributions(
        points, _panel_faces(result), doublet, pressure,
        screenshot=output_directory / "c208_panel_distributions.png",
        interactive=visualize,
    )
    return result, outputs, report


if __name__ == "__main__":
    main()
