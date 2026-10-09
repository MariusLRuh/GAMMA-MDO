"""Parameterize and validate Cessna 208 CAD geometry without moving a mesh.

Edit the five target values below, then run this module. Planform area uses the
full port/starboard three-station trapezoid reference; aspect ratio is full
span squared divided by that area. Coordinates and areas are SI; the pitch
angle is in degrees. The private helper contains the CAD and solver plumbing.
"""

import json
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from gamma_mdo.core.boundary_surface_movement.cessna_208_geometry_parameterization import (
    parameterize_c208_geometry,
    validate_c208_geometry,
)
from examples.cessna_208_geometry_preview import write_cad_preview


ROOT = Path(__file__).resolve().parents[1]
STEP_FILE = ROOT / "gamma_mdo/core/boundary_surface_movement/cessna_208.stp"
OUT = ROOT / "c208_geometry_results"

# Baseline targets measured from the shipped C208 STEP, in the units above.
WING_AREA_M2 = 25.985382406299443
WING_ASPECT_RATIO = 9.543398722310108
STAB_AREA_M2 = 6.507076017257924
STAB_ASPECT_RATIO = 6.000007139374481
STAB_ROTATION_DEG = 0.0  # Positive rotates the nose upward about +y.
EXPORT_CAD_PREVIEW = True  # Sampled CAD surfaces for ParaView; not the CFD mesh.


def main():
    """Solve geometry controls, reject failed attachments, and save coefficients."""
    OUT.mkdir(parents=True, exist_ok=True)
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        controls = {
            "wing_area_m2": csdl.Variable(value=np.array([WING_AREA_M2]), name="wing_area_m2"),
            "wing_aspect_ratio": csdl.Variable(
                value=np.array([WING_ASPECT_RATIO]), name="wing_aspect_ratio"
            ),
            "stab_area_m2": csdl.Variable(value=np.array([STAB_AREA_M2]), name="stab_area_m2"),
            "stab_aspect_ratio": csdl.Variable(
                value=np.array([STAB_ASPECT_RATIO]), name="stab_aspect_ratio"
            ),
            "stab_rotation_deg": csdl.Variable(
                value=np.array([STAB_ROTATION_DEG]), name="stab_rotation_deg"
            ),
        }
        graph = parameterize_c208_geometry(
            step_file=STEP_FILE, cache_directory=OUT / "cache", controls=controls
        )
        validation = validate_c208_geometry(graph)
        coefficient_order = ("strut", "wing", "fuselage", "stab")
        np.savez(
            OUT / "c208_geometry_coefficients.npz",
            **{name: graph.coefficients[name].value for name in coefficient_order},
        )
        preview_files = (
            write_cad_preview(graph, OUT) if EXPORT_CAD_PREVIEW else {}
        )
        report = {
            "units": {"length": "m", "area": "m^2", "angle": "deg"},
            "targets": {name: float(value.value.item()) for name, value in controls.items()},
            "baseline_metrics": graph.baseline_metrics,
            "achieved_metrics": {
                name: float(value.value.item()) for name, value in graph.metrics.items()
            },
            "sectional_states_m": {
                name: float(value.value.item()) for name, value in graph.states.items()
            },
            "coefficient_file": "c208_geometry_coefficients.npz",
            "coefficient_order": coefficient_order,
            "cad_preview_files": {
                name: path.name for name, path in preview_files.items()
            },
            "wing_ffd_anchor_coordinates": graph.wing_ffd_anchor_coordinates.tolist(),
            "validation": validation,
            "scope": "CAD geometry only; no mesh deformation or intersections",
        }
        (OUT / "c208_geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return graph, report
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
