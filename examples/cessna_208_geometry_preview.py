"""Export sampled C208 CAD surfaces for visual inspection in ParaView.

The quads here are display tessellation of the CAD functions. They are not the
existing CFD mesh and are never passed to mesh deformation or intersections.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from gamma_mdo.core.boundary_surface_movement import stack_component_coefficients_numpy


def _patch_preview(function, coefficients, u_samples, v_samples):
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, u_samples),
        np.linspace(0.0, 1.0, v_samples),
        indexing="ij",
    )
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = np.asarray(
        function.evaluate(uv, coefficients=coefficients, non_csdl=True), dtype=float
    )
    i, j = np.meshgrid(
        np.arange(u_samples - 1), np.arange(v_samples - 1), indexing="ij"
    )
    first = (i * v_samples + j).ravel()
    faces = np.column_stack((
        np.full(first.size, 4), first, first + v_samples,
        first + v_samples + 1, first + 1,
    )).ravel()
    return pv.PolyData(points, faces)


def write_cad_preview(graph, output_directory, *, u_samples=41, v_samples=97):
    """Write baseline and deformed `.vtm` files containing named CAD patches."""
    if u_samples < 2 or v_samples < 2:
        raise ValueError("Each CAD preview axis needs at least two samples.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    baseline = pv.MultiBlock()
    deformed = pv.MultiBlock()
    for name, component in zip(
        ("strut", "wing", "fuselage", "stab"), graph.components
    ):
        original = stack_component_coefficients_numpy(component)
        changed = np.asarray(graph.coefficients[name].value, dtype=float)
        offset = 0
        for patch_id in sorted(component.functions):
            function = component.functions[patch_id]
            shape = function.coefficients.shape
            count = int(np.prod(shape[:-1]))
            label = f"{name}_patch_{patch_id}"
            baseline[label] = _patch_preview(
                function, original[offset : offset + count].reshape(shape),
                u_samples, v_samples,
            )
            deformed[label] = _patch_preview(
                function, changed[offset : offset + count].reshape(shape),
                u_samples, v_samples,
            )
            offset += count
        if offset != changed.shape[0]:
            raise ValueError(f"Coefficient row count changed for {name}.")
    baseline_path = output_directory / "c208_cad_baseline.vtm"
    deformed_path = output_directory / "c208_cad_deformed.vtm"
    baseline.save(baseline_path)
    deformed.save(deformed_path)
    return {"baseline": baseline_path, "deformed": deformed_path}
