"""Geometry-only C208 FFD, attachment, inner-solve, and derivative gates."""

import contextlib

import csdl_alpha as csdl
import numpy as np
import pyvista as pv

from examples import cessna_208_geometry_parameterization as example
from examples.cessna_208_geometry_preview import write_cad_preview
from gamma_mdo.core.boundary_surface_movement import stack_component_coefficients_numpy
from gamma_mdo.core.boundary_surface_movement.cessna_208_geometry_parameterization import (
    parameterize_c208_geometry,
    validate_c208_geometry,
)


BASELINE = {
    "wing_area_m2": example.WING_AREA_M2,
    "wing_aspect_ratio": example.WING_ASPECT_RATIO,
    "stab_area_m2": example.STAB_AREA_M2,
    "stab_aspect_ratio": example.STAB_ASPECT_RATIO,
    "stab_rotation_deg": 0.0,
}


@contextlib.contextmanager
def _case(tmp_path, target_values):
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        controls = {
            name: csdl.Variable(value=np.array([value]), name=name)
            for name, value in target_values.items()
        }
        graph = parameterize_c208_geometry(
            step_file=example.STEP_FILE,
            cache_directory=tmp_path / "cad_cache",
            controls=controls,
        )
        yield recorder, graph
    finally:
        recorder.stop()


def test_baseline_is_identity_and_all_attachments_are_embedded(tmp_path):
    """The five baseline controls must reproduce every shipped CAD coefficient."""
    with _case(tmp_path, BASELINE) as (_, graph):
        report = validate_c208_geometry(graph)
        struts, wing, fuselage, stab = graph.components
        for name, component in (
            ("strut", struts), ("wing", wing),
            ("fuselage", fuselage), ("stab", stab),
        ):
            np.testing.assert_allclose(
                graph.coefficients[name].value,
                stack_component_coefficients_numpy(component),
                atol=3e-12, rtol=0,
            )
        assert report["inner_solver_converged_by_residual"]
        assert max(map(abs, report["inner_solver_residuals"].values())) < 1e-10
        assert max(report["symmetry_max_abs_m"].values()) == 0.0
        assert max(report["anchor_residuals"].values()) < 3e-12
        for host in ("wing", "fuselage"):
            projection = report["projection"][host]
            assert projection["all_interior"]
            assert not projection["branch_changed"]
            assert projection["nonconverged"] == 0
            assert projection["sign_ambiguous"] == 0
            assert max(projection["final_signed_distance_m"]) < 0.0
        assert graph.wing_ffd_anchor_coordinates.shape == (3, 3)
        assert np.all((graph.wing_ffd_anchor_coordinates > 0.0) & (
            graph.wing_ffd_anchor_coordinates < 1.0
        ))
        np.testing.assert_allclose(graph.strut_length.value, [3.3528], atol=1e-10)


def test_interior_design_targets_rotation_anchors_and_derivatives(tmp_path):
    """A smooth interior change solves and carries correct CSDL sensitivities."""
    targets = {
        "wing_area_m2": BASELINE["wing_area_m2"] * 1.02,
        "wing_aspect_ratio": BASELINE["wing_aspect_ratio"] * 1.03,
        "stab_area_m2": BASELINE["stab_area_m2"] * 0.98,
        "stab_aspect_ratio": BASELINE["stab_aspect_ratio"] * 1.02,
        "stab_rotation_deg": 3.0,
    }
    with _case(tmp_path, targets) as (recorder, graph):
        report = validate_c208_geometry(graph)
        for name, target in targets.items():
            if name != "stab_rotation_deg":
                np.testing.assert_allclose(graph.metrics[name].value, [target], atol=1e-10)
        assert report["inner_solver_converged_by_residual"]
        assert max(report["anchor_residuals"].values()) < 3e-12
        assert max(report["symmetry_max_abs_m"].values()) == 0.0
        assert report["projection"]["wing"]["all_interior"]
        assert not report["projection"]["wing"]["branch_changed"]
        assert report["projection"]["wing"]["nonconverged"] == 0
        # Host-material anchors preserve embedding; their Euclidean signed
        # distance need not remain exactly constant under chord/span scaling.
        assert 0.0 < report["projection"]["wing"]["max_signed_distance_change_m"] < 1e-4
        assert report["projection"]["fuselage"]["max_signed_distance_change_m"] == 0.0
        assert report["strut_centroid_length_m"] > 3.3528 + 0.05
        baseline_strut = stack_component_coefficients_numpy(graph.components[0])
        np.testing.assert_array_equal(
            graph.coefficients["strut"].value[:307], baseline_strut[:307]
        )

        # Compare the actual output coefficients, not only the compensated TE
        # expression. The first stabilizer coefficient is the root trailing edge.
        np.testing.assert_allclose(
            graph.coefficients["stab"].value[0],
            graph.baseline_root_trailing_edge, atol=2e-12, rtol=0,
        )
        stab_function = graph.components[3].functions[2]
        final_stab = graph.coefficients["stab"].value[:705].reshape((3, 235, 3))
        leading_v = np.array([0.5141602232483916])
        final_le = stab_function.evaluate(
            np.column_stack((np.zeros(1), leading_v)),
            coefficients=csdl.Variable(value=final_stab),
        ).value[0]
        final_chord = graph.baseline_root_trailing_edge - final_le
        before = graph.stab_pre_rotation_root_chord.value
        angle = np.degrees(
            np.arctan2(before[2], before[0])
            - np.arctan2(final_chord[2], final_chord[0])
        )
        np.testing.assert_allclose(angle, targets["stab_rotation_deg"], atol=1e-9)

        preview_files = write_cad_preview(
            graph, tmp_path / "preview", u_samples=5, v_samples=9
        )
        baseline_preview = pv.read(preview_files["baseline"])
        changed_preview = pv.read(preview_files["deformed"])
        assert len(changed_preview) == 7
        assert list(changed_preview.keys()) == list(baseline_preview.keys())
        assert np.max(np.abs(
            changed_preview["wing_patch_0"].points
            - baseline_preview["wing_patch_0"].points
        )) > 0.01
        assert np.max(np.abs(
            changed_preview["stab_patch_2"].points
            - baseline_preview["stab_patch_2"].points
        )) > 0.01

        outputs = {
            "wing_coeff_x": graph.coefficients["wing"][1000, 0],
            "strut_anchor_y": graph.anchors["wing_actual"][0, 1],
            "stab_coeff_z": graph.coefficients["stab"][500, 2],
            "fixed_root_te_x": graph.root_trailing_edge[0, 0],
        }
        pairs = (
            ("wing_coeff_x", "wing_area_m2"),
            ("wing_coeff_x", "wing_aspect_ratio"),
            ("strut_anchor_y", "wing_area_m2"),
            ("strut_anchor_y", "wing_aspect_ratio"),
            ("stab_coeff_z", "stab_area_m2"),
            ("stab_coeff_z", "stab_aspect_ratio"),
            ("stab_coeff_z", "stab_rotation_deg"),
            ("fixed_root_te_x", "stab_rotation_deg"),
        )
        for output_name, control_name in pairs:
            output = outputs[output_name]
            control = graph.controls[control_name]
            analytic = float(np.asarray(csdl.derivative(output, control).value).ravel()[0])
            original = control.value.copy()
            try:
                for h in (1e-4, 3e-5):
                    control.value = original + h
                    recorder.execute()
                    plus = float(np.asarray(output.value).ravel()[0])
                    control.value = original - h
                    recorder.execute()
                    minus = float(np.asarray(output.value).ravel()[0])
                    finite_difference = (plus - minus) / (2 * h)
                    np.testing.assert_allclose(analytic, finite_difference, atol=2e-7, rtol=2e-5)
            finally:
                control.value = original
                recorder.execute()
