"""Cessna 208 source, strut sensitivity, and mixed-cell example gates."""

import hashlib
import json

import csdl_alpha as csdl
import numpy as np
import pytest

from examples import cessna_208_strut_attachment_deformation as example
from bsm3.core.boundary_surface_movement import cessna_208_example as support


def test_inputs_are_the_approved_bytes():
    """Pin the two user-approved C208 inputs shipped with the example."""
    for path in (support.STEP_FILE, support.MESH_FILE):
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            support.ASSET_HASHES[path.name]
        )


def test_two_row_strut_map_is_mirrored_and_differentiable():
    """Check both attachment directions against centered differences."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        baseline = np.array([
            [[0., 0., 0.], [0.1, 0.1, 0.], [0., 0.2, 0.]],
            [[1., 1., 1.], [1.1, 1.1, 1.], [1., 1.2, 1.]],
        ])
        fractions = csdl.Variable(value=np.array([0.25, 0.30]))
        coefficients = support._two_row_strut_map(
            fractions=fractions,
            baseline=baseline,
            fuselage_centroid=np.array([0., 0., 0.]),
            wing_centroid=np.array([1., 1., 1.]),
            span=4., fuselage_length=3., nose_x=-0.9,
        )
        starboard = coefficients.value[:6].reshape((2, 3, 3))
        port = coefficients.value[6:].reshape((2, 3, 3))
        assert np.array_equal(
            port, starboard[:, ::-1] * np.array([1., -1., 1.]),
        )
        objective = csdl.sum(coefficients * coefficients)
        analytic = np.asarray(csdl.derivative(objective, fractions).value).reshape(-1)
        original = fractions.value.copy()
        for axis in range(2):
            delta = np.zeros(2)
            delta[axis] = 1e-5
            fractions.value = original + delta
            recorder.execute()
            plus = float(np.asarray(objective.value).reshape(-1)[0])
            fractions.value = original - delta
            recorder.execute()
            minus = float(np.asarray(objective.value).reshape(-1)[0])
            np.testing.assert_allclose(
                analytic[axis], (plus - minus) / (2e-5), atol=1e-7,
            )
    finally:
        recorder.stop()


def test_two_step_mode_warns_that_gradients_must_not_be_used():
    """Expose the measured two-step derivative discrepancy at runtime."""
    with pytest.warns(RuntimeWarning, match="gradients disagree"):
        support._validate_load_steps(2)


def test_invalid_final_seams_write_only_an_invalid_diagnostic(tmp_path):
    """Prevent an invalid seam from leaving a normal-looking result file."""
    bad = {
        "strut_wing": {
            "vertices": 1,
            "max_abs_host_sdf_m": float("nan"),
            "nonfinite": 1,
            "nonconverged": 0,
            "sign_ambiguous": 0,
        },
    }
    failure_file = tmp_path / "case.invalid.json"
    with pytest.raises(ValueError, match="Invalid final intersections"):
        support._check_final_seams(bad, failure_file=failure_file)
    assert json.loads(failure_file.read_text())["intersections_valid"] is False
    assert not (tmp_path / "case.json").exists()
    good = {"strut_wing": {**bad["strut_wing"],
                           "max_abs_host_sdf_m": 1e-10, "nonfinite": 0}}
    good_failure_file = tmp_path / "valid.invalid.json"
    support._check_final_seams(good, failure_file=good_failure_file)
    assert not good_failure_file.exists()


@pytest.mark.integration
def test_default_c208_example_preserves_seams_and_polygon_normals(
    monkeypatch, tmp_path,
):
    """Run the one-step mixed mesh and measure its inherited corner defects."""
    monkeypatch.setattr(example, "VISUALIZE", False)
    monkeypatch.setattr(example, "OUT", tmp_path)
    result = example.main()
    assert result.surface_fold_count == 0
    assert result.surface_projection_status.num_nonconverged == 0
    assert result.surface_quality_report.degenerate_elements == 0
    initial = set(result.initial_inversion_report.inverted_element_ids.tolist())
    final = set(result.surface_inversion_report.inverted_element_ids.tolist())
    assert len(initial) == 3
    assert len(final - initial) <= 5
    reports = list(tmp_path.glob("cessna_208_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["intersections_valid"] is True
    assert set(report["final_seams"]) == {
        "strut_wing", "strut_fuselage", "wing_fuselage", "stab_fuselage",
    }
    for seam in report["final_seams"].values():
        assert seam["vertices"] > 0
        assert seam["nonfinite"] == 0
        assert seam["nonconverged"] == 0
        assert seam["sign_ambiguous"] == 0
        assert seam["max_abs_host_sdf_m"] <= 1e-8

    assert report["settings"]["load_steps"] == 1
    gradients = report["sensitivity"]["x_coordinate_gradients"]
    assert set(gradients) == {"8115", "5000"}
    recorder = result.recorder
    recorder.start()
    try:
        controls = result.geometry._design_variables
        for name in (
            "wing_attachment_span_fraction",
            "fuselage_attachment_x_fraction",
        ):
            control = controls[name]
            baseline = np.asarray(control.value).copy()
            step = 1e-5
            control.value = baseline + step
            recorder.execute()
            plus = np.asarray(result.surface_coordinates.value)[[8115, 5000], 0].copy()
            control.value = baseline - step
            recorder.execute()
            minus = np.asarray(result.surface_coordinates.value)[[8115, 5000], 0].copy()
            control.value = baseline
            finite_difference = (plus - minus) / (2 * step)
            analytic = np.array([gradients[str(i)][name] for i in (8115, 5000)])
            np.testing.assert_allclose(
                analytic, finite_difference, atol=1e-6, rtol=1e-5,
            )
    finally:
        recorder.stop()
