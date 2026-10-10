"""End-to-end C208 CAD-coefficient handoff to the mixed-cell surface mesh."""

import json

import numpy as np
import pytest

from examples import cessna_208_parameterized_mesh_deformation as example
from tests.test_cessna_208_geometry_parameterization import BASELINE


@pytest.mark.integration
def test_parameterized_geometry_moves_the_surface_and_recomputes_seams(tmp_path):
    """The solved wing, strut, and stabilizer CAD must drive a valid CFD wall."""
    targets = {
        "wing_area_m2": BASELINE["wing_area_m2"] * 1.02,
        "wing_aspect_ratio": BASELINE["wing_aspect_ratio"] * 1.03,
        "stab_area_m2": BASELINE["stab_area_m2"] * 0.98,
        "stab_aspect_ratio": BASELINE["stab_aspect_ratio"] * 1.02,
        "stab_rotation_deg": 3.0,
    }
    result, graph, report = example.main(
        output_directory=tmp_path, visualize=False, targets=targets
    )
    assert report["targets"] == targets
    assert report["geometry_validation"]["inner_solver_converged_by_residual"]
    assert max(report["geometry_validation"]["anchor_residuals"].values()) < 1e-10
    assert report["surface_vertices"] == 26882
    assert report["surface_cell_counts"] == {"triangle": 7486, "quad": 23141}
    assert report["folds"] == 0
    assert report["initial_inverted"] == report["final_inverted"] == 3
    assert report["new_inverted"] == 0
    assert report["degenerate"] == 0
    assert report["projection_failures"] == 0
    assert set(report["final_seams"]) == {
        "strut_wing", "strut_fuselage", "wing_fuselage", "stab_fuselage"
    }
    # The 4 mm wing-root tolerance selects every wing-fuselage junction vertex,
    # including the trailing edge that the 2 mm default missed.
    assert report["final_seams"]["wing_fuselage"]["vertices"] == 158
    assert report["final_seams"]["stab_fuselage"]["vertices"] == 132
    for seam in report["final_seams"].values():
        assert seam["vertices"] > 0
        assert seam["max_abs_host_sdf_m"] < 1e-8
        assert seam["nonfinite"] == seam["nonconverged"] == seam["sign_ambiguous"] == 0
    assert report["coefficient_order"] == example.COEFFICIENT_ORDER
    assert report["coefficient_shapes"] == {
        "strut": [1228, 3], "wing": [2310, 3],
        "fuselage": [3298, 3], "stab": [1410, 3],
    }
    for name in example.COEFFICIENT_ORDER:
        record = next(item for item in result.geometry._component_records if item.name == name)
        component = graph.components[example.COEFFICIENT_ORDER.index(name)]
        assert record.coefficient_builder(component, 1.0, {}) is graph.coefficients[name]
    with np.load(tmp_path / report["surface_dump"]) as mesh:
        movement = np.linalg.norm(mesh["final_vertices"] - mesh["initial_vertices"], axis=1)
        assert movement[mesh["wing_ids"]].max() > 0.1
        assert movement[mesh["stab_ids"]].max() > 0.03
        assert movement[mesh["strut_ids"]].max() > 0.03
    saved = json.loads((tmp_path / "c208_parameterized_mesh_report.json").read_text())
    assert saved["targets"] == targets


def test_quads_folded_across_a_seam_are_found_by_their_diagonal():
    """A quad counts only when the seam runs along its diagonal, not an edge."""
    triangles = np.array([[0, 1, 2]])
    quads = np.array([
        [10, 11, 12, 13],  # seam 10-12 is the diagonal: folded
        [14, 12, 16, 15],  # seam 12-15 is the other diagonal: folded
        [10, 12, 17, 18],  # seam 10-12 is an edge: not folded
        [10, 12, 15, 19],  # three seam corners: not folded
    ])
    folded = example._quads_folded_across_seams(
        triangles, quads, {"wing_fuselage": np.array([10, 12, 15])}
    )
    assert folded.tolist() == [1, 2]
