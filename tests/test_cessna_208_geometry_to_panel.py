"""C208 geometry-to-panel example: stages 1-2 end to end, stage-3 settings offline."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples import cessna_208_geometry_to_panel as example
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
    graph, result, aerodynamics, report = example.main(
        output_directory=tmp_path, targets=targets,
        run_panel_analysis=False, visualize=False,
    )
    assert aerodynamics is None and report["aerodynamics"] is None
    assert report["targets"] == targets
    geometry = report["geometry"]
    assert geometry["validation"]["inner_solver_converged_by_residual"]
    assert max(geometry["validation"]["anchor_residuals"].values()) < 1e-10
    assert geometry["coefficient_shapes"] == {
        "strut": [1228, 3], "wing": [2310, 3],
        "fuselage": [3298, 3], "stab": [1410, 3],
    }
    motion = report["mesh_motion"]
    assert motion["surface_vertices"] == 26882
    assert motion["surface_cell_counts"] == {"triangle": 7486, "quad": 23141}
    assert motion["folds"] == 0
    assert motion["initial_inverted"] == 3
    assert motion["new_inverted"] == 0
    assert motion["degenerate"] == 0
    assert motion["projection_failures"] == 0
    # The 4 mm wing-root tolerance selects every wing-fuselage junction vertex,
    # including the trailing edge that the 2 mm default missed.
    assert motion["final_seams"]["wing_fuselage"]["vertices"] == 158
    assert motion["final_seams"]["stab_fuselage"]["vertices"] == 132
    assert set(motion["final_seams"]) == {
        "strut_wing", "strut_fuselage", "wing_fuselage", "stab_fuselage"
    }
    for seam in motion["final_seams"].values():
        assert seam["vertices"] > 0
        assert seam["max_abs_host_sdf_m"] < 1e-8
        assert seam["nonfinite"] == seam["nonconverged"] == seam["sign_ambiguous"] == 0
    for name in example.COEFFICIENT_ORDER:
        record = next(item for item in result.geometry._component_records if item.name == name)
        component = graph.components[example.COEFFICIENT_ORDER.index(name)]
        assert record.coefficient_builder(component, 1.0, {}) is graph.coefficients[name]
    with np.load(tmp_path / motion["surface_dump"]) as mesh:
        movement = np.linalg.norm(mesh["final_vertices"] - mesh["initial_vertices"], axis=1)
        assert movement[mesh["wing_ids"]].max() > 0.1
        assert movement[mesh["stab_ids"]].max() > 0.03
        assert movement[mesh["strut_ids"]].max() > 0.03
    saved = json.loads((tmp_path / "c208_report.json").read_text())
    assert saved["targets"] == targets
    with np.load(tmp_path / geometry["coefficient_file"]) as coefficients:
        assert sorted(coefficients.files) == sorted(example.COEFFICIENT_ORDER)


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


def test_plot_faces_follow_the_adapter_panel_order():
    """Cp and doublets are per panel: triangles first, then quads."""
    result = SimpleNamespace(surface_mesh=SimpleNamespace(cell_blocks={
        "triangle": np.array([[0, 1, 2]]),
        "quad": np.array([[1, 2, 3, 4], [2, 3, 4, 5]]),
        "line": np.array([[0, 1]]),
    }))
    assert example._panel_faces(result).tolist() == [
        3, 0, 1, 2, 4, 1, 2, 3, 4, 4, 2, 3, 4, 5
    ]


def _strip_result():
    """Build a strip of quads along x: strut 0-1, strut-wing seam 2-3, then wing."""
    return SimpleNamespace(
        initial_surface_coordinates=np.zeros((8, 3)),
        surface_mesh=SimpleNamespace(cell_blocks={
            "quad": np.array([[0, 2, 3, 1], [2, 4, 5, 3], [4, 6, 7, 5]]),
        }),
        surface_vertex_classification=SimpleNamespace(
            component_vertex_ids={"strut": np.array([0, 1])},
            intersection_vertex_ids={
                "strut_wing": np.array([2, 3]),
                "strut_fuselage": np.array([], dtype=np.int64),
                "wing_fuselage": np.array([2, 3]),
                "stab_fuselage": np.array([], dtype=np.int64),
            },
        ),
    )


def test_component_edges_include_the_component_seam():
    """Strut edges reach the seam nodes, which belong to the neighboring component."""
    assert example._component_edges(_strip_result(), "strut", ("strut_wing",)) == (
        (0, 1), (0, 2), (1, 3), (2, 3)
    )


def test_junction_edges_grow_one_edge_ring_at_a_time():
    """A gap of n rings drops every edge with a node within n - 1 edges of the seam."""
    result = _strip_result()
    assert example._junction_edges(result, ("wing_fuselage",), 1) == (
        (0, 2), (1, 3), (2, 3), (2, 4), (3, 5)
    )
    assert example._junction_edges(result, ("wing_fuselage",), 2) == (
        (0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)
    )


def test_panel_condition_uses_the_targets_and_wake_settings(monkeypatch):
    """Reference geometry follows the targets; strut and root edges shed no wake."""
    monkeypatch.setattr(example, "STRUT_WAKE", False)
    monkeypatch.setattr(example, "JUNCTION_WAKE_GAP_EDGES", 1)
    condition = example._panel_condition(
        _strip_result(), {"wing_area_m2": 25.0, "wing_aspect_ratio": 9.0}
    )
    assert condition.reference_area_m2 == 25.0
    assert np.isclose(condition.reference_chord_m, 25.0 / 15.0)
    assert condition.drag_type == "Trefftz"
    ignored = set(condition.trailing_edge_edges_to_ignore)
    assert set(example.TRAILING_EDGE_EDGES_TO_IGNORE) <= ignored
    assert {(0, 1), (2, 3), (2, 4)} <= ignored
    assert (6, 7) not in ignored
