"""Contract and derivative tests for the optional VortexAD adapter."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import csdl_alpha as csdl
import numpy as np
import pytest

import gamma_mdo.mesh_motion as mm
from gamma_mdo.core.boundary_surface_movement import panel_aerodynamics as panel
from gamma_mdo.preprocessing import MeshData


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "e175_fuel_burn_optimization.py"


def _mesh_result(surface_coordinates, *, nonconverged=()):
    """Build the public result surface consumed by the adapter."""
    initial = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mesh = MeshData(
        vertices=initial.copy(),
        connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_types=np.array(["quad"]),
        cell_blocks={"quad": np.array([[0, 1, 2, 3]], dtype=np.int64)},
    )
    return SimpleNamespace(
        surface_coordinates=surface_coordinates,
        initial_surface_coordinates=initial,
        surface_mesh=mesh,
        surface_vertex_classification=SimpleNamespace(
            intersection_vertex_ids={"root": np.array([0], dtype=np.int64)}
        ),
        surface_projection_status=SimpleNamespace(
            reprojected_vertex_ids=np.array([1, 2, 3], dtype=np.int64),
            nonconverged_vertex_ids=np.asarray(nonconverged, dtype=np.int64),
        ),
        aerodynamic_outputs={},
    )


def _fake_vortexad_module():
    """Return a differentiable stand-in for the pinned VortexAD API."""
    module = types.ModuleType("VortexAD")

    def find_cell_adjacency(*, points, cells):
        edges = {(0, 1): [0], (1, 2): [0], (2, 3): [0], (3, 0): [0]}
        points_to_cells = {index: [0] for index in range(len(points))}
        return points, cells, {"quad": [[]]}, edges, points_to_cells

    def detect_trailing_edges(**kwargs):
        del kwargs
        return [0], [0], [(1, 2)], np.array([1, 2], dtype=np.int64)

    class FakePanelMethod:
        """Small graph builder matching VortexAD's panel-method call shape."""

        def __init__(self, solver_input_dict, skip_geometry):
            assert skip_geometry is True
            self.settings = solver_input_dict

        def insert_grid_data(self, mesh, cell_adjacency_data, TE_properties):
            self.mesh = mesh
            self.adjacency = cell_adjacency_data
            self.trailing_edges = TE_properties

        def declare_outputs(self, outputs):
            self.outputs = tuple(outputs)

        def evaluate(self):
            lift_coefficient = 0.4 + 0.2 * csdl.sum(self.mesh[:, 2])
            induced_drag_coefficient = (
                0.015 + 0.01 * csdl.sum(self.mesh * self.mesh)
            )
            available = {
                "CL": lift_coefficient,
                "CDi": induced_drag_coefficient,
                "L": 1000.0 * lift_coefficient,
                "Di": 1000.0 * induced_drag_coefficient,
            }
            return {name: available[name] for name in self.outputs}

    module.PanelMethod = FakePanelMethod
    module.find_cell_adjacency = find_cell_adjacency
    module.TE_detection = detect_trailing_edges
    return module


def test_adapter_import_is_lazy_and_actionable(monkeypatch):
    """Import the module freely and fail only when the optional solve is used."""
    monkeypatch.setitem(sys.modules, "VortexAD", None)
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        result = _mesh_result(csdl.Variable(value=np.zeros((4, 3))))
        with pytest.raises(ImportError, match="optional GAMMA integration") as error:
            mm.build_panel_aerodynamics(
                result,
                mm.PanelCondition(reference_area_m2=10.0, reference_chord_m=2.0),
            )
    finally:
        recorder.stop()
    assert panel.VORTEXAD_REVISION in str(error.value)
    assert "--no-deps" in str(error.value)


def test_adapter_rejects_malformed_or_nonconverged_results():
    """Validate the public result contract before importing VortexAD."""
    condition = mm.PanelCondition(reference_area_m2=10.0, reference_chord_m=2.0)
    with pytest.raises(TypeError, match="surface_coordinates"):
        mm.build_panel_aerodynamics(SimpleNamespace(surface_coordinates=None), condition)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        result = _mesh_result(
            csdl.Variable(value=np.zeros((4, 3))), nonconverged=(2,)
        )
        with pytest.raises(ValueError, match="did not converge"):
            mm.build_panel_aerodynamics(result, condition)
    finally:
        recorder.stop()


def test_fake_panel_to_fuel_graph_matches_centered_fd(monkeypatch):
    """Carry a mesh design variable through the fake panel and fuel graph."""
    monkeypatch.setitem(sys.modules, "VortexAD", _fake_vortexad_module())
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        amplitude = csdl.Variable(name="panel_shape_amplitude", value=np.array([0.2]))
        amplitude.set_as_design_variable(lower=-0.5, upper=0.5)
        baseline = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        direction = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, 1.5], [0.0, 0.0, 0.5]]
        )
        coordinates = baseline + amplitude * direction
        result = _mesh_result(coordinates)
        outputs = mm.build_panel_aerodynamics(
            result,
            mm.PanelCondition(reference_area_m2=10.0, reference_chord_m=2.0),
        )
        fuel_burn = mm.compute_fuel_burn(
            outputs.lift_coefficient,
            outputs.induced_drag_coefficient + 0.02,
            mm.FuelBurnParameters(
                design_range_m=2.0e6,
                thrust_specific_fuel_consumption_kg_per_newton_second=1.7e-5,
                cruise_speed_m_s=230.0,
                initial_weight_newton=4.0e5,
            ),
        )
        fuel_burn.set_as_objective()
        analytic = csdl.derivative(fuel_burn, amplitude)
    finally:
        recorder.stop()

    assert isinstance(outputs, mm.PanelAerodynamicOutputs)
    assert result.aerodynamic_outputs == outputs.as_dict()
    assert mm.select_fd_objective is not None
    analytic_value = float(np.asarray(analytic.value).reshape(-1)[0])
    simulator = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    baseline_amplitude = np.asarray(amplitude.value).copy()
    errors = []
    for step in (1.0e-4, 1.0e-5, 1.0e-6):
        simulator[amplitude] = baseline_amplitude + step
        simulator.run()
        plus = float(np.asarray(simulator[fuel_burn]).reshape(-1)[0])
        simulator[amplitude] = baseline_amplitude - step
        simulator.run()
        minus = float(np.asarray(simulator[fuel_burn]).reshape(-1)[0])
        centered = (plus - minus) / (2.0 * step)
        errors.append(
            abs(analytic_value - centered)
            / max(abs(analytic_value), abs(centered), 1.0e-14)
        )
    simulator[amplitude] = baseline_amplitude
    assert abs(analytic_value) > 1.0e-6
    assert min(errors) < 1.0e-7


def test_registered_panel_output_is_selectable_for_fd(monkeypatch):
    """Expose named panel outputs through the existing public FD selector."""
    monkeypatch.setitem(sys.modules, "VortexAD", _fake_vortexad_module())
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        result = _mesh_result(csdl.Variable(value=np.zeros((4, 3))))
        outputs = mm.build_panel_aerodynamics(
            result,
            mm.PanelCondition(reference_area_m2=10.0, reference_chord_m=2.0),
        )
        selected = mm.select_fd_objective(result, "CL")
    finally:
        recorder.stop()
    assert selected is outputs.lift_coefficient


def test_optimization_example_keeps_the_five_stage_public_workflow():
    """Keep the tracked integration example high-level and optimization-ready."""
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert ast.get_docstring(tree)
    assert [source.index(f"# {index}.") for index in range(1, 6)] == sorted(
        source.index(f"# {index}.") for index in range(1, 6)
    )
    assert "mm.run(" in source
    assert "mm.build_panel_aerodynamics(" in source
    assert "mm.compute_fuel_burn(" in source
    assert ".set_as_objective(" in source
    assert "DerivativeCheck(enabled=True" not in source
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


@pytest.mark.integration
def test_pinned_vortexad_exports_the_adapter_surface():
    """Check the real optional API when the pinned dependency is available."""
    vortexad = pytest.importorskip("VortexAD")
    for name in ("PanelMethod", "TE_detection", "find_cell_adjacency"):
        assert callable(getattr(vortexad, name))
