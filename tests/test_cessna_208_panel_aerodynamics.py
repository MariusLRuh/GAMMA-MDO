"""Checks for the C208 panel-aerodynamics example that need no VortexAD solve."""

from types import SimpleNamespace

import numpy as np

from examples import cessna_208_panel_aerodynamics as example


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


def test_component_edges_include_the_component_seam():
    """Strut edges reach the seam nodes, which belong to the neighboring component."""
    result = SimpleNamespace(
        initial_surface_coordinates=np.zeros((6, 3)),
        surface_mesh=SimpleNamespace(cell_blocks={
            "quad": np.array([[0, 1, 2, 3]]),  # strut 0-1, seam 2, wing 3
            "triangle": np.array([[2, 3, 4], [3, 4, 5]]),  # wing only
        }),
        surface_vertex_classification=SimpleNamespace(
            component_vertex_ids={"strut": np.array([0, 1])},
            intersection_vertex_ids={"strut_wing": np.array([2])},
        ),
    )
    assert example._component_edges(result, "strut", ("strut_wing",)) == ((0, 1), (1, 2))
