from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

import bsm3
from bsm3 import plotting, preprocessing


ASSET_DIR = Path("bsm3/core/boundary_surface_movement")
E175_STEP = ASSET_DIR / "e175.stp"
E175_MESH = ASSET_DIR / "e175_r1_wall.msh"
E175_FIRST_VERTEX = [18.289553605559405, 11.9144025401861, 1.393874452294162]


class FakeFunction:
    def __init__(self, name):
        self.name = name


class FakeFunctionSet:
    def __init__(self, functions, space=None):
        self.functions = dict(functions)
        self.space = space

    def plot(self, **kwargs):
        elements = list(kwargs["additional_plotting_elements"])
        elements.append(
            {
                "mesh": f"mesh-{tuple(self.functions)}",
                "kwargs": {
                    "color": kwargs.get("color"),
                    "opacity": kwargs.get("opacity"),
                },
            }
        )
        return elements


class FakeProjectableComponent:
    def __init__(self, distances, patch_ids=None, uv=None):
        self.distances = np.asarray(distances, dtype=float)
        self.patch_ids = (
            np.zeros(self.distances.shape[0], dtype=np.int64)
            if patch_ids is None
            else np.asarray(patch_ids, dtype=np.int64)
        )
        self.uv = (
            np.zeros((self.distances.shape[0], 2), dtype=float)
            if uv is None
            else np.asarray(uv, dtype=float)
        )
        self.num_project_calls = 0
        self.projected_point_counts = []

    def project(self, points):
        self.num_project_calls += 1
        self.projected_point_counts.append(np.asarray(points).shape[0])
        assert np.asarray(points).shape[0] == self.distances.shape[0]
        return self.distances, {"patch_id": self.patch_ids, "uv": self.uv}


def test_e175_assets_exist():
    assert E175_STEP.is_file()
    assert E175_MESH.is_file()
    assert preprocessing.DEFAULT_STEP_FILE.name == E175_STEP.name
    assert plotting.DEFAULT_MESH_FILE.name == E175_MESH.name


def test_create_components_selects_by_search_names_and_keys():
    geometry = FakeFunctionSet(
        {
            0: FakeFunction("fuselage"),
            1: FakeFunction("left_wing_lifting_surface"),
            2: FakeFunction("right_wing_lifting_surface"),
            3: FakeFunction("HT"),
            4: FakeFunction("underbelly_fairing"),
        },
        space="fake-space",
    )

    wing, htail, fairing = bsm3.preprocessing.create_components(
        keys=[None, [3], None],
        search_names=[("wing", "lifting_surface"), "ignored", ("fairing", "underbelly")],
        geometry=geometry,
    )

    assert isinstance(wing, FakeFunctionSet)
    assert list(wing.functions) == [1, 2]
    assert list(htail.functions) == [3]
    assert list(fairing.functions) == [4]


def test_create_components_flat_keys_replace_manual_function_set_construction(monkeypatch):
    created = []

    class FakeLfsModule:
        @staticmethod
        def FunctionSet(functions):
            created.append(dict(functions))
            return FakeFunctionSet(functions)

    monkeypatch.setitem(sys.modules, "lsdo_function_spaces", FakeLfsModule)

    geometry = FakeFunctionSet(
        {
            0: FakeFunction("fuselage_0"),
            1: FakeFunction("fuselage_1"),
            2: FakeFunction("wing"),
        }
    )

    component = bsm3.preprocessing.create_components(
        keys=np.arange(0, 2),
        geometry=geometry,
    )

    assert isinstance(component, FakeFunctionSet)
    assert list(component.functions) == [0, 1]
    assert created == [{0: geometry.functions[0], 1: geometry.functions[1]}]


def test_create_components_single_search_name_returns_single_component():
    geometry = FakeFunctionSet(
        {
            0: FakeFunction("fuselage"),
            1: FakeFunction("left_wing_lifting_surface"),
            2: FakeFunction("right_wing_lifting_surface"),
        }
    )

    component = bsm3.preprocessing.create_components(
        search_names="wing",
        geometry=geometry,
    )

    assert isinstance(component, FakeFunctionSet)
    assert list(component.functions) == [1, 2]


def test_create_components_matches_component_aliases_from_step_surface_names():
    geometry = FakeFunctionSet(
        {
            0: FakeFunction("Surf_XYDWLBTLUM, Fuselage, 0, 0"),
            1: FakeFunction("Surf_XYDWLBTLUM, Fuselage, 0, 1"),
            2: FakeFunction("Surf_TNCJHEAWPQ, Fuselage_Fairing, 0, 0"),
            3: FakeFunction("Surf_OTGQOZDGCY, Wing, 0, 1, lower"),
            4: FakeFunction("Surf_LZMFONIUSU, HT, 0, 1, lower"),
            5: FakeFunction("Surf_LZMFONIUSU, HT, 0, 4, upper"),
        }
    )

    fuselage, fairing, horizontal_tail = bsm3.preprocessing.create_components(
        search_names=["fuselage", "fairing", "horizontal_tail"],
        geometry=geometry,
    )

    assert list(fuselage.functions) == [0, 1]
    assert list(fairing.functions) == [2]
    assert list(horizontal_tail.functions) == [4, 5]


def test_create_components_reports_available_options_for_bad_search_name():
    geometry = FakeFunctionSet(
        {
            0: FakeFunction("Surf_XYDWLBTLUM, Fuselage, 0, 0"),
            1: FakeFunction("Surf_XYDWLBTLUM, Fuselage, 0, 1"),
            2: FakeFunction("Surf_TNCJHEAWPQ, Fuselage_Fairing, 0, 0"),
            3: FakeFunction("Surf_OTGQOZDGCY, Wing, 0, 1, lower"),
            4: FakeFunction("Surf_LZMFONIUSU, HT, 0, 1, lower"),
        }
    )

    with pytest.raises(ValueError, match="Available components") as exc_info:
        bsm3.preprocessing.create_components(search_names=["canard"], geometry=geometry)

    message = str(exc_info.value)
    assert "Fuselage (keys 0-1" in message
    assert "Fuselage_Fairing (keys 2" in message
    assert "Wing (keys 3" in message
    assert "HT (keys 4; aliases: horizontal_tail" in message
    assert "Surf_XYDWLBTLUM" not in message


def test_import_mesh_reads_e175_msh_arrays():
    mesh = bsm3.preprocessing.import_mesh(E175_MESH)

    assert mesh.vertices.shape == (16400, 3)
    assert mesh.connectivity.shape == (32522, 3)
    assert mesh.cell_blocks["triangle"].shape == (32522, 3)
    assert mesh.cell_types.shape == (32522,)
    assert set(mesh.cell_types) == {"triangle"}
    assert mesh.node_ids[0] == 1
    assert mesh.element_ids["triangle"][0] == 1
    np.testing.assert_allclose(mesh.vertices[0], E175_FIRST_VERTEX)
    np.testing.assert_array_equal(mesh.connectivity[0], [0, 2, 1])
    assert mesh.metadata["format"] == "msh"


def test_read_mesh_alias_and_ascii_stl_import(tmp_path):
    stl_path = tmp_path / "square.stl"
    stl_path.write_text(
        """solid square
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 1 1 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 1 0
    vertex 0 1 0
  endloop
endfacet
endsolid square
""",
        encoding="utf8",
    )

    mesh = bsm3.preprocessing.read_mesh(stl_path)

    assert mesh.vertices.shape == (4, 3)
    assert mesh.connectivity.shape == (2, 3)
    assert mesh.cell_blocks["triangle"].shape == (2, 3)
    assert set(mesh.cell_types) == {"triangle"}
    assert mesh.metadata["format"] == "stl"
    np.testing.assert_allclose(
        mesh.vertices,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )


def test_import_mesh_rejects_unsupported_format(tmp_path):
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("", encoding="utf8")

    with pytest.raises(ValueError, match="Unsupported mesh format"):
        bsm3.preprocessing.import_mesh(mesh_path)


def test_create_symmetric_mesh_mirrors_half_mesh_across_xz_plane():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        connectivity=np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
        cell_types=np.asarray(["triangle", "triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64)},
    )

    symmetric_mesh = bsm3.preprocessing.create_symmetric_mesh(mesh)

    assert symmetric_mesh.vertices.shape == (6, 3)
    assert symmetric_mesh.cell_blocks["triangle"].shape == (4, 3)
    assert np.count_nonzero(symmetric_mesh.vertices[:, 1] > 0.0) == 2
    assert np.count_nonzero(symmetric_mesh.vertices[:, 1] < 0.0) == 2
    assert np.count_nonzero(np.isclose(symmetric_mesh.vertices[:, 1], 0.0)) == 2
    assert symmetric_mesh.metadata["symmetric"] is True
    np.testing.assert_allclose(
        symmetric_mesh.vertices[-2:],
        [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0]],
    )


def test_create_symmetric_mesh_can_save_msh(tmp_path):
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        connectivity=np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
        cell_types=np.asarray(["triangle", "triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64)},
    )
    mesh_path = tmp_path / "quad_dominant_mesh.msh"

    symmetric_mesh = bsm3.preprocessing.create_symmetric_mesh(
        mesh,
        convert_to_quad_dominant=True,
        save_as=mesh_path,
    )
    saved_mesh = bsm3.preprocessing.import_mesh(mesh_path)

    assert mesh_path.is_file()
    np.testing.assert_allclose(saved_mesh.vertices, symmetric_mesh.vertices)
    assert saved_mesh.cell_blocks["quad"].shape == symmetric_mesh.cell_blocks["quad"].shape


def test_export_mesh_rejects_unsupported_format(tmp_path):
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray([[0.0, 0.0, 0.0]]),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
    )

    with pytest.raises(ValueError, match="Unsupported export mesh format"):
        bsm3.preprocessing.export_mesh(mesh, tmp_path / "mesh.obj")


def test_create_symmetric_mesh_selects_dominant_side_from_full_mesh():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        ),
        connectivity=np.asarray([[0, 2, 1], [1, 2, 3], [0, 1, 4]], dtype=np.int64),
        cell_types=np.asarray(["triangle", "triangle", "triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 2, 1], [1, 2, 3], [0, 1, 4]], dtype=np.int64)},
    )

    symmetric_mesh = bsm3.preprocessing.create_symmetric_mesh(mesh)

    assert symmetric_mesh.metadata["symmetry_selected_side"] == "positive"
    assert symmetric_mesh.cell_blocks["triangle"].shape == (4, 3)
    assert symmetric_mesh.vertices.shape == (6, 3)


def test_create_symmetric_mesh_can_make_limited_quad_dominant_mesh():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        connectivity=np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
        cell_types=np.asarray(["triangle", "triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64)},
    )

    symmetric_mesh = bsm3.preprocessing.create_symmetric_mesh(
        mesh,
        convert_to_quad_dominant=True,
        num_quads=1,
    )

    assert "quad" in symmetric_mesh.cell_blocks
    assert symmetric_mesh.cell_blocks["quad"].shape == (2, 4)
    assert "triangle" not in symmetric_mesh.cell_blocks
    assert symmetric_mesh.metadata["quad_dominant"] is True
    assert symmetric_mesh.metadata["num_quads_created"] == 1
    assert symmetric_mesh.metadata["premerge_edge_flips"] >= 0
    assert symmetric_mesh.metadata["quad_candidate_count"] >= 1
    assert symmetric_mesh.metadata["quad_matching_method"] in {"max_weight_matching", "greedy"}
    assert symmetric_mesh.metadata["quad_quality_gates"]["min_angle_deg"] == 35.0


def test_create_symmetric_mesh_quality_gates_control_quad_candidates():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.15, 1.0, 0.0],
                [1.15, 0.15, 0.0],
            ]
        ),
        connectivity=np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
        cell_types=np.asarray(["triangle", "triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64)},
    )

    conservative_mesh = bsm3.preprocessing.create_symmetric_mesh(
        mesh,
        convert_to_quad_dominant=True,
    )
    aggressive_mesh = bsm3.preprocessing.create_symmetric_mesh(
        mesh,
        convert_to_quad_dominant=True,
        quality_gates="aggressive",
    )

    assert conservative_mesh.metadata["num_quads_created"] == 0
    assert aggressive_mesh.metadata["num_quads_created"] == 1
    assert "quad" in aggressive_mesh.cell_blocks


def test_quad_quality_gates_support_overrides():
    gates = bsm3.preprocessing.quad_quality_gates(
        min_angle_deg=25.0,
        max_quality_score=0.4,
    )

    assert isinstance(gates, bsm3.preprocessing.QuadQualityGates)
    assert gates.min_angle_deg == 25.0
    assert gates.max_quality_score == 0.4


def test_create_symmetric_mesh_quad_conversion_rejects_sharp_fold():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        connectivity=np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64),
        cell_types=np.asarray(["triangle", "triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 2, 1], [1, 2, 3]], dtype=np.int64)},
    )

    symmetric_mesh = bsm3.preprocessing.create_symmetric_mesh(
        mesh,
        convert_to_quad_dominant=True,
    )

    assert "quad" not in symmetric_mesh.cell_blocks
    assert symmetric_mesh.cell_blocks["triangle"].shape == (4, 3)
    assert symmetric_mesh.metadata["num_quads_created"] == 0


def test_create_symmetric_mesh_rejects_num_quads_without_quad_conversion():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        connectivity=np.asarray([[0, 1, 2]], dtype=np.int64),
        cell_types=np.asarray(["triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 1, 2]], dtype=np.int64)},
    )

    with pytest.raises(ValueError, match="num_quads"):
        bsm3.preprocessing.create_symmetric_mesh(mesh, num_quads=1)


def test_identify_intersection_vertices_returns_driving_parametric_coordinates():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
    )
    wing = FakeProjectableComponent(
        distances=[0.0, 5.0e-5, 2.0e-4, 0.0],
        patch_ids=[10, 11, 12, 13],
        uv=[
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ],
    )
    fuselage = FakeProjectableComponent(distances=[0.0, 5.0e-5, 0.0, 2.0e-4])

    parametric_coordinates, vertices, vertex_ids = (
        bsm3.preprocessing.identify_intersection_vertices(
            components=[wing, fuselage],
            driving_component=wing,
            mesh=mesh,
            intersection_tolerance=1.0e-4,
        )
    )

    np.testing.assert_array_equal(vertex_ids, [0, 1])
    np.testing.assert_allclose(vertices, mesh.vertices[[0, 1]])
    np.testing.assert_allclose(
        parametric_coordinates,
        [
            [10.0, 0.1, 0.2],
            [11.0, 0.3, 0.4],
        ],
    )


def test_identify_intersection_vertices_supports_three_component_intersections_and_cache():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
    )
    wing = FakeProjectableComponent(
        distances=[0.0, 0.0, 0.0],
        patch_ids=[1, 1, 2],
        uv=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    fairing = FakeProjectableComponent(distances=[0.0, 2.0e-4, 0.0])
    fuselage = FakeProjectableComponent(distances=[2.0e-4, 0.0, 0.0])
    projection_cache = {}

    _, _, pair_vertex_ids = bsm3.preprocessing.identify_intersection_vertices(
        components=[wing, fairing],
        driving_component=wing,
        mesh=mesh,
        intersection_tolerance=1.0e-4,
        projection_cache=projection_cache,
    )
    _, _, triple_vertex_ids = bsm3.preprocessing.identify_intersection_vertices(
        components=[wing, fairing, fuselage],
        driving_component=wing,
        mesh=mesh,
        intersection_tolerance=1.0e-4,
        projection_cache=projection_cache,
    )

    np.testing.assert_array_equal(pair_vertex_ids, [0, 2])
    np.testing.assert_array_equal(triple_vertex_ids, [2])
    assert wing.num_project_calls == 1
    assert fairing.num_project_calls == 1
    assert fuselage.num_project_calls == 1


def test_identify_intersection_vertices_auto_projects_symmetric_half_mesh():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [2.0, -1.0, 0.0],
            ]
        ),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
        metadata={
            "symmetric": True,
            "symmetry_axis": 1,
            "symmetry_selected_side": "positive",
        },
    )
    wing = FakeProjectableComponent(
        distances=[0.0, 0.0, 2.0e-4],
        patch_ids=[3, 4, 5],
        uv=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    fuselage = FakeProjectableComponent(distances=[0.0, 5.0e-5, 0.0])

    parametric_coordinates, vertices, vertex_ids = (
        bsm3.preprocessing.identify_intersection_vertices(
            components=[wing, fuselage],
            driving_component=wing,
            mesh=mesh,
            intersection_tolerance=1.0e-4,
        )
    )

    assert wing.projected_point_counts == [3]
    assert fuselage.projected_point_counts == [3]
    np.testing.assert_array_equal(vertex_ids, [0, 1])
    np.testing.assert_allclose(vertices, mesh.vertices[[0, 1]])
    np.testing.assert_allclose(parametric_coordinates, [[3.0, 0.0, 0.0], [4.0, 0.5, 0.5]])


def test_identify_intersection_vertices_can_force_full_symmetric_mesh_projection():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        ),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
        metadata={
            "symmetric": True,
            "symmetry_axis": 1,
            "symmetry_selected_side": "positive",
        },
    )
    wing = FakeProjectableComponent(
        distances=[0.0, 0.0, 0.0],
        patch_ids=[1, 2, 3],
        uv=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    fuselage = FakeProjectableComponent(distances=[0.0, 0.0, 0.0])

    _, _, vertex_ids = bsm3.preprocessing.identify_intersection_vertices(
        components=[wing, fuselage],
        driving_component=wing,
        mesh=mesh,
        intersection_tolerance=1.0e-4,
        symmetry_mode="none",
    )

    assert wing.projected_point_counts == [3]
    np.testing.assert_array_equal(vertex_ids, [0, 1, 2])


def test_identify_intersection_vertices_requires_driving_component_in_components():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray([[0.0, 0.0, 0.0]]),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
    )
    wing = FakeProjectableComponent(distances=[0.0])
    fuselage = FakeProjectableComponent(distances=[0.0])

    with pytest.raises(ValueError, match="driving_component"):
        bsm3.preprocessing.identify_intersection_vertices(
            components=[fuselage, fuselage],
            driving_component=wing,
            mesh=mesh,
        )


def test_identify_intersection_vertices_uses_bsm3_projection_model_for_function_sets(monkeypatch):
    from bsm3.preprocessing import intersections

    class FakeFunctionSetLike:
        functions = {0: object()}

        def project(self, points):
            raise AssertionError("FunctionSet-like components must not use native project().")

    class FakeProjectionModel:
        patch_ids = (7,)

        def __init__(self, component, **kwargs):
            self.component = component
            self.kwargs = kwargs

        def project(self, coefficients, points):
            assert coefficients == "stacked"
            assert np.asarray(points).shape == (2, 3)
            return (
                np.asarray([0.0, 1.0e-5]),
                {
                    "patch_id": np.asarray([7, 7]),
                    "uv": np.asarray([[0.25, 0.5], [0.75, 0.5]]),
                },
            )

    monkeypatch.setattr(
        intersections,
        "_repo_projection_model_api",
        lambda: (FakeProjectionModel, lambda component, patch_ids: "stacked"),
    )
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
    )
    component = FakeFunctionSetLike()

    parametric_coordinates, _, vertex_ids = (
        bsm3.preprocessing.identify_intersection_vertices(
            components=[component, component],
            driving_component=component,
            mesh=mesh,
            intersection_tolerance=1.0e-4,
            projection_options={"warm_start_nu": 5},
        )
    )

    np.testing.assert_array_equal(vertex_ids, [0, 1])
    np.testing.assert_allclose(
        parametric_coordinates,
        [[7.0, 0.25, 0.5], [7.0, 0.75, 0.5]],
    )


def test_create_components_requires_matching_keys_and_search_names_lengths():
    geometry = FakeFunctionSet({0: FakeFunction("fuselage"), 1: FakeFunction("wing")})

    with pytest.raises(ValueError, match="same length"):
        bsm3.preprocessing.create_components(
            keys=[np.arange(1), np.arange(1, 2)],
            search_names=["wing"],
            geometry=geometry,
        )


def test_plot_components_broadcasts_styles_and_preserves_existing_elements():
    components = [
        FakeFunctionSet({0: FakeFunction("wing")}),
        FakeFunctionSet({1: FakeFunction("tail")}),
    ]
    seed = [{"mesh": "seed", "kwargs": {"color": "black"}}]

    elements = bsm3.plotting.plot_components(
        components=components,
        colors="blue",
        opacity=[1.0, 0.4],
        plotting_elements=seed,
        show=False,
    )

    assert elements[0] == seed[0]
    assert elements[1]["kwargs"] == {"color": "blue", "opacity": 1.0}
    assert elements[2]["kwargs"] == {"color": "blue", "opacity": 0.4}


def test_plot_components_validates_style_lengths():
    components = [FakeFunctionSet({0: FakeFunction("wing")}), FakeFunctionSet({1: FakeFunction("tail")})]

    with pytest.raises(ValueError, match="colors"):
        bsm3.plotting.plot_components(components=components, colors=["red", "green", "blue"])


def test_plot_mesh_reads_e175_mesh_without_showing():
    elements = bsm3.plotting.plot_mesh(mesh=E175_MESH, color="gray", opacity=0.25, show=False)

    assert len(elements) == 1
    assert elements[0]["mesh"].n_points == 16400
    assert elements[0]["mesh"].n_cells == 32522
    assert elements[0]["kwargs"]["color"] == "gray"
    assert elements[0]["kwargs"]["opacity"] == 0.25


def test_plot_mesh_accepts_preprocessing_mesh_data():
    mesh = bsm3.preprocessing.import_mesh(E175_MESH)

    elements = bsm3.plotting.plot_mesh(mesh=mesh, color="gray", opacity=0.25, show=False)

    assert len(elements) == 1
    assert elements[0]["mesh"].n_points == 16400
    assert elements[0]["mesh"].n_cells == 32522
    assert elements[0]["kwargs"]["color"] == "gray"
    assert elements[0]["kwargs"]["opacity"] == 0.25


def test_plot_mesh_flattens_nested_plotting_elements():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        connectivity=np.asarray([[0, 1, 2]], dtype=np.int64),
        cell_types=np.asarray(["triangle"], dtype=object),
        cell_blocks={"triangle": np.asarray([[0, 1, 2]], dtype=np.int64)},
    )
    first_highlight = bsm3.plotting.highlight_mesh_nodes(
        mesh,
        nodes_to_highlight=[[0.0, 0.0, 0.0]],
        show=False,
    )
    second_highlight = bsm3.plotting.highlight_mesh_nodes(
        mesh,
        nodes_to_highlight=[[1.0, 0.0, 0.0]],
        show=False,
    )

    elements = bsm3.plotting.plot_mesh(
        mesh=mesh,
        plotting_elements=[first_highlight, second_highlight],
        show=False,
    )

    assert len(elements) == 3
    assert all(isinstance(element, dict) for element in elements)


def test_highlight_mesh_nodes_uses_e175_mesh_and_one_based_node_ids():
    elements = bsm3.plotting.highlight_mesh_nodes(
        E175_MESH,
        nodes_to_highlight=[1, 2, 3],
        node_colore=["red", "green", "blue"],
        show=False,
    )

    assert len(elements) == 3
    np.testing.assert_allclose(elements[0]["mesh"].points[0], E175_FIRST_VERTEX)
    assert [element["kwargs"]["color"] for element in elements] == ["red", "green", "blue"]


def test_highlight_mesh_nodes_accepts_preprocessing_mesh_data():
    mesh = bsm3.preprocessing.import_mesh(E175_MESH)

    elements = bsm3.plotting.highlight_mesh_nodes(
        mesh,
        nodes_to_highlight=[1],
        node_colore="red",
        show=False,
    )

    assert len(elements) == 1
    np.testing.assert_allclose(elements[0]["mesh"].points[0], E175_FIRST_VERTEX)


def test_highlight_mesh_nodes_accepts_point_coordinates_with_negative_values():
    mesh = bsm3.preprocessing.MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, -2.0, 3.0],
            ]
        ),
        connectivity=np.empty((0, 0), dtype=np.int64),
        cell_types=np.empty((0,), dtype=object),
        cell_blocks={},
    )

    elements = bsm3.plotting.highlight_mesh_nodes(
        mesh,
        nodes_to_highlight=np.asarray([[1.0, -2.0, 3.0]]),
        node_color="red",
        show=False,
    )

    assert len(elements) == 1
    np.testing.assert_allclose(elements[0]["mesh"].points[0], [1.0, -2.0, 3.0])


# ---------------------------------------------------------------------------
# Safe polygon .npz surface format
# ---------------------------------------------------------------------------
# Interleaved widths so a width-sorted reader cannot pass by accident.
_NPZ_FACES = [
    [0, 1, 2],
    [1, 2, 3, 4, 5],
    [0, 2, 3],
    [2, 3, 4, 5],
    [0, 1, 3, 4, 5, 6],
    [3, 4, 5],
    [1, 2, 4, 5],
]


def _write_polygon_npz(path, faces=None, **overrides):
    """Write a safe polygon archive, optionally corrupting one field."""
    faces = _NPZ_FACES if faces is None else faces
    offsets = np.zeros(len(faces) + 1, dtype=np.int64)
    flat = []
    for index, face in enumerate(faces):
        ids = np.asarray(face, dtype=np.int64)
        flat.append(ids)
        offsets[index + 1] = offsets[index] + ids.size
    arrays = {
        "vertices": np.arange(21, dtype=np.float64).reshape(7, 3),
        "connectivity": (
            np.concatenate(flat).astype(np.int64)
            if flat
            else np.empty(0, dtype=np.int64)
        ),
        "offsets": offsets,
    }
    arrays.update(overrides)
    arrays = {k: v for k, v in arrays.items() if v is not None}
    np.savez_compressed(path, **arrays)
    return path


def test_import_mesh_reads_polygon_npz_in_original_face_order(tmp_path):
    """Preserve the archive's face order while grouping blocks by width."""
    mesh = preprocessing.import_mesh(_write_polygon_npz(tmp_path / "s.npz"))

    assert mesh.vertices.shape == (7, 3)
    assert mesh.metadata["reader"] == "bsm3_safe_npz"
    assert mesh.metadata["source"].endswith("s.npz")

    # Original order, not width-sorted.
    assert list(mesh.cell_types) == [
        "triangle",
        "polygon5",
        "triangle",
        "quad",
        "polygon6",
        "triangle",
        "quad",
    ]
    assert mesh.connectivity.shape == (7,)
    for index, face in enumerate(_NPZ_FACES):
        np.testing.assert_array_equal(
            mesh.connectivity[index], np.asarray(face, dtype=np.int64)
        )

    # Blocks regroup the same faces in ascending width order.
    assert list(mesh.cell_blocks) == ["triangle", "quad", "polygon5", "polygon6"]
    np.testing.assert_array_equal(
        mesh.cell_blocks["triangle"], [[0, 1, 2], [0, 2, 3], [3, 4, 5]]
    )
    np.testing.assert_array_equal(
        mesh.cell_blocks["quad"], [[2, 3, 4, 5], [1, 2, 4, 5]]
    )
    np.testing.assert_array_equal(mesh.cell_blocks["polygon5"], [[1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(
        mesh.cell_blocks["polygon6"], [[0, 1, 3, 4, 5, 6]]
    )


def test_import_mesh_polygon_npz_uniform_width_is_two_dimensional(tmp_path):
    """Return a 2-D connectivity when every face has the same width."""
    path = _write_polygon_npz(tmp_path / "u.npz", faces=[[0, 1, 2], [2, 3, 4]])
    mesh = preprocessing.import_mesh(path)

    assert mesh.connectivity.shape == (2, 3)
    assert mesh.connectivity.dtype == np.int64
    assert list(mesh.cell_types) == ["triangle", "triangle"]
    assert list(mesh.cell_blocks) == ["triangle"]


def test_import_mesh_polygon_npz_refuses_object_arrays(tmp_path):
    """Reject an archive that can only be decoded by executing pickle."""
    path = tmp_path / "object.npz"
    np.savez_compressed(
        path,
        vertices=np.zeros((3, 3), dtype=np.float64),
        connectivity=np.array([[0, 1, 2], [0, 1]], dtype=object),
        offsets=np.array([0, 3], dtype=np.int64),
    )
    with pytest.raises(ValueError):
        preprocessing.import_mesh(path)

    # The reader must not fall back to an allow_pickle=True load.
    with pytest.raises(ValueError):
        np.load(path, allow_pickle=False)["connectivity"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"extra": np.zeros(1)}, "exactly"),
        ({"vertices": None}, "exactly"),
        ({"vertices": np.zeros((7, 2), dtype=np.float64)}, "shape"),
        (
            {"vertices": np.full((7, 3), np.nan, dtype=np.float64)},
            "non-finite",
        ),
        ({"vertices": np.zeros((7, 3), dtype=np.str_)}, "numeric"),
        ({"connectivity": np.zeros((2, 3), dtype=np.int64)}, "flat array"),
        ({"offsets": np.array([1, 4], dtype=np.int64)}, "start at 0"),
        ({"offsets": np.array([0, 3], dtype=np.int64)}, "end at the connectivity"),
    ],
)
def test_import_mesh_polygon_npz_rejects_malformed_archives(
    tmp_path, overrides, message
):
    """Give a clear ValueError for each representative schema violation."""
    path = _write_polygon_npz(tmp_path / "bad.npz", **overrides)
    with pytest.raises(ValueError, match=message):
        preprocessing.import_mesh(path)


def test_import_mesh_polygon_npz_rejects_short_faces_and_bad_node_ids(tmp_path):
    """Reject a two-node face span and an out-of-range node ID."""
    short = _write_polygon_npz(
        tmp_path / "short.npz", faces=[[0, 1, 2], [3, 4]]
    )
    with pytest.raises(ValueError, match="fewer than 3 nodes"):
        preprocessing.import_mesh(short)

    out_of_range = _write_polygon_npz(
        tmp_path / "range.npz", faces=[[0, 1, 2], [3, 4, 99]]
    )
    with pytest.raises(ValueError, match="node ID outside"):
        preprocessing.import_mesh(out_of_range)


def test_import_trusted_polygon_pickle_remains_an_opt_in_boundary(tmp_path):
    """Keep the explicit trusted-pickle compatibility path working."""
    import pickle

    path = tmp_path / "trusted.pkl"
    with path.open("wb") as stream:
        pickle.dump(
            {
                "points": np.arange(15, dtype=float).reshape(5, 3),
                "connectivity": [[0, 1, 2], [1, 2, 3, 4]],
            },
            stream,
        )

    mesh = preprocessing.import_trusted_polygon_pickle(path)
    assert mesh.vertices.shape == (5, 3)
    assert set(mesh.cell_blocks) == {"triangle", "quad"}
    assert mesh.metadata["reader"] == "bsm3_polygon_pickle"

    # It is reachable only on purpose: suffix dispatch still refuses pickles.
    with pytest.raises(ValueError, match="Unsupported mesh format"):
        preprocessing.import_mesh(path)


def test_plot_components_none_colors_preserves_component_colors():
    """Treat ``colors=None`` like the empty-string no-override sentinel."""

    class _Component:
        def __init__(self):
            self.kwargs = None

        def plot(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["additional_plotting_elements"]

    first, second = _Component(), _Component()
    elements = plotting.plot_components([first, second], colors=None)

    assert elements == []
    # No color keyword means each component keeps its own color.
    assert "color" not in first.kwargs
    assert "color" not in second.kwargs
    assert first.kwargs["opacity"] == 1.0
    assert first.kwargs["show"] is False

    # The empty string behaves identically.
    empty = _Component()
    plotting.plot_components([empty], colors="")
    assert "color" not in empty.kwargs

    # An explicit color is still forwarded.
    explicit = _Component()
    plotting.plot_components([explicit], colors="red")
    assert explicit.kwargs["color"] == "red"

    # Per-component lists and their length rule are unchanged.
    left, right = _Component(), _Component()
    plotting.plot_components([left, right], colors=["red", "blue"])
    assert (left.kwargs["color"], right.kwargs["color"]) == ("red", "blue")
    with pytest.raises(ValueError, match="colors"):
        plotting.plot_components([left, right], colors=["red", "blue", "green"])
