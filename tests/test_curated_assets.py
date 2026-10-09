"""Validation of the small E175 asset set shipped with the package.

The mixed N-gon ``wall_surface.npz`` is a local-only research asset; its tests
skip when it is absent.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from gamma_mdo.core.boundary_surface_movement import NgonAffineAssembler
from gamma_mdo.preprocessing import import_mesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIRECTORY = (
    REPOSITORY_ROOT / "gamma_mdo" / "core" / "boundary_surface_movement"
)
STEP_FILE = ASSET_DIRECTORY / "e175.stp"
R1_WALL_FILE = ASSET_DIRECTORY / "e175_r1_wall.msh"
R1_WALL_MAP_FILE = ASSET_DIRECTORY / "e175_r1_wall.volume_map.npz"
QUAD_PANEL_FILE = (
    ASSET_DIRECTORY / "e175_quad_panel.msh"
)
QUAD_PANEL_SHA256 = (
    "92feeeda05905a13d23a18c863e76b9596773beccb021148cc2d4e7016cd733c"
)
MIXED_NGON_FILE = ASSET_DIRECTORY / "wall_surface.npz"

# Decoded-array evidence for the safe curated wall surface, measured from the
# trusted legacy pickle before it was removed. These pin the exact arrays and
# the original face order, not merely the aggregate topology.
MIXED_NGON_ARRAY_SHA256 = {
    "vertices": "76ceaadd74a93eeb4153e51b784a4d4ff6dfa7079382510652cf8070ed3aad1a",
    "connectivity": "971e53f7bc46fd694955be2f00b1b4355e513d5e066c5ebe1499b72a463b4ff3",
    "offsets": "eff2e0fe7cd471ab5bac922b5290a0560db4d8d72d4fc3b2917ff70582083509",
}
MIXED_NGON_FACE_COUNTS = {3: 5, 4: 565, 5: 7891, 6: 28190, 7: 3927, 8: 126, 9: 2}
MIXED_NGON_WIDTH_TRANSITIONS = 14720


def _array_sha256(array: np.ndarray) -> str:
    """Hash an array's raw bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
CURATED_ASSETS = (
    STEP_FILE,
    R1_WALL_FILE,
    R1_WALL_MAP_FILE,
    QUAD_PANEL_FILE,
)
requires_curated_assets = pytest.mark.skipif(
    not all(path.is_file() for path in CURATED_ASSETS),
    reason="curated E175 assets are not available in this checkout",
)
requires_mixed_ngon = pytest.mark.skipif(
    not MIXED_NGON_FILE.is_file(),
    reason="the local-only mixed N-gon wall surface is not available",
)


def test_generic_mesh_import_does_not_dispatch_pickle():
    """Require callers to opt in explicitly before executing pickle data."""
    with pytest.raises(ValueError, match="Unsupported mesh format"):
        import_mesh(Path("untrusted.pkl"))
    with pytest.raises(ValueError, match="Unsupported mesh format"):
        import_mesh(Path("untrusted.pickle"))


@requires_mixed_ngon
def test_curated_wall_surface_is_a_safe_npz_archive():
    """Pin the decoded arrays of the curated wall surface, loaded without pickle."""
    with np.load(MIXED_NGON_FILE, allow_pickle=False) as archive:
        assert sorted(archive.files) == ["connectivity", "offsets", "vertices"]
        vertices = archive["vertices"]
        connectivity = archive["connectivity"]
        offsets = archive["offsets"]

    assert vertices.shape == (79207, 3)
    assert vertices.dtype == np.float64
    assert connectivity.shape == (239385,)
    assert connectivity.dtype == np.int64
    assert offsets.shape == (40707,)
    assert offsets.dtype == np.int64

    assert _array_sha256(vertices) == MIXED_NGON_ARRAY_SHA256["vertices"]
    assert _array_sha256(connectivity) == MIXED_NGON_ARRAY_SHA256["connectivity"]
    assert _array_sha256(offsets) == MIXED_NGON_ARRAY_SHA256["offsets"]

    assert offsets[0] == 0
    assert offsets[-1] == connectivity.size
    assert int(connectivity.min()) == 0
    assert int(connectivity.max()) == 79206

    widths = np.diff(offsets)
    counts = {int(w): int(c) for w, c in zip(*np.unique(widths, return_counts=True))}
    assert counts == MIXED_NGON_FACE_COUNTS
    assert widths.size == 40706
    # Source-order evidence: a width-sorted archive would have only 6 transitions.
    assert int(np.count_nonzero(widths[1:] != widths[:-1])) == (
        MIXED_NGON_WIDTH_TRANSITIONS
    )


@requires_mixed_ngon
def test_curated_wall_surface_imports_in_original_face_order():
    """Check generic import preserves face order while grouping blocks by width."""
    mesh = import_mesh(MIXED_NGON_FILE)
    assert mesh.metadata["reader"] == "gamma_mdo_safe_npz"

    with np.load(MIXED_NGON_FILE, allow_pickle=False) as archive:
        connectivity = archive["connectivity"]
        offsets = archive["offsets"]

    assert mesh.connectivity.shape == (40706,)
    assert mesh.cell_types.shape == (40706,)

    widths = np.diff(offsets)
    expected_types = [
        {3: "triangle", 4: "quad"}.get(int(w), f"polygon{int(w)}") for w in widths
    ]
    assert list(mesh.cell_types) == expected_types

    # Every face, in the archive's own order.
    for index in (0, 1, 2, 17, 4096, 20003, 40704, 40705):
        expected = connectivity[offsets[index]:offsets[index + 1]]
        np.testing.assert_array_equal(mesh.connectivity[index], expected)
    assert all(
        np.array_equal(
            mesh.connectivity[i], connectivity[offsets[i]:offsets[i + 1]]
        )
        for i in range(widths.size)
    )

    # Blocks regroup the same faces by ascending width.
    assert list(mesh.cell_blocks) == [
        "triangle",
        "quad",
        "polygon5",
        "polygon6",
        "polygon7",
        "polygon8",
        "polygon9",
    ]
    assert {
        int(block.shape[1]): int(block.shape[0])
        for block in mesh.cell_blocks.values()
    } == MIXED_NGON_FACE_COUNTS


@pytest.mark.integration
@requires_curated_assets
def test_curated_e175_surface_assets_load_with_expected_topology():
    """Load each curated surface mesh and verify its distinguishing topology."""
    assert STEP_FILE.is_file()

    triangle_wall = import_mesh(R1_WALL_FILE)
    assert triangle_wall.vertices.shape == (16400, 3)
    assert triangle_wall.cell_blocks["triangle"].shape == (32522, 3)

    quad_panel = import_mesh(QUAD_PANEL_FILE)
    assert hashlib.sha256(QUAD_PANEL_FILE.read_bytes()).hexdigest() == (
        QUAD_PANEL_SHA256
    )
    assert quad_panel.vertices.shape == (13262, 3)
    assert quad_panel.cell_blocks["triangle"].shape == (2804, 3)
    assert quad_panel.cell_blocks["quad"].shape == (11858, 4)


@pytest.mark.integration
@requires_mixed_ngon
def test_curated_mixed_surface_loads_with_expected_topology():
    """Load the local-only mixed N-gon wall surface."""
    mixed_ngon = import_mesh(MIXED_NGON_FILE)
    assert mixed_ngon.vertices.shape == (79207, 3)
    assert mixed_ngon.cell_blocks["polygon6"].shape == (28190, 6)
    assert {"quad", "polygon5", "polygon6", "polygon7"} <= set(
        mixed_ngon.cell_blocks
    )


@pytest.mark.integration
@requires_curated_assets
def test_curated_r1_wall_map_matches_the_triangle_wall():
    """Check the retained R1 wall-to-volume metadata against the wall mesh."""
    wall = import_mesh(R1_WALL_FILE)
    with np.load(R1_WALL_MAP_FILE) as mapping:
        wall_to_volume = np.asarray(mapping["wall_to_volume"], dtype=np.int64)
        baseline_wall = np.asarray(mapping["baseline_wall_vertices"], dtype=float)
        triangles = np.asarray(mapping["surface_triangles"], dtype=np.int64)
        stored_paths = (str(mapping["volume_mesh_path"]), str(mapping["wall_mesh_path"]))

    # Published maps name their meshes, not a local checkout path.
    assert stored_paths == ("e175_r1_volume.msh", "e175_r1_wall.msh")

    assert wall_to_volume.shape == (wall.vertices.shape[0],)
    np.testing.assert_allclose(baseline_wall, wall.vertices, atol=0.0)
    np.testing.assert_array_equal(triangles, wall.cell_blocks["triangle"])


@pytest.mark.integration
@requires_mixed_ngon
def test_curated_mixed_surface_has_the_expected_hourglass_mode_count():
    """Assemble, but do not solve, every trusted wall-surface N-gon mode."""
    mixed_ngon = import_mesh(MIXED_NGON_FILE)
    expected_modes = sum(
        (block.shape[1] - 3) * block.shape[0]
        for block in mixed_ngon.cell_blocks.values()
        if block.shape[1] >= 4
    )
    assert expected_modes == 117267

    system = NgonAffineAssembler().assemble(
        mixed_ngon,
        free_ids=np.arange(mixed_ngon.vertices.shape[0], dtype=np.int64),
        prescribed_ids=np.empty(0, dtype=np.int64),
    )
    assert system.num_hourglass_modes == expected_modes
