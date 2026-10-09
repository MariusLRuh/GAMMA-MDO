from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import csdl_alpha as csdl
import pytest

from gamma_mdo.core.boundary_surface_movement.mesh_motion_config import (
    InputFiles,
    PolygonRegularization,
    MeshMotion,
    SurfaceMotion,
    VolumeMotion,
)


PACKAGE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "gamma_mdo"
    / "core"
    / "boundary_surface_movement"
)


def test_public_e175_driver_does_not_use_cli_configuration():
    source = (
        PACKAGE_DIRECTORY / "cfd_mesh_movement_test.py"
    ).read_text(encoding="utf-8")
    assert "argparse" not in source
    assert "parse_args(" not in source
    assert "os.environ" not in source


def test_default_mesh_motion_is_surface_only_and_headless():
    """The safe default runs surface-only, headless, and writes nothing."""
    config = MeshMotion()
    assert config.surface.distance_weighting.enabled
    # Surface-only and headless by default; the DAFoam driver opts in to
    # volume motion and CFD diagnostics explicitly.
    assert config.volume.mode == "off"
    assert config.volume.write_meshes is False
    assert config.quality.surface is True
    assert config.quality.volume is False
    assert config.quality.gmsh_volume_metrics is False
    assert config.visualization.enabled is False
    assert config.derivative_check.enabled is False




def test_public_driver_exposes_explicit_matching_model_files():
    from gamma_mdo.core.boundary_surface_movement import cfd_mesh_movement_test

    movement_files = cfd_mesh_movement_test.MODEL_FILES
    assert isinstance(movement_files, InputFiles)
    assert movement_files.geometry_file.name == "e175.stp"
    assert movement_files.surface_mesh_file.name == "e175_r5_wall.msh"
    assert movement_files.volume_mesh_file.name == "e175_r5_volume.msh"
    assert (
        movement_files.volume_wall_map_file.name
        == "e175_r5_wall.volume_map.npz"
    )
    assert cfd_mesh_movement_test.MESH_MOTION.volume.mode == "off"
    assert (
        cfd_mesh_movement_test.MESH_MOTION.volume.load_mode
        == "synchronized"
    )
    # The wall extracted from the volume mesh is already the y >= 0 half.
    assert cfd_mesh_movement_test.MESH_MOTION.symmetry
@pytest.mark.integration
def test_driver_mesh_assets_exist_when_available():
    """Check the downloaded R5 driver meshes without requiring them in CI."""
    from gamma_mdo.core.boundary_surface_movement import cfd_mesh_movement_test

    model_files = cfd_mesh_movement_test.MODEL_FILES
    local_assets = (
        model_files.geometry_file,
        model_files.surface_mesh_file,
        model_files.volume_mesh_file,
        model_files.volume_wall_map_file,
    )
    missing = [path for path in local_assets if not path.is_file()]
    if missing:
        pytest.skip(
            "R5 driver meshes are unavailable: "
            + ", ".join(path.name for path in missing)
        )
    assert all(path.is_file() for path in local_assets)


def test_quad_diagonal_controls_are_validated():
    with pytest.raises(ValueError, match="finite and non-negative"):
        SurfaceMotion(quad_diagonal_weight=-0.1)
    with pytest.raises(ValueError, match="quad_bracing_mode"):
        SurfaceMotion(quad_bracing_mode="unknown")


def test_ngon_affine_weight_is_validated():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PolygonRegularization(weight=-0.1)


def test_invalid_synchronized_name_is_rejected():
    with pytest.raises(ValueError, match="final or synchronized"):
        VolumeMotion(load_mode="last")


def test_independent_synchronized_volume_steps_are_validated():
    config = VolumeMotion(
        load_mode="synchronized", synchronized_load_steps=5
    )
    assert config.synchronized_load_steps == 5
    with pytest.raises(ValueError, match="must be positive"):
        VolumeMotion(
            load_mode="synchronized", synchronized_load_steps=0
        )
    with pytest.raises(ValueError, match="requires synchronized mode"):
        VolumeMotion(load_mode="final", synchronized_load_steps=5)








