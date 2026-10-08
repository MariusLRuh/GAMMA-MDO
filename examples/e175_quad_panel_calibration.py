"""Run the E175 deformation on the curated quad-dominant panel mesh.

This advanced example keeps the geometry setup from the basic triangle
example, adds n-gon regularization, and inspects which surface vertices were
graph-moved, parametrically reevaluated, or solved as exact intersections.
The regularization weight remains an explicit calibration input; the value
below is the historical placeholder, not a universal recommendation.

Run it directly::

    python examples/e175_quad_panel_calibration.py
"""

from pathlib import Path
import tempfile

from e175_surface_deformation import ASSETS, main as deform_e175

QUAD_SURFACE_MESH_FILE = (
    ASSETS / "e175_quad_panel.msh"
)
CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "bsm3_e175_quad_panel_cache"


def _print_vertex_roles(result) -> None:
    """Print the public surface classification and projection status."""
    roles = result.surface_vertex_classification
    status = result.surface_projection_status
    print("\nsurface vertex roles")
    print(f"  graph free              : {roles.graph_free_vertex_ids.size}")
    print(
        "  graph prescribed        : "
        f"{roles.graph_prescribed_vertex_ids.size}"
    )
    print(
        "  parametrically prescribed: "
        f"{roles.parametrically_prescribed_vertex_ids.size}"
    )
    for name, vertex_ids in roles.intersection_vertex_ids.items():
        print(f"  exact intersection {name:<8}: {vertex_ids.size}")
    print(f"  closest-point projected : {status.num_reprojected}")
    print(f"  projection non-converged: {status.num_nonconverged}")


def main(
    *,
    geometry_file: Path = ASSETS / "e175.stp",
    surface_mesh_file: Path = QUAD_SURFACE_MESH_FILE,
    cache_directory: Path = CACHE_DIRECTORY,
    deformation_scale: float = 1.0,
    polygon_regularization_weight: float = 0.3,
    check_derivatives: bool = False,
    visualize: bool = False,
):
    """Deform the clean panel mesh and report its vertex roles.

    Parameters are ordinary Python values so calibration studies can call this
    function repeatedly with different n-gon weights. ``visualize=True`` opens
    the final deformed mesh; the returned classification supplies the global
    IDs needed to color individual roles in a custom plotting workflow.
    """
    result = deform_e175(
        geometry_file=geometry_file,
        surface_mesh_file=surface_mesh_file,
        cache_directory=cache_directory,
        deformation_scale=deformation_scale,
        polygon_regularization_weight=polygon_regularization_weight,
        check_derivatives=check_derivatives,
        visualize=visualize,
    )
    _print_vertex_roles(result)
    return result


if __name__ == "__main__":
    main()
