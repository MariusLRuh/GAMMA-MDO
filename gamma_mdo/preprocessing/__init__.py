"""Public preprocessing API.

The implementation is split across focused modules in this package while this
file keeps the stable user-facing import surface.
"""

from __future__ import annotations

from .components import DEFAULT_STEP_FILE, available_components, create_components
from .intersections import identify_intersection_vertices
from .mesh_io import (
    MeshData,
    export_mesh,
    import_mesh,
    import_trusted_polygon_pickle,
    read_mesh,
)
from .movement import (
    ComponentProjectionData,
    ProjectionMetadata,
    VertexEvaluationMetadata,
    get_projection_metadata,
    identify_deformation_vertices,
    identify_reevaluated_vertices,
    identify_vertices_by_components,
    project_mesh_onto_geometry,
    project_mesh_onto_components,
)
from .quad_conversion import QuadQualityGates, quad_quality_gates
from .symmetry import (
    SymmetrySplit,
    create_symmetric_mesh,
    detect_symmetry,
    reconstruct_full_from_half,
    split_symmetric_mesh,
)

__all__ = [
    "DEFAULT_STEP_FILE",
    "ComponentProjectionData",
    "MeshData",
    "ProjectionMetadata",
    "QuadQualityGates",
    "SymmetrySplit",
    "VertexEvaluationMetadata",
    "available_components",
    "create_components",
    "create_symmetric_mesh",
    "detect_symmetry",
    "reconstruct_full_from_half",
    "split_symmetric_mesh",
    "export_mesh",
    "get_projection_metadata",
    "identify_deformation_vertices",
    "identify_intersection_vertices",
    "identify_reevaluated_vertices",
    "identify_vertices_by_components",
    "import_mesh",
    "import_trusted_polygon_pickle",
    "quad_quality_gates",
    "read_mesh",
    "project_mesh_onto_geometry",
    "project_mesh_onto_components",
]
