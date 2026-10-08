"""Symmetry-plane mesh construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh_io import MeshData, _as_mesh_data, _make_mesh_data, export_mesh
from .quad_conversion import QuadQualityGates, _convert_triangles_to_quads, quad_quality_gates


@dataclass(frozen=True)
class SymmetrySplit:
    """Half mesh plus the maps to reconstruct the mirror-symmetric full mesh.

    ``half_mesh`` is the kept side (its vertices re-indexed 0..n_half-1).  Run
    the whole deformation pipeline on it, producing ``n_half`` deformed
    vertices, then call :func:`reconstruct_full_from_half` to get the full
    ``n_full`` deformed vertices (differentiable).

    * ``half_to_full`` -- full vertex id for each half-local row.
    * ``full_to_half`` -- half-local row for each full vertex (-1 if mirrored).
    * ``plane_local_ids`` -- half-local rows lying on the symmetry plane.
    * ``gather_index`` / ``mirror_sign`` -- for each full vertex, the half-local
      row it is built from and the per-axis sign (``-1`` on ``axis`` for the
      mirrored side).
    """

    half_mesh: MeshData
    half_to_full: np.ndarray
    full_to_half: np.ndarray
    plane_local_ids: np.ndarray
    gather_index: np.ndarray
    mirror_sign: np.ndarray
    axis: int
    num_full_vertices: int


def detect_symmetry(mesh, *, axis: int = 1, tol: float = 1e-6) -> bool:
    """Return whether the mesh is mirror-symmetric about ``coord[axis] = 0``.

    True when every off-plane vertex has a mirror partner within ``tol`` and no
    cell straddles the plane (a clean node seam on the plane).
    """
    mesh_data = _as_mesh_data(mesh)
    vertices = np.asarray(mesh_data.vertices, dtype=float)
    if vertices.shape[0] == 0:
        return False
    if not _has_mirror_partners(vertices, axis=axis, tol=tol):
        return False
    plane = np.abs(vertices[:, axis]) <= tol
    for cells in mesh_data.cell_blocks.values():
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.size == 0:
            continue
        cell_axis = vertices[cell_array, axis]
        if np.any(np.any(cell_axis < -tol, axis=1) & np.any(cell_axis > tol, axis=1)):
            return False
    # need vertices on both sides (otherwise it is already a half mesh)
    return bool(np.any(vertices[:, axis] > tol) and np.any(vertices[:, axis] < -tol))


def split_symmetric_mesh(
    mesh, *, axis: int = 1, tol: float = 1e-6, keep_side: str = "auto"
) -> SymmetrySplit:
    """Split a mirror-symmetric mesh into a half plus reconstruction maps."""
    mesh_data = _as_mesh_data(mesh)
    vertices = np.asarray(mesh_data.vertices, dtype=float)
    num_full = vertices.shape[0]

    if keep_side == "auto":
        side = _select_symmetry_side(mesh_data, tol=tol)
    elif keep_side in ("positive", "negative"):
        side = keep_side
    else:
        raise ValueError("keep_side must be 'auto', 'positive', or 'negative'.")
    sign = 1.0 if side == "positive" else -1.0

    keep_mask = sign * vertices[:, axis] >= -tol  # kept side plus the plane
    half_to_full = np.where(keep_mask)[0].astype(np.int64)
    full_to_half = np.full(num_full, -1, dtype=np.int64)
    full_to_half[half_to_full] = np.arange(half_to_full.size, dtype=np.int64)

    half_blocks: dict[str, np.ndarray] = {}
    for cell_type, cells in mesh_data.cell_blocks.items():
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.size == 0:
            half_blocks[cell_type] = cell_array.reshape((0, cell_array.shape[-1] if cell_array.ndim == 2 else 0))
            continue
        cell_axis = vertices[cell_array, axis]
        if np.any(np.any(cell_axis < -tol, axis=1) & np.any(cell_axis > tol, axis=1)):
            raise ValueError("Mesh has cells that cross the symmetry plane; cannot split.")
        keep = np.all(keep_mask[cell_array], axis=1)
        half_blocks[cell_type] = full_to_half[cell_array[keep]]

    half_mesh = _make_mesh_data(
        vertices=vertices[half_to_full],
        cell_blocks=half_blocks,
        node_ids=None,
        element_ids=None,
        element_tags=None,
        metadata={**(mesh_data.metadata or {}), "symmetry_half_side": side},
    )

    plane_full = np.where(np.abs(vertices[:, axis]) <= tol)[0].astype(np.int64)
    plane_local_ids = full_to_half[plane_full]
    plane_local_ids = np.sort(plane_local_ids[plane_local_ids >= 0])

    # Reconstruction: kept vertices map to themselves; mirrored vertices map to
    # their mirror partner on the kept side with the axis coordinate negated.
    gather_index = np.empty(num_full, dtype=np.int64)
    mirror_sign = np.ones((num_full, 3), dtype=float)
    partner = _mirror_partner_index(vertices, axis=axis, tol=tol)
    for full_id in range(num_full):
        if keep_mask[full_id]:
            gather_index[full_id] = full_to_half[full_id]
        else:
            mirror_full = int(partner[full_id])
            if mirror_full < 0 or full_to_half[mirror_full] < 0:
                raise ValueError(
                    f"Vertex {full_id} on the mirrored side has no kept-side partner."
                )
            gather_index[full_id] = full_to_half[mirror_full]
            mirror_sign[full_id, axis] = -1.0

    # Plane vertices belong exactly on the symmetry plane; pin their axis
    # coordinate to 0 so the reconstructed mesh is exactly mirror-symmetric
    # (the projection back-end otherwise leaves the centerline ~sub-mm off).
    # For a symmetric deformation the plane's axis coordinate is a constant 0,
    # so this is consistent with the true derivative.
    mirror_sign[plane_full, axis] = 0.0

    return SymmetrySplit(
        half_mesh=half_mesh,
        half_to_full=half_to_full,
        full_to_half=full_to_half,
        plane_local_ids=plane_local_ids,
        gather_index=gather_index,
        mirror_sign=mirror_sign,
        axis=axis,
        num_full_vertices=num_full,
    )


def reconstruct_full_from_half(half_values, split: SymmetrySplit):
    """Mirror ``n_half`` deformed vertices into ``n_full`` (differentiable).

    ``half_values`` is a CSDL ``(n_half, 3)`` variable of deformed positions on
    the kept half; the result is a CSDL ``(n_full, 3)`` variable in the original
    full-mesh vertex order.
    """
    import scipy.sparse as sp

    import csdl_alpha as csdl

    num_full = split.num_full_vertices
    num_half = int(split.half_to_full.size)
    selection = sp.csr_matrix(
        (
            np.ones(num_full, dtype=float),
            (np.arange(num_full), split.gather_index),
        ),
        shape=(num_full, num_half),
    )
    gathered = csdl.sparse.matmat(selection, half_values)
    return gathered * split.mirror_sign


def _has_mirror_partners(vertices: np.ndarray, *, axis: int, tol: float) -> bool:
    partner = _mirror_partner_index(vertices, axis=axis, tol=tol)
    plane = np.abs(vertices[:, axis]) <= tol
    return bool(np.all(plane | (partner >= 0)))


def _mirror_partner_index(vertices: np.ndarray, *, axis: int, tol: float) -> np.ndarray:
    """Index of each vertex's mirror image across ``coord[axis]=0`` (-1 if none)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(vertices)
    mirrored = vertices.copy()
    mirrored[:, axis] *= -1.0
    distances, indices = tree.query(mirrored, k=1)
    return np.where(distances <= tol, indices, -1).astype(np.int64)

def create_symmetric_mesh(
    mesh,
    convert_to_quad_dominant: bool = False,
    num_quads: int | None = None,
    quality_gates: QuadQualityGates | dict[str, float | int] | str | None = None,
    save_as: str | Path | None = None,
) -> MeshData:
    """Mirror a surface mesh across the x-z symmetry plane.

    Parameters
    ----------
    mesh : MeshData, mesh-like, str, or Path
        Input mesh data or path to a supported mesh file.
    convert_to_quad_dominant : bool, optional
        If ``True``, merge high-quality adjacent triangle pairs into quads
        before mirroring.
    num_quads : int, optional
        Maximum number of triangle pairs to merge on the selected half mesh.
        This is an upper bound, not a target; quality gates determine how many
        candidates are eligible.
    quality_gates : QuadQualityGates, dict, str, or None, optional
        Quad conversion thresholds. Strings select presets from
        :func:`quad_quality_gates`.
    save_as : str or Path, optional
        If provided, write the returned symmetric mesh to this path.

    Returns
    -------
    MeshData
        Symmetric mesh with shared vertices on the ``y=0`` plane.

    Notes
    -----
    The x-z symmetry plane is ``y=0``. If the input has cells on both sides of
    the plane, the side with more cells is kept and mirrored. If the input is
    already a half mesh, that half is mirrored.
    """
    mesh_data = _as_mesh_data(mesh)
    if num_quads is not None and num_quads < 0:
        raise ValueError("num_quads must be non-negative or None.")

    gates = _normalize_quad_quality_gates(quality_gates)
    half_mesh = _extract_symmetry_half_mesh(mesh_data, tol=1e-8)
    if convert_to_quad_dominant:
        half_mesh = _convert_triangles_to_quads(
            half_mesh,
            max_num_quads=num_quads,
            quality_gates=gates,
        )
    elif num_quads is not None or quality_gates is not None:
        raise ValueError(
            "num_quads and quality_gates can only be provided when "
            "convert_to_quad_dominant=True."
        )

    symmetric_mesh = _mirror_mesh_data_across_xz_plane(half_mesh, tol=1e-8)
    if save_as is not None:
        export_mesh(symmetric_mesh, save_as)
    return symmetric_mesh


def _normalize_quad_quality_gates(
    quality_gates: QuadQualityGates | dict[str, float | int] | str | None,
) -> QuadQualityGates:
    if quality_gates is None:
        return quad_quality_gates()
    if isinstance(quality_gates, QuadQualityGates):
        return quality_gates
    if isinstance(quality_gates, str):
        return quad_quality_gates(quality_gates)
    if isinstance(quality_gates, dict):
        return quad_quality_gates(**quality_gates)
    raise TypeError("quality_gates must be None, a preset string, a dict, or QuadQualityGates.")


def _extract_symmetry_half_mesh(mesh: MeshData, *, tol: float) -> MeshData:
    """Select the source half mesh that will be mirrored."""
    side = _select_symmetry_side(mesh, tol=tol)
    selected_blocks: dict[str, np.ndarray] = {}
    y_coordinates = mesh.vertices[:, 1]

    for cell_type, cells in mesh.cell_blocks.items():
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.size == 0:
            selected_blocks[cell_type] = cell_array.copy()
            continue
        cell_y = y_coordinates[cell_array]
        crosses = (np.any(cell_y < -tol, axis=1) & np.any(cell_y > tol, axis=1))
        if np.any(crosses):
            raise ValueError(
                "Mesh contains cells that cross the x-z symmetry plane; split "
                "or remesh those cells before creating a symmetric mesh."
            )
        if side == "positive":
            mask = np.all(cell_y >= -tol, axis=1)
        else:
            mask = np.all(cell_y <= tol, axis=1)
        selected_blocks[cell_type] = cell_array[mask]

    return _reindex_mesh_data(mesh.vertices, selected_blocks, metadata={
        **(mesh.metadata or {}),
        "symmetry_selected_side": side,
    })


def _select_symmetry_side(mesh: MeshData, *, tol: float) -> str:
    y_coordinates = mesh.vertices[:, 1]
    positive_vertices = int(np.count_nonzero(y_coordinates > tol))
    negative_vertices = int(np.count_nonzero(y_coordinates < -tol))
    if positive_vertices > 0 and negative_vertices == 0:
        return "positive"
    if negative_vertices > 0 and positive_vertices == 0:
        return "negative"

    positive_cells = 0
    negative_cells = 0
    for cells in mesh.cell_blocks.values():
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.size == 0:
            continue
        cell_y = y_coordinates[cell_array]
        positive_cells += int(np.count_nonzero(np.all(cell_y >= -tol, axis=1)))
        negative_cells += int(np.count_nonzero(np.all(cell_y <= tol, axis=1)))
    return "positive" if positive_cells >= negative_cells else "negative"


def _reindex_mesh_data(
    vertices: np.ndarray,
    cell_blocks: dict[str, np.ndarray],
    *,
    metadata: dict[str, object],
) -> MeshData:
    used_arrays = [
        np.asarray(cells, dtype=np.int64).ravel()
        for cells in cell_blocks.values()
        if np.asarray(cells).size > 0
    ]
    if not used_arrays:
        return _make_mesh_data(
            vertices=np.empty((0, 3), dtype=float),
            cell_blocks={
                cell_type: np.empty((0, np.asarray(cells).shape[1]), dtype=np.int64)
                for cell_type, cells in cell_blocks.items()
            },
            node_ids=None,
            element_ids=None,
            element_tags=None,
            metadata=metadata,
        )

    used = np.unique(np.concatenate(used_arrays))
    old_to_new = -np.ones((vertices.shape[0],), dtype=np.int64)
    old_to_new[used] = np.arange(used.size, dtype=np.int64)
    reindexed_blocks = {
        cell_type: old_to_new[np.asarray(cells, dtype=np.int64)]
        for cell_type, cells in cell_blocks.items()
    }
    return _make_mesh_data(
        vertices=np.asarray(vertices, dtype=float)[used],
        cell_blocks=reindexed_blocks,
        node_ids=None,
        element_ids={
            cell_type: np.arange(cells.shape[0], dtype=np.int64)
            for cell_type, cells in reindexed_blocks.items()
        },
        element_tags=None,
        metadata=metadata,
    )


def _mirror_mesh_data_across_xz_plane(mesh: MeshData, *, tol: float) -> MeshData:
    """Mirror vertices and cells across ``y=0`` while preserving orientation."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    mirror_map = np.arange(vertices.shape[0], dtype=np.int64)
    mirrored_vertices = [vertices.copy()]
    new_vertex_count = vertices.shape[0]

    for vertex_index, point in enumerate(vertices):
        if abs(point[1]) <= tol:
            continue
        mirrored_point = point.copy()
        mirrored_point[1] *= -1.0
        mirrored_vertices.append(mirrored_point[None, :])
        mirror_map[vertex_index] = new_vertex_count
        new_vertex_count += 1

    full_vertices = np.vstack(mirrored_vertices)
    full_blocks: dict[str, np.ndarray] = {}
    for cell_type, cells in mesh.cell_blocks.items():
        cell_array = np.asarray(cells, dtype=np.int64)
        mirrored_cells = mirror_map[cell_array]
        if mirrored_cells.size > 0:
            mirrored_cells = mirrored_cells[:, _mirrored_cell_order(mirrored_cells.shape[1])]
        full_blocks[cell_type] = (
            np.vstack([cell_array, mirrored_cells])
            if cell_array.size > 0
            else mirrored_cells
        )

    metadata = {
        **(mesh.metadata or {}),
        "symmetric": True,
        "symmetry_plane": "x-z",
        "symmetry_axis": 1,
    }
    return _make_mesh_data(
        vertices=full_vertices,
        cell_blocks=full_blocks,
        node_ids=np.arange(1, full_vertices.shape[0] + 1, dtype=np.int64),
        element_ids={
            cell_type: np.arange(1, cells.shape[0] + 1, dtype=np.int64)
            for cell_type, cells in full_blocks.items()
        },
        element_tags=None,
        metadata=metadata,
    )


def _mirrored_cell_order(num_nodes: int) -> np.ndarray:
    if num_nodes == 3:
        return np.asarray([0, 2, 1], dtype=np.int64)
    if num_nodes == 4:
        return np.asarray([0, 3, 2, 1], dtype=np.int64)
    return np.arange(num_nodes - 1, -1, -1, dtype=np.int64)
