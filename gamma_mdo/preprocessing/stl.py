"""STL surface mesh import helpers."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .mesh_io import MeshData, _make_mesh_data

def _import_stl(path: Path) -> MeshData:
    vertices, triangles = _read_stl_triangles(path)
    return _make_mesh_data(
        vertices=vertices,
        cell_blocks={"triangle": triangles},
        node_ids=None,
        element_ids={"triangle": np.arange(triangles.shape[0], dtype=np.int64)},
        element_tags=None,
        metadata={"path": str(path), "format": "stl", "reader": "gamma_mdo_stl"},
    )


def _read_stl_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) >= 84:
        num_triangles = struct.unpack("<I", raw[80:84])[0]
        if len(raw) == 84 + 50 * num_triangles:
            return _read_binary_stl_triangles(raw, num_triangles)
    return _read_ascii_stl_triangles(path)


def _read_binary_stl_triangles(raw: bytes, num_triangles: int) -> tuple[np.ndarray, np.ndarray]:
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    triangles = np.empty((num_triangles, 3), dtype=np.int64)
    offset = 84
    for triangle_index in range(num_triangles):
        record = raw[offset : offset + 50]
        offset += 50
        values = struct.unpack("<12fH", record)
        for local_index in range(3):
            xyz = tuple(float(value) for value in values[3 + 3 * local_index : 6 + 3 * local_index])
            triangles[triangle_index, local_index] = _vertex_index(xyz, vertex_map, vertices)
    return np.asarray(vertices, dtype=float), triangles


def _read_ascii_stl_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    triangle_vertices: list[int] = []

    with path.open("r", encoding="utf8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
                triangle_vertices.append(_vertex_index(xyz, vertex_map, vertices))

    if not triangle_vertices:
        raise ValueError(f"{path} does not contain STL triangle vertices.")
    if len(triangle_vertices) % 3 != 0:
        raise ValueError(f"{path} has an invalid number of STL vertex records.")

    return (
        np.asarray(vertices, dtype=float),
        np.asarray(triangle_vertices, dtype=np.int64).reshape((-1, 3)),
    )


def _vertex_index(
    xyz: tuple[float, float, float],
    vertex_map: dict[tuple[float, float, float], int],
    vertices: list[tuple[float, float, float]],
) -> int:
    if xyz not in vertex_map:
        vertex_map[xyz] = len(vertices)
        vertices.append(xyz)
    return vertex_map[xyz]
