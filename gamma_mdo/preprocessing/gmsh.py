"""Gmsh ASCII mesh import and export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh_io import MeshData, _array_from_variable_rows, _make_mesh_data, _merge_cell_blocks

def _write_gmsh_ascii(mesh: MeshData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    writable_blocks = []
    for cell_type, cells in mesh.cell_blocks.items():
        element_type_id = _gmsh_element_type_id(cell_type)
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.ndim != 2:
            raise ValueError(f"Cannot export cell block {cell_type!r}; expected a 2D connectivity array.")
        expected_width = _gmsh_element_num_nodes(element_type_id)
        if cell_array.shape[1] != expected_width:
            raise ValueError(
                f"Cannot export cell block {cell_type!r}; expected {expected_width} "
                f"nodes per element, received {cell_array.shape[1]}."
            )
        writable_blocks.append((element_type_id, cell_array))

    num_elements = sum(cells.shape[0] for _, cells in writable_blocks)
    with path.open("w", encoding="utf8") as stream:
        stream.write("$MeshFormat\n")
        stream.write("2.2 0 8\n")
        stream.write("$EndMeshFormat\n")
        stream.write("$Nodes\n")
        stream.write(f"{vertices.shape[0]}\n")
        for point_index, point in enumerate(vertices, start=1):
            stream.write(
                f"{point_index} "
                f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n"
            )
        stream.write("$EndNodes\n")
        stream.write("$Elements\n")
        stream.write(f"{num_elements}\n")
        element_index = 1
        for element_type_id, cells in writable_blocks:
            for cell in cells:
                node_ids = " ".join(str(int(node_index) + 1) for node_index in cell)
                stream.write(f"{element_index} {element_type_id} 0 {node_ids}\n")
                element_index += 1
        stream.write("$EndElements\n")


def _gmsh_element_type_id(cell_type: str) -> int:
    element_type_ids = {
        "line": 1,
        "triangle": 2,
        "quad": 3,
        "tetra": 4,
        "hexahedron": 5,
        "wedge": 6,
        "pyramid": 7,
        "line3": 8,
        "triangle6": 9,
        "quad9": 10,
        "tetra10": 11,
        "hexahedron27": 12,
        "wedge18": 13,
        "pyramid14": 14,
        "vertex": 15,
        "quad8": 16,
        "hexahedron20": 17,
        "wedge15": 18,
        "pyramid13": 19,
    }
    try:
        return element_type_ids[cell_type]
    except KeyError as exc:
        supported = ", ".join(sorted(element_type_ids))
        raise ValueError(f"Cannot export unsupported cell type {cell_type!r}. Supported types: {supported}.") from exc


def _gmsh_element_num_nodes(element_type_id: int) -> int:
    num_nodes = {
        1: 2,
        2: 3,
        3: 4,
        4: 4,
        5: 8,
        6: 6,
        7: 5,
        8: 3,
        9: 6,
        10: 9,
        11: 10,
        12: 27,
        13: 18,
        14: 14,
        15: 1,
        16: 8,
        17: 20,
        18: 15,
        19: 13,
    }
    return num_nodes[element_type_id]


def _import_msh(path: Path) -> MeshData:
    try:
        import meshio
    except ImportError:
        return _read_gmsh_ascii(path)

    mesh = meshio.read(path)
    cell_blocks: dict[str, list[np.ndarray]] = {}
    element_ids: dict[str, list[np.ndarray]] = {}
    running_element_id = 1
    for cell_block in mesh.cells:
        cell_type = str(cell_block.type)
        data = np.asarray(cell_block.data, dtype=np.int64)
        cell_blocks.setdefault(cell_type, []).append(data)
        ids = np.arange(
            running_element_id,
            running_element_id + data.shape[0],
            dtype=np.int64,
        )
        element_ids.setdefault(cell_type, []).append(ids)
        running_element_id += data.shape[0]

    metadata = {
        "path": str(path),
        "format": "msh",
        "reader": "meshio",
        "point_data": getattr(mesh, "point_data", {}),
        "cell_data": getattr(mesh, "cell_data", {}),
    }
    return _make_mesh_data(
        vertices=np.asarray(mesh.points, dtype=float),
        cell_blocks=_merge_cell_blocks(cell_blocks),
        # meshio normalizes connectivity to row order and does not expose the
        # original Gmsh node tags.  The writer and the meshes used by GAMMA use
        # contiguous one-based tags, so preserve that public convention here.
        node_ids=np.arange(1, mesh.points.shape[0] + 1, dtype=np.int64),
        element_ids={
            cell_type: np.concatenate(id_blocks).astype(np.int64)
            for cell_type, id_blocks in element_ids.items()
        },
        element_tags=None,
        metadata=metadata,
    )


def _read_gmsh_ascii(path: Path) -> MeshData:
    with path.open("r", encoding="utf8") as stream:
        lines = iter(stream)
        vertices = None
        node_ids = None
        node_id_to_index = {}
        cell_blocks: dict[str, list[np.ndarray]] = {}
        element_ids: dict[str, list[int]] = {}
        element_tags: dict[str, list[list[int]]] = {}
        mesh_format = None

        for line in lines:
            marker = line.strip()
            if marker == "$MeshFormat":
                mesh_format = next(lines).strip()
                parts = mesh_format.split()
                if len(parts) >= 2 and parts[1] != "0":
                    raise ValueError(
                        f"{path} is a binary Gmsh file. Install meshio to read binary .msh files."
                    )
            elif marker == "$Nodes":
                num_nodes = int(next(lines).strip())
                vertices = np.empty((num_nodes, 3), dtype=float)
                node_ids = np.empty((num_nodes,), dtype=np.int64)
                for point_index in range(num_nodes):
                    parts = next(lines).split()
                    node_id = int(parts[0])
                    node_ids[point_index] = node_id
                    node_id_to_index[node_id] = point_index
                    vertices[point_index] = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif marker == "$Elements":
                num_elements = int(next(lines).strip())
                for _ in range(num_elements):
                    parts = next(lines).split()
                    element_id = int(parts[0])
                    element_type_id = int(parts[1])
                    num_tags = int(parts[2])
                    tags = [int(value) for value in parts[3 : 3 + num_tags]]
                    node_ids_for_element = [int(value) for value in parts[3 + num_tags :]]
                    cell_type = _gmsh_element_type_name(element_type_id)
                    connectivity = np.asarray(
                        [node_id_to_index[node_id] for node_id in node_ids_for_element],
                        dtype=np.int64,
                    )
                    cell_blocks.setdefault(cell_type, []).append(connectivity)
                    element_ids.setdefault(cell_type, []).append(element_id)
                    element_tags.setdefault(cell_type, []).append(tags)

        if vertices is None:
            raise ValueError(f"{path} does not contain a $Nodes section.")
        if not cell_blocks:
            raise ValueError(f"{path} does not contain a $Elements section with cells.")

    metadata = {
        "path": str(path),
        "format": "msh",
        "reader": "gamma_mdo_gmsh_ascii",
        "mesh_format": mesh_format,
    }
    return _make_mesh_data(
        vertices=vertices,
        cell_blocks={
            cell_type: np.vstack(cells).astype(np.int64)
            for cell_type, cells in cell_blocks.items()
        },
        node_ids=node_ids,
        element_ids={
            cell_type: np.asarray(ids, dtype=np.int64)
            for cell_type, ids in element_ids.items()
        },
        element_tags={
            cell_type: _array_from_variable_rows(tags)
            for cell_type, tags in element_tags.items()
        },
        metadata=metadata,
    )


def _gmsh_element_type_name(element_type_id: int) -> str:
    element_type_names = {
        1: "line",
        2: "triangle",
        3: "quad",
        4: "tetra",
        5: "hexahedron",
        6: "wedge",
        7: "pyramid",
        8: "line3",
        9: "triangle6",
        10: "quad9",
        11: "tetra10",
        12: "hexahedron27",
        13: "wedge18",
        14: "pyramid14",
        15: "vertex",
        16: "quad8",
        17: "hexahedron20",
        18: "wedge15",
        19: "pyramid13",
    }
    return element_type_names.get(element_type_id, f"gmsh_{element_type_id}")
