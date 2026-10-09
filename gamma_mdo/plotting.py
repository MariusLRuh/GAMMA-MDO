"""Optional PyVista-backed plotting helpers for components and surface meshes.

Every entry point builds and returns a list of *plotting elements* -- plain
dictionaries of the form ``{"mesh": pyvista.PolyData, "kwargs": {...}}`` -- so
calls can be chained by threading one call's returned list into the next call's
``plotting_elements`` argument. Nothing is rendered unless ``show=True``, which
opens a blocking window as a side effect and returns the same list.

PyVista is imported lazily, only when mesh data is actually converted, and a
missing install raises :exc:`ImportError`. ``meshio`` is likewise lazy: without
it, only ASCII Gmsh 2.2 ``.msh`` files can be read. Rendering prefers
``lsdo_function_spaces.show_plot`` when importable and otherwise falls back to a
direct :class:`pyvista.Plotter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_MESH_FILE = (
    Path(__file__).resolve().parent
    / "core"
    / "boundary_surface_movement"
    / "e175_r1_wall.msh"
)


def plot_components(
    components=None,
    colors="",
    show=False,
    opacity=1.0,
    plotting_elements=None,
):
    """Build plotting elements for a sequence of components.

    Parameters
    ----------
    components
        Iterable of components. Each entry must either expose a ``.plot(...)``
        method, which is called with ``show=False`` so its elements accumulate,
        or already be a plotting-element dictionary containing a ``"mesh"``
        key. ``None`` is treated as an empty sequence. Any other entry raises
        :exc:`TypeError`.
    colors
        A single color applied to every component, or one color per component.
        The empty string and ``None`` both mean "leave the component's own color
        alone", so no color keyword is passed for that component. A
        per-component sequence whose length does not match ``components``
        raises :exc:`ValueError`.
    show
        When ``True``, render the accumulated elements in a blocking window
        before returning them.
    opacity
        A single opacity applied to every component, or one per component,
        broadcast under the same rules as ``colors``.
    plotting_elements
        Existing elements to append to. ``None`` starts a new list. The list is
        not copied defensively, so passing one in may extend it in place.

    Returns
    -------
    list of dict
        The accumulated plotting elements, components in input order.

    Raises
    ------
    TypeError
        If a component neither exposes ``.plot(...)`` nor is a plotting-element
        dictionary with a ``"mesh"`` entry.
    """
    component_list = [] if components is None else list(components)
    elements = _normalize_plotting_elements(plotting_elements)
    # None is the same "leave the component's own color alone" request as the
    # empty-string sentinel; broadcasting it directly would fail on list(None).
    color_values = _broadcast(
        "" if colors is None else colors, len(component_list), "colors"
    )
    opacity_values = _broadcast(opacity, len(component_list), "opacity")

    for component, color, component_opacity in zip(component_list, color_values, opacity_values):
        if hasattr(component, "plot"):
            kwargs = {
                "point_types": ["evaluated_points"],
                "plot_types": ["function"],
                "opacity": component_opacity,
                "additional_plotting_elements": elements,
                "show": False,
            }
            if color not in ("", None):
                kwargs["color"] = color
            elements = component.plot(**kwargs)
        elif isinstance(component, dict) and "mesh" in component:
            element = dict(component)
            element_kwargs = dict(element.get("kwargs", {}))
            if color not in ("", None):
                element_kwargs["color"] = color
            element_kwargs["opacity"] = component_opacity
            element["kwargs"] = element_kwargs
            elements.append(element)
        else:
            raise TypeError(
                "Each component must provide a .plot(...) method or be a plotting "
                "element dictionary with a 'mesh' entry."
            )

    if show:
        _show_plot(elements)
    return elements


def plot_mesh(
    mesh=None,
    color="lightsteelblue",
    opacity=1.0,
    plotting_elements=None,
    show=False,
    inverted_elements=None,
    inverted_color="red",
    inverted_line_width=3.0,
    inverted_opacity=1.0,
):
    """Create a PyVista surface mesh plotting element from a mesh object or file.

    ``inverted_elements`` highlights specific surface cells in ``inverted_color``.
    Pass either the id array or the report returned by
    :func:`gamma_mdo.core.boundary_surface_movement.check_element_inversion` -- its
    ``inverted_element_ids`` index directly, because the plot builds cells in the
    same triangle-then-quad order the quality module uses.  The highlighted cells
    are nudged slightly along their normals so they do not z-fight with the base
    surface.

    Parameters
    ----------
    mesh
        Surface to draw, as a path, a :class:`pyvista.PolyData`, a ``MeshData``
        with ``vertices``/``cell_blocks``, or a meshio-like object. ``None``
        loads the packaged default E175 surface mesh.
    color
        Face color of the base surface.
    opacity
        Opacity of the base surface.
    plotting_elements
        Existing elements to append to; ``None`` starts a new list.
    show
        When ``True``, render the accumulated elements in a blocking window
        before returning them.
    inverted_elements
        Cells to highlight: an id array, an ``ElementInversionReport`` whose
        ``inverted_element_ids`` are used, or ``None`` for no highlight. When
        the selection is empty no highlight element is appended.
    inverted_color
        Face and edge color of the highlighted cells.
    inverted_line_width
        Edge width used for the highlighted cells.
    inverted_opacity
        Opacity of the highlighted cells.

    Returns
    -------
    list of dict
        The accumulated plotting elements: the base surface, followed by one
        highlight element when any cell was selected.
    """
    pv_mesh = _as_pyvista_mesh(DEFAULT_MESH_FILE if mesh is None else mesh)
    elements = _normalize_plotting_elements(plotting_elements)
    elements.append(
        {
            "mesh": pv_mesh,
            "kwargs": {
                "color": color,
                "opacity": opacity,
                "show_edges": True,
                "edge_color": "black",
                "line_width": 0.1,
            },
        }
    )

    cell_ids = _normalize_cell_ids(inverted_elements, pv_mesh.n_cells)
    if cell_ids.size:
        elements.append(
            {
                "mesh": _extract_offset_cells(pv_mesh, cell_ids),
                "kwargs": {
                    "color": inverted_color,
                    "opacity": inverted_opacity,
                    "show_edges": True,
                    "edge_color": inverted_color,
                    "line_width": inverted_line_width,
                },
            }
        )

    if show:
        _show_plot(elements)
    return elements


def _normalize_cell_ids(value, num_cells: int) -> np.ndarray:
    """Accept an id array, an ElementInversionReport, or None."""
    if value is None:
        return np.empty((0,), dtype=np.int64)
    ids = getattr(value, "inverted_element_ids", value)
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= num_cells):
        raise ValueError(
            f"cell ids must lie in [0, {num_cells}); got "
            f"[{ids.min()}, {ids.max()}]."
        )
    return ids


def _extract_offset_cells(pv_mesh, cell_ids: np.ndarray, offset_fraction: float = 2e-3):
    """Pull out selected cells and lift them slightly off the base surface."""
    subset = pv_mesh.extract_cells(cell_ids)
    surface = subset.extract_surface() if hasattr(subset, "extract_surface") else subset
    try:
        bounds = np.asarray(pv_mesh.bounds, dtype=float).reshape(3, 2)
        scale = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
        oriented = surface.compute_normals(
            cell_normals=False, point_normals=True, auto_orient_normals=False
        )
        normals = np.asarray(oriented.point_data["Normals"], dtype=float)
        surface.points = np.asarray(surface.points, dtype=float) + (
            offset_fraction * scale * normals
        )
    except Exception:  # normals are optional polish, never fail the plot
        pass
    return surface


def highlight_mesh_nodes(
    mesh,
    nodes_to_highlight,
    node_color=None,
    node_colore=None,
    plotting_elements=None,
    show=False,
    point_size=10,
):
    """Add highlighted mesh nodes to a plotting-element list.

    ``nodes_to_highlight`` may be zero/one-based node indices or an ``(n, 3)``
    array of point coordinates. ``node_color`` may be a single color or one
    color per node. ``node_colore`` is accepted as the misspelled legacy alias.

    Each highlighted node becomes its own single-point element rendered as a
    sphere, so the returned list grows by one entry per node.

    Parameters
    ----------
    mesh
        Surface the node indices refer to, in any form
        :func:`read_surface_mesh` accepts. It is only converted when
        ``nodes_to_highlight`` holds indices rather than coordinates.
    nodes_to_highlight
        Either an ``(n, 3)`` array of point coordinates, used directly, or node
        indices into ``mesh``. Indices may be zero- or one-based.
    node_color
        A single color for every node, or one color per node. Exactly ``None``
        defers to ``node_colore``; any value that is then still empty, whether
        ``None``, ``""``, or an empty sequence, becomes ``"red"``.
    node_colore
        Misspelled legacy alias for ``node_color``, retained for backward
        compatibility. It is consulted **only** when ``node_color`` is exactly
        ``None``; an explicitly empty ``node_color`` such as ``""`` bypasses
        the alias and falls through to ``"red"``. Prefer ``node_color`` in new
        code.
    plotting_elements
        Existing elements to append to; ``None`` starts a new list.
    show
        When ``True``, render the accumulated elements in a blocking window
        before returning them.
    point_size
        Rendered diameter of each highlighted node.

    Returns
    -------
    list of dict
        The accumulated plotting elements, with one single-point element per
        highlighted node appended in order.
    """
    coordinate_points = _as_highlight_coordinate_points(nodes_to_highlight)
    if coordinate_points is None:
        pv_mesh = _as_pyvista_mesh(mesh)
        points = _normalize_highlight_points(nodes_to_highlight, pv_mesh)
    else:
        points = coordinate_points
    if node_color is None and node_colore is not None:
        node_color = node_colore
    color_input = "red" if _empty_color_value(node_color) else node_color
    colors = _broadcast(color_input, len(points), "node_color")

    elements = _normalize_plotting_elements(plotting_elements)
    for point, color in zip(points, colors):
        elements.append(
            {
                "mesh": _points_to_polydata(np.asarray(point, dtype=float).reshape((1, 3))),
                "kwargs": {
                    "color": color,
                    "point_size": point_size,
                    "render_points_as_spheres": True,
                },
            }
        )

    if show:
        _show_plot(elements)
    return elements


def read_surface_mesh(mesh_file: str | Path):
    """Read a surface mesh into a PyVista ``PolyData`` object.

    Parameters
    ----------
    mesh_file
        Path to the mesh file. ``meshio`` reads it when importable; otherwise
        only ASCII Gmsh 2.2 ``.msh`` files are supported and any other suffix
        raises :exc:`ImportError`.

    Returns
    -------
    pyvista.PolyData
        The surface, containing only its triangle and quadrangle faces.

    Raises
    ------
    ImportError
        If PyVista is unavailable, or ``meshio`` is unavailable and the file is
        not an ASCII ``.msh``.
    """
    return _as_pyvista_mesh(mesh_file)


def _broadcast(value, length: int, name: str) -> list:
    if length == 0:
        return []
    if isinstance(value, str):
        return [value] * length
    if np.isscalar(value):
        return [value] * length

    values = list(value)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"{name} must be a scalar or have length {length}; received {len(values)}.")
    return values


def _empty_color_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        return len(value) == 0
    except TypeError:
        return False


def _as_pyvista_mesh(mesh):
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise ImportError("PyVista is required for plotting mesh data.") from exc

    if isinstance(mesh, pv.PolyData):
        return mesh
    if isinstance(mesh, (str, Path)):
        return _read_mesh_file(Path(mesh))
    if hasattr(mesh, "vertices") and hasattr(mesh, "cell_blocks"):
        return _mesh_data_to_polydata(mesh)
    if hasattr(mesh, "points") and hasattr(mesh, "cells"):
        return _meshio_to_polydata(mesh)
    if hasattr(mesh, "points") and hasattr(mesh, "faces"):
        return pv.PolyData(np.asarray(mesh.points, dtype=float), np.asarray(mesh.faces, dtype=np.int64))
    raise TypeError("mesh must be a path, MeshData, meshio-like object, or pyvista.PolyData.")


def _read_mesh_file(path: Path):
    try:
        import meshio

        return _meshio_to_polydata(meshio.read(path))
    except ImportError:
        if path.suffix.lower() != ".msh":
            raise ImportError("meshio is required to read mesh files other than ASCII .msh files.")
        points, cells = _read_gmsh_22_ascii_surface(path)
        return _points_and_cells_to_polydata(points, cells)


def _meshio_to_polydata(mesh):
    cells = []
    for cell_block in mesh.cells:
        cell_type = getattr(cell_block, "type", None)
        data = np.asarray(getattr(cell_block, "data", cell_block), dtype=np.int64)
        if cell_type in ("triangle", "quad") or (data.ndim == 2 and data.shape[1] in (3, 4)):
            cells.append(data)
    if not cells:
        raise ValueError("Mesh does not contain triangle or quad surface cells.")
    return _points_and_cells_to_polydata(np.asarray(mesh.points, dtype=float), cells)


def _mesh_data_to_polydata(mesh):
    # Gather every polygon block (triangle/quad first, then any polygonN block of
    # a hex-dominant CFD mesh) in the same order the quality module's
    # _surface_cells uses, so an ElementInversionReport's ids index the built
    # cells directly.  PyVista PolyData supports arbitrary n-gon faces.
    ordered_types = ["triangle", "quad"] + [
        t for t in mesh.cell_blocks if t not in ("triangle", "quad")
    ]
    cells = []
    for cell_type in ordered_types:
        block = mesh.cell_blocks.get(cell_type)
        if block is None:
            continue
        block_array = np.asarray(block, dtype=np.int64)
        if block_array.ndim == 2 and block_array.shape[1] >= 3:
            cells.append(block_array)

    if not cells:
        connectivity = np.asarray(mesh.connectivity, dtype=object)
        if connectivity.ndim == 2 and connectivity.shape[1] >= 3:
            cells.append(np.asarray(connectivity, dtype=np.int64))

    if not cells:
        raise ValueError("MeshData does not contain surface cells to plot.")
    return _points_and_cells_to_polydata(np.asarray(mesh.vertices, dtype=float), cells)


def _read_gmsh_22_ascii_surface(path: Path):
    with path.open("r", encoding="utf8") as stream:
        lines = iter(stream)
        points = None
        cells: list[np.ndarray] = []
        node_id_to_index = {}

        for line in lines:
            marker = line.strip()
            if marker == "$Nodes":
                num_nodes = int(next(lines).strip())
                points = np.empty((num_nodes, 3), dtype=float)
                for point_index in range(num_nodes):
                    parts = next(lines).split()
                    node_id = int(parts[0])
                    node_id_to_index[node_id] = point_index
                    points[point_index] = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif marker == "$Elements":
                num_elements = int(next(lines).strip())
                for _ in range(num_elements):
                    parts = next(lines).split()
                    element_type = int(parts[1])
                    num_tags = int(parts[2])
                    node_ids = [int(value) for value in parts[3 + num_tags :]]
                    if element_type in (2, 3):
                        cells.append(np.asarray([node_id_to_index[node_id] for node_id in node_ids], dtype=np.int64))

        if points is None:
            raise ValueError(f"{path} does not contain a $Nodes section.")
        if not cells:
            raise ValueError(f"{path} does not contain triangle or quad elements.")

    triangles = [cell for cell in cells if cell.size == 3]
    quads = [cell for cell in cells if cell.size == 4]
    cell_blocks = []
    if triangles:
        cell_blocks.append(np.vstack(triangles))
    if quads:
        cell_blocks.append(np.vstack(quads))
    return points, cell_blocks


def _points_and_cells_to_polydata(points: np.ndarray, cell_blocks: Sequence[np.ndarray]):
    return _points_to_polydata(points, _pyvista_faces(cell_blocks))


def _pyvista_faces(cell_blocks: Sequence[np.ndarray]) -> np.ndarray:
    face_blocks = []
    for cells in cell_blocks:
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.ndim != 2 or cell_array.shape[1] < 3:
            raise ValueError("Surface cell blocks must be two-dimensional polygons with >= 3 vertices.")
        block = np.empty((cell_array.shape[0], cell_array.shape[1] + 1), dtype=np.int64)
        block[:, 0] = cell_array.shape[1]
        block[:, 1:] = cell_array
        face_blocks.append(block.ravel())
    return np.concatenate(face_blocks)


def _points_to_polydata(points: np.ndarray, faces: np.ndarray | None = None):
    import pyvista as pv

    point_array = np.asarray(points, dtype=float)
    if faces is None:
        return pv.PolyData(point_array)
    return pv.PolyData(point_array, np.asarray(faces, dtype=np.int64))


def _normalize_highlight_points(nodes_to_highlight, pv_mesh) -> np.ndarray:
    point_indices = _normalize_node_indices(nodes_to_highlight, pv_mesh.n_points)
    return np.asarray(pv_mesh.points, dtype=float)[point_indices]


def _as_highlight_coordinate_points(nodes_to_highlight) -> np.ndarray | None:
    raw = np.asarray(nodes_to_highlight)
    if raw.ndim != 2 or raw.shape[1] != 3:
        return None
    points = np.asarray(raw, dtype=float)
    if points.shape[0] == 0:
        raise ValueError("nodes_to_highlight is empty.")
    return points


def _normalize_node_indices(nodes_to_highlight, num_points: int) -> np.ndarray:
    if isinstance(nodes_to_highlight, range):
        raw = np.asarray(list(nodes_to_highlight), dtype=np.int64)
    else:
        raw = np.asarray(list(nodes_to_highlight), dtype=np.int64).ravel()
    if raw.size == 0:
        raise ValueError("nodes_to_highlight is empty.")

    if np.any(raw < 0):
        raise IndexError("nodes_to_highlight cannot contain negative indices.")

    if np.any(raw >= num_points):
        if np.all((raw >= 1) & (raw <= num_points)):
            raw = raw - 1
        else:
            raise IndexError(f"Node indices must be within [0, {num_points - 1}] or [1, {num_points}].")
    elif 0 not in raw and np.all((raw >= 1) & (raw <= num_points)):
        raw = raw - 1

    return raw


def _normalize_plotting_elements(plotting_elements) -> list:
    if plotting_elements is None:
        return []
    return list(_iter_plotting_elements(plotting_elements))


def _iter_plotting_elements(plotting_elements):
    if isinstance(plotting_elements, dict) and "mesh" in plotting_elements:
        yield plotting_elements
        return
    if isinstance(plotting_elements, tuple) and len(plotting_elements) == 2:
        yield plotting_elements
        return
    if isinstance(plotting_elements, list):
        for element in plotting_elements:
            yield from _iter_plotting_elements(element)
        return
    yield plotting_elements


def _show_plot(plotting_elements):
    plotting_elements = _normalize_plotting_elements(plotting_elements)
    try:
        import lsdo_function_spaces as lfs

        lfs.show_plot(plotting_elements=plotting_elements)
        return
    except Exception:
        pass

    import pyvista as pv

    plotter = pv.Plotter()
    for element in plotting_elements:
        plotter.add_mesh(element["mesh"], **element.get("kwargs", {}))
    plotter.show()
