from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy. Install the visualization dependencies with "
        "`python -m pip install numpy h5py meshio matplotlib imageio pillow`."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_HDF5 = REPO_ROOT / "wing_rotation_defomration.hdf5"
DEFAULT_MESH = SCRIPT_DIR / "e175_quad_panel.msh"
DEFAULT_CP_BOUNDS = (-1.5, 1.0)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gamma_mdo_matplotlib"))


@dataclass(frozen=True)
class SurfaceMesh:
    points: np.ndarray
    cell_blocks: tuple[np.ndarray, ...]

    @property
    def num_cells(self) -> int:
        return int(sum(block.shape[0] for block in self.cell_blocks))


@dataclass(frozen=True)
class CameraSettings:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    elev: float
    azim: float
    roll: float
    zoom: float


@dataclass(frozen=True)
class FrameData:
    name: str
    vertices: np.ndarray
    values: np.ndarray
    metrics: dict[str, float]


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: h5py. Install the visualization dependencies with "
            "`python -m pip install numpy h5py meshio matplotlib imageio pillow`."
        ) from exc
    return h5py


def _resolve_path(value: str | Path, *, fallback_dirs: Sequence[Path]) -> Path:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(base / path for base in fallback_dirs)

    if path.name == "wing_rotation_deformation.hdf5":
        typo_name = "wing_rotation_defomration.hdf5"
        candidates.extend(candidate.with_name(typo_name) for candidate in list(candidates))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find {value!s}")


def _load_mesh(mesh_path: Path) -> SurfaceMesh:
    try:
        import meshio
    except ImportError:
        return _load_gmsh22_ascii(mesh_path)

    mesh = meshio.read(mesh_path)
    blocks: list[np.ndarray] = []
    for cell_block in mesh.cells:
        if cell_block.type not in ("triangle", "quad"):
            continue
        data = np.asarray(cell_block.data, dtype=np.int64)
        if data.size:
            blocks.append(data)

    if not blocks:
        raise ValueError(f"{mesh_path} does not contain triangle or quad surface cells.")
    return SurfaceMesh(points=np.asarray(mesh.points, dtype=float), cell_blocks=tuple(blocks))


def _load_gmsh22_ascii(mesh_path: Path) -> SurfaceMesh:
    lines = mesh_path.read_text().splitlines()

    try:
        node_start = lines.index("$Nodes")
        num_nodes = int(lines[node_start + 1].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{mesh_path} is not a readable ASCII Gmsh 2.2 file.") from exc

    points = np.empty((num_nodes, 3), dtype=float)
    node_to_index: dict[int, int] = {}
    for local_index, line in enumerate(lines[node_start + 2 : node_start + 2 + num_nodes]):
        fields = line.split()
        node_tag = int(fields[0])
        node_to_index[node_tag] = local_index
        points[local_index] = [float(fields[1]), float(fields[2]), float(fields[3])]

    try:
        element_start = lines.index("$Elements")
        num_elements = int(lines[element_start + 1].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{mesh_path} is missing a Gmsh $Elements section.") from exc

    blocks: list[np.ndarray] = []
    active_width: int | None = None
    active_cells: list[list[int]] = []

    def flush_active_block() -> None:
        nonlocal active_width, active_cells
        if active_cells:
            blocks.append(np.asarray(active_cells, dtype=np.int64))
        active_width = None
        active_cells = []

    for line in lines[element_start + 2 : element_start + 2 + num_elements]:
        fields = line.split()
        element_type = int(fields[1])
        num_tags = int(fields[2])
        if element_type not in (2, 3):
            continue
        width = 3 if element_type == 2 else 4
        node_tags = [int(tag) for tag in fields[3 + num_tags : 3 + num_tags + width]]
        cell = [node_to_index[tag] for tag in node_tags]
        if active_width != width:
            flush_active_block()
            active_width = width
        active_cells.append(cell)

    flush_active_block()
    if not blocks:
        raise ValueError(f"{mesh_path} does not contain triangle or quad surface cells.")
    return SurfaceMesh(points=points, cell_blocks=tuple(blocks))


def _iteration_sort_key(name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"iteration_(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def _parse_iteration_selector(selector: str, iteration_names: Sequence[str]) -> list[str]:
    if selector == "all":
        return list(iteration_names)

    selected_indices: list[int] = []
    if ":" in selector:
        parts = selector.split(":")
        if len(parts) > 3:
            raise ValueError("Iteration slice must be start:stop[:step].")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        selected_indices = list(range(len(iteration_names))[slice(start, stop, step)])
    else:
        selected_indices = [int(part) for part in selector.split(",") if part.strip()]

    try:
        return [iteration_names[index] for index in selected_indices]
    except IndexError as exc:
        raise ValueError(f"Iteration selector {selector!r} is out of range.") from exc


def _dataset_name_for_field(group: Any, field_name: str) -> str:
    if field_name == "auto":
        for candidate in ("Cp", "mu", "None_stacked"):
            if candidate in group:
                return candidate
        raise ValueError("No default scalar field found. Try `--list-fields`.")
    if field_name == "mu" and "mu" not in group and "None_stacked" in group:
        return "None_stacked"
    if field_name not in group:
        raise ValueError(f"Field {field_name!r} was not found in {group.name}.")
    return field_name


def _scalar_cell_values(array: np.ndarray, *, num_cells: int, component: int) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        if values.size != num_cells:
            raise ValueError(f"Expected {num_cells} cell values, got {values.size}.")
        return values

    squeezed = np.squeeze(values)
    if squeezed.ndim == 1 and squeezed.size == num_cells:
        return squeezed.astype(float, copy=False)

    if values.ndim == 2:
        if values.shape[0] == num_cells:
            if not 0 <= component < values.shape[1]:
                raise ValueError(f"Field component {component} is out of range for {values.shape}.")
            return values[:, component]
        if values.shape[1] == num_cells:
            if not 0 <= component < values.shape[0]:
                raise ValueError(f"Field component {component} is out of range for {values.shape}.")
            return values[component, :]

    raise ValueError(f"Could not interpret field shape {values.shape} as cell data.")


def _vertices_from_group(group: Any, *, dataset_name: str, num_points: int) -> np.ndarray:
    if dataset_name not in group:
        raise ValueError(f"Dataset {dataset_name!r} was not found in {group.name}.")
    vertices = np.asarray(group[dataset_name][()], dtype=float)
    vertices = np.squeeze(vertices)
    if vertices.shape != (num_points, 3):
        raise ValueError(
            f"{group.name}/{dataset_name} has shape {vertices.shape}; expected {(num_points, 3)}."
        )
    return vertices


def _read_scalar_metric(group: Any, name: str) -> float | None:
    if name not in group:
        return None
    values = np.asarray(group[name][()], dtype=float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        return None
    return float(values[0])


def _load_frames(
    hdf5_path: Path,
    *,
    field_name: str,
    vertices_name: str,
    iteration_selector: str,
    num_points: int,
    num_cells: int,
    field_component: int,
) -> tuple[list[FrameData], str]:
    h5py = _require_h5py()
    with h5py.File(hdf5_path, "r") as h5_file:
        iteration_names = sorted(h5_file.keys(), key=_iteration_sort_key)
        selected_names = _parse_iteration_selector(iteration_selector, iteration_names)
        if not selected_names:
            raise ValueError("No iterations were selected.")

        resolved_field = _dataset_name_for_field(h5_file[selected_names[0]], field_name)
        frames: list[FrameData] = []
        for name in selected_names:
            group = h5_file[name]
            resolved_field = _dataset_name_for_field(group, resolved_field)
            vertices = _vertices_from_group(group, dataset_name=vertices_name, num_points=num_points)
            values = _scalar_cell_values(
                group[resolved_field][()],
                num_cells=num_cells,
                component=field_component,
            )
            metrics = {
                metric: value
                for metric in ("wing_root_rotation_degrees", "CL", "CDi", "L", "Di", "variable_0")
                if (value := _read_scalar_metric(group, metric)) is not None
            }
            frames.append(FrameData(name=name, vertices=vertices, values=values, metrics=metrics))
    return frames, resolved_field


def _list_fields(hdf5_path: Path, *, num_points: int, num_cells: int) -> None:
    h5py = _require_h5py()
    with h5py.File(hdf5_path, "r") as h5_file:
        iteration_names = sorted(h5_file.keys(), key=_iteration_sort_key)
        if not iteration_names:
            print("No iteration groups found.")
            return
        group = h5_file[iteration_names[0]]
        print(f"Iterations: {len(iteration_names)} ({iteration_names[0]} ... {iteration_names[-1]})")
        print(f"Expected mesh data: {num_points} points, {num_cells} cells")
        print("Datasets in first iteration:")
        for name in sorted(group.keys()):
            dataset = group[name]
            shape = tuple(dataset.shape)
            role = ""
            size = int(np.prod(shape)) if shape else 1
            if shape == (num_points, 3):
                role = "vertices"
            elif size == num_cells:
                role = "cell field"
            elif size == 1:
                role = "scalar metric"
            print(f"  {name}: shape={shape} dtype={dataset.dtype} {role}")


def _parse_vector(value: str | None, *, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three numbers.")
    return tuple(float(part) for part in parts)


def _global_bounds(vertices_per_frame: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for vertices in vertices_per_frame:
        mins.append(np.min(vertices, axis=0))
        maxs.append(np.max(vertices, axis=0))
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def _camera_from_args(args: argparse.Namespace, bounds: tuple[np.ndarray, np.ndarray]) -> CameraSettings:
    mins, maxs = bounds
    center = 0.5 * (mins + maxs)
    focal = _parse_vector(args.focal_point, name="--focal-point") or tuple(float(v) for v in center)
    view_up = _parse_vector(args.view_up, name="--view-up") or (0.0, 0.0, 1.0)
    explicit_position = _parse_vector(args.camera_position, name="--camera-position")

    if explicit_position is not None:
        direction = np.asarray(explicit_position, dtype=float) - np.asarray(focal, dtype=float)
        radius = max(float(np.linalg.norm(direction)), 1e-12)
        elev = math.degrees(math.asin(float(direction[2]) / radius))
        azim = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        position = explicit_position
    else:
        elev = float(args.elev)
        azim = float(args.azim)
        diagonal = max(float(np.linalg.norm(maxs - mins)), 1e-12)
        distance = float(args.distance) * diagonal
        elev_rad = math.radians(elev)
        azim_rad = math.radians(azim)
        offset = distance * np.array(
            [
                math.cos(elev_rad) * math.cos(azim_rad),
                math.cos(elev_rad) * math.sin(azim_rad),
                math.sin(elev_rad),
            ],
            dtype=float,
        )
        position = tuple(float(v) for v in np.asarray(focal, dtype=float) + offset)

    return CameraSettings(
        position=tuple(float(v) for v in position),
        focal_point=tuple(float(v) for v in focal),
        view_up=tuple(float(v) for v in view_up),
        elev=float(elev),
        azim=float(azim),
        roll=float(args.roll),
        zoom=float(args.zoom),
    )


def _print_camera(camera: CameraSettings) -> None:
    print(
        "[camera] "
        f"position={camera.position} "
        f"focal_point={camera.focal_point} "
        f"view_up={camera.view_up} "
        f"elev={camera.elev:.6g} "
        f"azim={camera.azim:.6g} "
        f"roll={camera.roll:.6g}"
    )


def _field_limits(
    frames: Sequence[FrameData],
    clim: Sequence[float] | None,
    *,
    field_label: str,
    cp_bounds: Sequence[float] | None,
) -> tuple[float, float]:
    if clim is not None:
        if len(clim) != 2:
            raise ValueError("--clim requires exactly two values.")
        return float(clim[0]), float(clim[1])
    if field_label == "Cp":
        if cp_bounds is None:
            return DEFAULT_CP_BOUNDS
        if len(cp_bounds) != 2:
            raise ValueError("--cp-bounds requires exactly two values.")
        return float(cp_bounds[0]), float(cp_bounds[1])
    values = np.concatenate([frame.values[np.isfinite(frame.values)] for frame in frames])
    if values.size == 0:
        raise ValueError("Selected field contains no finite values.")
    return float(np.min(values)), float(np.max(values))


def _format_title(frame: FrameData, field_label: str) -> str:
    parts = [frame.name, field_label]
    if "wing_root_rotation_degrees" in frame.metrics:
        parts.append(f"rotation={frame.metrics['wing_root_rotation_degrees']:.3f} deg")
    if "CL" in frame.metrics:
        parts.append(f"CL={frame.metrics['CL']:.4f}")
    if "CDi" in frame.metrics:
        parts.append(f"CDi={frame.metrics['CDi']:.4g}")
    return " | ".join(parts)


def _all_polygons(vertices: np.ndarray, mesh: SurfaceMesh) -> list[np.ndarray]:
    return [vertices[cell] for block in mesh.cell_blocks for cell in block]


def _render_gif_matplotlib(
    frames: Sequence[FrameData],
    mesh: SurfaceMesh,
    *,
    output_path: Path,
    field_label: str,
    camera: CameraSettings,
    bounds: tuple[np.ndarray, np.ndarray],
    clim: tuple[float, float],
    cmap_name: str,
    fps: float,
    window_size: tuple[int, int],
    show_edges: bool,
    show_axes: bool,
) -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm = colors.Normalize(vmin=clim[0], vmax=clim[1], clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    width, height = window_size
    dpi = 110
    mins, maxs = bounds
    span = np.maximum(maxs - mins, 1e-12)
    pad = 0.04 * span

    with imageio.get_writer(output_path, mode="I", duration=1.0 / max(float(fps), 1e-12), loop=0) as writer:
        for frame_index, frame in enumerate(frames, start=1):
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            polygons = _all_polygons(frame.vertices, mesh)
            collection = Poly3DCollection(
                polygons,
                linewidths=0.12 if show_edges else 0.0,
                edgecolors=(0.05, 0.05, 0.05, 0.38) if show_edges else "none",
                antialiased=False,
            )
            collection.set_facecolor(cmap(norm(frame.values)))
            ax.add_collection3d(collection)

            ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
            ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
            ax.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
            try:
                ax.set_box_aspect(tuple(span), zoom=camera.zoom)
            except TypeError:
                ax.set_box_aspect(tuple(span))
            try:
                ax.view_init(elev=camera.elev, azim=camera.azim, roll=camera.roll)
            except TypeError:
                ax.view_init(elev=camera.elev, azim=camera.azim)
            if not show_axes:
                ax.set_axis_off()

            scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_mappable.set_array([])
            colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.62, pad=0.02)
            colorbar.set_label(field_label)
            ax.set_title(_format_title(frame, field_label), fontsize=9)
            fig.tight_layout(pad=0.3)

            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(image)
            plt.close(fig)
            print(f"wrote frame {frame_index}/{len(frames)}: {frame.name}")


def _pyvista_faces(mesh: SurfaceMesh) -> np.ndarray:
    face_blocks: list[np.ndarray] = []
    for cells in mesh.cell_blocks:
        width = cells.shape[1]
        block = np.empty((cells.shape[0], width + 1), dtype=np.int64)
        block[:, 0] = width
        block[:, 1:] = cells
        face_blocks.append(block.ravel())
    return np.concatenate(face_blocks)


def _render_gif_pyvista(
    frames: Sequence[FrameData],
    mesh: SurfaceMesh,
    *,
    output_path: Path,
    field_label: str,
    camera: CameraSettings,
    clim: tuple[float, float],
    cmap_name: str,
    fps: float,
    window_size: tuple[int, int],
    show_edges: bool,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyVista backend requested but pyvista is not installed. "
            "Use `--backend matplotlib` or install pyvista."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pv_mesh = pv.PolyData(np.ascontiguousarray(frames[0].vertices), _pyvista_faces(mesh))
    pv_mesh.cell_data[field_label] = frames[0].values

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(
        pv_mesh,
        scalars=field_label,
        cmap=cmap_name,
        clim=clim,
        show_edges=show_edges,
        show_scalar_bar=True,
    )
    text_actor = None
    plotter.set_background("white")
    plotter.add_axes(line_width=2)
    plotter.camera_position = [camera.position, camera.focal_point, camera.view_up]
    plotter.open_gif(str(output_path), fps=float(fps))

    for frame_index, frame in enumerate(frames, start=1):
        pv_mesh.points = np.ascontiguousarray(frame.vertices)
        pv_mesh.cell_data[field_label] = frame.values
        if text_actor is not None:
            try:
                plotter.remove_actor(text_actor, reset_camera=False, render=False)
            except TypeError:
                plotter.remove_actor(text_actor, reset_camera=False)
        text_actor = plotter.add_text(
            _format_title(frame, field_label),
            position="upper_left",
            font_size=10,
        )
        plotter.render()
        plotter.write_frame()
        print(f"wrote frame {frame_index}/{len(frames)}: {frame.name}")
    plotter.close()


def _show_pyvista_camera_probe(
    frame: FrameData,
    mesh: SurfaceMesh,
    *,
    field_label: str,
    camera: CameraSettings,
    clim: tuple[float, float],
    cmap_name: str,
    show_edges: bool,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Interactive camera probing requires pyvista.") from exc

    pv_mesh = pv.PolyData(np.ascontiguousarray(frame.vertices), _pyvista_faces(mesh))
    pv_mesh.cell_data[field_label] = frame.values
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(
        pv_mesh,
        scalars=field_label,
        cmap=cmap_name,
        clim=clim,
        show_edges=show_edges,
        show_scalar_bar=True,
    )
    plotter.add_text(_format_title(frame, field_label), position="upper_left", font_size=10)
    plotter.camera_position = [camera.position, camera.focal_point, camera.view_up]
    plotter.set_background("white")
    plotter.add_axes(line_width=2)

    def print_current_camera(*_args: Any, **_kwargs: Any) -> None:
        current = plotter.camera_position
        print(
            "[camera] "
            f"position={tuple(float(v) for v in current[0])} "
            f"focal_point={tuple(float(v) for v in current[1])} "
            f"view_up={tuple(float(v) for v in current[2])}",
            flush=True,
        )

    observer_added = False
    iren = getattr(plotter, "iren", None)
    add_observer = getattr(iren, "add_observer", None)
    if callable(add_observer):
        add_observer("InteractionEvent", print_current_camera)
        add_observer("EndInteractionEvent", print_current_camera)
        observer_added = True
    else:
        add_observer = getattr(plotter, "add_observer", None)
        if callable(add_observer):
            add_observer("InteractionEvent", print_current_camera)
            add_observer("EndInteractionEvent", print_current_camera)
            observer_added = True

    if not observer_added:
        print("[camera] warning: camera interaction observers are not available in this PyVista version.")
    print_current_camera()
    plotter.show(auto_close=True)


def _select_interactive_frame(frames: Sequence[FrameData], selector: str | None) -> FrameData:
    if selector is None:
        return frames[-1]

    for frame in frames:
        if frame.name == selector:
            return frame

    if re.fullmatch(r"-?\d+", selector):
        iteration_name = f"iteration_{int(selector)}"
        for frame in frames:
            if frame.name == iteration_name:
                return frame
        try:
            return frames[int(selector)]
        except IndexError as exc:
            raise ValueError(f"Interactive iteration {selector!r} is out of range.") from exc

    raise ValueError(
        f"Interactive iteration {selector!r} was not found. "
        "Use an integer index or an HDF5 group name like `iteration_3`."
    )


def _choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return "matplotlib"
    return "pyvista"


def _window_size(values: Sequence[int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError("--window-size requires width and height.")
    width, height = int(values[0]), int(values[1])
    if width <= 0 or height <= 0:
        raise ValueError("--window-size values must be positive.")
    return width, height


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot HDF5 optimization iteration cell fields on the moving wing/fuselage mesh."
    )
    parser.add_argument("--hdf5", default=str(DEFAULT_HDF5), help="Optimization HDF5 file.")
    parser.add_argument("--mesh", default=str(DEFAULT_MESH), help="Corresponding Gmsh/meshio mesh file.")
    parser.add_argument(
        "--field",
        default="auto",
        help="Cell field to plot. Use `Cp`, `None_stacked`, or `mu` for doublet strengths.",
    )
    parser.add_argument("--vertices-dataset", default="surface_mesh_vertices")
    parser.add_argument("--iterations", default="all", help="`all`, comma indices like `0,2,5`, or a slice.")
    parser.add_argument("--field-component", type=int, default=0)
    parser.add_argument("--output", default=None, help="GIF output path.")
    parser.add_argument("--backend", choices=("auto", "matplotlib", "pyvista"), default="auto")
    parser.add_argument("--fps", type=float, default=1.5)
    parser.add_argument("--window-size", nargs=2, type=int, default=(1100, 760), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--clim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--cp-bounds",
        nargs=2,
        type=float,
        default=DEFAULT_CP_BOUNDS,
        metavar=("MIN", "MAX"),
        help="Cp color bounds used when plotting Cp and --clim is not supplied.",
    )
    parser.add_argument("--camera-position", default="-15.949790460168874, -19.552118758282074, 11.932912176782372", help="Camera position as `x,y,z`.")
    parser.add_argument("--focal-point", default="12.5, 0.0, -0.0008540016693756591", help="Camera focal point as `x,y,z`.")
    parser.add_argument("--view-up", default="0.19261896258213604, 0.29159676709963983, 0.9369467757940256", help="Camera view-up vector as `x,y,z`.")
    parser.add_argument("--elev", type=float, default=22.0, help="Matplotlib-style elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-62.0, help="Matplotlib-style azimuth in degrees.")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll in degrees where supported.")
    parser.add_argument("--distance", type=float, default=1.9, help="Camera distance in bounding-box diagonals.")
    parser.add_argument("--zoom", type=float, default=0.88, help="Matplotlib 3D zoom factor.")
    parser.add_argument("--print-camera", action="store_true", help="Print the resolved camera settings.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open one PyVista view and print camera settings after rotations.",
    )
    parser.add_argument(
        "--interactive-iteration",
        default=None,
        help="Iteration for --interactive. Accepts `3` or `iteration_3`; defaults to the last selected frame.",
    )
    parser.add_argument(
        "--probe-camera",
        action="store_true",
        help="Alias for --interactive.",
    )
    parser.add_argument("--show-edges", action="store_true")
    parser.add_argument("--show-axes", action="store_true")
    parser.add_argument("--list-fields", action="store_true", help="Print HDF5 datasets and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hdf5_path = _resolve_path(args.hdf5, fallback_dirs=(Path.cwd(), REPO_ROOT))
    mesh_path = _resolve_path(args.mesh, fallback_dirs=(Path.cwd(), SCRIPT_DIR, REPO_ROOT))

    mesh = _load_mesh(mesh_path)
    if args.list_fields:
        _list_fields(hdf5_path, num_points=mesh.points.shape[0], num_cells=mesh.num_cells)
        return 0

    frames, resolved_field = _load_frames(
        hdf5_path,
        field_name=args.field,
        vertices_name=args.vertices_dataset,
        iteration_selector=args.iterations,
        num_points=mesh.points.shape[0],
        num_cells=mesh.num_cells,
        field_component=args.field_component,
    )
    field_label = "mu" if args.field == "mu" and resolved_field == "None_stacked" else resolved_field
    output_path = Path(args.output).expanduser() if args.output else hdf5_path.with_name(
        f"{hdf5_path.stem}_{field_label}.gif"
    )
    bounds = _global_bounds(frame.vertices for frame in frames)
    camera = _camera_from_args(args, bounds)
    clim = _field_limits(frames, args.clim, field_label=field_label, cp_bounds=args.cp_bounds)
    backend = _choose_backend(args.backend)
    window_size = _window_size(args.window_size)

    print(f"loaded mesh: {mesh.points.shape[0]} points, {mesh.num_cells} cells")
    print(f"loaded HDF5: {hdf5_path}")
    print(f"selected frames: {len(frames)}")
    print(f"field: {field_label} clim={clim}")
    print(f"backend: {backend}")
    if args.print_camera:
        _print_camera(camera)

    if args.interactive or args.probe_camera:
        interactive_frame = _select_interactive_frame(frames, args.interactive_iteration)
        _show_pyvista_camera_probe(
            interactive_frame,
            mesh,
            field_label=field_label,
            camera=camera,
            clim=clim,
            cmap_name=args.cmap,
            show_edges=args.show_edges,
        )
        return 0

    if backend == "pyvista":
        _render_gif_pyvista(
            frames,
            mesh,
            output_path=output_path,
            field_label=field_label,
            camera=camera,
            clim=clim,
            cmap_name=args.cmap,
            fps=args.fps,
            window_size=window_size,
            show_edges=args.show_edges,
        )
    else:
        _render_gif_matplotlib(
            frames,
            mesh,
            output_path=output_path,
            field_label=field_label,
            camera=camera,
            bounds=bounds,
            clim=clim,
            cmap_name=args.cmap,
            fps=args.fps,
            window_size=window_size,
            show_edges=args.show_edges,
            show_axes=args.show_axes,
        )

    print(f"wrote GIF: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
