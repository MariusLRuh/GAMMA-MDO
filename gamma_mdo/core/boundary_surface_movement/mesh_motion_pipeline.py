"""Reusable, configuration-driven surface and volume mesh-motion pipeline.

The user-facing deformation and DAFoam drivers instantiate CSDL design
variables and configuration dataclasses explicitly, then call the builder in
this module. No run configuration is read from command-line arguments or
environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np

import gamma_mdo
from .geometry_model import GeometryModel
from .mesh_motion_config import (
    _IntersectionRecord,
    MeshMotionResult,
    QualityChecks,
    InputFiles,
    MeshMotion,
    SurfaceProjectionStatus,
    SurfaceVertexClassification,
    VolumeMotion,
)


def _ordered_surface_cells(mesh):
    """Return the surface cells in the imported mesh's own stored order.

    Faces are kept as polygons *without* triangulating. The CFD wall is
    hex-dominant; fan-triangulating an n-gon invents sliver cells that report
    spurious inversions and hands the elastic solve the wrong (diagonal) edges.

    This reads ``mesh.connectivity``, so it preserves whatever face sequence the
    reader decoded. For the curated ``.npz`` wall that is the original source
    face order, which is what the fold diagnostics are defined against. It is
    deliberately *not* the same ordering as :func:`_cfd_surface_cells`, which
    regroups by width for the quality and inversion reports.

    Parameters
    ----------
    mesh
        Imported surface, whose ``connectivity`` is either a uniform-width 2-D
        array or an object array of variable-width rows.

    Returns
    -------
    list of numpy.ndarray
        One ``int64`` node-ID array per face, in stored order.
    """
    connectivity = mesh.connectivity
    if connectivity is None:
        return []
    array = np.asarray(connectivity, dtype=object) if connectivity.dtype == object else np.asarray(connectivity, dtype=np.int64)
    if array.dtype == object:
        return [np.asarray(face, dtype=np.int64).reshape(-1) for face in array]
    if array.ndim != 2:
        return []
    return [np.asarray(row, dtype=np.int64).reshape(-1) for row in array]


def _cfd_surface_cells(mesh):
    """Return the cells in the ordering the quality report uses.

    The order is triangles, then quads, then each ``polygonN`` block, so
    inverted-element ids from ``check_element_inversion`` map to the right cell.
    """
    ordered = ["triangle", "quad"] + [
        t for t in mesh.cell_blocks if t not in ("triangle", "quad")
    ]
    cells = []
    for cell_type in ordered:
        block = mesh.cell_blocks.get(cell_type)
        if block is None:
            continue
        for cell in np.asarray(block, dtype=np.int64):
            cells.append(cell)
    return cells


def _polygon_normals(vertices, connectivity):
    """Compute the area-weighted (Newell) normal of every polygon.

    Used for a fold check that is meaningful on the original polygonal cells
    rather than on fan triangles.

    Vectorized over all directed edges of all polygons (scatter-add by polygon
    id) so it is O(total-edges) in NumPy rather than a Python loop per cell.
    """
    poly_ids = np.concatenate(
        [np.full(face.size, pid, dtype=np.int64) for pid, face in enumerate(connectivity)]
    )
    a_idx = np.concatenate([face for face in connectivity])
    b_idx = np.concatenate([np.roll(face, -1) for face in connectivity])
    a = vertices[a_idx]
    b = vertices[b_idx]
    contrib = np.empty_like(a)
    contrib[:, 0] = (a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2])
    contrib[:, 1] = (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0])
    contrib[:, 2] = (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])
    normals = np.zeros((len(connectivity), 3), dtype=float)
    np.add.at(normals, poly_ids, contrib)
    return normals


def _surface_has_ngons(connectivity) -> bool:
    """Report whether any cell has four or more sides.

    Parameters
    ----------
    connectivity
        Sequence of per-cell vertex-index arrays.

    Returns
    -------
    bool
        ``True`` when at least one cell can carry an affine hourglass mode.
    """
    return any(len(cell) >= 4 for cell in connectivity)


def _count_polygon_folds(initial_vertices, final_vertices, connectivity):
    """Count polygons whose area-weighted normal flipped direction.

    A polygon has folded when the dot product of its undeformed and deformed
    area-weighted normals is negative.
    """
    n0 = _polygon_normals(initial_vertices, connectivity)
    n1 = _polygon_normals(final_vertices, connectivity)
    dots = np.einsum("ij,ij->i", n0, n1)
    return int(np.sum(dots < 0.0)), dots


def _setup_code_signature(extra_params):
    """Fingerprint the code and parameters that determine the cached setup.

    The setup covers baseline projection ownership and seams.  Hashing the
    *contents* of every
    module the setup depends on means the cache is invalidated whenever any of
    that logic changes -- otherwise a cache written by an older/broken code
    version is silently reused and produces garbage (folded mesh) from a benign
    deformation, which is impossible to diagnose from the run alone.
    """
    import hashlib
    import importlib

    digest = hashlib.sha1()
    module_names = [
        "gamma_mdo.preprocessing.movement",
        "gamma_mdo.preprocessing.intersections",
        "gamma_mdo.core.projections.warm_start_candidate_projection_numpy",
        "gamma_mdo.core.projections.orthogonality_projection_numpy",
        "gamma_mdo.core.projections.function_set_closest_distance_custom_op",
        "gamma_mdo.core.boundary_surface_movement.intersections",
    ]
    for name in module_names:
        try:
            module = importlib.import_module(name)
            digest.update(Path(module.__file__).read_bytes())
        except Exception:
            digest.update(name.encode())
    digest.update(repr(extra_params).encode())
    return digest.hexdigest()


def _run_volume_motion(
    volume_mesh,
    final_wall_positions,
    wall_position_history,
    load_fractions,
    *,
    volume_config: VolumeMotion,
    quality_config: QualityChecks,
    output_directory: Path,
    surface_mesh_file: Path,
    surface_distortion_weight: float,
):
    """Run and write the requested volume propagators.

    ``VolumeMotion.load_mode`` selects the volume path independently of the number
    of surface load steps:

    * ``final`` -- one differentiable reference-stiffness ``evaluate`` solve
      driven by the final wall, factored once on the baseline volume mesh.  This
      is the default and stays in the CSDL graph even for multi-step surfaces.
    * ``synchronized`` -- reassemble/refactor from every intermediate deformed
      volume state in NumPy; forward-only, deferred derivatives.  When
      ``synchronized_load_steps`` is set, the volume gets its own uniform
      baseline-to-final continuation so surface and volume robustness controls
      remain independent.

    The exact wall history and per-step wall meshes are always written so the
    volume-only replay tool can reproduce either path.
    """
    volume_load_mode = volume_config.load_mode
    volume_methods = volume_config.methods
    graph_stiffening = volume_config.graph_stiffening_exponent
    elasticity_poisson_ratio = volume_config.elasticity_poisson_ratio
    elasticity_stiffening = volume_config.elasticity_stiffening_exponent
    evaluate_gmsh_quality = quality_config.gmsh_volume_metrics
    output_directory = Path(output_directory).expanduser()

    surface_fractions = tuple(float(value) for value in load_fractions)
    if not surface_fractions:
        raise ValueError("Volume motion requires at least one load fraction.")
    surface_targets = [
        np.asarray(
            item.value if hasattr(item, "value") else item,
            dtype=float,
        )
        for item in wall_position_history
    ]
    final_target = np.asarray(final_wall_positions.value, dtype=float)
    surface_targets[-1] = final_target
    fractions = surface_fractions
    targets = surface_targets
    if (
        volume_load_mode == "synchronized"
        and volume_config.synchronized_load_steps is not None
    ):
        step_count = int(volume_config.synchronized_load_steps)
        fractions = tuple(np.linspace(1.0 / step_count, 1.0, step_count))
        baseline_wall = volume_mesh.vertices[volume_mesh.aircraft_nodes]
        targets = [
            baseline_wall + fraction * (final_target - baseline_wall)
            for fraction in fractions
        ]
    history_path = (
        output_directory
        / f"e175_surface_wall_history_load{len(fractions)}.npz"
    )
    gamma_mdo.core.boundary_surface_movement.write_wall_position_history(
        history_path,
        volume_mesh,
        targets,
        fractions,
    )
    print(
        f"[volume] exact surface history written to {history_path.resolve()}",
        flush=True,
    )
    wall_mesh_paths = []
    for step, target in enumerate(targets, start=1):
        wall_mesh_path = (
            output_directory
            / (
                f"e175_surface_wall_load{len(fractions)}"
                f"_step{step:02d}.msh"
            )
        )
        gamma_mdo.core.boundary_surface_movement.write_aircraft_wall_gmsh22(
            volume_mesh,
            target,
            wall_mesh_path,
        )
        wall_mesh_paths.append(str(wall_mesh_path.resolve()))
    baseline_quality = (
        gamma_mdo.core.boundary_surface_movement.evaluate_volume_quality(
            volume_mesh.vertices,
            volume_mesh.vertices,
            volume_mesh.tetrahedra,
            pyramids=volume_mesh.pyramids,
        )
    )
    summary = {
        "volume_mesh": str(volume_mesh.source_path),
        "surface_mesh": str(surface_mesh_file.resolve()),
        "surface_load_steps": len(surface_fractions),
        "surface_load_fractions": list(surface_fractions),
        "volume_load_mode": volume_load_mode,
        "load_steps": len(fractions),
        "load_fractions": list(fractions),
        "surface_wall_history": str(history_path.resolve()),
        "surface_wall_meshes": wall_mesh_paths,
        "surface_distortion_lambda": surface_distortion_weight,
        "methods_requested": list(volume_methods),
        "baseline_quality": baseline_quality.as_dict(),
        "methods": {},
    }
    if evaluate_gmsh_quality:
        quality_started = time.perf_counter()
        summary["baseline_gmsh_quality"] = (
            gamma_mdo.core.boundary_surface_movement.evaluate_gmsh_tetra_quality(
                volume_mesh.source_path
            )
        )
        summary["baseline_gmsh_quality_seconds"] = (
            time.perf_counter() - quality_started
        )

    volume_outputs = {}
    for method in volume_methods:
        method_started = time.perf_counter()
        if volume_load_mode == "final":
            # One differentiable reference-stiffness solve driven by the final
            # wall, regardless of the number of *surface* load steps.  The
            # operator is assembled/factored once on the baseline volume mesh.
            if method == "graph":
                system = (
                    gamma_mdo.core.boundary_surface_movement
                    .assemble_graph_volume_system(
                        volume_mesh,
                        stiffening_exponent=graph_stiffening,
                    )
                )
            else:
                system = (
                    gamma_mdo.core.boundary_surface_movement
                    .assemble_elastic_volume_system(
                        volume_mesh,
                        poisson_ratio=elasticity_poisson_ratio,
                        stiffening_exponent=elasticity_stiffening,
                    )
                )
            solve_started = time.perf_counter()
            volume_variable = system.evaluate(final_wall_positions)
            solve_seconds = time.perf_counter() - solve_started
            final_vertices = np.asarray(volume_variable.value, dtype=float)
            volume_outputs[method] = volume_variable
            method_summary = {
                "mode": "differentiable_reference_stiffness",
                "differentiable": True,
                "assembly_seconds": system.assembly_seconds,
                "factorization_seconds": system.factorization_seconds,
                "solve_seconds": solve_seconds,
                "factor_backend": (
                    system.factor._backend
                    if method == "elasticity"
                    else {
                        "xz": system.factor_xz._backend,
                        "y": system.factor_y._backend,
                    }
                ),
            }
            if method == "elasticity":
                method_summary["stiffness_nnz"] = system.stiffness_nnz
        else:
            load_result = (
                gamma_mdo.core.boundary_surface_movement
                .run_forward_volume_load_steps(
                    volume_mesh,
                    targets,
                    fractions,
                    method=method,
                    graph_stiffening_exponent=graph_stiffening,
                    elasticity_poisson_ratio=elasticity_poisson_ratio,
                    elasticity_stiffening_exponent=elasticity_stiffening,
                )
            )
            final_vertices = load_result.vertices
            method_summary = {
                "mode": "forward_load_stepping_with_refactorization",
                "differentiable": False,
                "steps": [record.as_dict() for record in load_result.records],
                "total_assembly_seconds": sum(
                    record.assembly_seconds for record in load_result.records
                ),
                "total_factorization_seconds": sum(
                    record.factorization_seconds
                    for record in load_result.records
                ),
                "total_solve_seconds": sum(
                    record.solve_seconds for record in load_result.records
                ),
            }

        final_quality = (
            gamma_mdo.core.boundary_surface_movement.evaluate_volume_quality(
                volume_mesh.vertices,
                final_vertices,
                volume_mesh.tetrahedra,
                pyramids=volume_mesh.pyramids,
            )
        )
        method_summary["final_quality"] = final_quality.as_dict()
        output_path = (
            output_directory
            / (
                f"e175_volume_{method}_{volume_load_mode}"
                f"_load{len(fractions)}.msh"
            )
        )
        gamma_mdo.core.boundary_surface_movement.write_deformed_gmsh22(
            volume_mesh,
            final_vertices,
            output_path,
        )
        method_summary["output_mesh"] = str(output_path.resolve())
        if evaluate_gmsh_quality:
            quality_started = time.perf_counter()
            method_summary["gmsh_quality"] = (
                gamma_mdo.core.boundary_surface_movement
                .evaluate_gmsh_tetra_quality(output_path)
            )
            method_summary["gmsh_quality_seconds"] = (
                time.perf_counter() - quality_started
            )
        method_summary["wall_maximum_error"] = float(
            np.max(
                np.abs(
                    final_vertices[volume_mesh.aircraft_nodes]
                    - np.asarray(final_wall_positions.value, dtype=float)
                )
            )
        )
        method_summary["total_wall_seconds"] = (
            time.perf_counter() - method_started
        )
        summary["methods"][method] = method_summary

        gmsh_metrics = method_summary.get("gmsh_quality", {})
        min_sicn = gmsh_metrics.get("minSICN", {}).get("minimum")
        min_sige = gmsh_metrics.get("minSIGE", {}).get("minimum")
        print(
            f"[volume:{method}] inverted={final_quality.inverted_tetrahedra} "
            f"relative-J min/p001={final_quality.minimum_relative_jacobian:.4g}/"
            f"{final_quality.relative_jacobian_p001:.4g} "
            f"mean-ratio min/p001={final_quality.minimum_mean_ratio:.4g}/"
            f"{final_quality.mean_ratio_p001:.4g} "
            f"minSICN={min_sicn} minSIGE={min_sige} "
            f"time={method_summary['total_wall_seconds']:.2f}s",
            flush=True,
        )

    summary_path = (
        output_directory
        / (
            f"e175_volume_comparison_{volume_load_mode}"
            f"_load{len(fractions)}.json"
        )
    )
    gamma_mdo.core.boundary_surface_movement.write_comparison_summary(
        summary_path, summary
    )
    print(f"[volume] comparison written to {summary_path.resolve()}", flush=True)
    return volume_outputs, summary


reference_wing_area = 70.0
reference_wing_aspect_ratio = 8.4

intersection_tolerance = 2e-3
bisection_tolerance = 1e-10
bisection_max_iter = 80



@dataclass(frozen=True)
class _IntersectionSetup:
    spec: _IntersectionRecord
    parametric_coordinates: np.ndarray
    vertices: np.ndarray
    vertex_ids: np.ndarray


@dataclass
class _GeometrySetup:
    started_at: float
    loaded_at: float
    mesh_file: Path
    volume_output_directory: Path
    mesh: Any
    full_mesh: Any
    initial_full_vertices: np.ndarray
    initial_vertices: np.ndarray
    polygon_connectivity: list[np.ndarray]
    volume_mesh: Any | None
    symmetry_split: Any | None
    symmetry_plane_vertex_ids: np.ndarray
    components: dict[str, Any]
    component_ids: dict[str, np.ndarray]
    intersections: dict[str, _IntersectionSetup]
    initial_parametric_coordinates: np.ndarray


@dataclass(frozen=True)
class _DeformationSetup:
    load_fractions: tuple[float, ...]
    component_coefficient_steps: list[dict[int, Any]]


@dataclass
class _SurfaceSystem:
    motion: Any
    projection_ids: dict[str, np.ndarray]
    deformation_vertex_ids: np.ndarray


@dataclass
class _SurfaceResult:
    load_step_result: Any
    preprojected_mesh_vertices: csdl.Variable
    final_mesh_vertices: csdl.Variable


@dataclass(frozen=True)
class _SurfaceDiagnostics:
    inversion_report: Any
    quality_report: Any
    fold_count: int = 0
    initial_inversion_report: Any = None
    preprojection_inversion_report: Any = None


def _resolve_model_paths(
    input_files: InputFiles,
    config: MeshMotion,
) -> dict[str, Any]:
    """Resolve and validate all paths before expensive geometry setup."""
    geometry_file = input_files.geometry_file.resolve()
    mesh_file = input_files.surface_mesh_file.resolve()
    volume_mesh_file = (
        input_files.volume_mesh_file.resolve()
        if input_files.volume_mesh_file is not None
        else None
    )
    volume_wall_map_file = (
        input_files.volume_wall_map_file.resolve()
        if input_files.volume_wall_map_file is not None
        else None
    )
    cache_directory = (
        input_files.cache_directory.resolve()
        if input_files.cache_directory is not None
        else geometry_file.parent
    )
    volume_output_directory = (
        Path(config.volume.output_directory).expanduser().resolve()
        if config.volume.output_directory is not None
        else (
            volume_mesh_file.parent
            if volume_mesh_file is not None
            else mesh_file.parent
        )
        / "deformation_results"
    )
    if not geometry_file.is_file():
        raise FileNotFoundError(f"STEP geometry not found: {geometry_file}")
    surface_exists = mesh_file.is_file()
    if config.volume.methods:
        if volume_mesh_file is None or volume_wall_map_file is None:
            raise ValueError(
                "Volume motion requires volume_mesh_file and "
                "volume_wall_map_file."
            )
        if not volume_mesh_file.is_file():
            raise FileNotFoundError(f"Volume mesh not found: {volume_mesh_file}")
        wall_map_exists = volume_wall_map_file.is_file()
        if surface_exists != wall_map_exists:
            raise FileNotFoundError(
                "The surface mesh and volume-wall map are a matched pair and "
                "must either both exist or both be absent."
            )
    elif not surface_exists:
        raise FileNotFoundError(
            f"Surface mesh not found for surface-only motion: {mesh_file}"
        )
    cache_directory.mkdir(parents=True, exist_ok=True)
    if config.volume.methods:
        volume_output_directory.mkdir(parents=True, exist_ok=True)
    print(
        "[inputs] "
        f"STEP={geometry_file} surface={mesh_file} "
        f"volume={volume_mesh_file} wall_map={volume_wall_map_file}",
        flush=True,
    )
    return {
        "geometry_file": geometry_file,
        "mesh_file": mesh_file,
        "volume_mesh_file": volume_mesh_file,
        "volume_wall_map_file": volume_wall_map_file,
        "cache_directory": cache_directory,
        "volume_output_directory": volume_output_directory,
        "surface_exists": surface_exists,
    }


def _setup_geometry_and_mesh(
    input_files: InputFiles,
    geometry: GeometryModel,
    config: MeshMotion,
) -> _GeometrySetup:
    """Import geometry/meshes and cache baseline ownership and intersections."""
    started_at = time.perf_counter()
    paths = _resolve_model_paths(input_files, config)
    component_specs = tuple(geometry._component_records)
    intersection_specs = tuple(geometry._intersection_records)
    # ``lsdo_function_spaces`` writes its STEP-import cache relative to the
    # process working directory. Contain that third-party side effect next to
    # the configured cache directory so a run never dirties the checkout.
    _cache_root = paths.get("cache_directory") or paths["geometry_file"].parent
    _previous_directory = Path.cwd()
    try:
        Path(_cache_root).mkdir(parents=True, exist_ok=True)
        os.chdir(_cache_root)
        geometry = lfs.import_file_patched(
            paths["geometry_file"], parallelize=False
        )
    finally:
        os.chdir(_previous_directory)
    imported_components = gamma_mdo.preprocessing.create_components(
        search_names=[spec.search_name for spec in component_specs],
        geometry=geometry,
    )
    components = {
        spec.name: component
        for spec, component in zip(component_specs, imported_components)
    }

    volume_mesh = None
    if not paths["surface_exists"]:
        volume_mesh = gamma_mdo.core.boundary_surface_movement.read_gmsh22_volume(
            paths["volume_mesh_file"]
        )
        gamma_mdo.core.boundary_surface_movement.extract_aircraft_wall_mesh(
            volume_mesh,
            paths["mesh_file"],
            paths["volume_wall_map_file"],
        )
        print(
            "[volume] extracted exact aircraft wall from "
            f"{paths['volume_mesh_file'].name}",
            flush=True,
        )
    # Always load through the generic safe importer; suffix dispatch there
    # covers .msh, .stl, and the safe polygon .npz. The fold diagnostics use the
    # reader's stored face order, while the quality/inversion reports use the
    # width-grouped order from _cfd_surface_cells.
    mesh = gamma_mdo.preprocessing.import_mesh(paths["mesh_file"])
    polygon_connectivity = _ordered_surface_cells(mesh)
    block_summary = ", ".join(
        f"{key}:{np.asarray(block).shape[0]}"
        for key, block in mesh.cell_blocks.items()
    )
    print(
        f"[cfd] loaded {mesh.vertices.shape[0]} vertices, "
        f"{len(polygon_connectivity)} polygons (no triangulation) "
        f"[{block_summary}]"
    )
    if config.volume.methods:
        if volume_mesh is None:
            volume_mesh = gamma_mdo.core.boundary_surface_movement.read_gmsh22_volume(
                paths["volume_mesh_file"]
            )
        wall_to_volume = (
            gamma_mdo.core.boundary_surface_movement.load_wall_to_volume_map(
                paths["volume_wall_map_file"],
                volume_mesh,
                wall_vertices=np.asarray(mesh.vertices, dtype=float),
            )
        )
        topology = (
            f"{volume_mesh.tetrahedra.shape[0]} tetrahedra, "
            f"{volume_mesh.pyramids.shape[0]} pyramids"
        )
        print(
            f"[volume] exact wall map validated: {wall_to_volume.size} wall "
            f"nodes -> {volume_mesh.vertices.shape[0]} volume nodes, {topology}",
            flush=True,
        )
    loaded_at = time.perf_counter()

    full_mesh = mesh
    initial_full_vertices = np.asarray(full_mesh.vertices, dtype=float).copy()
    symmetry_split = None
    if config.symmetry and gamma_mdo.preprocessing.detect_symmetry(mesh):
        symmetry_split = gamma_mdo.preprocessing.split_symmetric_mesh(mesh)
        mesh = symmetry_split.half_mesh
        print(
            f"[symmetry] half-mesh solve: {symmetry_split.half_to_full.size} of "
            f"{full_mesh.vertices.shape[0]} vertices, "
            f"{symmetry_split.plane_local_ids.size} on the y=0 plane"
        )
    initial_vertices = np.asarray(mesh.vertices, dtype=float)
    symmetry_plane_vertex_ids = (
        symmetry_split.plane_local_ids
        if symmetry_split is not None
        else gamma_mdo.core.boundary_surface_movement.identify_symmetry_plane_vertices(
            initial_vertices,
            axis=1,
            tolerance=config.symmetry_plane_tolerance,
        )
    )
    print(
        "[symmetry] hard y=0 constraint on "
        f"{symmetry_plane_vertex_ids.size} baseline plane vertices",
        flush=True,
    )

    setup_projection_options = {
        "warm_start_nu": config.setup_projection_resolution,
        "warm_start_nv": config.setup_projection_resolution,
    }
    cache_path = paths["cache_directory"] / (
        f"_setup_cache_{paths['mesh_file'].stem}_{paths['geometry_file'].stem}"
        f"_nu{config.setup_projection_resolution}_sym{int(config.symmetry)}.npz"
    )
    mesh_mtime = paths["mesh_file"].stat().st_mtime
    geometry_mtime = paths["geometry_file"].stat().st_mtime
    code_signature = _setup_code_signature(
        (
            sorted(setup_projection_options.items()),
            tuple((spec.name, spec.search_name) for spec in component_specs),
            tuple(
                (
                    spec.name,
                    spec.driving_component,
                    spec.query_component,
                    spec.bisection_search_direction,
                )
                for spec in intersection_specs
            ),
            float(intersection_tolerance),
            float(bisection_tolerance),
            int(bisection_max_iter),
        )
    )
    setup_keys = ["initial_parametric_coordinates"]
    setup_keys.extend(f"{spec.name}_ids" for spec in component_specs)
    for spec in intersection_specs:
        setup_keys.extend(
            (
                f"{spec.name}_parametric",
                f"{spec.name}_vertices",
                f"{spec.name}_ids",
            )
        )
    cached = None
    if cache_path.exists() and not config.rebuild_setup_cache:
        with np.load(cache_path) as cache:
            if (
                float(cache["_mesh_mtime"]) == mesh_mtime
                and float(cache["_geom_mtime"]) == geometry_mtime
                and int(cache["_num_vertices"]) == initial_vertices.shape[0]
                and "_code_sig" in cache
                and str(cache["_code_sig"]) == code_signature
                and all(key in cache for key in setup_keys)
            ):
                cached = {key: cache[key] for key in setup_keys}
                print(
                    f"[cache] loaded setup projections from {cache_path.name} "
                    f"({time.perf_counter() - started_at:.1f}s)",
                    flush=True,
                )
            else:
                print("[cache] setup cache stale; rebuilding", flush=True)
    if cached is None:
        projection_cache: dict[Any, Any] = {}
        component_list = [components[spec.name] for spec in component_specs]
        projection_data = gamma_mdo.preprocessing.project_mesh_onto_components(
            mesh_vertices=initial_vertices,
            components=component_list,
            projection_cache=projection_cache,
            projection_options=setup_projection_options,
        )
        component_distances = np.column_stack(
            [np.abs(item.distances) for item in projection_data]
        )
        ownership = np.argmin(component_distances, axis=1)
        initial_parametric_coordinates = np.empty(
            (initial_vertices.shape[0], 3)
        )
        for index, projection in enumerate(projection_data):
            owned = ownership == index
            initial_parametric_coordinates[owned] = (
                projection.parametric_coordinates[owned]
            )
        patch_ids = np.rint(initial_parametric_coordinates[:, 0]).astype(np.int64)
        cached = {
            "initial_parametric_coordinates": initial_parametric_coordinates
        }
        for spec in component_specs:
            cached[f"{spec.name}_ids"] = np.where(
                np.isin(patch_ids, list(components[spec.name].functions))
            )[0].astype(np.int64)
        for spec in intersection_specs:
            parametric, vertices, vertex_ids = (
                gamma_mdo.preprocessing.identify_intersection_vertices(
                    components=[
                        components[spec.driving_component],
                        components[spec.query_component],
                    ],
                    driving_component=components[spec.driving_component],
                    mesh=mesh,
                    intersection_tolerance=intersection_tolerance,
                    projection_cache=projection_cache,
                    projection_options=setup_projection_options,
                )
            )
            cached[f"{spec.name}_parametric"] = parametric
            cached[f"{spec.name}_vertices"] = vertices
            cached[f"{spec.name}_ids"] = vertex_ids
        np.savez(
            cache_path,
            _mesh_mtime=mesh_mtime,
            _geom_mtime=geometry_mtime,
            _num_vertices=initial_vertices.shape[0],
            _code_sig=code_signature,
            **cached,
        )
        print(f"[cache] saved setup projections to {cache_path.name}", flush=True)

    component_ids = {
        spec.name: cached[f"{spec.name}_ids"] for spec in component_specs
    }
    intersections = {
        spec.name: _IntersectionSetup(
            spec=spec,
            parametric_coordinates=cached[f"{spec.name}_parametric"],
            vertices=cached[f"{spec.name}_vertices"],
            vertex_ids=cached[f"{spec.name}_ids"],
        )
        for spec in intersection_specs
    }
    return _GeometrySetup(
        started_at=started_at,
        loaded_at=loaded_at,
        mesh_file=paths["mesh_file"],
        volume_output_directory=paths["volume_output_directory"],
        mesh=mesh,
        full_mesh=full_mesh,
        initial_full_vertices=initial_full_vertices,
        initial_vertices=initial_vertices,
        polygon_connectivity=polygon_connectivity,
        volume_mesh=volume_mesh,
        symmetry_split=symmetry_split,
        symmetry_plane_vertex_ids=symmetry_plane_vertex_ids,
        components=components,
        component_ids=component_ids,
        intersections=intersections,
        initial_parametric_coordinates=cached["initial_parametric_coordinates"],
    )


def _parameterize_geometry(
    setup: _GeometrySetup,
    geometry: GeometryModel,
    config: MeshMotion,
) -> _DeformationSetup:
    """Evaluate driver-supplied component deformations at every load step."""
    load_fractions = tuple(
        gamma_mdo.core.boundary_surface_movement.linear_load_fractions(
            config.surface.load_steps
        )
    )
    intersection_vertices = {
        name: data.vertices for name, data in setup.intersections.items()
    }
    steps: list[dict[int, Any]] = []
    for load_fraction in load_fractions:
        step: dict[int, Any] = {}
        for spec in geometry._component_records:
            component = setup.components[spec.name]
            step[id(component)] = spec.coefficient_builder(
                component,
                float(load_fraction),
                intersection_vertices,
            )
        steps.append(step)
    return _DeformationSetup(load_fractions, steps)


def _concatenate_ids(blocks) -> np.ndarray:
    arrays = [np.asarray(block, dtype=np.int64) for block in blocks]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays)).astype(np.int64)


def _build_intersections_and_graph(
    setup: _GeometrySetup,
    deformation: _DeformationSetup,
    geometry: GeometryModel,
    config: MeshMotion,
) -> _SurfaceSystem:
    """Build exact intersection constraints and the graph-Laplacian system."""
    projection_options = {
        "warm_start_nu": config.projection_warm_start_resolution,
        "warm_start_nv": config.projection_warm_start_resolution,
    }
    intersection_parameters = []
    for data in setup.intersections.values():
        spec = data.spec
        intersection_parameters.append(
            gamma_mdo.core.boundary_surface_movement.IntersectionParameters(
                name=spec.solver_name,
                parametric_coords=data.parametric_coordinates,
                vertex_ids=data.vertex_ids,
                driving_component=setup.components[spec.driving_component],
                sdf_query_component=setup.components[spec.query_component],
                bisection_search_direction=spec.bisection_search_direction,
                bisection_tolerance=bisection_tolerance,
                bisection_max_iter=bisection_max_iter,
                projection_options=projection_options,
            )
        )
    seam_ids = _concatenate_ids(
        data.vertex_ids for data in setup.intersections.values()
    )
    free_regions = [
        spec.free_region_factory(setup.components[spec.name])
        for spec in geometry._component_records
    ]
    free_ids = gamma_mdo.core.boundary_surface_movement.select_free_vertices(
        free_regions=free_regions,
        component_vertex_ids={
            id(setup.components[name]): vertex_ids
            for name, vertex_ids in setup.component_ids.items()
        },
        mesh_vertices=setup.initial_vertices,
        exclude_ids=seam_ids,
    )
    projection_ids: dict[str, np.ndarray] = {}
    claimed = np.empty(0, dtype=np.int64)
    for spec in geometry._component_records:
        component_free_ids = np.intersect1d(
            free_ids, setup.component_ids[spec.name]
        )
        driven_seams = [
            data.vertex_ids
            for data in setup.intersections.values()
            if data.spec.driving_component == spec.name
        ]
        candidate = _concatenate_ids([component_free_ids, *driven_seams])
        projection_ids[spec.name] = np.setdiff1d(candidate, claimed)
        claimed = np.union1d(claimed, projection_ids[spec.name])
    deformation_vertex_ids = np.concatenate(
        [projection_ids[spec.name] for spec in geometry._component_records]
    ).astype(np.int64)

    distance = config.surface.distance_weighting
    distance_weighting = None
    if distance.enabled:
        seed_names = (
            tuple(setup.intersections)
            if distance.seed_intersections is None
            else distance.seed_intersections
        )
        unknown = set(seed_names).difference(setup.intersections)
        if unknown:
            raise ValueError(
                "Unknown graph-distance seed intersections: "
                + ", ".join(sorted(unknown))
            )
        distance_seed_ids = _concatenate_ids(
            setup.intersections[name].vertex_ids for name in seed_names
        )
        distance_band_ids = np.union1d(
            free_ids,
            gamma_mdo.core.boundary_surface_movement.graph_neighbors(
                setup.mesh, free_ids
            ),
        )
        distance_weighting = (
            gamma_mdo.core.boundary_surface_movement.build_graph_distance_weighting(
                setup.mesh,
                distance_seed_ids,
                beta=distance.beta,
                length=distance.length_scale,
                cap=distance.cap,
                decay=distance.decay,
                rational_power=distance.power,
                restrict_vertex_ids=distance_band_ids,
            )
        )
        band_edges = set()
        for block in setup.mesh.cell_blocks.values():
            for cell in np.asarray(block, dtype=np.int64):
                for vertex_a, vertex_b in zip(cell, np.roll(cell, -1)):
                    vertex_a, vertex_b = int(vertex_a), int(vertex_b)
                    if vertex_a != vertex_b:
                        band_edges.add(
                            (min(vertex_a, vertex_b), max(vertex_a, vertex_b))
                        )
        summary = distance_weighting.summary(
            np.asarray(sorted(band_edges), dtype=np.int64)
        )
        print(
            "[graph-distance] "
            f"seeds={','.join(seed_names)}({distance_seed_ids.size}) "
            f"beta={distance.beta:g} length={distance.length_scale:g} "
            f"cap={distance.cap:g} decay={distance.decay} "
            f"reachable={summary['num_reachable_vertices']} "
            f"max_dist={summary['max_finite_distance']:.3g} "
            "mult[min/med/max]="
            f"{summary['multiplier_min']:.3g}/"
            f"{summary['multiplier_median']:.3g}/"
            f"{summary['multiplier_max']:.3g}",
            flush=True,
        )
    surface = config.surface
    prescribed_ids = None
    if surface.distortion_penalty.weight > 0.0 or surface.polygon_regularization.weight > 0.0:
        prescribed_ids = gamma_mdo.core.boundary_surface_movement.element_neighbors(
            setup.mesh, free_ids
        )
    motion = gamma_mdo.core.boundary_surface_movement.ElasticityMotionSolver(
        mesh=setup.mesh,
        free_ids=free_ids,
        intersection_params=intersection_parameters,
        parametric_coordinates=setup.initial_parametric_coordinates,
        components=list(setup.components.values()),
        component_reevaluations=(),
        use_query_seam_reference=config.query_seam_reference,
        symmetry_plane_ids=setup.symmetry_plane_vertex_ids,
        stiffening_exponent=surface.stiffening_exponent,
        prescribed_ids=prescribed_ids,
        symmetry_use_element_neighbors=(surface.polygon_regularization.weight > 0.0),
        distance_weighting=distance_weighting,
        quad_diagonal_weight=surface.quad_diagonal_weight,
        quad_bracing_mode=surface.quad_bracing_mode,
    )
    return _SurfaceSystem(motion, projection_ids, deformation_vertex_ids)


def _lifting_surface_metadata(
    component,
    projection_ids: np.ndarray,
    parent_coordinates: np.ndarray,
    initial_vertices: np.ndarray,
    name: str,
    mode: str,
) -> list[Any]:
    """Create projection metadata for a lifting-surface component."""
    if projection_ids.size == 0:
        return []
    if mode == "parent":
        return [
            gamma_mdo.preprocessing.get_projection_metadata(
                component=component,
                vertices=initial_vertices[projection_ids],
                vertex_ids=projection_ids,
                para_coords=parent_coordinates,
                name=name,
            )
        ]
    if mode == "any":
        return [
            gamma_mdo.preprocessing.get_projection_metadata(
                component=component,
                vertices=initial_vertices[projection_ids],
                vertex_ids=projection_ids,
                para_coords=None,
                name=name,
            )
        ]
    sides = gamma_mdo.core.boundary_surface_movement.classify_patch_sides(component)
    node_patch = np.rint(parent_coordinates[projection_ids, 0]).astype(np.int64)
    node_side = np.array([sides.get(int(patch), 0) for patch in node_patch])
    groups = []
    for side in sorted(set(node_side.tolist())):
        selected = projection_ids[node_side == side]
        if selected.size == 0:
            continue
        allowed_patch_ids = (
            None
            if side == 0
            else [patch for patch, value in sides.items() if value == side]
        )
        groups.append(
            gamma_mdo.preprocessing.get_projection_metadata(
                component=component,
                vertices=initial_vertices[selected],
                vertex_ids=selected,
                para_coords=None,
                allowed_patch_ids=allowed_patch_ids,
                name=f"{name}_side{side:+d}",
            )
        )
    return groups


def _reproject_and_reevaluate(
    setup: _GeometrySetup,
    deformation: _DeformationSetup,
    system: _SurfaceSystem,
    geometry: GeometryModel,
    config: MeshMotion,
) -> _SurfaceResult:
    """Solve load steps, reproject selected nodes, and reevaluate the rest."""
    projection_metadata = []
    for spec in geometry._component_records:
        component = setup.components[spec.name]
        vertex_ids = system.projection_ids[spec.name]
        if spec.projection_metadata_builder is not None:
            projection_metadata.extend(spec.projection_metadata_builder(
                component=component,
                vertex_ids=vertex_ids,
                initial_parametric_coordinates=setup.initial_parametric_coordinates,
                initial_vertices=setup.initial_vertices,
            ))
        elif spec.projection_mode == "lifting_surface":
            parent_coordinates = setup.initial_parametric_coordinates.copy()
            for data in setup.intersections.values():
                if data.spec.driving_component == spec.name:
                    parent_coordinates[data.vertex_ids] = (
                        data.parametric_coordinates
                    )
            projection_metadata.extend(
                _lifting_surface_metadata(
                    component,
                    vertex_ids,
                    parent_coordinates,
                    setup.initial_vertices,
                    spec.projection_name,
                    config.lifting_surface_patch_mode,
                )
            )
        else:
            projection_metadata.append(
                gamma_mdo.preprocessing.get_projection_metadata(
                    component=component,
                    vertices=setup.initial_vertices[vertex_ids],
                    vertex_ids=vertex_ids,
                    para_coords=None,
                    name=spec.projection_name,
                )
            )
    reevaluation_metadata = gamma_mdo.preprocessing.identify_reevaluated_vertices(
        mesh=setup.mesh,
        vertex_ids=system.deformation_vertex_ids,
        parametric_coords=setup.initial_parametric_coordinates,
        components=[
            setup.components[spec.name]
            for spec in geometry._component_records
        ],
        names=[spec.projection_name for spec in geometry._component_records],
    )
    surface = config.surface
    distortion = surface.distortion_penalty
    polygon_regularization = surface.polygon_regularization
    projection_options = {
        "warm_start_nu": config.projection_warm_start_resolution,
        "warm_start_nv": config.projection_warm_start_resolution,
    }
    load_step_result = gamma_mdo.core.boundary_surface_movement.run_graph_load_steps(
        motion=system.motion,
        mesh=setup.mesh,
        initial_deformation_vertices=csdl.Variable(
            name="deformation_vertices",
            value=setup.initial_vertices[system.deformation_vertex_ids],
        ),
        deformation_vertex_ids=system.deformation_vertex_ids,
        component_coefficient_steps=deformation.component_coefficient_steps,
        projection_metadata=projection_metadata,
        reevaluation_metadata=reevaluation_metadata,
        projection_options=projection_options,
        load_fractions=deformation.load_fractions,
        distortion_config=(
            None
            if distortion.weight == 0.0
            else gamma_mdo.core.boundary_surface_movement.QuadraticDistortionConfig(
                lambda_dist=distortion.weight,
                mode=distortion.mode,
                coefficients=(
                    gamma_mdo.core.boundary_surface_movement.DistortionModeCoefficients(
                        area=distortion.area,
                        deviatoric=distortion.deviatoric,
                        shear=distortion.shear,
                        rotation=distortion.rotation,
                        normal=distortion.normal,
                    )
                ),
            )
        ),
        # Positive polygon regularization is applicable-if-present: a
        # triangle-only surface has no hourglass modes, so the affine model is
        # skipped rather than assembled on a structurally zero matrix.
        ngon_affine_config=(
            None
            if polygon_regularization.weight == 0.0
            or not _surface_has_ngons(setup.polygon_connectivity)
            else gamma_mdo.core.boundary_surface_movement.NgonAffineConfig(
                lambda_ngon=polygon_regularization.weight,
            )
        ),
        symmetry_plane_vertex_ids=setup.symmetry_plane_vertex_ids,
        symmetry_plane_axis=1,
    )
    preprojected = gamma_mdo.core.boundary_surface_movement.enforce_symmetry_plane(
        load_step_result.final_preprojected_mesh_vertices,
        vertex_ids=setup.symmetry_plane_vertex_ids,
        axis=1,
    )
    final = gamma_mdo.core.boundary_surface_movement.enforce_symmetry_plane(
        load_step_result.final_mesh_vertices,
        vertex_ids=setup.symmetry_plane_vertex_ids,
        axis=1,
    )
    final = gamma_mdo.core.boundary_surface_movement.enforce_symmetry_plane(
        final,
        vertex_ids=setup.symmetry_plane_vertex_ids,
        axis=1,
    )
    print(
        "[time] after project-onto-OML: "
        f"{time.perf_counter() - setup.started_at:.1f}s",
        flush=True,
    )
    if setup.symmetry_split is not None:
        preprojected = gamma_mdo.preprocessing.reconstruct_full_from_half(
            preprojected, setup.symmetry_split
        )
        final = gamma_mdo.preprocessing.reconstruct_full_from_half(
            final, setup.symmetry_split
        )
    return _SurfaceResult(load_step_result, preprojected, final)


def _run_volume_handoff(
    setup: _GeometrySetup,
    deformation: _DeformationSetup,
    surface_result: _SurfaceResult,
    config: MeshMotion,
) -> tuple[dict[str, csdl.Variable], dict | None]:
    """Extend the surface displacement into the optional volume mesh."""
    if setup.volume_mesh is None:
        return {}, None
    if config.surface.load_steps == 1:
        wall_history = (surface_result.final_mesh_vertices,)
    else:
        wall_history = surface_result.load_step_result.projected_mesh_history
        if setup.symmetry_split is not None:
            wall_history = tuple(
                gamma_mdo.preprocessing.reconstruct_full_from_half(
                    item, setup.symmetry_split
                )
                for item in wall_history
            )
    return _run_volume_motion(
        setup.volume_mesh,
        surface_result.final_mesh_vertices,
        wall_history,
        deformation.load_fractions,
        volume_config=config.volume,
        quality_config=config.quality,
        output_directory=setup.volume_output_directory,
        surface_mesh_file=setup.mesh_file,
        surface_distortion_weight=config.surface.distortion_penalty.weight,
    )


def _global_surface_ids(
    setup: _GeometrySetup,
    half_mesh_ids,
) -> np.ndarray:
    """Expand internal half-mesh IDs into complete input-mesh IDs."""
    ids = np.asarray(half_mesh_ids, dtype=np.int64).reshape(-1)
    if setup.symmetry_split is None:
        return np.unique(ids).astype(np.int64)
    return np.where(
        np.isin(setup.symmetry_split.gather_index, ids)
    )[0].astype(np.int64)


def _global_intersection_vertices(
    setup: _GeometrySetup,
    half_mesh_ids,
    half_mesh_vertices,
    global_ids: np.ndarray,
) -> np.ndarray:
    """Align intersection coordinates with complete-mesh vertex IDs."""
    local_ids = np.asarray(half_mesh_ids, dtype=np.int64).reshape(-1)
    local_vertices = np.asarray(half_mesh_vertices, dtype=float)
    if local_vertices.shape != (local_ids.size, 3):
        raise ValueError(
            "Intersection vertex IDs and coordinates must be aligned row-wise."
        )

    row_for_local_id = np.full(
        setup.initial_vertices.shape[0], -1, dtype=np.int64
    )
    row_for_local_id[local_ids] = np.arange(local_ids.size, dtype=np.int64)
    if setup.symmetry_split is None:
        source_ids = global_ids
        signs = None
    else:
        source_ids = setup.symmetry_split.gather_index[global_ids]
        signs = setup.symmetry_split.mirror_sign[global_ids]

    rows = row_for_local_id[source_ids]
    if np.any(rows < 0):
        raise ValueError(
            "Complete-mesh intersection IDs do not map to retained-half "
            "coordinates."
        )
    vertices = local_vertices[rows]
    return vertices if signs is None else vertices * signs


def _surface_vertex_classification(
    setup: _GeometrySetup,
    system: _SurfaceSystem,
    surface_result: _SurfaceResult,
) -> SurfaceVertexClassification:
    """Build the public surface classification in global mesh indexing."""
    deformation_ids = _global_surface_ids(
        setup, system.deformation_vertex_ids
    )
    closest_ids = _global_surface_ids(
        setup,
        surface_result.load_step_result.final_reprojected_vertex_ids,
    )
    return SurfaceVertexClassification(
        deformation_vertex_ids=deformation_ids,
        closest_projection_vertex_ids=closest_ids,
        parametrically_prescribed_vertex_ids=np.setdiff1d(
            np.arange(setup.initial_full_vertices.shape[0], dtype=np.int64),
            deformation_ids,
            assume_unique=True,
        ),
        graph_free_vertex_ids=_global_surface_ids(
            setup, system.motion.free_ids
        ),
        graph_prescribed_vertex_ids=_global_surface_ids(
            setup, system.motion.prescribed_ids
        ),
        symmetry_plane_vertex_ids=_global_surface_ids(
            setup, setup.symmetry_plane_vertex_ids
        ),
        component_vertex_ids={
            name: _global_surface_ids(setup, vertex_ids)
            for name, vertex_ids in setup.component_ids.items()
        },
        intersection_vertex_ids={
            name: _global_surface_ids(setup, data.vertex_ids)
            for name, data in setup.intersections.items()
        },
    )


def _surface_projection_status(
    setup: _GeometrySetup,
    surface_result: _SurfaceResult,
) -> SurfaceProjectionStatus:
    """Build the public final-step closest-projection convergence status."""
    load_result = surface_result.load_step_result
    return SurfaceProjectionStatus(
        reprojected_vertex_ids=_global_surface_ids(
            setup, load_result.final_reprojected_vertex_ids
        ),
        nonconverged_vertex_ids=_global_surface_ids(
            setup, load_result.final_nonconverged_vertex_ids
        ),
    )


def _write_diagnostic_dump(
    setup: _GeometrySetup,
    surface_result: _SurfaceResult,
    classification: SurfaceVertexClassification,
    inversion_report,
    final_vertices: np.ndarray,
    dump_path: Path,
) -> None:
    """Write generic component/intersection diagnostics to an NPZ archive."""
    parametric = (
        setup.initial_parametric_coordinates
        if setup.symmetry_split is None
        else setup.initial_parametric_coordinates[
            setup.symmetry_split.gather_index
        ]
    )
    payload = {
        "preprojected_vertices": np.asarray(
            surface_result.preprojected_mesh_vertices.value, dtype=float
        ),
        "final_vertices": final_vertices,
        "initial_vertices": np.asarray(setup.full_mesh.vertices, dtype=float),
        "triangles": np.asarray(
            setup.full_mesh.cell_blocks.get("triangle", np.empty((0, 3))),
            dtype=np.int64,
        ),
        "quads": np.asarray(
            setup.full_mesh.cell_blocks.get("quad", np.empty((0, 4))),
            dtype=np.int64,
        ),
        "inverted_element_ids": inversion_report.inverted_element_ids,
        "initial_parametric_coordinates": parametric,
        "deformation_vertex_ids": classification.deformation_vertex_ids,
        "closest_projection_vertex_ids": (
            classification.closest_projection_vertex_ids
        ),
        "parametrically_prescribed_vertex_ids": (
            classification.parametrically_prescribed_vertex_ids
        ),
        "graph_free_ids": classification.graph_free_vertex_ids,
        "graph_prescribed_ids": classification.graph_prescribed_vertex_ids,
        "symmetry_plane_vertex_ids": (
            classification.symmetry_plane_vertex_ids
        ),
    }
    for name, vertex_ids in classification.component_vertex_ids.items():
        payload[f"{name}_ids"] = vertex_ids
    for name, vertex_ids in classification.intersection_vertex_ids.items():
        intersection = setup.intersections[name]
        payload[f"{name}_vertices"] = _global_intersection_vertices(
            setup,
            intersection.vertex_ids,
            intersection.vertices,
            vertex_ids,
        )
        payload[f"{name}_ids"] = vertex_ids
    np.savez(dump_path, **payload)
    print(f"[diagnostics] dumped arrays to {dump_path}")


def _evaluate_surface_diagnostics(
    setup: _GeometrySetup,
    surface_result: _SurfaceResult,
    classification: SurfaceVertexClassification,
    config: MeshMotion,
) -> _SurfaceDiagnostics:
    """Evaluate and report surface quality without changing pipeline values."""
    preprojected = surface_result.preprojected_mesh_vertices
    final = surface_result.final_mesh_vertices
    # Three states, one metric: the untouched input mesh, the deformed surface
    # before reprojection, and the final reprojected surface. Comparing any two
    # of these is only meaningful because they use the same check.
    initial_inversion = (
        gamma_mdo.core.boundary_surface_movement.check_element_inversion(
            mesh=setup.full_mesh,
            final_mesh_vertices=np.asarray(
                setup.initial_full_vertices, dtype=float
            ),
        )
    )
    pre_inversion = gamma_mdo.core.boundary_surface_movement.check_element_inversion(
        mesh=setup.full_mesh, final_mesh_vertices=preprojected
    )
    pre_quality = gamma_mdo.core.boundary_surface_movement.evaluate_mesh_quality(
        mesh=setup.full_mesh, vertices=preprojected
    )
    inversion = gamma_mdo.core.boundary_surface_movement.check_element_inversion(
        mesh=setup.full_mesh, final_mesh_vertices=final
    )
    quality = gamma_mdo.core.boundary_surface_movement.evaluate_mesh_quality(
        mesh=setup.full_mesh, vertices=final
    )
    final_np = np.asarray(final.value, dtype=float)
    preprojected_np = np.asarray(preprojected.value, dtype=float)
    full_plane_ids = (
        setup.symmetry_plane_vertex_ids
        if setup.symmetry_split is None
        else setup.symmetry_split.half_to_full[setup.symmetry_plane_vertex_ids]
    )
    pre_plane_error = (
        float(np.max(np.abs(preprojected_np[full_plane_ids, 1])))
        if full_plane_ids.size
        else 0.0
    )
    final_plane_error = (
        float(np.max(np.abs(final_np[full_plane_ids, 1])))
        if full_plane_ids.size
        else 0.0
    )
    surface_cells = _cfd_surface_cells(setup.full_mesh)
    seam_blocks = [data.vertices for data in setup.intersections.values()]
    seam_vertices = np.vstack(seam_blocks) if seam_blocks else np.empty((0, 3))
    if setup.symmetry_split is not None and seam_vertices.size:
        mirrored = seam_vertices.copy()
        mirrored[:, setup.symmetry_split.axis] *= -1.0
        seam_vertices = np.vstack((seam_vertices, mirrored))
    near_seam = 0
    for element_id in inversion.inverted_element_ids:
        centroid = final_np[surface_cells[int(element_id)]].mean(axis=0)
        if seam_vertices.size and float(
            np.min(np.linalg.norm(seam_vertices - centroid, axis=1))
        ) < 1.0:
            near_seam += 1
    surface = config.surface
    load_result = surface_result.load_step_result
    print(
        f"[diagnostics] elasticity=graph chi={surface.stiffening_exponent} "
        f"quad_bracing_mode={surface.quad_bracing_mode} "
        f"quad_diagonal_weight={surface.quad_diagonal_weight:g} "
        f"surface_load_steps={surface.load_steps} "
        f"volume_load_mode={config.volume.load_mode} "
        f"distortion_lambda={surface.distortion_penalty.weight:g} "
        f"distortion_mode={surface.distortion_penalty.mode} "
        f"ngon_affine_lambda={surface.polygon_regularization.weight:g}"
    )
    if load_result.distortion_normalization_scale is not None:
        print(
            "[diagnostics] distortion "
            f"normalization={load_result.distortion_normalization_scale:.6g} "
            f"redundancy={load_result.distortion_redundancy:.6f} "
            f"ear_clipped={load_result.distortion_num_ear_clipped} "
            f"max_warp={load_result.distortion_maximum_warp_ratio:.6g}"
        )
    if load_result.ngon_affine_normalization_scale is not None:
        print(
            "[diagnostics] polygon_regularization "
            f"normalization={load_result.ngon_affine_normalization_scale:.6g} "
            f"elements={load_result.ngon_affine_num_elements} "
            f"modes={load_result.ngon_affine_num_modes} "
            f"max_warp={load_result.ngon_affine_maximum_warp_ratio:.6g}"
        )
    print(
        "[diagnostics] symmetry-plane max |y| "
        f"PRE/POST: {pre_plane_error:.3e}/{final_plane_error:.3e}"
    )
    print(
        f"[diagnostics] PRE inverted elements: {pre_inversion.num_inverted}; "
        f"min scaled Jacobian: {pre_quality.minimum_scaled_jacobian:.4f}"
    )
    print(
        f"[diagnostics] POST inverted elements: {inversion.num_inverted} "
        f"(within 1 m of a seam: {near_seam})"
    )
    print(f"[diagnostics] POST inverted corners: {quality.inverted_corners}")
    print(
        "[diagnostics] POST min scaled Jacobian: "
        f"{quality.minimum_scaled_jacobian:.4f}"
    )
    print(
        f"[diagnostics] p05 scaled Jacobian: {quality.scaled_jacobian_p05:.4f}"
    )
    print(
        f"[diagnostics] max aspect ratio: {quality.maximum_aspect_ratio:.2f} "
        f"(p95 {quality.aspect_ratio_p95:.2f})"
    )
    print(
        "[diagnostics] area ratio range: "
        f"[{quality.minimum_area_ratio:.3f}, {quality.maximum_area_ratio:.3f}]"
    )
    if config.diagnostic_dump:
        _write_diagnostic_dump(
            setup,
            surface_result,
            classification,
            inversion,
            final_np,
            config.diagnostic_dump,
        )
    if inversion.num_inverted:
        print("[diagnostics] inverted element centroids (deformed):")
        for element_id in inversion.inverted_element_ids[:20]:
            centroid = final_np[surface_cells[int(element_id)]].mean(axis=0)
            print(
                f"    cell {int(element_id):6d}  "
                f"({centroid[0]:8.3f}, {centroid[1]:8.3f}, "
                f"{centroid[2]:8.3f})"
            )
        if inversion.num_inverted > 20:
            print(f"    ... {inversion.num_inverted - 20} more")
    cfd_folds, _ = _count_polygon_folds(
        np.asarray(setup.full_mesh.vertices, dtype=float),
        final_np,
        setup.polygon_connectivity,
    )
    print(
        f"[cfd] polygon folds (normal-flip): {cfd_folds} of "
        f"{len(setup.polygon_connectivity)} polygons"
    )
    return _SurfaceDiagnostics(
        inversion, quality, int(cfd_folds), initial_inversion, pre_inversion
    )


def run_mesh_motion(
    *,
    recorder: csdl.Recorder,
    input_files: InputFiles,
    geometry: GeometryModel,
    config: MeshMotion,
    aerodynamic_analysis: Callable[
        [csdl.Variable], Mapping[str, csdl.Variable]
    ]
    | None = None,
    aerodynamic_volume_method: str = "elasticity",
) -> MeshMotionResult:
    """Build and evaluate a differentiable geometry-to-volume mesh pipeline.

    Parameters
    ----------
    recorder
        Active recorder that owns the supplied design variables.
    input_files
        Geometry, surface-mesh, and optional volume-mesh inputs.
    geometry
        User-supplied component/intersection declarations and differentiable
        coefficient builders.
    config
        Surface, volume, projection, quality, and visualization settings.
    aerodynamic_analysis
        Optional downstream CSDL builder receiving the selected final volume
        coordinates and returning named aerodynamic outputs.
    aerodynamic_volume_method
        Enabled volume-motion method forwarded to ``aerodynamic_analysis``.

    Returns
    -------
    MeshMotionResult
        Differentiable mesh outputs and forward quality diagnostics.
    """
    _started_at = time.perf_counter()
    geometry.validate()

    _ = recorder
    setup = _setup_geometry_and_mesh(
        input_files, geometry, config
    )
    deformation = _parameterize_geometry(
        setup, geometry, config
    )
    system = _build_intersections_and_graph(
        setup, deformation, geometry, config
    )
    surface_result = _reproject_and_reevaluate(
        setup, deformation, system, geometry, config
    )
    volume_outputs, volume_summary = _run_volume_handoff(
        setup, deformation, surface_result, config
    )
    classification = _surface_vertex_classification(
        setup, system, surface_result
    )
    projection_status = _surface_projection_status(setup, surface_result)
    diagnostics = _evaluate_surface_diagnostics(
        setup, surface_result, classification, config
    )
    _ngon_mode_count = (
        getattr(surface_result.load_step_result, "ngon_affine_num_modes", 0)
        or 0
    )
    aerodynamic_outputs: dict[str, csdl.Variable] = {}
    if aerodynamic_analysis is not None:
        if aerodynamic_volume_method not in volume_outputs:
            raise ValueError(
                "The requested aerodynamic volume method "
                f"{aerodynamic_volume_method!r} is unavailable. Enable that "
                "method in VolumeMotion before adding CFD."
            )
        aerodynamic_outputs = dict(
            aerodynamic_analysis(volume_outputs[aerodynamic_volume_method])
        )
    elapsed = time.perf_counter() - setup.started_at
    print(
        f"[cfd] END-TO-END pipeline time: {elapsed:.1f}s for "
        f"{setup.mesh.vertices.shape[0]} vertices "
        f"(load {setup.loaded_at - setup.started_at:.1f}s)"
    )
    setup.full_mesh.nodes = surface_result.final_mesh_vertices.value
    if config.visualization.enabled:
        gamma_mdo.plotting.plot_mesh(
            mesh=setup.full_mesh,
            plotting_elements=None,
            opacity=config.visualization.opacity,
            show=True,
            inverted_elements=diagnostics.inversion_report,
        )
    return MeshMotionResult(
        input_files=input_files,
        geometry=geometry,
        initial_surface_coordinates=setup.initial_full_vertices,
        preprojected_surface_coordinates=(
            surface_result.preprojected_mesh_vertices
        ),
        surface_coordinates=surface_result.final_mesh_vertices,
        volume_coordinates=volume_outputs,
        aerodynamic_outputs=aerodynamic_outputs,
        surface_inversion_report=diagnostics.inversion_report,
        surface_quality_report=diagnostics.quality_report,
        volume_quality_summary=volume_summary,
        volume_mesh=setup.volume_mesh,
        surface_mesh=setup.full_mesh,
        recorder=recorder,
        elapsed_seconds=time.perf_counter() - _started_at,
        surface_fold_count=diagnostics.fold_count,
        surface_cell_count=len(setup.polygon_connectivity),
        surface_ngon_mode_count=int(_ngon_mode_count),
        initial_inversion_report=diagnostics.initial_inversion_report,
        preprojection_inversion_report=(
            diagnostics.preprojection_inversion_report
        ),
        surface_vertex_classification=classification,
        surface_projection_status=projection_status,
    )


def select_fd_objective(
    result: MeshMotionResult,
    objective_name: str,
) -> csdl.Variable:
    """Select a scalar objective for the optional finite-difference sweep.

    Parameters
    ----------
    result
        Completed mesh-motion pipeline result.
    objective_name
        Surface, enabled volume-method, or aerodynamic output name.

    Returns
    -------
    csdl.Variable
        Scalar objective registered on the active recorder.

    Raises
    ------
    ValueError
        If ``objective_name`` is not available in ``result``.
    """
    if objective_name == "surface_coordinates":
        objective = csdl.sum(result.surface_coordinates)
    elif objective_name.startswith("volume_"):
        method = objective_name.removeprefix("volume_")
        if method not in result.volume_coordinates:
            raise ValueError(
                f"FD objective {objective_name!r} requires volume method "
                f"{method!r} to be enabled."
            )
        objective = csdl.sum(result.volume_coordinates[method])
    elif objective_name in result.aerodynamic_outputs:
        objective = result.aerodynamic_outputs[objective_name]
    else:
        choices = [
            "surface_coordinates",
            *(
                f"volume_{name}"
                for name in result.volume_coordinates
            ),
            *result.aerodynamic_outputs,
        ]
        raise ValueError(
            f"Unknown FD objective {objective_name!r}; choose one of {choices}."
        )
    objective.name = f"fd_objective_{objective_name}"
    objective.set_as_objective()
    return objective


def run_fd_sweep(
    recorder: csdl.Recorder,
    step_sizes: tuple[float, ...],
) -> dict[str, dict[float, float]]:
    """Run and print a finite-difference convergence sweep.

    Parameters
    ----------
    recorder
        Recorder containing the objective and design variables.
    step_sizes
        Perturbation sizes evaluated by the JAX simulator.

    Returns
    -------
    dict[str, dict[float, float]]
        Relative errors keyed first by design-variable name and then by step
        size.
    """
    simulator = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    sweep: dict[str, dict[float, float]] = {}
    for step in step_sizes:
        results = simulator.check_optimization_derivatives(
            step_size=step,
            print_results=False,
            raise_on_error=False,
        )
        for key, entry in results.items():
            name = str(entry.get("wrt_name", key[1]))
            sweep.setdefault(name, {})[step] = float(entry["rel_error"])

    names = sorted(sweep)
    print("\n=== FD derivative convergence sweep (relative error) ===")
    print("  step      " + "".join(f"{name[:20]:>22s}" for name in names))
    for step in step_sizes:
        print(
            f"  {step:<9.0e}"
            + "".join(f"{sweep[name][step]:>22.3e}" for name in names)
        )
    print()
    for name in names:
        errors = [sweep[name][step] for step in step_sizes]
        best = int(np.argmin(errors))
        converging = errors[best] < errors[0]
        print(
            f"  {name}: min rel error {errors[best]:.3e} at step "
            f"{step_sizes[best]:.0e}  "
            f"({'converges' if converging else 'NO CONVERGENCE'})"
        )
    return sweep


__all__ = [
    "run_mesh_motion",
    "run_fd_sweep",
    "select_fd_objective",
]
