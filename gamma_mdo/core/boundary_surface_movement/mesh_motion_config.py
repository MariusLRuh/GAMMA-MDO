"""Configuration, parameterization, and result types for mesh motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import csdl_alpha as csdl
import numpy as np


@dataclass(frozen=True)
class InputFiles:
    """Geometry and surface/optional-volume mesh inputs for one analysis.

    When volume motion is enabled, ``surface_mesh_file`` must be the
    aircraft-wall boundary extracted from ``volume_mesh_file`` and
    ``volume_wall_map_file`` must store the corresponding surface-to-volume
    node map. Surface-only analyses may omit both volume paths.

    Parameters
    ----------
    geometry_file
        STEP file containing the source geometry.
    surface_mesh_file
        Surface mesh whose nodes are moved and reprojected.
    volume_mesh_file
        Optional volume mesh associated with the surface mesh.
    volume_wall_map_file
        Optional surface-to-volume node map.
    cache_directory
        Optional directory for reusable setup data.
    """

    geometry_file: Path
    surface_mesh_file: Path
    volume_mesh_file: Path | None = None
    volume_wall_map_file: Path | None = None
    cache_directory: Path | None = None

    def __post_init__(self):
        """Normalize the configured paths.

        Each supplied path is expanded and resolved and stored back on the
        instance; ``None`` entries are left as-is.
        """
        for name in (
            "geometry_file",
            "surface_mesh_file",
            "volume_mesh_file",
            "volume_wall_map_file",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())
        if self.cache_directory is not None:
            object.__setattr__(
                self,
                "cache_directory",
                Path(self.cache_directory).expanduser(),
            )


@dataclass(frozen=True)
class DistanceWeighting:
    """Configure distance-dependent graph-edge stiffening.

    Parameters
    ----------
    enabled
        Whether graph-distance weighting is active.
    beta
        Nonnegative magnitude of the edge-weight increase.
    length_scale
        Positive physical decay length.
    cap
        Upper bound on the resulting multiplier.
    decay
        Decay law, either ``"exp"`` or ``"rational"``.
    power
        Exponent used by the rational decay law.
    seed_intersections
        Optional intersection names used as distance seeds.

    Raises
    ------
    ValueError
        If a numeric bound, decay law, or seed-name list is invalid.
    """

    enabled: bool = True
    beta: float = 2.0
    length_scale: float = 4.0
    cap: float = np.inf
    decay: str = "exp"
    power: float = 1.0
    seed_intersections: tuple[str, ...] | None = None

    def __post_init__(self):
        """Validate the graph-distance weighting parameters.

        Raises
        ------
        ValueError
            If ``beta`` is negative, ``length_scale`` is not positive, ``cap``
            is below one, ``decay`` is not ``"exp"`` or ``"rational"``, or the
            seed-intersection names are malformed.
        """
        if self.beta < 0.0 or self.length_scale <= 0.0 or self.cap < 1.0:
            raise ValueError("Invalid graph-distance weighting parameters.")
        if self.decay not in ("exp", "rational"):
            raise ValueError("Distance decay must be exp or rational.")
        if self.seed_intersections is not None:
            names = tuple(str(name) for name in self.seed_intersections)
            if not names or any(not name for name in names):
                raise ValueError(
                    "Distance seed_intersections must contain nonempty names."
                )
            if len(set(names)) != len(names):
                raise ValueError(
                    "Distance seed_intersections must not contain duplicates."
                )
            object.__setattr__(self, "seed_intersections", names)


@dataclass(frozen=True)
class DistortionPenalty:
    """Configure the fixed quadratic element-distortion penalty.

    Parameters
    ----------
    weight
        Nonnegative global regularization strength.
    mode
        Distortion formulation selected by the assembler.
    area
        Relative area-change penalty.
    deviatoric
        Relative deviatoric-strain penalty.
    shear
        Relative shear penalty.
    rotation
        Relative in-plane rotation penalty.
    normal
        Relative out-of-plane normal penalty.

    Raises
    ------
    ValueError
        If ``weight`` is negative.
    """

    weight: float = 0.0
    mode: str = "strain_distortion"
    area: float = 1.0
    deviatoric: float = 1.0
    shear: float = 1.0
    rotation: float = 0.0
    normal: float = 0.5

    def __post_init__(self):
        """Validate the distortion regularization weight.

        Raises
        ------
        ValueError
            If ``weight`` is negative.
        """
        if self.weight < 0.0:
            raise ValueError("Distortion regularization weight cannot be negative.")


@dataclass(frozen=True)
class PolygonRegularization:
    """Configure the element-local affine-residual polygon penalty.

    Parameters
    ----------
    weight
        Finite nonnegative penalty applied to polygons with at least four
        vertices.

    Raises
    ------
    ValueError
        If ``weight`` is negative or non-finite.
    """

    weight: float = 0.0

    def __post_init__(self):
        """Validate the n-gon affine regularization weight.

        Raises
        ------
        ValueError
            If ``weight`` is non-finite or negative.
        """
        value = float(self.weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "N-gon affine regularization weight must be finite and nonnegative."
            )
        object.__setattr__(self, "weight", value)


@dataclass(frozen=True)
class SurfaceMotion:
    """Configure graph-Laplacian surface-mesh propagation.

    Parameters
    ----------
    load_steps
        Positive number of continuation increments.
    stiffening_exponent
        Reference-area exponent used by graph edge weights.
    quad_diagonal_weight
        Nonnegative weight for optional quadrilateral bracing.
    quad_bracing_mode
        Bracing topology used when diagonal weighting is active.
    distance_weighting
        Settings for distance-dependent graph stiffening.
    distortion
        Optional quadratic distortion penalty.
    polygon_regularization
        Optional affine-residual polygon penalty.

    Raises
    ------
    ValueError
        If an option is invalid or incompatible regularizers are enabled.
    """

    load_steps: int = 2
    stiffening_exponent: float = 1.5
    quad_diagonal_weight: float = 0.0
    quad_bracing_mode: str = "both_diagonals"
    distance_weighting: DistanceWeighting = field(
        default_factory=DistanceWeighting
    )
    distortion_penalty: DistortionPenalty = field(
        default_factory=DistortionPenalty
    )
    polygon_regularization: PolygonRegularization = field(
        default_factory=PolygonRegularization
    )

    def __post_init__(self):
        """Validate the surface-motion settings.

        Raises
        ------
        ValueError
            If ``load_steps`` is below one, or the quad-diagonal weight is
            non-finite or negative.
        """
        if self.load_steps < 1:
            raise ValueError("Surface load_steps must be positive.")
        if (
            not np.isfinite(float(self.quad_diagonal_weight))
            or float(self.quad_diagonal_weight) < 0.0
        ):
            raise ValueError(
                "Surface quad_diagonal_weight must be finite and non-negative."
            )
        if self.quad_bracing_mode not in (
            "single_diagonal",
            "both_diagonals",
            "virtual_center",
        ):
            raise ValueError(
                "Surface quad_bracing_mode must be single_diagonal, "
                "both_diagonals, or virtual_center."
            )
        if (
            self.distortion_penalty.weight > 0.0
            and self.polygon_regularization.weight > 0.0
        ):
            raise ValueError(
                "The first-milestone n-gon study does not combine affine and "
                "quadratic distortion regularization."
            )


@dataclass(frozen=True)
class VolumeMotion:
    """Configure propagation from the surface into a volume mesh.

    Parameters
    ----------
    mode
        Enabled volume solver: ``"off"``, ``"graph"``, ``"elasticity"``,
        or ``"both"``.
    load_mode
        Apply only the final surface state or synchronized increments.
    synchronized_load_steps
        Optional positive override for synchronized continuation.
    graph_stiffening_exponent
        Cell-size exponent for graph volume motion.
    elasticity_poisson_ratio
        Poisson ratio for linear elasticity.
    elasticity_stiffening_exponent
        Cell-size exponent for elasticity stiffness.
    output_directory
        Optional directory for generated volume meshes.
    write_meshes
        Whether to write deformed meshes to disk.

    Raises
    ------
    ValueError
        If a mode or physical/numerical parameter is invalid.
    """

    mode: str = "off"
    load_mode: str = "final"
    synchronized_load_steps: int | None = None
    graph_stiffening_exponent: float = 0.3
    elasticity_poisson_ratio: float = 0.3
    elasticity_stiffening_exponent: float = 0.75
    output_directory: Path | None = None
    write_meshes: bool = False

    def __post_init__(self):
        """Validate the volume-motion settings and normalize the output path.

        Raises
        ------
        ValueError
            If ``mode`` is not one of ``"off"``, ``"graph"``, ``"elasticity"``,
            or ``"both"``; if ``load_mode`` is not ``"final"`` or
            ``"synchronized"``; or if another volume setting is outside its
            permitted range.
        """
        if self.mode not in ("off", "graph", "elasticity", "both"):
            raise ValueError("Invalid volume motion mode.")
        if self.load_mode not in ("final", "synchronized"):
            raise ValueError("Volume load_mode must be final or synchronized.")
        if (
            self.synchronized_load_steps is not None
            and self.synchronized_load_steps < 1
        ):
            raise ValueError("Volume synchronized_load_steps must be positive.")
        if (
            self.synchronized_load_steps is not None
            and self.load_mode != "synchronized"
        ):
            raise ValueError(
                "Volume synchronized_load_steps requires synchronized mode."
            )
        if not (0.0 <= self.elasticity_poisson_ratio < 0.5):
            raise ValueError("Elasticity Poisson ratio must lie in [0, 0.5).")
        if self.elasticity_stiffening_exponent < 0.0:
            raise ValueError("Elasticity stiffening exponent must be nonnegative.")

    @property
    def methods(self) -> tuple[str, ...]:
        """Return the concrete volume solvers selected by ``mode``.

        Returns
        -------
        tuple[str, ...]
            Zero, one, or both of ``"graph"`` and ``"elasticity"``.
        """
        return {
            "off": (),
            "graph": ("graph",),
            "elasticity": ("elasticity",),
            "both": ("graph", "elasticity"),
        }[self.mode]


@dataclass(frozen=True)
class QualityChecks:
    """Configure surface and volume quality evaluation.

    Parameters
    ----------
    surface
        Whether to evaluate surface metrics.
    volume
        Whether to evaluate volume metrics.
    gmsh_volume_metrics
        Whether to request Gmsh-specific volume metrics.
    fail_on_surface_inversion
        Whether inverted surface elements raise an error.
    fail_on_volume_inversion
        Whether inverted volume elements raise an error.
    """

    surface: bool = True
    volume: bool = False
    gmsh_volume_metrics: bool = False
    fail_on_surface_inversion: bool = True
    fail_on_volume_inversion: bool = True


@dataclass(frozen=True)
class Visualization:
    """Configure optional interactive surface visualization.

    Parameters
    ----------
    enabled
        Whether visualization is produced.
    opacity
        Surface opacity supplied to the plotting backend.
    """

    enabled: bool = False
    opacity: float = 1.0


@dataclass(frozen=True)
class DerivativeCheck:
    """Configure the optional driver-level finite-difference sweep.

    Parameters
    ----------
    enabled
        Whether to run the sweep.
    objective
        Named scalar objective selected from the pipeline result.
    step_sizes
        Perturbation sizes evaluated by the derivative checker.
    """

    enabled: bool = False
    objective: str = "surface_coordinates"
    step_sizes: tuple[float, ...] = (
        1.0e-2,
        1.0e-3,
        1.0e-4,
        1.0e-5,
        1.0e-6,
    )


@dataclass(frozen=True)
class SurfaceVertexClassification:
    """Classify surface vertices in the complete input-mesh index space.

    The arrays contain zero-based global vertex IDs from the complete surface
    mesh, even when a symmetric half mesh was used internally. Mirrored
    vertices inherit the classification of the half-mesh vertex from which
    they were reconstructed.

    Attributes
    ----------
    deformation_vertex_ids
        Vertices participating in graph motion, exact intersection motion, or
        closest-point reprojection.
    closest_projection_vertex_ids
        Deformation vertices passed through closest-point OML reprojection.
        Exact intersection vertices are deliberately excluded.
    parametrically_prescribed_vertex_ids
        Vertices outside the deformation set, updated by fixed-parametric
        reevaluation on their owning component.
    graph_free_vertex_ids
        Unknown rows solved by the graph-Laplacian system.
    graph_prescribed_vertex_ids
        Boundary rows prescribed to the graph-Laplacian system.
    symmetry_plane_vertex_ids
        Vertices constrained to the configured symmetry plane.
    component_vertex_ids
        Global vertex IDs keyed by declared component name.
    intersection_vertex_ids
        Exact bracketed-intersection vertex IDs keyed by intersection name.
    """

    deformation_vertex_ids: np.ndarray
    closest_projection_vertex_ids: np.ndarray
    parametrically_prescribed_vertex_ids: np.ndarray
    graph_free_vertex_ids: np.ndarray
    graph_prescribed_vertex_ids: np.ndarray
    symmetry_plane_vertex_ids: np.ndarray
    component_vertex_ids: dict[str, np.ndarray]
    intersection_vertex_ids: dict[str, np.ndarray]


@dataclass(frozen=True)
class SurfaceProjectionStatus:
    """Report closest-point convergence in global surface-mesh indexing.

    Attributes
    ----------
    reprojected_vertex_ids
        Vertices actually passed through the closest-point operation. Exact
        intersection vertices and fixed-parametric reevaluation vertices are
        not included.
    nonconverged_vertex_ids
        Reprojected vertices whose final warm-started Newton solve did not
        report convergence. The pipeline returns their best candidates rather
        than raising, and derivatives at those points are not guaranteed.
    """

    reprojected_vertex_ids: np.ndarray
    nonconverged_vertex_ids: np.ndarray

    @property
    def num_reprojected(self) -> int:
        """Return the number of vertices sent through closest projection."""
        return int(self.reprojected_vertex_ids.size)

    @property
    def num_nonconverged(self) -> int:
        """Return the number of closest projections that did not converge."""
        return int(self.nonconverged_vertex_ids.size)


@dataclass(frozen=True)
class MeshMotion:
    """Collect all mesh-motion pipeline settings.

    Parameters
    ----------
    surface
        Surface graph-motion configuration.
    volume
        Optional volume-motion configuration.
    quality
        Quality evaluation and failure policy.
    visualization
        Interactive visualization settings.
    derivative_check
        Optional derivative-sweep settings.
    symmetry
        Whether a fixed symmetry plane is enforced.
    symmetry_plane_tolerance
        Coordinate tolerance used to identify symmetry-plane vertices.
    setup_projection_resolution
        Sampling resolution for setup-time projection.
    projection_warm_start_resolution
        Sampling resolution for projection warm starts.
    rebuild_setup_cache
        Whether cached setup data is ignored and rebuilt.
    query_seam_reference
        Whether seam reference points follow the query component.
    lifting_surface_patch_mode
        Patch restriction used for lifting-surface projection.
    diagnostic_dump
        Optional path for diagnostic output.

    Raises
    ------
    ValueError
        If a tolerance, resolution, or patch mode is invalid.
    """

    surface: SurfaceMotion = field(
        default_factory=SurfaceMotion
    )
    volume: VolumeMotion = field(
        default_factory=VolumeMotion
    )
    quality: QualityChecks = field(
        default_factory=QualityChecks
    )
    visualization: Visualization = field(
        default_factory=Visualization
    )
    derivative_check: DerivativeCheck = field(
        default_factory=DerivativeCheck
    )
    symmetry: bool = False
    symmetry_plane_tolerance: float = 1.0e-8
    setup_projection_resolution: int = 80
    projection_warm_start_resolution: int = 130
    rebuild_setup_cache: bool = False
    query_seam_reference: bool = True
    lifting_surface_patch_mode: str = "side"
    diagnostic_dump: Path | None = None

    def __post_init__(self):
        """Validate the pipeline tolerances, resolutions, and modes.

        Raises
        ------
        ValueError
            If the symmetry-plane tolerance is negative, a resolution is below
            one, or another pipeline setting is outside its permitted range.
        """
        if self.symmetry_plane_tolerance < 0.0:
            raise ValueError("Symmetry-plane tolerance cannot be negative.")
        if self.setup_projection_resolution < 1:
            raise ValueError("Setup projection resolution must be positive.")
        if self.projection_warm_start_resolution < 1:
            raise ValueError("Projection warm-start resolution must be positive.")
        if self.lifting_surface_patch_mode not in ("side", "any", "parent"):
            raise ValueError(
                "Lifting-surface patch mode must be side, any, or parent."
            )


@dataclass(frozen=True)
class _ComponentRecord:
    """Describe one geometry component without embedding aircraft-specific logic.

    Parameters
    ----------
    name
        Stable identifier used in mappings and cache keys.
    search_name
        Name supplied to the geometry importer's component search.
    coefficient_builder
        Driver-supplied callable returning the component coefficients for one
        load fraction. It receives the imported component, the load fraction,
        and the baseline intersection-vertex mapping.
    free_region_factory
        Driver-supplied callable creating the component's graph free region.
    projection_name
        Diagnostic name used by projection and reevaluation metadata.
    projection_mode
        ``"lifting_surface"`` uses the configured patch-side restriction;
        ``"all"`` projects against every patch on the component.
    projection_metadata_builder
        Optional callable defining component-specific projection groups.
    """

    name: str
    search_name: str
    coefficient_builder: Callable[[Any, float, Mapping[str, np.ndarray]], Any]
    free_region_factory: Callable[[Any], Any]
    projection_name: str | None = None
    projection_mode: str = "all"
    projection_metadata_builder: Callable[..., Any] | None = None

    def __post_init__(self):
        if not self.name or not self.name.isidentifier():
            raise ValueError("_ComponentRecord.name must be a nonempty identifier.")
        if not self.search_name:
            raise ValueError("_ComponentRecord.search_name cannot be empty.")
        if self.projection_mode not in ("lifting_surface", "all"):
            raise ValueError(
                "_ComponentRecord.projection_mode must be lifting_surface or all."
            )
        if self.projection_metadata_builder is not None and not callable(
            self.projection_metadata_builder
        ):
            raise ValueError("projection_metadata_builder must be callable.")
        if self.projection_name is None:
            object.__setattr__(self, "projection_name", self.name)


@dataclass(frozen=True)
class _IntersectionRecord:
    """Describe one independent closed component-intersection curve.

    Parameters
    ----------
    name
        Stable identifier for the intersection.
    driving_component
        Component whose parametric line drives the solve.
    query_component
        Component providing the signed-distance residual.
    bisection_search_direction
        Parametric coordinate varied by the bracketed solve.
    solver_name
        Optional diagnostic name; defaults to ``name``.

    Raises
    ------
    ValueError
        If the name is invalid or both component names are equal.
    """

    name: str
    driving_component: str
    query_component: str
    bisection_search_direction: str = "u"
    solver_name: str | None = None

    def __post_init__(self):
        if not self.name or not self.name.isidentifier():
            raise ValueError(
                "_IntersectionRecord.name must be a nonempty identifier."
            )
        if self.driving_component == self.query_component:
            raise ValueError(
                "An intersection must use different driving and query components."
            )
        if self.solver_name is None:
            object.__setattr__(self, "solver_name", self.name)

@dataclass
class MeshMotionResult:
    """Collect differentiable outputs and forward diagnostics.

    Attributes
    ----------
    model_files
        Input file contracts used to build the model.
    geometry_parameterization
        User-supplied geometry controls and declarations.
    initial_surface_coordinates
        Baseline surface coordinates.
    preprojected_surface_coordinates
        Surface coordinates before OML reprojection.
    surface_coordinates
        Final reprojected surface coordinates.
    volume_coordinates
        Deformed volume coordinates keyed by motion method.
    aerodynamic_outputs
        Optional downstream aerodynamic results.
    initial_inversion_report
        Surface-orientation diagnostics of the untouched input mesh.
    preprojection_inversion_report
        Diagnostics of the deformed surface before OML reprojection.
    surface_inversion_report
        Diagnostics of the final reprojected surface. All three use the same
        metric, so any two may be compared directly.
    surface_quality_report
        Aggregate surface-quality diagnostics.
    volume_quality_summary
        Optional volume-quality metrics.
    volume_mesh
        Optional loaded volume-mesh object.
    surface_mesh
        Loaded surface-mesh object.
    surface_vertex_classification
        Global vertex-ID sets identifying graph, parametric, symmetry, and
        exact-intersection roles.
    surface_projection_status
        Closest-point reprojection IDs and non-converged subset. The final
        surface is composite: graph-moved vertices use differentiable
        closest-point reprojection, exact intersections retain the bracketed
        solve, and all remaining vertices use fixed-parametric reevaluation.
    """

    input_files: InputFiles
    geometry: Any
    initial_surface_coordinates: np.ndarray
    preprojected_surface_coordinates: csdl.Variable
    surface_coordinates: csdl.Variable
    volume_coordinates: dict[str, csdl.Variable]
    aerodynamic_outputs: dict[str, csdl.Variable]
    surface_inversion_report: Any
    surface_quality_report: Any
    volume_quality_summary: dict | None
    volume_mesh: Any | None
    surface_mesh: Any
    surface_vertex_classification: SurfaceVertexClassification
    surface_projection_status: SurfaceProjectionStatus
    recorder: Any = None
    elapsed_seconds: float = 0.0
    surface_fold_count: int = 0
    surface_cell_count: int = 0
    surface_ngon_mode_count: int = 0
    initial_inversion_report: Any = None
    preprojection_inversion_report: Any = None

    def print_summary(self) -> None:
        """Print a concise forward-diagnostic summary of this solve.

        Reports mesh size, fold and inversion counts, projection convergence,
        elapsed wall-clock time, and aggregate surface-quality metrics already
        computed by the pipeline. Nothing is recomputed here.
        """
        vertices = int(self.initial_surface_coordinates.shape[0])
        print("mesh motion summary")
        print(f"  surface vertices    : {vertices}")
        print(f"  surface cells       : {self.surface_cell_count}")
        print(f"  n-gon modes         : {self.surface_ngon_mode_count}")
        print(f"  elapsed             : {self.elapsed_seconds:.1f} s")
        print(f"  folds               : {self.surface_fold_count}")
        for label, report in (
            ("input", self.initial_inversion_report),
            ("preprojection", self.preprojection_inversion_report),
            ("final", self.surface_inversion_report),
        ):
            if report is not None:
                print(
                    f"  inverted ({label:<13}): {report.num_inverted}"
                )
        report = self.surface_quality_report
        if report is not None:
            print(f"  degenerate elements : {report.degenerate_elements}")
            print(
                "  min scaled Jacobian : "
                f"{report.minimum_scaled_jacobian:.4g}"
            )
        status = self.surface_projection_status
        print(f"  closest projected   : {status.num_reprojected}")
        print(f"  projection failures : {status.num_nonconverged}")


__all__ = [
    "DerivativeCheck",
    "DistanceWeighting",
    "DistortionPenalty",
    "InputFiles",
    "MeshMotion",
    "MeshMotionResult",
    "PolygonRegularization",
    "QualityChecks",
    "SurfaceMotion",
    "SurfaceProjectionStatus",
    "SurfaceVertexClassification",
    "Visualization",
    "VolumeMotion",
]
