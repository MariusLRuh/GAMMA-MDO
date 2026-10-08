"""E175 geometry -> surface motion -> volume motion -> DAFoam in CSDL.

This executable is intentionally configured in Python: there is no argument
parser and no environment-variable configuration. Fill in ``CASE.case_directory``
when an OpenFOAM/DAFoam case template is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

# Support module execution and direct execution by absolute file path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import csdl_alpha as csdl
import numpy as np

from bsm3.core.boundary_surface_movement.dafoam_csdl import (
    DAFoamAnalysisOperation,
    PYDAFoamBackend,
    add_csdl_inputs_to_da_options,
    make_patch_velocity,
)
from bsm3.core.boundary_surface_movement.geometry_volume_backend import (
    MeshMotionVolumeBackend,
    read_gmsh_volume_point_count,
)
from bsm3.core.boundary_surface_movement.geometry_volume_mpi import (
    is_root,
    resolve_comm,
)
from bsm3.core.boundary_surface_movement.geometry_volume_operation import (
    GeometryVolumeOperation,
)
from bsm3.core.boundary_surface_movement.geometry_model import GeometryModel
from bsm3.core.boundary_surface_movement.mesh_motion_config import (
    DerivativeCheck,
    DistanceWeighting,
    MeshMotionResult,
    QualityChecks,
    InputFiles,
    MeshMotion,
    SurfaceMotion,
    Visualization,
    VolumeMotion,
)
from bsm3.core.boundary_surface_movement.mesh_motion_pipeline import (
    run_mesh_motion,
    run_fd_sweep,
    select_fd_objective,
)
from bsm3.core.boundary_surface_movement.run_dafoam_gmsh import (
    FlowConfig,
    build_da_options,
    convert_and_check_mesh,
    set_control_dict_max_iterations,
    validate_case_template,
    validate_reused_mesh,
)
from bsm3.core.boundary_surface_movement.volume_mesh_motion import (
    read_gmsh22_volume,
)


ASSET_DIRECTORY = Path(__file__).resolve().parent
# Local-only: this OpenVSP Euler mesh carries the OpenFOAM patch names that
# CASE expects (aircraft, symmetry, inlet, outlet, farfield). It is not
# published with GAMMA; supply a mesh with the same patches to run DAFoam.
OPENVSP_MESH_DIRECTORY = ASSET_DIRECTORY / "openvsp_euler_volume_mesh"


# ---------------------------------------------------------------------------
# 1. Geometry and matching mesh files
# ---------------------------------------------------------------------------
MODEL_FILES = InputFiles(
    geometry_file=ASSET_DIRECTORY / "e175.stp",
    surface_mesh_file=OPENVSP_MESH_DIRECTORY / "e175_openvsp_aircraft_wall.msh",
    volume_mesh_file=OPENVSP_MESH_DIRECTORY / "e175_euler_volume.msh",
    volume_wall_map_file=(
        OPENVSP_MESH_DIRECTORY
        / "e175_openvsp_aircraft_wall.volume_map.npz"
    ),
    cache_directory=ASSET_DIRECTORY,
)
DEFORMATION_OUTPUT_DIRECTORY = OPENVSP_MESH_DIRECTORY / "deformation_results"


# ---------------------------------------------------------------------------
# 2. Geometry design point
# ---------------------------------------------------------------------------
# Small nonzero changes avoid evaluating derivative checks exactly at CAD seams.
GEOMETRY_VALUES = {
    "wing_translation_x": 0.01,
    "wing_rotation_degrees": 0.01,
    "tail_rotation_degrees": 0.01,
    "wing_area": 70.01,
    "wing_aspect_ratio": 8.41,
    "fuselage_diameter_scale": 1.001,
}


#: Names of the six geometric design variables this driver parameterizes.
GEOMETRY_VARIABLE_NAMES: tuple[str, ...] = (
    "wing_translation_x",
    "wing_rotation_degrees",
    "tail_rotation_degrees",
    "wing_area",
    "wing_aspect_ratio",
    "fuselage_diameter_scale",
)


def _populate_e175_geometry(
    geometry: GeometryModel,
    variables: Mapping[str, Any],
) -> GeometryModel:
    """Declare the E175 components and intersections on ``geometry``.

    The single source of truth for the driver's geometry declaration. Both
    :func:`create_geometry_model` and
    :func:`create_geometry_parameterization_from_variables` call it, so the two
    entry points cannot drift apart.

    Parameters
    ----------
    geometry
        Model to populate. It is modified in place and returned.
    variables
        The six driving quantities keyed by
        :data:`GEOMETRY_VARIABLE_NAMES`. Each value is used as supplied, so it
        may be a registered design variable or any caller-owned CSDL
        expression.

    Returns
    -------
    GeometryModel
        The same object, with two lifting surfaces, one body, and two
        intersections declared.
    """
    geometry.add_lifting_surface(
        name="wing",
        search_name="wing",
        pivot_intersection="wing_fuse",
        translation_x=variables["wing_translation_x"],
        rotation_y_degrees=variables["wing_rotation_degrees"],
        area=variables["wing_area"],
        aspect_ratio=variables["wing_aspect_ratio"],
        reference_area=70.0,
        reference_aspect_ratio=8.4,
    )
    geometry.add_lifting_surface(
        name="tail",
        search_name="HT",
        pivot_intersection="tail_fuse",
        rotation_y_degrees=variables["tail_rotation_degrees"],
        projection_name="horizontal_tail",
    )
    geometry.add_body(
        name="fuselage",
        search_name="fuselage",
        diameter_scale=variables["fuselage_diameter_scale"],
    )
    geometry.connect(
        name="wing_fuse",
        driving_component="wing",
        query_component="fuselage",
        solver_name="wing_fuselage",
    )
    geometry.connect(
        name="tail_fuse",
        driving_component="tail",
        query_component="fuselage",
        solver_name="tail_fuselage",
    )
    return geometry


def create_geometry_parameterization_from_variables(
    variables: Mapping[str, Any],
) -> GeometryModel:
    """Build the E175 geometry model from caller-owned variables.

    The rank-0 geometry-to-volume backend owns its own private recorder and
    supplies its own CSDL variables, so this entry point registers **no** design
    variables and owns no recorder. The supplied expressions are used exactly as
    given, which keeps the caller's derivative graph intact.

    The resulting components and intersections are identical to
    :func:`create_geometry_model`; only the source of the six driving
    quantities differs.

    Parameters
    ----------
    variables
        Mapping of exactly :data:`GEOMETRY_VARIABLE_NAMES` to CSDL variables or
        expressions.

    Returns
    -------
    GeometryModel
        Populated model with no registered design variables.

    Raises
    ------
    KeyError
        If any required name is missing, or any unexpected name is present.
        Both are reported rather than silently ignored.
    """
    supplied = set(variables)
    required = set(GEOMETRY_VARIABLE_NAMES)
    missing = sorted(required - supplied)
    unexpected = sorted(supplied - required)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise KeyError(
            "create_geometry_parameterization_from_variables requires exactly "
            f"{sorted(required)}; got {'; '.join(details)}."
        )
    return _populate_e175_geometry(GeometryModel(), variables)


def create_geometry_model() -> GeometryModel:
    """Build the E175 geometry model with its registered design variables.

    Returns
    -------
    GeometryModel
        Driver-owned design variables, components, and intersections. The
        motion is identical to the previous hand-written declaration; the
        general mechanism now lives in :class:`GeometryModel`. The component
        and intersection declaration itself is shared with
        :func:`create_geometry_parameterization_from_variables`.
    """
    geometry = GeometryModel()
    wing_translation_x = geometry.design_variable(
        "wing_translation_x",
        GEOMETRY_VALUES["wing_translation_x"],
        lower=-4.0,
        upper=4.0,
        scaler=1.0 / 3.0,
    )
    wing_rotation_degrees = geometry.design_variable(
        "wing_rotation_degrees",
        GEOMETRY_VALUES["wing_rotation_degrees"],
        lower=-5.0,
        upper=5.0,
        scaler=1.0 / 5.0,
    )
    tail_rotation_degrees = geometry.design_variable(
        "tail_rotation_degrees",
        GEOMETRY_VALUES["tail_rotation_degrees"],
        lower=-8.0,
        upper=8.0,
        scaler=1.0 / 8.0,
    )
    wing_area = geometry.design_variable(
        "wing_area",
        GEOMETRY_VALUES["wing_area"],
        lower=56.0,
        upper=84.0,
        scaler=1.0 / 70.0,
    )
    wing_aspect_ratio = geometry.design_variable(
        "wing_aspect_ratio",
        GEOMETRY_VALUES["wing_aspect_ratio"],
        lower=6.3,
        upper=10.08,
        scaler=1.0 / 8.4,
    )
    fuselage_diameter_scale = geometry.design_variable(
        "fuselage_diameter_scale",
        GEOMETRY_VALUES["fuselage_diameter_scale"],
        lower=0.8,
        upper=1.25,
        scaler=1.0,
    )

    return _populate_e175_geometry(
        geometry,
        {
            "wing_translation_x": wing_translation_x,
            "wing_rotation_degrees": wing_rotation_degrees,
            "tail_rotation_degrees": tail_rotation_degrees,
            "wing_area": wing_area,
            "wing_aspect_ratio": wing_aspect_ratio,
            "fuselage_diameter_scale": fuselage_diameter_scale,
        },
    )


# ---------------------------------------------------------------------------
# 3. Surface and volume deformation settings
# ---------------------------------------------------------------------------
MESH_MOTION = MeshMotion(
    surface=SurfaceMotion(
        load_steps=2,
        stiffening_exponent=1.5,
        distance_weighting=DistanceWeighting(
            enabled=True,
            beta=2.0,
            length_scale=4.0,
            decay="exp",
        ),
    ),
    volume=VolumeMotion(
        mode="elasticity",
        # One differentiable reference-stiffness volume solve after the final
        # surface load step is the default CFD/optimization path.
        load_mode="final",
        elasticity_poisson_ratio=0.3,
        elasticity_stiffening_exponent=0.75,
        output_directory=DEFORMATION_OUTPUT_DIRECTORY,
        write_meshes=True,
    ),
    quality=QualityChecks(
        # These diagnostics are evaluated before DAFoam is called.
        surface=True,
        volume=True,
        gmsh_volume_metrics=True,
        fail_on_surface_inversion=True,
        fail_on_volume_inversion=True,
    ),
    visualization=Visualization(enabled=False),
    derivative_check=DerivativeCheck(
        enabled=False,
        objective="CD",
        step_sizes=(1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5),
    ),
    symmetry=False,
)


# ---------------------------------------------------------------------------
# 4. DAFoam flow and OpenFOAM case settings
# ---------------------------------------------------------------------------
FLOW = FlowConfig(
    solver_name="DARhoSimpleCFoam",
    velocity_m_per_s=242.52,
    angle_of_attack_deg=0.0,
    pressure_pa=30089.6,
    temperature_k=228.714,
    nu_tilda_m2_per_s=4.5e-5,
    reference_area_m2=70.0,
    normal_axis="z",
    use_wall_functions=False,
    # Tight primal tolerance for derivative diagnostics; primalMinResTolDiff=1.1
    # rejects a solve that stalls more than 10% above the requested residual.
    primal_min_res_tol=1.0e-9,
    primal_min_res_tol_difference=1.1,
    primal_min_iterations=1,
    primal_max_iterations=10_000,
    adjoint_gmres_relative_tolerance=1.0e-4,
    adjoint_gmres_absolute_tolerance=1.0e-14,
    adjoint_gmres_max_iterations=1_000,
    adjoint_gmres_restart=1_000,
)


@dataclass(frozen=True)
class OpenFOAMCaseConfig:
    """Location and patch layout of the OpenFOAM case this driver runs in.

    Describes a concrete E175/OpenFOAM case; it is not a generic geometry
    contract. The patch-name tuples must match the case's ``boundary`` file,
    because they drive both mesh conversion and the DAFoam options.

    Attributes
    ----------
    case_directory
        Complete OpenFOAM case containing ``0/``, ``constant/``, and
        ``system/``. ``None`` means DAFoam is not configured yet and the driver
        raises before doing expensive setup.
    reuse_openfoam_mesh
        Validate and reuse the ``polyMesh`` already in the case instead of
        converting the Gmsh volume mesh again.
    overwrite_existing_polymesh
        Permit conversion to replace an existing ``polyMesh``. Ignored when the
        mesh is reused.
    run_check_mesh
        Run OpenFOAM's ``checkMesh`` during preparation, and let DAFoam reject a
        deformed mesh during the primal.
    wall_patches
        Patch names forming the aircraft wall, deformed by mesh motion.
    farfield_patches
        Patch names forming the farfield, where the patch-velocity input
        applies.
    symmetry_patches
        Patch names forming the symmetry plane.
    function_names
        Aerodynamic functions DAFoam evaluates and exposes as CSDL outputs.
    results_file
        File the driver writes its JSON result payload to.
    """

    # TODO: set this once the E175 OpenFOAM case template exists.
    case_directory: Path | None = None
    reuse_openfoam_mesh: bool = False
    overwrite_existing_polymesh: bool = False
    run_check_mesh: bool = True
    wall_patches: tuple[str, ...] = ("aircraft",)
    farfield_patches: tuple[str, ...] = ("inlet", "outlet", "farfield")
    symmetry_patches: tuple[str, ...] = ("symmetry",)
    function_names: tuple[str, ...] = ("CL", "CD")
    results_file: Path = Path("bsm3_dafoam_results.json")


CASE = OpenFOAMCaseConfig(
    case_directory=Path(
        os.environ.get(
            "DAFOAM_CASE_DIRECTORY",
            "/tscc/lustre/ddn/scratch/mruh/"
            "dafoam_cases/da_foam_csdl",
        )
    ),
    reuse_openfoam_mesh=True,
    overwrite_existing_polymesh=False,
)


# ---------------------------------------------------------------------------
# 4b. MPI execution model
# ---------------------------------------------------------------------------
# Geometry / mesh-motion execution:
#   "rank0"      -- geometry -> surface -> volume runs only on MPI rank 0 via
#                   GeometryVolumeOperation; the global volume coordinates are
#                   broadcast and DAFoam stays distributed.  This is the target
#                   architecture.
#   "replicated" -- the original path: the full mesh-motion CSDL graph is built
#                   and executed on every rank.  Kept for A/B validation of the
#                   rank-0 forward output against the established pipeline.
GEOMETRY_VOLUME_MODE = "rank0"

# DAFoam volume-coordinate gradient assembly:
#   "replicated" -- Allreduce; every rank holds the full global gradient.
#   "root"       -- Reduce; only rank 0 owns the assembled gradient.
# Validate the two against one another before dropping "replicated".
DAFOAM_VOLUME_GRADIENT_OWNERSHIP = "replicated"

# Verify that the replicated design variables agree across ranks before rank 0
# uses its own copy.  Cheap; leave on until the coupling is trusted.
GEOMETRY_VOLUME_DEBUG = True


@dataclass(frozen=True)
class EndToEndDerivativeCheckConfig:
    """Optimization-level FD check through geometry, mesh motion, and DAFoam.

    Attributes
    ----------
    enabled
        Run the check. When ``False`` nothing is registered and no check runs.
    lift_constraint_target
        Value ``CL`` is constrained to equal when the optimization outputs are
        registered.
    step_size
        Finite-difference step handed to CSDL's total-derivative check. Must be
        positive.
    print_results
        Print CSDL's comparison table.
    raise_on_error
        Let CSDL raise when a derivative disagrees, instead of only reporting.

    Raises
    ------
    ValueError
        At construction, if ``step_size`` is not positive.
    """

    enabled: bool = False
    lift_constraint_target: float = 0.5
    step_size: float = 1.0e-5
    print_results: bool = True
    raise_on_error: bool = False

    def __post_init__(self):
        """Validate the finite-difference step.

        Raises
        ------
        ValueError
            If ``step_size`` is not positive.
        """
        if self.step_size <= 0.0:
            raise ValueError("Derivative-check step_size must be positive.")


END_TO_END_DERIVATIVE_CHECK = EndToEndDerivativeCheckConfig(
    enabled=False,
    lift_constraint_target=0.5,
    step_size=1.0e-5,
)


@dataclass
class E175DAFoamResult:
    """Coupled mesh-motion and DAFoam outputs for one E175 analysis.

    A plain record holding references to the CSDL variables built by the
    driver; it performs no computation.

    Attributes
    ----------
    mesh_motion
        Full mesh-motion result, including surface and volume coordinates and
        the quality diagnostics. In the rank-0 execution path it is a snapshot
        of the backend's ``last_mesh_motion_result`` taken while this object is
        built, so it is ``None`` on every non-root rank, which holds no
        backend, and can **also** be ``None`` on the root rank when the custom
        operation has not executed inline before that snapshot. Being on root
        is therefore not a guarantee that it is populated, which is why the
        annotation is optional.
    flow_outputs
        Every aerodynamic function DAFoam produced, keyed by function name.
    cl
        Lift-coefficient variable, the ``"CL"`` entry of ``flow_outputs``.
    cd
        Drag-coefficient variable, the ``"CD"`` entry of ``flow_outputs``.
    """

    mesh_motion: MeshMotionResult | None
    flow_outputs: dict[str, csdl.Variable]
    cl: csdl.Variable
    cd: csdl.Variable


def _require_case_directory(config: OpenFOAMCaseConfig) -> Path:
    if config.case_directory is None:
        raise RuntimeError(
            "DAFoam is not configured yet: set CASE.case_directory to a "
            "complete OpenFOAM case containing 0/, constant/, and system/."
        )
    return Path(config.case_directory).expanduser().resolve()


def prepare_openfoam_case(
    config: OpenFOAMCaseConfig,
    input_files: InputFiles,
    flow: FlowConfig,
    comm,
) -> Path:
    """Convert/validate the mesh on rank zero and synchronize all ranks.

    Collective: every rank must call this. Rank 0 does the file work inside a
    try/except and the outcome is broadcast before the closing barrier, so a
    failure raises on every rank rather than stranding the others.

    Parameters
    ----------
    config
        Case location and patch layout. Its patch tuples are concatenated into
        the set of expected patch names.
    input_files
        Input file contracts; the Gmsh volume mesh is read from here when the
        case mesh is converted rather than reused.
    flow
        Flow settings; its primal iteration limit is written into the case's
        ``controlDict``.
    comm
        MPI communicator. Rank 0 performs the preparation.

    Returns
    -------
    pathlib.Path
        The resolved case directory.

    Raises
    ------
    RuntimeError
        On every rank when ``case_directory`` is unset, or when rank 0's
        validation or conversion failed, carrying that rank's error text.
    """
    case_directory = _require_case_directory(config)
    expected_patches = (
        config.wall_patches
        + config.farfield_patches
        + config.symmetry_patches
    )
    error_message = None
    if comm.rank == 0:
        try:
            validate_case_template(case_directory)
            set_control_dict_max_iterations(
                case_directory,
                flow.primal_max_iterations,
            )
            if config.reuse_openfoam_mesh:
                validate_reused_mesh(
                    case_directory,
                    expected_patches,
                    not config.run_check_mesh,
                )
            else:
                convert_and_check_mesh(
                    case_dir=case_directory,
                    mesh_file=input_files.volume_mesh_file,
                    wall_patches=config.wall_patches,
                    farfield_patches=config.farfield_patches,
                    symmetry_patches=config.symmetry_patches,
                    overwrite_existing=config.overwrite_existing_polymesh,
                    skip_check_mesh=not config.run_check_mesh,
                )
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
    error_message = comm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(
            "OpenFOAM mesh/case preparation failed: " + error_message
        )
    comm.Barrier()
    return case_directory


def create_dafoam_backend(
    flow: FlowConfig,
    case: OpenFOAMCaseConfig,
    input_files: InputFiles,
    comm,
) -> PYDAFoamBackend:
    """Prepare the case and construct the live DAFoam backend.

    Collective: it calls :func:`prepare_openfoam_case` first, so every rank must
    participate. The differentiable volume-coordinate and patch-velocity inputs
    are registered in the DAFoam options, and the Gmsh volume mesh supplies the
    reference coordinates used to build the local-to-global point map.

    Parameters
    ----------
    flow
        Flow settings, supplying the DAFoam options and the normal axis of the
        patch-velocity input.
    case
        Case location and patch layout, supplying the wall and farfield patches,
        the function names, and the mesh-check setting.
    input_files
        Input file contracts; the Gmsh volume mesh provides the reference
        coordinates.
    comm
        MPI communicator handed to DAFoam.

    Returns
    -------
    PYDAFoamBackend
        Backend configured with this module's
        ``DAFOAM_VOLUME_GRADIENT_OWNERSHIP`` setting.

    Raises
    ------
    RuntimeError
        If case preparation failed, or the DAFoam environment was not sourced.
    """
    case_directory = prepare_openfoam_case(case, input_files, flow, comm)
    da_options = add_csdl_inputs_to_da_options(
        build_da_options(
            flow,
            case.wall_patches,
            case.farfield_patches,
        ),
        volume_input_name="aero_vol_coords",
        patch_velocity_input_name="patch_velocity",
        farfield_patches=case.farfield_patches,
        flow_axis="x",
        normal_axis=flow.normal_axis,
    )
    reference_mesh = read_gmsh22_volume(input_files.volume_mesh_file)
    return PYDAFoamBackend.from_options(
        da_options,
        comm,
        reference_mesh.vertices,
        case_directory=case_directory,
        volume_input_name="aero_vol_coords",
        function_names=case.function_names,
        check_mesh=case.run_check_mesh,
        volume_gradient_ownership=DAFOAM_VOLUME_GRADIENT_OWNERSHIP,
    )


def build_cfd_analysis(
    recorder: csdl.Recorder,
    input_files: InputFiles,
    geometry: GeometryModel,
    mesh_motion: MeshMotion,
    flow: FlowConfig,
    backend: PYDAFoamBackend,
) -> E175DAFoamResult:
    """Build the end-to-end graph and return CL/CD as CSDL variables.

    The replicated path: the full mesh-motion graph is built on every rank and
    the aerodynamic analysis is attached as the pipeline's callback, so DAFoam
    sees the volume coordinates produced in-graph. Compare
    :func:`build_cfd_analysis_rank0`, which confines mesh motion to rank 0.

    Parameters
    ----------
    recorder
        Active CSDL recorder the graph is built into.
    input_files
        Input file contracts for the geometry and meshes.
    geometry
        Geometry model carrying the registered design variables.
    mesh_motion
        Mesh-motion configuration. It must use ``volume.load_mode='final'`` and
        include the ``elasticity`` volume method.
    flow
        Flow settings supplying the freestream speed and angle of attack, which
        become CSDL variables feeding the patch-velocity input.
    backend
        Live DAFoam backend evaluating the aerodynamic functions.

    Returns
    -------
    E175DAFoamResult
        The mesh-motion result together with every aerodynamic output and the
        ``CL`` and ``CD`` variables.

    Raises
    ------
    ValueError
        If ``volume.load_mode`` is not ``'final'``, because the synchronized
        re-factorization path is forward-only, or if the elasticity volume
        method is absent.
    """
    if mesh_motion.volume.load_mode != "final":
        raise ValueError(
            "DAFoam coupling requires volume.load_mode='final'; the "
            "synchronized re-factorization path is forward-only."
        )
    if "elasticity" not in mesh_motion.volume.methods:
        raise ValueError(
            "DAFoam coupling requires the elasticity volume-motion method."
        )
    operation = DAFoamAnalysisOperation(backend)

    def aerodynamic_analysis(volume_coordinates):
        airspeed = csdl.Variable(
            name="dafoam_airspeed_m_per_s",
            value=np.array([flow.velocity_m_per_s]),
        )
        angle_of_attack = csdl.Variable(
            name="dafoam_angle_of_attack_deg",
            value=np.array([flow.angle_of_attack_deg]),
        )
        outputs = operation.evaluate(
            volume_coordinates,
            patch_velocity=make_patch_velocity(
                airspeed,
                angle_of_attack,
            ),
        )
        for name, variable in outputs.items():
            variable.add_name(f"dafoam_{name}")
        return outputs

    motion_result = run_mesh_motion(
        recorder=recorder,
        input_files=input_files,
        geometry=geometry,
        config=mesh_motion,
        aerodynamic_analysis=aerodynamic_analysis,
        aerodynamic_volume_method="elasticity",
    )
    outputs = motion_result.aerodynamic_outputs
    return E175DAFoamResult(
        mesh_motion=motion_result,
        flow_outputs=outputs,
        cl=outputs["CL"],
        cd=outputs["CD"],
    )


def build_cfd_analysis_rank0(
    recorder: csdl.Recorder,
    input_files: InputFiles,
    geometry: GeometryModel,
    mesh_motion: MeshMotion,
    flow: FlowConfig,
    backend: PYDAFoamBackend,
    comm,
    *,
    geometry_values: dict[str, float],
    debug: bool = False,
    seed_ownership: str = "replicated",
) -> E175DAFoamResult:
    """Build the rank-0 geometry->volume + distributed-DAFoam graph.

    The whole geometry -> surface -> volume pipeline is encapsulated in an
    ``MeshMotionVolumeBackend`` that lives only on rank 0.  The outer CSDL
    graph is the two-custom-operation chain ``d -> X -> (CL, CD)``:
    ``GeometryVolumeOperation`` runs the mesh motion on rank 0 and broadcasts the
    global coordinates; ``DAFoamAnalysisOperation`` extracts each rank's local
    OpenFOAM partition and runs the collective primal/adjoint.

    Collective: every rank must call this and declare the same design variables
    in the same order.

    Parameters
    ----------
    recorder
        Present for signature symmetry with :func:`build_cfd_analysis`; the
        active recorder owns the variables built here, so it is not used
        directly.
    input_files
        Input file contracts. The Gmsh volume mesh supplies the global point
        count, which every rank needs to declare the operation's output shape
        without building the geometry backend.
    geometry
        Geometry model. Only rank 0 builds a live backend from it.
    mesh_motion
        Mesh-motion configuration, subject to the same ``load_mode`` and
        elasticity requirements as :func:`build_cfd_analysis`.
    flow
        Flow settings supplying the freestream speed and angle of attack.
    backend
        Live DAFoam backend, distributed across all ranks.
    comm
        MPI communicator. It is passed through ``resolve_comm``, so ``None``
        selects ``MPI.COMM_WORLD`` when ``mpi4py`` is importable and a
        single-rank ``SerialComm`` only when it is not.
    geometry_values
        Baseline design-variable values, replicated on every rank.
    debug
        Verify that the replicated design variables agree across ranks before
        rank 0 uses its own copy.
    seed_ownership
        Ownership contract for the incoming volume-coordinate cotangent,
        ``"replicated"`` or ``"root"``. It must agree with the DAFoam backend's
        volume-gradient ownership.

    Returns
    -------
    E175DAFoamResult
        Every aerodynamic output together with the ``CL`` and ``CD`` variables.
        Its ``mesh_motion`` field is a snapshot of the rank-0 backend's
        ``last_mesh_motion_result`` taken as the result is constructed, so it is
        ``None`` on every non-root rank, which builds no backend, and may still
        be ``None`` on the root rank if the custom operation has not executed
        inline by then. The field is annotated ``MeshMotionResult | None`` to
        express both cases.

    Raises
    ------
    ValueError
        If ``volume.load_mode`` is not ``'final'``, or the elasticity volume
        method is absent.
    """
    _ = recorder  # the active recorder owns the variables built below
    comm = resolve_comm(comm)
    if mesh_motion.volume.load_mode != "final":
        raise ValueError(
            "DAFoam coupling requires volume.load_mode='final'; the "
            "synchronized re-factorization path is forward-only."
        )
    if "elasticity" not in mesh_motion.volume.methods:
        raise ValueError(
            "DAFoam coupling requires the elasticity volume-motion method."
        )

    # Every rank needs the global volume-point count to declare the operation's
    # output shape.  Read it on rank 0 (geometry I/O stays on the root) and
    # broadcast the small integer.
    num_points = None
    if is_root(comm):
        num_points = read_gmsh_volume_point_count(input_files.volume_mesh_file)
    num_points = comm.bcast(num_points, root=0)
    output_shape = (int(num_points), 3)

    design_variable_map = dict(geometry.design_variables)
    # Derive the declared shapes from the actual CSDL variables (scalar design
    # variables are shape (1,), not ()), so the VJP cotangent shapes match.
    design_variable_specs = {
        name: tuple(variable.shape)
        for name, variable in design_variable_map.items()
    }

    geometry_backend = None
    if is_root(comm):
        geometry_backend = MeshMotionVolumeBackend(
            input_files=input_files,
            geometry_values=geometry_values,
            pipeline_config=mesh_motion,
            parameterization_factory=(
                create_geometry_parameterization_from_variables
            ),
            aerodynamic_volume_method="elasticity",
        )

    geometry_operation = GeometryVolumeOperation(
        geometry_backend,
        comm,
        output_shape=output_shape,
        design_variable_specs=design_variable_specs,
        debug=debug,
        # The reverse-pass seed is DAFoam's volume-coordinate cotangent, whose
        # ownership must match DAFoam's gradient assembly mode.
        seed_ownership=seed_ownership,
    )
    volume_coordinates = geometry_operation.evaluate(design_variable_map)
    volume_coordinates.add_name("global_volume_coordinates")

    dafoam_operation = DAFoamAnalysisOperation(backend)
    airspeed = csdl.Variable(
        name="dafoam_airspeed_m_per_s",
        value=np.array([flow.velocity_m_per_s]),
    )
    angle_of_attack = csdl.Variable(
        name="dafoam_angle_of_attack_deg",
        value=np.array([flow.angle_of_attack_deg]),
    )
    outputs = dafoam_operation.evaluate(
        volume_coordinates,
        patch_velocity=make_patch_velocity(airspeed, angle_of_attack),
    )
    for name, variable in outputs.items():
        variable.add_name(f"dafoam_{name}")

    mesh_motion_result = (
        geometry_backend.last_mesh_motion_result
        if geometry_backend is not None
        else None
    )
    return E175DAFoamResult(
        mesh_motion=mesh_motion_result,
        flow_outputs=outputs,
        cl=outputs["CL"],
        cd=outputs["CD"],
    )


def configure_end_to_end_derivative_check(
    result: E175DAFoamResult,
    config: EndToEndDerivativeCheckConfig,
) -> None:
    """Register the aerodynamic optimization outputs before stopping CSDL.

    Mutates the CSDL variables in ``result``: ``CL`` becomes an equality
    constraint and ``CD`` the objective. It must be called while the recorder
    is still active, and it runs no check itself.

    Parameters
    ----------
    result
        Coupled analysis result whose ``cl`` and ``cd`` variables are
        registered.
    config
        Check configuration supplying the lift-constraint target. Its
        ``enabled`` flag is not consulted here; the caller decides.

    Returns
    -------
    None
        The variables are modified in place.
    """
    result.cl.set_as_constraint(equals=config.lift_constraint_target)
    result.cd.set_as_objective()


def run_end_to_end_derivative_check(
    recorder: csdl.Recorder,
    config: EndToEndDerivativeCheckConfig,
):
    """Run CSDL's Python-backend total-derivative finite-difference check.

    Builds a :class:`PySimulator` over the stopped recorder and runs CSDL's own
    check, which constructs analytical-derivative operations. It is therefore
    not the forward-only checker in
    :mod:`bsm3.core.boundary_surface_movement.forward_only_fd_checker`, and each
    perturbation may re-run the coupled primal and adjoint.

    Parameters
    ----------
    recorder
        Recorder holding the built graph, with the objective and constraint
        already registered.
    config
        Check configuration supplying the step size, printing, and whether a
        disagreement raises.

    Returns
    -------
    Any
        CSDL's comparison result, as returned by
        ``PySimulator.check_optimization_derivatives``.
    """
    simulator = csdl.experimental.PySimulator(recorder=recorder)
    return simulator.check_optimization_derivatives(
        step_size=config.step_size,
        print_results=config.print_results,
        raise_on_error=config.raise_on_error,
    )


def _quality_payload(
    result: MeshMotionResult,
    config: QualityChecks,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config.surface:
        payload["surface"] = {
            "inverted_elements": int(
                result.surface_inversion_report.num_inverted
            ),
            "minimum_scaled_jacobian": float(
                result.surface_quality_report.minimum_scaled_jacobian
            ),
            "scaled_jacobian_p05": float(
                result.surface_quality_report.scaled_jacobian_p05
            ),
            "maximum_aspect_ratio": float(
                result.surface_quality_report.maximum_aspect_ratio
            ),
        }
    if config.volume:
        payload["volume"] = result.volume_quality_summary
    return payload


def main() -> E175DAFoamResult:
    """Run the configured E175 DAFoam analysis once under MPI.

    A concrete E175/OpenFOAM driver: there is no argument parser and no
    environment-variable configuration beyond the case directory, so the design
    point, mesh files, flow settings, and MPI mode come from the module-level
    constants above. It must be launched with ``mpirun`` in a sourced DAFoam
    environment.

    Fails fast when no case directory is configured, before any expensive
    mesh-motion setup. Which graph is built depends on ``GEOMETRY_VOLUME_MODE``:
    ``"rank0"`` confines mesh motion to rank 0, ``"replicated"`` builds it on
    every rank for A/B validation. Writing the JSON results file and running the
    optional derivative check are side effects of the module-level settings.

    Returns
    -------
    E175DAFoamResult
        The coupled mesh-motion and aerodynamic result.

    Raises
    ------
    RuntimeError
        If no OpenFOAM case directory is configured, or ``mpi4py`` is
        unavailable because the DAFoam environment was not sourced.
    """
    # Fail before an expensive mesh-motion setup when no OpenFOAM case exists.
    _require_case_directory(CASE)
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise RuntimeError(
            "mpi4py is unavailable. Source the DAFoam environment and launch "
            "this module with mpirun."
        ) from error

    comm = MPI.COMM_WORLD
    backend = create_dafoam_backend(FLOW, CASE, MODEL_FILES, comm)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    geometry = create_geometry_model()
    if GEOMETRY_VOLUME_MODE == "rank0":
        result = build_cfd_analysis_rank0(
            recorder,
            MODEL_FILES,
            geometry,
            MESH_MOTION,
            FLOW,
            backend,
            comm,
            geometry_values=GEOMETRY_VALUES,
            debug=GEOMETRY_VOLUME_DEBUG,
            seed_ownership=DAFOAM_VOLUME_GRADIENT_OWNERSHIP,
        )
    elif GEOMETRY_VOLUME_MODE == "replicated":
        result = build_cfd_analysis(
            recorder,
            MODEL_FILES,
            geometry,
            MESH_MOTION,
            FLOW,
            backend,
        )
    else:
        raise ValueError(
            "GEOMETRY_VOLUME_MODE must be 'rank0' or 'replicated'; got "
            f"{GEOMETRY_VOLUME_MODE!r}."
        )

    if (
        END_TO_END_DERIVATIVE_CHECK.enabled
        and MESH_MOTION.derivative_check.enabled
    ):
        raise ValueError(
            "Enable either END_TO_END_DERIVATIVE_CHECK or the generic "
            "MESH_MOTION finite-difference sweep, not both."
        )
    if GEOMETRY_VOLUME_MODE == "rank0" and (
        END_TO_END_DERIVATIVE_CHECK.enabled
        or MESH_MOTION.derivative_check.enabled
    ):
        # In rank-0 mode the mesh motion lives in the backend's private
        # recorder, not the outer graph, so the legacy CSDL FD paths do not
        # apply.  Use the forward-only ladder in e175_derivative_ladder.py, which
        # never reruns the adjoint during finite differences.
        raise ValueError(
            "GEOMETRY_VOLUME_MODE='rank0' does not support the legacy "
            "END_TO_END_DERIVATIVE_CHECK or MESH_MOTION.derivative_check "
            "sweeps. Run e175_derivative_ladder.py for the forward-only "
            "derivative validation, or set GEOMETRY_VOLUME_MODE='replicated'."
        )
    if END_TO_END_DERIVATIVE_CHECK.enabled:
        configure_end_to_end_derivative_check(
            result,
            END_TO_END_DERIVATIVE_CHECK,
        )
    elif MESH_MOTION.derivative_check.enabled:
        select_fd_objective(
            result.mesh_motion,
            MESH_MOTION.derivative_check.objective,
    )
    recorder.stop()

    if END_TO_END_DERIVATIVE_CHECK.enabled:
        run_end_to_end_derivative_check(
            recorder,
            END_TO_END_DERIVATIVE_CHECK,
        )
    elif MESH_MOTION.derivative_check.enabled:
        run_fd_sweep(
            recorder,
            MESH_MOTION.derivative_check.step_sizes,
        )

    if comm.rank == 0:
        case_directory = _require_case_directory(CASE)
        results_path = CASE.results_file
        if not results_path.is_absolute():
            results_path = case_directory / results_path
        payload = {
            "case_directory": str(case_directory),
            "mpi_ranks": int(comm.size),
            "geometry_design_variables": {
                name: np.asarray(variable.value).tolist()
                for name, variable in (
                    geometry.design_variables.items()
                )
            },
            "flow_config": asdict(FLOW),
            "mesh_quality": _quality_payload(
                result.mesh_motion,
                MESH_MOTION.quality,
            ),
            "functions": {
                name: float(np.asarray(variable.value).reshape(-1)[0])
                for name, variable in result.flow_outputs.items()
            },
        }
        results_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
