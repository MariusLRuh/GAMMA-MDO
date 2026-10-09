"""
Run a steady compressible DAFoam primal analysis from a Gmsh volume mesh.

Designed for DAFoam v4.0.4 / OpenFOAM-v1812 on TSCC.

The script performs this sequence:

1. Validate the OpenFOAM case template (0/, constant/, and system/).
2. Convert the supplied Gmsh .msh volume mesh with gmshToFoam.
3. Set and validate OpenFOAM boundary patch types.
4. Run OpenFOAM checkMesh.
5. Instantiate PYDAFOAM and run the primal solver.
6. Evaluate CD and CL and write dafoam_results.json.

The .msh file must contain a 3-D volume mesh and named 2-D Physical Surface
groups. Their names must match --wall-patches, --farfield-patches, and any
--symmetry-patches. A .msh file alone is not a complete CFD case: the case
directory must already contain compatible OpenFOAM field and dictionary files.

TSCC example
------------
source "$HOME/dafoam/loadDAFoam.sh"
cd /path/to/working/directory

mpirun --oversubscribe -np 4 python run_dafoam_gmsh.py \
    --case /path/to/openfoam_case \
    --mesh /path/to/aircraft.msh \
    --wall-patches wall \
    --farfield-patches farfield \
    --symmetry-patches symmetry

On TSCC, use mpirun even for one rank:

mpirun --oversubscribe -np 1 python run_dafoam_gmsh.py ...

Subsequent runs may reuse the converted mesh:

mpirun --oversubscribe -np 4 python run_dafoam_gmsh.py \
    --case /path/to/openfoam_case \
    --reuse-openfoam-mesh

Important
---------
Review the USER-EDITABLE DEFAULTS below and the files in 0/ before running.
For a transonic case, the default solver is DARhoSimpleCFoam. Use
--solver DARhoSimpleFoam for a subsonic compressible case.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

@runtime_checkable
class _Communicator(Protocol):
    """Structural type for the MPI communicator features this module uses.

    ``mpi4py`` is an optional, environment-provided dependency, so annotating
    against ``mpi4py.MPI.Comm`` would require importing it eagerly just to
    resolve a type. This protocol describes only what the functions below
    actually touch, so the annotations resolve at runtime with no MPI
    installed, and a real ``MPI.Comm`` satisfies it structurally.

    Attributes
    ----------
    rank
        Index of this process within the communicator.
    size
        Number of processes in the communicator.
    """

    rank: int
    size: int

    def Barrier(self) -> None:
        """Block until every rank reaches this call."""

    def bcast(self, obj: object, root: int = 0) -> object:
        """Broadcast a picklable object from ``root`` to every rank."""


# Avoid accidental oversubscription when one Python process is launched per MPI
# rank. Override these in the shell before launching if you intentionally want
# threaded BLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# =============================================================================
# USER-EDITABLE DEFAULTS
# =============================================================================


@dataclass(frozen=True)
class FlowConfig:
    """Freestream and reference quantities used by DAFoam.

    Attributes
    ----------
    solver_name
        DAFoam solver class to instantiate, for example
        ``"DARhoSimpleCFoam"``.
    velocity_m_per_s
        Freestream speed in meters per second.
    angle_of_attack_deg
        Angle of attack in **degrees**, used to build the flow and force
        direction vectors.
    pressure_pa
        Freestream static pressure in pascals.
    temperature_k
        Freestream static temperature in kelvin.
    nu_tilda_m2_per_s
        Spalart-Allmaras working variable in square meters per second, applied
        as the farfield ``nuTilda0`` primal boundary condition. Must be finite
        and non-negative.
    reference_area_m2
        Reference area in square meters, normalizing the force coefficients.
    gas_constant_j_per_kg_k
        Specific gas constant in joules per kilogram-kelvin.
    normal_axis
        Axis label, ``"y"`` or ``"z"``, spanning the lift direction together
        with the flow direction.
    use_wall_functions
        Select wall functions instead of resolving the near-wall layer. It sets
        ``primalBC/useWallFunction`` in the DAFoam options. On the CLI it is
        controlled by ``--wall-functions`` / ``--no-wall-functions``, whose
        default is this dataclass default.
    primal_min_res_tol
        Residual the steady primal must reach. Must be positive.
    primal_min_iterations
        Fewest primal iterations before convergence may be declared. At least
        one.
    primal_max_iterations
        Iteration ceiling, written into ``controlDict`` as ``endTime``. Must be
        at least ``primal_min_iterations``.
    primal_min_res_tol_difference
        Factor of ``primal_min_res_tol`` within which the final residual must
        fall for DAFoam to accept the primal; ``1.1`` rejects a solve stalling
        more than 10% above the request. The default ``1e2`` is DAFoam's
        lenient value. Must be positive.
    adjoint_gmres_relative_tolerance
        Relative GMRES tolerance for the adjoint solve. Must be positive.
    adjoint_gmres_absolute_tolerance
        Absolute GMRES tolerance for the adjoint solve. Must be positive.
    adjoint_gmres_max_iterations
        GMRES iteration ceiling for the adjoint solve. At least one.
    adjoint_gmres_restart
        GMRES restart length for the adjoint solve. At least one.

    Raises
    ------
    ValueError
        At construction, when any tolerance is non-positive, an iteration count
        is below one, or ``primal_max_iterations`` is below
        ``primal_min_iterations``.
    """

    solver_name: str = "DARhoSimpleCFoam"
    velocity_m_per_s: float = 242.52
    angle_of_attack_deg: float = 0.0
    pressure_pa: float = 30089.6
    temperature_k: float = 228.714
    nu_tilda_m2_per_s: float = 4.5e-5
    reference_area_m2: float = 70.0
    gas_constant_j_per_kg_k: float = 287.0
    normal_axis: str = "z"
    use_wall_functions: bool = False
    primal_min_res_tol: float = 1.0e-7
    primal_min_iterations: int = 1
    primal_max_iterations: int = 10_000
    # DAFoam accepts a converged primal only if the final residual is within
    # this factor of primalMinResTol; e.g. 1.1 rejects a solve that stalls more
    # than 10% above the requested residual. Defaults to DAFoam's lenient 1e2.
    primal_min_res_tol_difference: float = 1.0e2
    adjoint_gmres_relative_tolerance: float = 1.0e-4
    adjoint_gmres_absolute_tolerance: float = 1.0e-14
    adjoint_gmres_max_iterations: int = 1_000
    adjoint_gmres_restart: int = 1_000

    def __post_init__(self) -> None:
        """Validate the freestream, convergence, and adjoint solver settings.

        Raises
        ------
        ValueError
            If ``nu_tilda_m2_per_s`` is non-finite or negative, any tolerance is
            non-positive, an iteration count is below one, or
            ``primal_max_iterations`` is below ``primal_min_iterations``.
        """
        if not math.isfinite(self.nu_tilda_m2_per_s) or self.nu_tilda_m2_per_s < 0.0:
            raise ValueError("nu_tilda_m2_per_s must be finite and non-negative")
        if self.primal_min_res_tol <= 0.0:
            raise ValueError("primal_min_res_tol must be positive")
        if self.primal_min_iterations < 1:
            raise ValueError("primal_min_iterations must be at least one")
        if self.primal_max_iterations < self.primal_min_iterations:
            raise ValueError(
                "primal_max_iterations must be greater than or equal to "
                "primal_min_iterations"
            )
        if self.primal_min_res_tol_difference <= 0.0:
            raise ValueError("primal_min_res_tol_difference must be positive")
        if self.adjoint_gmres_relative_tolerance <= 0.0:
            raise ValueError(
                "adjoint_gmres_relative_tolerance must be positive"
            )
        if self.adjoint_gmres_absolute_tolerance <= 0.0:
            raise ValueError(
                "adjoint_gmres_absolute_tolerance must be positive"
            )
        if self.adjoint_gmres_max_iterations < 1:
            raise ValueError(
                "adjoint_gmres_max_iterations must be at least one"
            )
        if self.adjoint_gmres_restart < 1:
            raise ValueError("adjoint_gmres_restart must be at least one")


# Add case-specific DAFoam options here without changing build_da_options().
# Nested dictionaries are merged recursively.
DAFOAM_OPTIONS_OVERRIDES: dict[str, Any] = {}


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def rank0_print(comm: _Communicator, *items: object) -> None:
    """Print on rank zero only, flushing immediately.

    Parameters
    ----------
    comm
        MPI communicator whose rank decides whether anything is printed.
    *items
        Values passed straight to :func:`print`, space separated.

    Returns
    -------
    None
        Printing is a side effect; non-root ranks do nothing.
    """
    if comm.rank == 0:
        print(*items, flush=True)


def csv_names(value: str) -> list[str]:
    """Parse a comma-separated patch-name list, as an argparse type.

    Parameters
    ----------
    value
        Comma-separated names. Surrounding whitespace is stripped and empty
        entries are dropped.

    Returns
    -------
    list of str
        The non-empty names in order.

    Raises
    ------
    argparse.ArgumentTypeError
        If no non-empty name remains, so argparse reports a usage error.
    """
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected at least one comma-separated patch name")
    return names


def run_command(command: list[str], cwd: Path) -> None:
    """Run a command with its output streamed to the terminal.

    There is **no rank check here**: the caller decides where this runs, and
    under MPI every rank that reaches it will launch the command.

    Parameters
    ----------
    command
        Argument list; each item is stringified for the echoed banner.
    cwd
        Working directory the command runs in.

    Returns
    -------
    None
        Output goes to the terminal; nothing is captured.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with a nonzero status.
    """
    printable = " ".join(str(item) for item in command)
    print(f"\n>>> {printable}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary in place.

    Parameters
    ----------
    base
        Dictionary to update. **Mutated in place**, including its nested
        dictionaries.
    updates
        Values to merge in. A nested dictionary is merged recursively; any
        other value replaces the existing entry outright.

    Returns
    -------
    dict
        ``base`` itself, for convenient chaining.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def jsonable(value: Any) -> Any:
    """Convert NumPy-like scalars/arrays and nested containers to JSON values.

    Parameters
    ----------
    value
        Value to convert. Dictionaries and sequences are walked recursively,
        with dictionary keys stringified; anything exposing ``tolist`` or
        ``item`` is converted through it; everything else is returned
        unchanged, so an unsupported object still fails at serialization time.

    Returns
    -------
    Any
        A structure of plain Python types, with tuples becoming lists.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


# =============================================================================
# GMSH -> OPENFOAM PREPROCESSING
# =============================================================================


def validate_case_template(case_dir: Path) -> None:
    """Check that the OpenFOAM case template has its required files.

    A Gmsh file supplies the mesh only, so the case must already contain the
    ``0/``, ``constant/``, and ``system/`` directories together with
    ``controlDict``, ``fvSchemes``, and ``fvSolution``.

    Parameters
    ----------
    case_dir
        Case directory to check.

    Returns
    -------
    None
        Returns normally when every required path exists.

    Raises
    ------
    FileNotFoundError
        Listing every missing path.
    """
    required = [
        case_dir / "0",
        case_dir / "constant",
        case_dir / "system",
        case_dir / "system" / "controlDict",
        case_dir / "system" / "fvSchemes",
        case_dir / "system" / "fvSolution",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The OpenFOAM case template is incomplete. Missing:\n  "
            f"{formatted}\n"
            "The Gmsh file supplies the mesh only; DAFoam still requires the "
            "OpenFOAM 0/, constant/, and system/ case files."
        )


def set_control_dict_max_iterations(
    case_dir: Path,
    maximum_iterations: int,
) -> None:
    """Set the steady primal iteration ceiling through controlDict endTime.

    Rewrites the case's ``system/controlDict`` in place, since a steady
    OpenFOAM run treats ``endTime`` as its iteration limit.

    Parameters
    ----------
    case_dir
        Case directory holding ``system/controlDict``.
    maximum_iterations
        Iteration ceiling written as ``endTime``.

    Returns
    -------
    None
        The file is modified as a side effect.

    Raises
    ------
    ValueError
        If ``maximum_iterations`` is below one, or the ``endTime`` entry could
        not be located and rewritten exactly once.
    """
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least one")

    control_dict = case_dir / "system" / "controlDict"
    text = control_dict.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?m)^([ \t]*endTime[ \t]+)([^;\n]+)([ \t]*;)"
    )
    updated, count = pattern.subn(
        rf"\g<1>{int(maximum_iterations)}\g<3>",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(
            f"Expected one active 'endTime ...;' entry in {control_dict}"
        )
    if updated != text:
        control_dict.write_text(updated, encoding="utf-8")


def read_gmsh_mesh_format(mesh_file: Path) -> tuple[str, int]:
    """Return (version, file_type), where file_type 0=ASCII and 1=binary.

    Only the first 4096 bytes are read, so this stays cheap for a large mesh.

    Parameters
    ----------
    mesh_file
        Gmsh ``.msh`` file to inspect.

    Returns
    -------
    tuple
        ``(version, file_type)``, the version as the string Gmsh wrote, such
        as ``"2.2"``, and the file type as ``0`` for ASCII or ``1`` for binary.

    Raises
    ------
    ValueError
        If no ``$MeshFormat`` section appears in the leading bytes, or its
        content line is missing.
    """
    with mesh_file.open("rb") as stream:
        header = stream.read(4096)

    marker = b"$MeshFormat"
    position = header.find(marker)
    if position < 0:
        raise ValueError(f"{mesh_file} does not contain a Gmsh $MeshFormat section")

    lines = header[position:].splitlines()
    if len(lines) < 2:
        raise ValueError(f"Could not read the Gmsh mesh format from {mesh_file}")

    fields = lines[1].decode("ascii", errors="strict").split()
    if len(fields) < 2:
        raise ValueError(f"Malformed Gmsh $MeshFormat line in {mesh_file}")

    return fields[0], int(fields[1])


def prepare_gmsh_22_ascii(mesh_file: Path, case_dir: Path) -> tuple[Path, bool]:
    """
    Return a Gmsh 2.2 ASCII file suitable for OpenFOAM-v1812.

    If conversion is required, use the Gmsh executable and place a temporary
    file in the case directory. Physical group names are retained by Gmsh.

    Parameters
    ----------
    mesh_file
        Source mesh. Returned unchanged when it is already 2.2 ASCII.
    case_dir
        Directory the converted temporary file is written into.

    Returns
    -------
    tuple
        ``(path, converted)``: the file to hand to ``gmshToFoam``, and whether
        a temporary file was created that the caller should clean up.

    Raises
    ------
    RuntimeError
        If ``gmsh`` is not on ``PATH``, or the conversion did not produce a 2.2
        ASCII mesh.
    """
    version, file_type = read_gmsh_mesh_format(mesh_file)
    if version.startswith("2.2") and file_type == 0:
        return mesh_file, False

    gmsh = shutil.which("gmsh")
    if gmsh is None:
        kind = "binary" if file_type == 1 else "ASCII"
        raise RuntimeError(
            f"The input is Gmsh {version} {kind}, whereas this OpenFOAM-v1812 "
            "workflow expects Gmsh 2.2 ASCII. Install/load the gmsh executable "
            "or export the mesh as 'Legacy MSH 2.2 ASCII' before running."
        )

    converted = case_dir / ".dafoam_mesh_import_msh22.msh"
    run_command(
        [
            gmsh,
            str(mesh_file),
            "-save",
            "-format",
            "msh2",
            "-setnumber",
            "Mesh.Binary",
            "0",
            "-o",
            str(converted),
        ],
        cwd=case_dir,
    )
    converted_version, converted_type = read_gmsh_mesh_format(converted)
    if not converted_version.startswith("2.2") or converted_type != 0:
        raise RuntimeError(f"Gmsh did not produce a 2.2 ASCII mesh at {converted}")
    return converted, True


def backup_existing_polymesh(poly_mesh_dir: Path) -> Path:
    """Rename an existing ``polyMesh`` aside, timestamping the backup.

    Parameters
    ----------
    poly_mesh_dir
        Directory to move aside. It is renamed, not copied, so the original
        path no longer exists afterwards.

    Returns
    -------
    pathlib.Path
        The backup path, a sibling named ``polyMesh.backup_<timestamp>`` at
        one-second resolution.

    Raises
    ------
    FileExistsError
        If that backup path already exists, rather than overwriting it.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = poly_mesh_dir.with_name(f"polyMesh.backup_{timestamp}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup {backup}")
    poly_mesh_dir.rename(backup)
    return backup


def find_patch_block(lines: list[str], patch_name: str) -> tuple[int, int]:
    """Locate the inclusive { ... } block belonging to patch_name.

    Parameters
    ----------
    lines
        Boundary-file lines, as read with line endings kept.
    patch_name
        Patch to find. Quoted and unquoted spellings both match.

    Returns
    -------
    tuple
        ``(start, end)`` line indices of the block, inclusive of the opening
        and closing braces.

    Raises
    ------
    KeyError
        If no block for that patch is found.
    """
    accepted_names = {patch_name, f'"{patch_name}"'}
    for name_index, line in enumerate(lines):
        if line.strip() not in accepted_names:
            continue

        brace_index = name_index + 1
        while brace_index < len(lines) and not lines[brace_index].strip():
            brace_index += 1
        if brace_index >= len(lines) or lines[brace_index].strip() != "{":
            continue

        depth = 0
        for end_index in range(brace_index, len(lines)):
            depth += lines[end_index].count("{")
            depth -= lines[end_index].count("}")
            if depth == 0:
                return brace_index, end_index

    raise KeyError(
        f"Patch '{patch_name}' was not found in constant/polyMesh/boundary. "
        "Check that the Gmsh model contains a 2-D Physical Surface with exactly "
        "this name."
    )


def set_openfoam_patch_types(boundary_file: Path, patch_types: dict[str, str]) -> None:
    """Set `type` entries for named patches in an OpenFOAM boundary file.

    Rewrites ``boundary_file`` in place, replacing the first ``type`` entry
    inside each named patch block and leaving every other line untouched.

    Parameters
    ----------
    boundary_file
        OpenFOAM ``constant/polyMesh/boundary`` file to modify.
    patch_types
        Mapping of patch name to the OpenFOAM patch type to set.

    Returns
    -------
    None
        The file is modified as a side effect.

    Raises
    ------
    KeyError
        Propagated from :func:`find_patch_block` if a named patch has no block
        in the file.
    ValueError
        If a named patch's block exists but contains no ``type`` entry to
        rewrite.
    """
    lines = boundary_file.read_text(encoding="utf-8").splitlines(keepends=True)
    type_pattern = re.compile(r"^(\s*)type\s+[^;]+;")

    for patch_name, patch_type in patch_types.items():
        block_start, block_end = find_patch_block(lines, patch_name)
        for index in range(block_start + 1, block_end):
            match = type_pattern.match(lines[index])
            if match:
                newline = "\n" if lines[index].endswith("\n") else ""
                lines[index] = f"{match.group(1)}type            {patch_type};{newline}"
                break
        else:
            raise ValueError(f"Patch '{patch_name}' has no type entry in {boundary_file}")

    boundary_file.write_text("".join(lines), encoding="utf-8")


def validate_patch_names(boundary_file: Path, names: Iterable[str]) -> None:
    """Check that every named patch appears in the boundary file.

    Parameters
    ----------
    boundary_file
        OpenFOAM ``constant/polyMesh/boundary`` file to read.
    names
        Patch names that must be present. Quoted and unquoted spellings both
        match.

    Returns
    -------
    None
        Returns normally when every name is found.

    Raises
    ------
    KeyError
        From the underlying block lookup, naming the first patch not found.
    """
    lines = boundary_file.read_text(encoding="utf-8").splitlines(keepends=True)
    for name in names:
        find_patch_block(lines, name)


def convert_and_check_mesh(
    case_dir: Path,
    mesh_file: Path,
    wall_patches: list[str],
    farfield_patches: list[str],
    symmetry_patches: list[str],
    overwrite_existing: bool,
    skip_check_mesh: bool,
) -> None:
    """Convert the Gmsh mesh, set boundary types, and run checkMesh.

    Converts the mesh to 2.2 ASCII if needed, runs ``gmshToFoam``, assigns the
    wall, farfield, and symmetry patch types, and optionally validates the
    result. Intended to run on one rank.

    Parameters
    ----------
    case_dir
        Case directory the mesh is converted into.
    mesh_file
        Gmsh volume mesh whose named 2-D physical groups supply the patches.
    wall_patches
        Patch names set to the wall type.
    farfield_patches
        Patch names set to the farfield type.
    symmetry_patches
        Patch names set to the symmetry type.
    overwrite_existing
        Permit replacing an existing ``polyMesh``, which is first moved aside
        as a timestamped backup.
    skip_check_mesh
        Skip the external ``checkMesh`` run.

    Returns
    -------
    None
        The case directory is modified as a side effect.

    Raises
    ------
    RuntimeError
        If ``gmshToFoam`` or ``checkMesh`` is not on ``PATH``, or the Gmsh
        conversion did not produce a 2.2 ASCII mesh.
    FileExistsError
        Either when a ``polyMesh`` already exists and ``overwrite_existing`` is
        ``False``, or when overwriting is permitted but the timestamped backup
        destination is already taken.
    FileNotFoundError
        If ``gmshToFoam`` completed without creating the boundary file.
    KeyError
        Propagated from the patch-type assignment if an expected patch is
        absent from the converted boundary file.
    subprocess.CalledProcessError
        If an external command exits nonzero.
    """
    gmsh_to_foam = shutil.which("gmshToFoam")
    if gmsh_to_foam is None:
        raise RuntimeError(
            "gmshToFoam is not on PATH. Source $HOME/dafoam/loadDAFoam.sh "
            "before launching this script."
        )

    poly_mesh = case_dir / "constant" / "polyMesh"
    if poly_mesh.exists():
        if not overwrite_existing:
            raise FileExistsError(
                f"{poly_mesh} already exists. Use --reuse-openfoam-mesh to keep "
                "it, or --overwrite-existing-polymesh to move it to a timestamped "
                "backup and import the supplied .msh file."
            )
        backup = backup_existing_polymesh(poly_mesh)
        print(f"Existing polyMesh moved to {backup}", flush=True)

    import_mesh, is_temporary = prepare_gmsh_22_ascii(mesh_file, case_dir)
    try:
        run_command(
            [gmsh_to_foam, "-case", str(case_dir), str(import_mesh)],
            cwd=case_dir,
        )
    finally:
        if is_temporary and import_mesh.exists():
            import_mesh.unlink()

    boundary_file = poly_mesh / "boundary"
    if not boundary_file.is_file():
        raise FileNotFoundError(
            f"gmshToFoam completed without creating {boundary_file}"
        )

    patch_types: dict[str, str] = {}
    patch_types.update({name: "wall" for name in wall_patches})
    patch_types.update({name: "patch" for name in farfield_patches})
    patch_types.update({name: "symmetryPlane" for name in symmetry_patches})
    set_openfoam_patch_types(boundary_file, patch_types)
    validate_patch_names(boundary_file, patch_types)

    print("\nBoundary patch types set by this script:", flush=True)
    for name, patch_type in patch_types.items():
        print(f"  {name}: {patch_type}", flush=True)

    if not skip_check_mesh:
        check_mesh = shutil.which("checkMesh")
        if check_mesh is None:
            raise RuntimeError(
                "checkMesh is not on PATH. Source the DAFoam/OpenFOAM environment."
            )
        run_command(
            [
                check_mesh,
                "-case",
                str(case_dir),
                "-allGeometry",
                "-allTopology",
            ],
            cwd=case_dir,
        )


def validate_reused_mesh(
    case_dir: Path,
    expected_patches: list[str],
    skip_check_mesh: bool,
) -> None:
    """Validate a ``polyMesh`` that is being reused instead of converted.

    Parameters
    ----------
    case_dir
        Case directory whose existing ``constant/polyMesh`` is checked.
    expected_patches
        Patch names that must appear in the boundary file.
    skip_check_mesh
        Skip the external ``checkMesh`` run, validating patch names only.

    Returns
    -------
    None
        Returns normally when the mesh is acceptable.

    Raises
    ------
    FileNotFoundError
        If the case has no ``constant/polyMesh/boundary`` file.
    RuntimeError
        If ``checkMesh`` was requested but is not on ``PATH``.
    subprocess.CalledProcessError
        If ``checkMesh`` itself fails.
    """
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary_file.is_file():
        raise FileNotFoundError(
            f"--reuse-openfoam-mesh was requested, but {boundary_file} is missing"
        )
    validate_patch_names(boundary_file, expected_patches)

    if not skip_check_mesh:
        check_mesh = shutil.which("checkMesh")
        if check_mesh is None:
            raise RuntimeError("checkMesh is not on PATH")
        run_command(
            [
                check_mesh,
                "-case",
                str(case_dir),
                "-allGeometry",
                "-allTopology",
            ],
            cwd=case_dir,
        )


# =============================================================================
# DAFOAM SETUP AND RUN
# =============================================================================


def flow_and_force_directions(
    angle_deg: float, normal_axis: str
) -> tuple[list[float], list[float]]:
    """
    Return freestream and positive-lift unit vectors.

    The streamwise axis is x. The aerodynamic normal axis is selected with
    --normal-axis (z for a conventional x-z aircraft convention, or y for a
    conventional 2-D x-y airfoil convention).

    Parameters
    ----------
    angle_deg
        Angle of attack in **degrees**, converted to radians internally.
    normal_axis
        Axis label spanning the lift direction with the streamwise axis,
        ``"z"`` or ``"y"``.

    Returns
    -------
    tuple
        ``(flow_direction, lift_direction)``, each a three-component unit
        vector as a plain list, in that order.

    Raises
    ------
    ValueError
        If ``normal_axis`` is not a supported axis label.
    """
    alpha = math.radians(angle_deg)
    cosine = math.cos(alpha)
    sine = math.sin(alpha)

    if normal_axis == "z":
        return [cosine, 0.0, sine], [-sine, 0.0, cosine]
    if normal_axis == "y":
        return [cosine, sine, 0.0], [-sine, cosine, 0.0]
    raise ValueError("normal_axis must be 'y' or 'z'")


def build_da_options(
    config: FlowConfig,
    wall_patches: list[str],
    farfield_patches: list[str],
) -> dict[str, Any]:
    """Build native DAFoam v4 options for a steady compressible primal run.

    Patch-name sequences are converted to plain lists, because DAFoam requires
    them. Any entries in the module-level ``DAFOAM_OPTIONS_OVERRIDES`` are
    merged in recursively, so case-specific settings need no change here.

    Parameters
    ----------
    config
        Flow and reference quantities, including the solver name, freestream
        state, convergence controls, and adjoint GMRES settings.
    wall_patches
        Patch names forming the aircraft wall, used for the force functions.
    farfield_patches
        Patch names forming the farfield, used for the boundary conditions.

    Returns
    -------
    dict
        A DAFoam options dictionary for the configured primal.

    Raises
    ------
    ValueError
        If the freestream velocity is not positive, or another configured
        quantity is rejected while the options are assembled.
    """
    # DAFoam requires native Python lists; callers may pass tuples.
    wall_patches = [str(name) for name in wall_patches]
    farfield_patches = [str(name) for name in farfield_patches]

    if config.velocity_m_per_s <= 0.0:
        raise ValueError("Freestream velocity must be positive")
    if config.pressure_pa <= 0.0 or config.temperature_k <= 0.0:
        raise ValueError("Freestream pressure and temperature must be positive")
    if config.reference_area_m2 <= 0.0:
        raise ValueError("Reference area must be positive")

    flow_direction, lift_direction = flow_and_force_directions(
        config.angle_of_attack_deg, config.normal_axis
    )
    velocity = [
        config.velocity_m_per_s * component for component in flow_direction
    ]
    density = (
        config.pressure_pa
        / (config.gas_constant_j_per_kg_k * config.temperature_k)
    )
    dynamic_pressure = 0.5 * density * config.velocity_m_per_s**2
    force_scale = 2.0 / (dynamic_pressure * config.reference_area_m2)  # Half-model symmetry factor

    options: dict[str, Any] = {
        "designSurfaces": wall_patches,
        "solverName": config.solver_name,
        "primalMinResTol": config.primal_min_res_tol,
        "primalMinResTolDiff": config.primal_min_res_tol_difference,
        "primalMinIters": config.primal_min_iterations,
        "primalBC": {
            "U0": {
                "variable": "U",
                "patches": farfield_patches,
                "value": velocity,
            },
            "p0": {
                "variable": "p",
                "patches": farfield_patches,
                "value": [config.pressure_pa],
            },
            "T0": {
                "variable": "T",
                "patches": farfield_patches,
                "value": [config.temperature_k],
            },
            "nuTilda0": {
                "variable": "nuTilda",
                "patches": farfield_patches,
                "value": [config.nu_tilda_m2_per_s],
            },
            "useWallFunction": bool(config.use_wall_functions),
        },
        "primalVarBounds": {
            "pMin": max(1.0, 0.05 * config.pressure_pa),
            "rhoMin": 0.01,
        },
        "function": {
            "CD": {
                "type": "force",
                "source": "patchToFace",
                "patches": wall_patches,
                "directionMode": "fixedDirection",
                "direction": flow_direction,
                "scale": force_scale,
            },
            "CL": {
                "type": "force",
                "source": "patchToFace",
                "patches": wall_patches,
                "directionMode": "fixedDirection",
                "direction": lift_direction,
                "scale": force_scale,
            },
        },
        "normalizeStates": {
            "U": config.velocity_m_per_s,
            "p": config.pressure_pa,
            "T": config.temperature_k,
            "phi": 1.0,
        },
        "adjStateOrdering": "cell",
        "adjEqnOption": {
            "gmresRelTol": config.adjoint_gmres_relative_tolerance,
            "gmresAbsTol": config.adjoint_gmres_absolute_tolerance,
            "gmresMaxIters": config.adjoint_gmres_max_iterations,
            "gmresRestart": config.adjoint_gmres_restart,
            "pcFillLevel": 1,
            "jacMatReOrdering": "natural",
        },
        "checkMeshThreshold": {
            "maxAspectRatio": 10000.0,
            "maxNonOrth": 90.0,
            "maxSkewness": 10.0,
            "maxIncorrectlyOrientedFaces": 0,
        },
    }

    if config.solver_name == "DARhoSimpleCFoam":
        options["transonicPCOption"] = 2
        options["adjPCLag"] = 5

    return deep_update(options, DAFOAM_OPTIONS_OVERRIDES)


def apply_custom_mesh_deformation(
    da_solver: Any,
    case_dir: Path,
    comm: _Communicator,
) -> None:
    """
    Extension hook for the user's future surface/volume deformation code.

    This baseline intentionally performs no deformation. A future integration
    can compute each rank's local DAFoam-ordered volume coordinates and call:

        local_volume_coordinates = your_volume_deformer(...)
        da_solver.setVolCoords(local_volume_coordinates)

    If the custom implementation starts from surface motion, it should first
    map/deform the volume mesh, then pass the final volume coordinates to
    DAFoam. Keeping that operation in this hook avoids any dependency on IDWarp
    or CSDL in the baseline primal runner.

    Parameters
    ----------
    da_solver
        Live DAFoam solver whose volume coordinates a future implementation
        would set. Unused here.
    case_dir
        Case directory the solver runs in. Unused here.
    comm
        MPI communicator. Unused here.

    Returns
    -------
    None
        This baseline deliberately performs no deformation, so the mesh is
        left exactly as ``gmshToFoam`` produced it.
    """
    del da_solver, case_dir, comm


def run_dafoam(
    case_dir: Path,
    config: FlowConfig,
    wall_patches: list[str],
    farfield_patches: list[str],
    comm: _Communicator,
) -> dict[str, Any]:
    """Instantiate PYDAFOAM, run the primal, and evaluate CD and CL.

    Imports DAFoam lazily, so the module stays importable without a sourced
    DAFoam environment. Collective: every rank participates in the primal.

    Parameters
    ----------
    case_dir
        Prepared OpenFOAM case the solver runs in.
    config
        Flow and reference quantities supplying the DAFoam options.
    wall_patches
        Patch names forming the aircraft wall.
    farfield_patches
        Patch names forming the farfield.
    comm
        MPI communicator handed to DAFoam.

    Returns
    -------
    dict
        The mapping DAFoam's ``evalFunctions`` fills in, holding one entry per
        evaluated function such as ``CD`` and ``CL``. It carries no additional
        run metadata.

    Raises
    ------
    RuntimeError
        If DAFoam cannot be imported because its environment was not sourced,
        or the primal failed to converge.
    """
    try:
        from dafoam import PYDAFOAM
    except ImportError as error:
        raise RuntimeError(
            "Could not import dafoam.PYDAFOAM. Source "
            "$HOME/dafoam/loadDAFoam.sh before launching Python."
        ) from error

    options = build_da_options(config, wall_patches, farfield_patches)

    original_directory = Path.cwd()
    os.chdir(case_dir)
    try:
        rank0_print(comm, "\nInitializing PYDAFOAM...")
        da_solver = PYDAFOAM(options=options, comm=comm)

        # This no-op hook is where the user's custom surface/volume deformation
        # implementation can be connected later.
        apply_custom_mesh_deformation(da_solver, case_dir, comm)

        comm.Barrier()
        rank0_print(comm, "\nRunning the DAFoam primal solver...")
        da_solver()
        comm.Barrier()

        primal_fail = int(getattr(da_solver, "primalFail", 0))
        if primal_fail != 0:
            raise RuntimeError(
                f"DAFoam reported primalFail={primal_fail}. Inspect the solver "
                "residual history and OpenFOAM case files."
            )

        functions: dict[str, Any] = {}
        da_solver.evalFunctions(functions)
        return functions
    finally:
        os.chdir(original_directory)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for this driver.

    The numerical flow and solver options take their defaults from a
    default-constructed :class:`FlowConfig`, as does the ``--wall-functions``
    flag, which uses :class:`argparse.BooleanOptionalAction` so both
    ``--wall-functions`` and ``--no-wall-functions`` are accepted. The case,
    path, and patch-name options do **not** come from the dataclass: they carry
    their own parser defaults.

    This is the CLI surface of a standalone driver script; it is unrelated to
    the no-CLI user example.

    Returns
    -------
    argparse.ArgumentParser
        Parser covering the mesh, case, patch-name, and flow options, with the
        module docstring as its description.
    """
    defaults = FlowConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--case",
        type=Path,
        default=Path.cwd(),
        help="OpenFOAM case directory (default: current directory)",
    )
    mesh_group = parser.add_mutually_exclusive_group(required=True)
    mesh_group.add_argument(
        "--mesh",
        type=Path,
        help="Gmsh .msh volume mesh to convert before the DAFoam run",
    )
    mesh_group.add_argument(
        "--reuse-openfoam-mesh",
        action="store_true",
        help="Reuse case/constant/polyMesh instead of importing a .msh file",
    )
    parser.add_argument(
        "--overwrite-existing-polymesh",
        action="store_true",
        help="Move an existing polyMesh to a timestamped backup before conversion",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Convert/check the mesh and stop before importing DAFoam",
    )
    parser.add_argument(
        "--skip-check-mesh",
        action="store_true",
        help="Skip the external OpenFOAM checkMesh command (not recommended)",
    )

    parser.add_argument(
        "--wall-patches",
        type=csv_names,
        default=["wall"],
        help="Comma-separated aerodynamic wall patch names (default: wall)",
    )
    parser.add_argument(
        "--farfield-patches",
        type=csv_names,
        default=["farfield"],
        help="Comma-separated farfield patch names (default: farfield)",
    )
    parser.add_argument(
        "--symmetry-patches",
        type=csv_names,
        default=[],
        help="Comma-separated symmetry patch names (default: none)",
    )

    parser.add_argument("--solver", default=defaults.solver_name)
    parser.add_argument(
        "--velocity",
        type=float,
        default=defaults.velocity_m_per_s,
        help=f"Freestream speed in m/s (default: {defaults.velocity_m_per_s})",
    )
    parser.add_argument(
        "--aoa",
        type=float,
        default=defaults.angle_of_attack_deg,
        help=f"Angle of attack in degrees (default: {defaults.angle_of_attack_deg})",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=defaults.pressure_pa,
        help=f"Freestream static pressure in Pa (default: {defaults.pressure_pa})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature_k,
        help=f"Freestream static temperature in K (default: {defaults.temperature_k})",
    )
    parser.add_argument(
        "--nu-tilda",
        type=float,
        default=defaults.nu_tilda_m2_per_s,
        help=(
            "Freestream Spalart-Allmaras nuTilda in m^2/s "
            f"(default: {defaults.nu_tilda_m2_per_s})"
        ),
    )
    parser.add_argument(
        "--reference-area",
        type=float,
        default=defaults.reference_area_m2,
        help=f"Force reference area in m^2 (default: {defaults.reference_area_m2})",
    )
    parser.add_argument(
        "--normal-axis",
        choices=["y", "z"],
        default=defaults.normal_axis,
        help="Aerodynamic lift/angle-of-attack axis (default: z)",
    )
    parser.add_argument(
        "--wall-functions",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_wall_functions,
        help=(
            "Use wall functions instead of resolving the near-wall layer; "
            "sets primalBC/useWallFunction "
            f"(default: {defaults.use_wall_functions})"
        ),
    )
    parser.add_argument(
        "--primal-tolerance",
        type=float,
        default=defaults.primal_min_res_tol,
        help=f"DAFoam primalMinResTol (default: {defaults.primal_min_res_tol})",
    )
    parser.add_argument(
        "--primal-min-iterations",
        type=int,
        default=defaults.primal_min_iterations,
        help=(
            "DAFoam primalMinIters "
            f"(default: {defaults.primal_min_iterations})"
        ),
    )
    parser.add_argument(
        "--primal-max-iterations",
        type=int,
        default=defaults.primal_max_iterations,
        help=(
            "Steady primal iteration ceiling written to controlDict endTime "
            f"(default: {defaults.primal_max_iterations})"
        ),
    )
    parser.add_argument(
        "--adjoint-relative-tolerance",
        type=float,
        default=defaults.adjoint_gmres_relative_tolerance,
        help=(
            "Adjoint GMRES relative tolerance "
            f"(default: {defaults.adjoint_gmres_relative_tolerance})"
        ),
    )
    parser.add_argument(
        "--adjoint-absolute-tolerance",
        type=float,
        default=defaults.adjoint_gmres_absolute_tolerance,
        help=(
            "Adjoint GMRES absolute tolerance "
            f"(default: {defaults.adjoint_gmres_absolute_tolerance})"
        ),
    )
    parser.add_argument(
        "--adjoint-max-iterations",
        type=int,
        default=defaults.adjoint_gmres_max_iterations,
        help=(
            "Adjoint GMRES iteration ceiling "
            f"(default: {defaults.adjoint_gmres_max_iterations})"
        ),
    )
    parser.add_argument(
        "--adjoint-restart",
        type=int,
        default=defaults.adjoint_gmres_restart,
        help=(
            "Adjoint GMRES restart interval "
            f"(default: {defaults.adjoint_gmres_restart})"
        ),
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("dafoam_results.json"),
        help="JSON results path, relative to the case unless absolute",
    )
    return parser


def main() -> int:
    """Run the command-line DAFoam primal analysis end to end.

    A CLI driver entry point, not a library function: it parses ``sys.argv``,
    prepares the OpenFOAM case, runs the primal across all ranks, and writes
    the results file. It must be launched with ``mpirun`` in a sourced DAFoam
    environment.

    Returns
    -------
    int
        Process exit status: ``0`` on success and nonzero on failure, suitable
        for ``sys.exit``.
    """
    args = make_parser().parse_args()

    try:
        from mpi4py import MPI
    except ImportError:
        print(
            "ERROR: mpi4py is unavailable. Source $HOME/dafoam/loadDAFoam.sh "
            "and launch this script through mpirun on TSCC.",
            file=sys.stderr,
        )
        return 2

    comm = MPI.COMM_WORLD

    case_dir = args.case.expanduser().resolve()
    mesh_file = args.mesh.expanduser().resolve() if args.mesh else None
    expected_patches = (
        args.wall_patches + args.farfield_patches + args.symmetry_patches
    )

    preprocessing_error: str | None = None
    if comm.rank == 0:
        try:
            validate_case_template(case_dir)
            if args.reuse_openfoam_mesh:
                validate_reused_mesh(
                    case_dir,
                    expected_patches,
                    args.skip_check_mesh,
                )
            else:
                assert mesh_file is not None
                if not mesh_file.is_file():
                    raise FileNotFoundError(f"Gmsh mesh not found: {mesh_file}")
                if mesh_file.suffix.lower() != ".msh":
                    raise ValueError(f"Expected a .msh file, got: {mesh_file}")
                convert_and_check_mesh(
                    case_dir=case_dir,
                    mesh_file=mesh_file,
                    wall_patches=args.wall_patches,
                    farfield_patches=args.farfield_patches,
                    symmetry_patches=args.symmetry_patches,
                    overwrite_existing=args.overwrite_existing_polymesh,
                    skip_check_mesh=args.skip_check_mesh,
                )
        except Exception as error:  # broadcast failure so other ranks do not hang
            preprocessing_error = f"{type(error).__name__}: {error}"

    preprocessing_error = comm.bcast(preprocessing_error, root=0)
    if preprocessing_error is not None:
        if comm.rank == 0:
            print(f"\nERROR during mesh/case preprocessing:\n{preprocessing_error}", file=sys.stderr)
        return 2

    comm.Barrier()
    if args.convert_only:
        rank0_print(comm, "\nMesh conversion and validation completed.")
        return 0

    config = FlowConfig(
        solver_name=args.solver,
        velocity_m_per_s=args.velocity,
        angle_of_attack_deg=args.aoa,
        pressure_pa=args.pressure,
        temperature_k=args.temperature,
        nu_tilda_m2_per_s=args.nu_tilda,
        reference_area_m2=args.reference_area,
        normal_axis=args.normal_axis,
        use_wall_functions=args.wall_functions,
        primal_min_res_tol=args.primal_tolerance,
        primal_min_iterations=args.primal_min_iterations,
        primal_max_iterations=args.primal_max_iterations,
        adjoint_gmres_relative_tolerance=args.adjoint_relative_tolerance,
        adjoint_gmres_absolute_tolerance=args.adjoint_absolute_tolerance,
        adjoint_gmres_max_iterations=args.adjoint_max_iterations,
        adjoint_gmres_restart=args.adjoint_restart,
    )

    configuration_error: str | None = None
    if comm.rank == 0:
        try:
            set_control_dict_max_iterations(
                case_dir,
                config.primal_max_iterations,
            )
        except Exception as error:
            configuration_error = f"{type(error).__name__}: {error}"
    configuration_error = comm.bcast(configuration_error, root=0)
    if configuration_error is not None:
        if comm.rank == 0:
            print(
                "\nERROR while configuring the primal iteration limit:\n"
                f"{configuration_error}",
                file=sys.stderr,
            )
        return 2

    comm.Barrier()
    try:
        functions = run_dafoam(
            case_dir=case_dir,
            config=config,
            wall_patches=args.wall_patches,
            farfield_patches=args.farfield_patches,
            comm=comm,
        )
    except Exception as error:
        # DAFoam/PETSc normally aborts all ranks together for fatal failures.
        # This message handles ordinary Python/API failures.
        print(f"[rank {comm.rank}] ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 3

    if comm.rank == 0:
        results_path = args.results_file.expanduser()
        if not results_path.is_absolute():
            results_path = case_dir / results_path
        results_path = results_path.resolve()

        payload = {
            "case_directory": str(case_dir),
            "mesh_source": (
                "reused constant/polyMesh" if mesh_file is None else str(mesh_file)
            ),
            "mpi_ranks": comm.size,
            "flow_config": asdict(config),
            "wall_patches": args.wall_patches,
            "farfield_patches": args.farfield_patches,
            "symmetry_patches": args.symmetry_patches,
            "functions": jsonable(functions),
        }
        results_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print("\nDAFoam primal analysis completed.", flush=True)
        print("Functions:", json.dumps(jsonable(functions), indent=2), flush=True)
        print(f"Results written to {results_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
