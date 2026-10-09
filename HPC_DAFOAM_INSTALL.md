# Installing GAMMA beside DAFoam on TSCC

GAMMA must be installed from the same sourced environment used to run DAFoam.
It deliberately declares no automatic pip dependencies so that installing it
cannot replace the MPI, PETSc, NumPy, SciPy, or DAFoam packages selected by the
cluster environment.

## Safe editable installation

```bash
source /path/to/DAFoam/loadDAFoam.sh
git clone --branch dev https://github.com/MariusLRuh/GAMMA-MDO.git
cd GAMMA-MDO
python -m pip install --no-deps --no-build-isolation -e .
```

Both flags are intentional:

- `--no-deps` prevents dependency resolution, installation, and upgrades.
- `--no-build-isolation` prevents pip from creating a build environment and
  downloading build requirements.

Do not run `pip install -r requirements.txt`; that file is an inactive
prerequisite checklist rather than an environment specification.

## Preflight check

Before installation, verify that the active Python belongs to the intended
DAFoam environment and that all imports resolve there:

```bash
which python
python - <<'PY'
import dafoam
import mpi4py
import petsc4py
import numpy
import scipy
import csdl_alpha
import lsdo_function_spaces
import gmsh
print("DAFoam/GAMMA prerequisite imports passed")
PY
```

After installation:

```bash
python -c "import gamma_mdo; print(gamma_mdo.__version__, gamma_mdo.__file__)"
python -m pytest tests -q
```

A complete E175 DAFoam analysis driver is not yet published. The
deformation-only driver can be tested after downloading its R5 meshes:

```bash
python -m gamma_mdo.assets download e175-r5
python -m gamma_mdo.core.boundary_surface_movement.cfd_mesh_movement_test
```
