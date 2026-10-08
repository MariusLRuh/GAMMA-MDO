# Installation

Install GAMMA from a checkout into an environment you control. The validated
setup uses exact Git revisions for CSDL_alpha and lsdo_function_spaces rather
than relying on whichever releases a package registry currently provides.

GAMMA deliberately declares **no mandatory pip dependencies**. Its own install
therefore uses `--no-deps`, so it does not replace packages in an existing
geometry or DAFoam environment whose MPI, PETSc, NumPy, SciPy, and solver
versions are under an administrator's control. The dependency-bootstrap step
below intentionally installs packages and belongs in a new core/developer
environment, not an administrator-managed solver environment.

## Validated core stack

The following combination is the one the test suite and numerical acceptance
runs used. This is the exact validated core/developer environment, not a claim
about the smallest possible runtime dependency set.

| Component | Validated version |
| --- | --- |
| Python | 3.12 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| JAX / jaxlib | 0.4.38 |
| CSDL_alpha | `73a9efd1033016a835779db10a9b9e81ed2254ce` |
| lsdo_function_spaces | `307ad3aabfff31c6fb44ddf51bc0dcc41a60c420` |

```bash
conda create -n bsm3_py312_main python=3.12
conda activate bsm3_py312_main

# Install the complete tested dependency set. requirements-ci.txt contains the
# exact CSDL_alpha revision shown in the table as well as the tested numerical,
# geometry, diagnostics, and development packages.
python -m pip install -r requirements-ci.txt

# Install official LFS without allowing its dependency resolver to replace the
# validated versions installed in the preceding step.
python -m pip install --no-deps \
  "lsdo_function_spaces @ git+https://github.com/LSDOlab/lsdo_function_spaces.git@307ad3aabfff31c6fb44ddf51bc0dcc41a60c420"

# Install GAMMA itself without resolving or building dependencies, so an
# externally managed solver environment is left untouched.
python -m pip install --no-deps --no-build-isolation -e .
```

Both `--no-deps` choices are load-bearing:

- `--no-deps` on `lsdo_function_spaces` preserves the complete validated stack
  installed from `requirements-ci.txt`, including the CSDL revision.
- `--no-deps --no-build-isolation` on GAMMA keeps pip from resolving or
  rebuilding anything in the surrounding environment.

## Developer extras

`requirements-ci.txt` includes the packages used by tests and diagnostics,
including `pytest`, `meshio`, and PyVista. Interactive GAMMA visualization is
optional, but PyVista is currently imported eagerly by the pinned LFS package,
so it remains part of this validated environment even for headless runs.

## Optional solver environment

DAFoam, OpenFOAM, `mpi4py`, and `petsc4py` are **not** installed by the steps
above and are not needed for surface mesh motion. They come from an existing
sourced solver environment and are only required for the optional aerodynamic
coupling. See [Integrations](integrations.md) for what is and is not exercised.

## Inputs are local files

A run needs a STEP geometry file and a surface mesh file. These are ordinary
local paths that you supply:

```python
inputs = mm.InputFiles(
    geometry_file=Path("/path/to/geometry.stp"),
    surface_mesh_file=Path("/path/to/surface.msh"),
    cache_directory=Path("/path/to/writable/cache"),
)
```

`cache_directory` holds reusable setup data such as baseline projections and
seam identification. The first run populates it; later runs with the same
geometry and mesh are substantially faster. Point it somewhere writable and
outside your source checkout.

### Example assets

The E175 geometry, its quad-dominant panel mesh, the R1 wall mesh and the
Cessna 208 inputs ship with the package and are enough to run the surface
examples. The large R1 and R5 volume meshes are downloaded on demand and
verified by SHA-256:

```bash
python -m bsm3.assets list
python -m bsm3.assets download e175-r1-volume   # or e175-r5, or --all
```

Downloads are cached in `~/.cache/gamma/assets` (set `GAMMA_ASSET_DIR` to
change it). In code, `bsm3.assets.asset_path(name)` gives a file's location.
Tests that need a downloaded mesh skip when it is absent. The assets are
licensed under CC BY 4.0; see the
[E175](../examples/e175/ASSETS.md) and
[C208](../examples/cessna_208/ASSETS.md) asset records.

## Documentation is not yet hosted

This site builds from the repository. Build it locally with:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs /tmp/bsm3-docs-html
```

There is no published Read the Docs deployment yet; `.readthedocs.yaml` is
configured and ready for one.
