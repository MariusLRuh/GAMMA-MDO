# GAMMA

**GAMMA — Geometry-Aware Mesh Movement Analysis** performs differentiable
boundary-surface mesh motion. Given a CAD outer mould line and a surface mesh
that must follow it, GAMMA moves every mesh node as the geometry deforms,
carries analytic derivatives through the CSDL graph, and reports mesh-quality
and inversion diagnostics so validity is a measured outcome rather than an
assumption.

The installable distribution is `gamma-mdo`; the Python import namespace
remains `bsm3`.

It also provides tetrahedral volume-mesh motion and an optional CSDL/DAFoam
coupling.

## Quickstart

The complete tracked E175 example is the quickest executable starting point:

```bash
python examples/e175_surface_deformation.py
```

Its first uncached run can take several minutes. All high-level inputs are
editable in the script's `main` function.

`bsm3.mesh_motion` is the intended library entry point and is deliberately
small. The following is the call shape, not a standalone example: at least one
component and its deformed coefficients or built-in motion must be registered
on `geometry` before `run` is called.

```python
from pathlib import Path

import csdl_alpha as csdl

import bsm3.mesh_motion as mm

recorder = csdl.Recorder(inline=True)
recorder.start()
try:
    geometry = mm.GeometryModel()
    # Required before run: register at least one component, either with
    # geometry.add_component(...) and external deformed coefficients or with
    # the optional built-in helpers demonstrated by the E175 example.

    result = mm.run(
        inputs=mm.InputFiles(
            geometry_file=Path("geometry.stp"),
            surface_mesh_file=Path("surface.msh"),
            cache_directory=Path("/tmp/bsm3_cache"),
        ),
        geometry=geometry,
        motion=mm.MeshMotion(quality=mm.QualityChecks(surface=True)),
        recorder=recorder,
    )
finally:
    recorder.stop()

result.print_summary()
```

The caller owns the public mesh-motion recorder: `GeometryModel` and `mm.run`
do not create, start, or stop it.

A complete, runnable example is tracked at
[`examples/e175_surface_deformation.py`](examples/e175_surface_deformation.py).

A second example moves both Cessna 208 strut attachments on a mixed
triangle/quad mesh:

```bash
python examples/cessna_208_strut_attachment_deformation.py
```

The STEP and mesh inputs ship with GAMMA. Edit the settings at the top of the
script; it has no CLI and opens the built-in mesh viewer by default. The
one-step default prints the analytic sensitivity of the mean final mesh node
(x, y, z) to each attachment variable. Set `CHECK_DERIVATIVES = True` to
compare it with complete-pipeline centered finite differences in the same run.
Two load steps are available for mesh viewing only: their gradients disagree
with finite differences and must not be used for optimization. See the
[example details](docs/src/examples.md#cessna-208-strut-attachment-deformation).

## Documentation

The documentation source is in [`docs/`](docs/). There is no hosted
deployment yet. Build it locally:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs /tmp/bsm3-docs-html
```

It covers installation, the E175 example, the external-parameterization
contract, the public API, background, and integration status.

## Installation

The validated setup installs GAMMA from a checkout and pins its geometry
dependencies to exact Git revisions. GAMMA deliberately declares no automatic
dependencies, so its own editable install cannot replace packages in an
externally managed DAFoam, MPI, or PETSc stack.

The validated stack is Python 3.12 with NumPy 2.0.2, SciPy 1.13.1, and JAX
0.4.38, against these exact revisions:

```bash
conda create -n bsm3_py312_main python=3.12
conda activate bsm3_py312_main

# Installs the exact tested dependency set, including CSDL_alpha at
# 73a9efd1033016a835779db10a9b9e81ed2254ce.
python -m pip install -r requirements-ci.txt

# LFS is installed separately so it cannot replace the validated stack.
python -m pip install --no-deps \
  "lsdo_function_spaces @ git+https://github.com/LSDOlab/lsdo_function_spaces.git@307ad3aabfff31c6fb44ddf51bc0dcc41a60c420"

python -m pip install --no-deps --no-build-isolation -e .
```

See [docs/src/getting_started.md](docs/src/getting_started.md) for details, and
[HPC_DAFOAM_INSTALL.md](HPC_DAFOAM_INSTALL.md) for the TSCC/DAFoam prerequisite
audit.

## Running

```bash
python examples/e175_surface_deformation.py
python -m pytest tests -q
```

DAFoam, OpenFOAM, `mpi4py`, and real MPI are optional integrations that need an
existing sourced solver environment and a case you supply. They are not run in
the standard test suite.

## License

GAMMA is licensed under the GNU Lesser General Public License v3.0 or later.

The example geometry and meshes (E175 and Cessna 208) are separately
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see
the [E175](docs/examples/e175/ASSETS.md) and
[C208](docs/examples/cessna_208/ASSETS.md) asset records for attribution.
They are approximate research models, not manufacturer data.
