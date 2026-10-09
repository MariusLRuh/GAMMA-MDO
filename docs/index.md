# GAMMA

**GAMMA — Geometry-Aware Mesh Movement Analysis** performs differentiable
boundary-surface mesh motion. Given a CAD outer mold line and a surface mesh
that must follow it, GAMMA moves every mesh node as the geometry deforms,
carries analytic derivatives through the CSDL graph, and reports mesh-quality
and inversion diagnostics so validity is checked rather than guaranteed.

Install the `gamma-mdo` distribution and import its public Python namespace as
`gamma_mdo`.

:::{warning}
**Alpha release.** GAMMA is alpha (pre-release) software under active
development. Interfaces, default settings and outputs may change between
releases without notice, and the methods have not been validated for
production or engineering use. Use it at your own risk: GAMMA is provided
without warranty, as stated in its
[license](https://github.com/MariusLRuh/GAMMA-MDO/blob/main/LICENSE.txt).
Please report problems through
[GitHub issues](https://github.com/MariusLRuh/GAMMA-MDO/issues).
:::

## E175 mesh deformation

<div style="display: flex; flex-wrap: wrap; gap: 1rem;">
  <figure style="flex: 1 1 18rem; margin: 0; text-align: center;">
    <a href="https://github.com/MariusLRuh/GAMMA-MDO/releases/download/v0.2.0a1/e175_mesh_deformation_tri_full.gif">
      <img
        src="https://github.com/MariusLRuh/GAMMA-MDO/releases/download/v0.2.0a1/e175_mesh_deformation_tri_full.gif"
        alt="Animation of the E175 triangular surface mesh deforming with its geometry"
        width="1400"
        height="640"
        loading="lazy"
        decoding="async"
        style="width: 100%; height: auto;"
      >
    </a>
    <figcaption>Triangular surface mesh</figcaption>
  </figure>
  <figure style="flex: 1 1 18rem; margin: 0; text-align: center;">
    <a href="https://github.com/MariusLRuh/GAMMA-MDO/releases/download/v0.2.0a1/e175_mesh_deformation_quad_full.gif">
      <img
        src="https://github.com/MariusLRuh/GAMMA-MDO/releases/download/v0.2.0a1/e175_mesh_deformation_quad_full.gif"
        alt="Animation of the E175 quad-dominant surface mesh deforming with its geometry"
        width="1400"
        height="640"
        loading="lazy"
        decoding="async"
        style="width: 100%; height: auto;"
      >
    </a>
    <figcaption>Quad-dominant surface mesh</figcaption>
  </figure>
</div>

The animations are served from the
[v0.2.0a1 release](https://github.com/MariusLRuh/GAMMA-MDO/releases/tag/v0.2.0a1),
so they are not included in repository clones or package installations.

The pipeline is:

```text
geometry parameterization
    -> deformed component coefficients
    -> intersection curves between components
    -> graph-Laplacian surface motion with regularization
    -> reprojection onto the deformed outer mold line
    -> quality and inversion diagnostics
    -> optional volume-mesh motion
```

`gamma_mdo.mesh_motion` is the intended entry point and is deliberately small: input
files, a `GeometryModel` describing what moves, a `MeshMotion` settings object
describing how the mesh follows, and `run` to evaluate it.

```python
import gamma_mdo.mesh_motion as mm

result = mm.run(
    inputs=mm.InputFiles(
        geometry_file=step_path,
        surface_mesh_file=surface_path,
        cache_directory=cache_path,
    ),
    geometry=geometry,
    motion=mm.MeshMotion(quality=mm.QualityChecks(surface=True)),
    recorder=recorder,
)
result.print_summary()
```

## Where to start

- **[Installation](src/getting_started.md)** — the validated Python 3.12
  dependency stack and the install flags that protect an existing solver
  environment.
- **[E175 example](src/examples.md)** — the five-stage workflow, based on the
  tracked `examples/e175_surface_deformation.py`.
- **[External parameterization](src/external_parameterization.md)** — the
  generic contract: bring your own differentiable coefficients.
- **[API reference](src/api.md)** — the public `gamma_mdo.mesh_motion` surface.
- **[Background](src/background.md)** — what each pipeline stage does and why.
- **[Integrations and troubleshooting](src/integrations.md)** — honest status
  of DAFoam/OpenFOAM, MPI, VortexAD, mesh generation, and assets.

```{toctree}
:maxdepth: 2
:hidden:

src/getting_started
src/examples
src/external_parameterization
src/api
src/background
src/integrations
```
