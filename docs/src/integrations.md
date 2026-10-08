# Integrations and troubleshooting

This page states what is actually exercised and what is not.

## Status

### DAFoam and OpenFOAM — optional, not in the standard suite

The aerodynamic coupling requires an **existing sourced solver environment**, an
OpenFOAM installation, and a case directory you supply. DAFoam, `mpi4py`, and
`petsc4py` are imported lazily, so importing GAMMA and running its test suite
needs none of them.

**No DAFoam or OpenFOAM solve runs in the standard CI suite.** The rank-0
geometry-to-volume *construction* path is covered by tests that mock the
downstream operation, which proves the API boundary holds. That is a
construction guarantee, not evidence that a real DAFoam solve was executed.

### Real MPI — optional, not in the standard suite

The distributed coupling is designed around rank-0 execution with broadcast
coordinates. The communication protocol, ownership contracts, and matrix-free
VJP are tested with scripted single-process communicators and a structural
communicator protocol that resolves without `mpi4py` installed. No real
multi-rank MPI job runs in the standard suite.

### VortexAD — optional tracked adapter

GAMMA provides `mm.build_panel_aerodynamics`, a differentiable adapter from the
public `mm.run` result to VortexAD's steady panel method. It targets official
VortexAD `main` at commit
`8c5bc86fda5fa1e359fecde24c6c6e8527c773b5`. Install that optional dependency
explicitly after validating it against your environment:

```bash
python -m pip install --no-deps \
  "VortexAD @ git+https://github.com/LSDOlab/VortexAD.git@8c5bc86fda5fa1e359fecde24c6c6e8527c773b5"
```

VortexAD is not a base dependency and is not installed by the standard test or
documentation environments. The standard suite exercises the adapter and its
analytic derivative with a fake panel solver implementing the pinned API; a
real VortexAD import check is an integration test and skips when the optional
package is absent. No real panel solve is claimed as a standard-CI result.

The adapter accepts triangle/quad connectivity, derives adjacency and trailing
edges from the undeformed curated mesh, rejects non-converged OML projections
by default, and registers `CL`, `CDi`, `L`, and `Di` in
`MeshMotionResult.aerodynamic_outputs`. The caller owns the active recorder;
the adapter never starts or stops it.

### Mesh generation — separate

Generating a surface or volume mesh is outside the surface-motion quickstart.
GAMMA consumes meshes you already have.

## Assets and file formats

Surface meshes are read by suffix through `gamma_mdo.preprocessing.import_mesh`:
`.msh`, `.stl`, and `.npz`.

The `.npz` polygon format is **safe**: it is loaded with `allow_pickle=False`
and holds exactly three non-object arrays — `vertices`, a flattened
`connectivity`, and `offsets` — so a malformed or hostile file cannot execute
code. The curated mixed-N-gon wall asset uses this format.

Pickle is deliberately **not** part of suffix dispatch. Two explicitly named
entry points remain for trusted local files you produced yourself:

- `gamma_mdo.preprocessing.import_trusted_polygon_pickle(path)`
- `gamma_mdo.core.projections.warm_start_projections.load_function_set_from_trusted_pickle(path)`

Both take a mandatory path, carry an execution warning, and are opt-in. Python
pickle executes arbitrary code on load; never point either at an untrusted or
remote file. No curated asset and no retained pipeline loads a pickle.

## Troubleshooting

**A path does not exist.** Geometry and mesh paths are local user inputs, not
repository assets. Larger volume meshes and any private CFD case are untracked;
tests that need them skip automatically. Check the path before assuming a
pipeline failure.

**The cache seems stale or the run is unexpectedly slow.** The first run with a
given geometry and mesh populates `cache_directory`; later runs reuse it. Set
`MeshMotion(rebuild_setup_cache=True)` to force a rebuild. Point the cache
somewhere writable and outside your source checkout.

**`RuntimeError` about the recorder, or derivatives that vanish.** The public
`GeometryModel` and `mm.run` interfaces do not create, start, or stop the
caller's recorder. Create it, start it, pass the *same* recorder to `mm.run`,
and stop it in a `finally` block. Every design variable your coefficients
depend on must belong to that recorder; a variable created under a different
recorder is not part of the graph GAMMA evaluates.

**Coefficient shape or patch-ID errors.** Component topology and coefficient
layout must stay compatible with the STEP component found by `search_name`.
These are validated after the component is imported and the canonical patch IDs
are known, so a mismatch surfaces during the run rather than at the
`add_component` call.

**A headless run should not open a plot.** Interactive GAMMA visualization is
optional; leave `Visualization(enabled=False)`. PyVista is nevertheless part
of the validated environment because the pinned LFS package currently imports
it eagerly, even when no interactive window is requested.

**New inverted elements after a large deformation.** Compare the three
inversion reports: an element inverted in the input mesh was not introduced by
the solve. Reduce the deformation, raise `load_steps`, or check the
quad-dominant panel specifically, which is more sliver-sensitive than the
triangle wall.
