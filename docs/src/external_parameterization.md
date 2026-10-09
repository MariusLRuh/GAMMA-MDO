# External parameterization

GAMMA does not require you to use its geometry helpers. The general contract is:

```text
external CSDL/LFS parameterization
    -> deformed coefficients
    -> GeometryModel.add_component(...)
    -> GAMMA projection, surface motion, quality, and optional volume chain
```

`GeometryModel` is a **declaration and binding envelope, not a
parameterization**. `add_component` is the general entry point and the
universal boundary: you supply deformed component coefficients produced by any
differentiable CSDL/LFS-compatible parameterization, owned entirely outside
GAMMA, and GAMMA places no constraint on how they were built.

`add_lifting_surface` and `add_body` are **optional conveniences** oriented at
the E175 example. They build their own private records rather than calling
through `add_component`, and they exist only for callers who would rather
describe a motion than supply coefficients. They are not the generic mechanism.

## Who owns what

The external parameterization owns its design variables **and** the recorder.
`GeometryModel` and `mm.run` never create, start, or stop that recorder, and
`add_component` never registers design variables on your behalf.

```python
import csdl_alpha as csdl

import gamma_mdo.mesh_motion as mm

# Your recorder, your design variables.
recorder = csdl.Recorder(inline=True)
recorder.start()
try:
    sweep = csdl.Variable(name="sweep", value=0.0)
    sweep.set_as_design_variable()

    # Your parameterization produces deformed coefficients however it likes.
    wing_coefficients = my_parameterization(baseline_wing_coefficients, sweep)

    geometry = mm.GeometryModel()
    geometry.add_component(
        name="wing",
        search_name="wing",
        deformed_coefficients=wing_coefficients,
    )
    geometry.add_component(
        name="fuselage",
        search_name="fuselage",
        deformed_coefficients=fuselage_coefficients,
    )
    geometry.connect(
        name="wing_root",
        driving_component="wing",
        query_component="fuselage",
    )

    result = mm.run(
        inputs=inputs,
        geometry=geometry,
        motion=mm.MeshMotion(quality=mm.QualityChecks(surface=True)),
        recorder=recorder,
    )
finally:
    recorder.stop()
```

Everything downstream — intersection curves, graph-Laplacian surface motion,
reprojection onto the deformed outer mold line, quality diagnostics, and the
optional volume chain — is identical whether the coefficients came from the
built-in helpers or from your own parameterization.

## Coefficient formats

`deformed_coefficients` accepts either:

- one stacked `(N, 3)` CSDL variable or array, with patches concatenated by row
  in sorted patch-ID order; or
- a mapping from patch ID to that patch's coefficient block.

CSDL expressions are preserved, so derivatives through the final reprojected
mesh stay analytic.

## The compatibility boundary

Three conditions must hold. They are the current limits of the contract:

1. **Component topology must stay compatible.** The STEP component found by
   `search_name` supplies the canonical patch IDs. Your coefficients must
   describe that same component.
2. **Coefficient layout must stay compatible.** Shapes and patch IDs are
   validated once the STEP component is imported and the canonical patch IDs
   are known — that is, *after* the call to `add_component`, not at call time.
   A layout mismatch therefore surfaces during the run, not during declaration.
3. **Every dependent variable must belong to the caller-owned recorder** that
   you pass to `gamma_mdo.mesh_motion.run`. A variable created under a different
   recorder is not part of the graph GAMMA evaluates.

## Restricting motion

`free_region` restricts which coordinates a component may move in. `None`
leaves the whole component free; otherwise pass a mapping of `"x"`, `"y"`, or
`"z"` to a `(lower, upper, mode)` triple.

`projection_name` sets the diagnostic projection name, defaulting to `name`,
and `projection_mode` is either `"all"` or `"lifting_surface"`.
