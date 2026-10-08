# E175 example

The basic `examples/e175_surface_deformation.py` deforms the tracked triangular
E175 CFD wall at the full example design point and reports its quality. Run it
directly:

```bash
python examples/e175_surface_deformation.py
```

It is an ordinary Python script configured by editing values in the file or by
calling `main(...)` with keyword arguments. There is no command-line interface
and no environment-variable configuration.

The script walks the five stages the pipeline performs, in order.

## 1. Choose geometry and mesh files

A STEP body supplies the outer mould line; a surface mesh supplies the nodes
that must follow it.

```python
from pathlib import Path
import tempfile

import csdl_alpha as csdl

import bsm3.mesh_motion as mm

inputs = mm.InputFiles(
    geometry_file=step_file,
    surface_mesh_file=surface_mesh_file,
    cache_directory=Path(tempfile.gettempdir()) / "bsm3_e175_example_cache",
)
```

The basic example deliberately stays on the triangle wall. Panel-mesh
regularization is a separate calibration problem covered by the advanced
example below.

## 2. Declare the geometry motion

The recorder is created, started, and stopped **by the caller**. The public
`GeometryModel` and `mm.run` interfaces do not create, start, or stop that
recorder, which is what lets mesh motion compose inside a larger graph. Open
the `try` immediately so a failure in any later stage still stops the recorder.

```python
recorder = csdl.Recorder(inline=True)
recorder.start()
try:
    geometry = mm.GeometryModel()

    wing_shift = geometry.design_variable("wing_shift", 0.0)
    wing_incidence = geometry.design_variable("wing_incidence", 0.0)
    wing_area = geometry.design_variable("wing_area", 70.0)
    tail_incidence = geometry.design_variable("tail_incidence", 0.0)
    fuselage_width = geometry.design_variable("fuselage_width", 1.0)
```

This example uses the **optional** built-in helpers, because they keep the
declaration short and readable. They are not the required entry point: an
external parameterization replaces this stage with
[`add_component`](external_parameterization.md), and stages 3 to 5 are
identical either way.

```python
    geometry.add_lifting_surface(
        name="wing",
        search_name="wing",
        pivot_intersection="wing_root",
        translation_x=wing_shift,
        rotation_y_degrees=wing_incidence,
        area=wing_area,
        reference_area=70.0,
        reference_aspect_ratio=8.4,
    )
    geometry.add_lifting_surface(
        name="tail",
        search_name="HT",
        pivot_intersection="tail_root",
        rotation_y_degrees=tail_incidence,
        projection_name="horizontal_tail",
    )
    geometry.add_body(
        name="fuselage",
        search_name="fuselage",
        diameter_scale=fuselage_width,
    )
```

`connect` names the closed intersection curve between two components. The
`pivot_intersection` above refers to one of these by name.

```python
    geometry.connect(
        name="wing_root",
        driving_component="wing",
        query_component="fuselage",
    )
    geometry.connect(
        name="tail_root",
        driving_component="tail",
        query_component="fuselage",
    )
```

## 3. Choose mesh-motion and quality settings

```python
    motion = mm.MeshMotion(
        surface=mm.SurfaceMotion(
            load_steps=2,
            stiffening_exponent=1.5,
            distance_weighting=mm.DistanceWeighting(
                enabled=True,
                beta=5.0,
                length_scale=10.0,
                decay="exp",
            ),
        ),
        quality=mm.QualityChecks(surface=True),
        symmetry=True,
    )
```

`symmetry=True` declares that the surface is a half model about the symmetry
plane. See [Background](background.md) for what each setting controls.

## 4. Run the differentiable model

```python
    result = mm.run(
        inputs=inputs,
        geometry=geometry,
        motion=motion,
        recorder=recorder,
    )
finally:
    recorder.stop()
```

## 5. Inspect the result

```python
result.print_summary()
```

`print_summary` reports vertex, cell, and n-gon-mode counts; elapsed time; fold
and inversion counts; and, when available, degenerate-element and minimum
scaled-Jacobian quality fields. The arrays are on the result object:
`result.surface_coordinates` is the final reprojected surface as a CSDL
variable, with `initial_surface_coordinates` and
`preprojected_surface_coordinates` alongside it for comparison, plus the
inversion and quality reports. See the [API reference](api.md).

Set `visualize=True` when calling `main(...)` to open an interactive view of
the final deformed mesh. Set `check_derivatives=True` to register the surface
coordinate objective and run the public finite-difference convergence sweep.

## Deformation scale

The example interpolates the whole design point with a single
`deformation_scale`, so one number moves every target coherently. Its default
is `1.0`, the complete documented design point. The tracked triangle-wall
integration test exercises that scale without folds or inversions.

## Advanced quad-panel calibration

`examples/e175_quad_panel_calibration.py` runs the same geometry motion on the
clean curated `e175_quad_panel.msh` panel. It
keeps `polygon_regularization_weight` explicit because the best weight depends
on mesh topology and deformation; `0.3` remains a historical placeholder while
the calibration sweep is reviewed, not a universal recommendation.

The advanced example prints the new public classifications:

- graph-free and graph-prescribed IDs;
- parametrically prescribed IDs, which are reevaluated at fixed surface
  coordinates;
- exact intersection IDs from the bracketed component-intersection solve;
- the IDs actually sent through closest-point projection; and
- the non-converged closest-point subset.

Every array uses the complete input mesh's zero-based index space. This also
makes custom visualization direct: use `result.surface_coordinates.value` for
the final coordinates and color rows selected by the classification arrays.

## Fuel-burn optimization with VortexAD

`examples/e175_fuel_burn_optimization.py` is the tracked advanced composition:

```text
geometry design variables
    -> GAMMA surface motion and reprojection
    -> VortexAD lift and induced drag
    -> total drag with an explicit parasite contribution
    -> Breguet cruise fuel burn
    -> lift constraint and fuel-burn objective
```

Run it after installing the exact optional VortexAD revision documented under
[Integrations](integrations.md):

```bash
python examples/e175_fuel_burn_optimization.py
```

The script keeps `MeshMotion.derivative_check` disabled because that debug
convenience would register its own objective. It registers fuel burn directly
and optionally calls the public FD sweep for the complete analytic graph. Its
mission and parasite-drag values are clearly labeled illustrative rather than
validated E175 performance data; replace them with sourced analysis inputs for
an actual design study.

## Cessna 208 strut-attachment deformation

The `examples/cessna_208_strut_attachment_deformation.py` script
slides both strut wing ends spanwise and both fuselage ends fore/aft while
keeping the two struts mirrored. It runs directly, with no command-line
arguments:

```bash
python examples/cessna_208_strut_attachment_deformation.py
```

Edit the settings at the top of the script. The default moves the starboard
wing attachment inboard by 5% of the full CAD span and the fuselage attachment
aft by 5% of the fuselage-body length, in one load step. The source mesh has
triangles and quads, so the example uses positive n-gon affine regularization.
It also holds the baseline chordwise CAD coordinate near the wing leading edge
(`WING_LEADING_EDGE_HOLD_M`, fading out by `WING_LEADING_EDGE_FADE_M`) while
allowing spanwise sliding. All four seams (strut-wing, strut-fuselage,
wing-fuselage and stab-fuselage) are always connected. The built-in interactive viewer is enabled by
default; set `VISUALIZE = False` for headless runs.

This is one sampled design, not a claim that every combination of offsets is
valid. In particular, the opposite 5%/5% corner has a partly exposed fuselage
end and does not define four of its strut-fuselage intersection brackets. The
example measures its final seam residuals and rejects an invalid result.
Intermediate load-state seams are not checked by this example. The viewer
can open before the final seam check; a failed check raises and writes a
clearly marked invalid diagnostic rather than a normal result. Its source
inputs and checksums are recorded in the [C208 asset record](../examples/cessna_208/ASSETS.md).

At the default setting, no polygon normal flips occur. Eight quads have a
negative corner; three already do in the source mesh. The example prints these
quality figures and highlights the flagged cells in the viewer. A negative
corner in a recombined quad is distinct from inversion of the whole polygon.

The script reports the analytic sensitivity of the mean final mesh node
(x, y, z) to both attachment variables. Set `CHECK_DERIVATIVES = True` (one
load step only) to re-run the complete pipeline at each design variable
plus and minus every step in `DERIVATIVE_CHECK_STEP_SIZES` and compare each
vector with centered finite differences; the check passes within 0.5% of the
vector length. The example test runs the same comparison at the default
design. The final-mesh coordinate
map is piecewise smooth: projected vertices can jump between nearby
closest points at CAD crease lines (on the strut and the wing leading edge),
which scatters whole-mesh finite differences by up to about 0.3%. A gradient
therefore describes the selected branch and must be checked for the design
and objective of interest.

Two load steps are available for mesh viewing, with a runtime warning. Their
mesh gradients measurably disagree with finite differences and **must not be
used for optimization**. The cause remains under investigation.
