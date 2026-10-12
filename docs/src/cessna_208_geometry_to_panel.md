# Cessna 208: from CAD parameters to panel aerodynamics

`examples/cessna_208_geometry_to_panel.py` takes the Cessna 208 from five
planform design targets to a deformed CFD surface mesh and, optionally, a
panel-method solution. The three stages are assembled in one CSDL graph;
[Validation status](#validation-status) lists which results and derivatives
have been checked:

1. **Parameterize the CAD** with [lsdo_geo](https://github.com/LSDOlab/lsdo_geo):
   wing area and aspect ratio, stabilizer area and aspect ratio, and
   stabilizer pitch, with both struts kept attached.
2. **Deform the surface mesh** with GAMMA: the four intersections are
   recomputed on the new CAD and the remaining vertices follow.
3. **Solve the steady panel method** with
   [VortexAD](https://github.com/LSDOlab/VortexAD) on the deformed surface and
   report the lift and induced drag. This stage is optional and expensive.

Like the other examples, it is an ordinary Python script configured by editing
the settings at the top of the file, or by calling `main(...)` with keyword
arguments. Run it from the repository root:

```bash
python examples/cessna_208_geometry_to_panel.py
```

The default targets reproduce the baseline CAD. Change a multiplier, for
example `WING_ASPECT_RATIO = 9.543398722310108 * 0.7`, to move the design.
The STEP file and the surface mesh ship with GAMMA; see the
[C208 asset record](../examples/cessna_208/ASSETS.md) for their provenance and
license.

## Before you start

Stages 1 and 2 need the validated core stack from
[Installation](getting_started.md) plus `lsdo_geo` at the tested revision
(it imports `vedo`, which `requirements-ci.txt` pins):

```bash
python -m pip install --no-deps \
  "lsdo_geo @ git+https://github.com/LSDOlab/lsdo_geo.git@3e9cb8872bfb1a4116757de3f436d6ac3f5476c9"
```

Stage 3 also needs VortexAD at the revision GAMMA's adapter targets:

```bash
python -m pip install --no-deps \
  "VortexAD @ git+https://github.com/LSDOlab/VortexAD.git@c8027dfa137445ce7a1ba34bd3644d099111f770"
```

Stages 1 and 2 take a few minutes on a laptop; the first run also caches the
baseline projections in `c208_results/cache`. Stage 3 is off by default; see
[its cost](#cost) before turning it on.

## Step 1: parameterize the CAD

```python
WING_AREA_M2 = 25.985382406299443 * 1.0
WING_ASPECT_RATIO = 9.543398722310108 * 1.0
STAB_AREA_M2 = 6.507076017257924 * 1.0
STAB_ASPECT_RATIO = 6.000007139374481 * 1.0
STAB_ROTATION_DEG = 0.0  # Positive rotates the stabilizer nose up about +y.
```

The baseline numbers are measured from the shipped STEP file:

| Quantity | Wing | Stabilizer |
| --- | ---: | ---: |
| Full span | 15.748 m | 6.248 m |
| Reference area | 25.985 m² | 6.507 m² |
| Aspect ratio | 9.543 | 6.000 |

**Reference metrics.** Each wing and stabilizer side has three spanwise rows of
B-spline coefficients. The chord of a row is the 3D distance from its leading
edge (the minimum-x point) to its trailing edge. The reference area is the sum
of the two starboard trapezoids between the rows, doubled for the mirrored port
side; the span is the tip-to-tip leading-edge distance, and the aspect ratio is
span squared over area. This is a planform reference, not the curved CAD
surface area.

**Inner solve.** Each wing and stabilizer is embedded in an lsdo_geo free-form
deformation (FFD) block with two states: a uniform chordwise stretch and a
spanwise translation of the tip. Four equality constraints meet the area and
aspect-ratio targets. The port side is an exact mirror of starboard, and the
fuselage CAD does not change.

**Strut attachments.** Five further equalities keep each strut attached: its
wing end moves with the wing material in the same FFD block, and its fuselage
end stays at its baseline position on the fuselage. The strut changes length
and can shear as the wing end moves.

**Stabilizer pitch.** After the area and aspect-ratio solve, the stabilizer
rotates about +y through its root quarter chord and is then translated so its
root trailing edge stays at its baseline position.

The stage checks that the inner solve converged by residual, that all
attachment residuals are at round-off, and that every attachment marker lies
inside its host. The solved coefficients are saved to
`c208_geometry_coefficients.npz`, one `(n, 3)` array per component (`strut`,
`wing`, `fuselage`, `stab`).

## Step 2: deform the surface mesh

The solved coefficient arrays enter `GeometryModel.add_component` directly, as
CSDL variables, so the deformed mesh stays connected to the five targets in
the graph. The four intersections are declared with `GeometryModel.connect`:
strut–wing, strut–fuselage, wing–fuselage and stab–fuselage.

```python
WING_FREE_REGION = {"y": (0.0, 0.8, "abs")}
FUSELAGE_FREE_REGION = {"x": (0.1, 0.97, "extent")}
STAB_FREE_REGION = {"y": (0.0, 0.5, "abs")}
STRUT_VERTICES_FREE = True
NGON_REGULARIZATION_WEIGHT = 100.0
WING_ROOT_SEAM_TOLERANCE_M = 4e-3
```

**Free regions.** A free region lists the vertices that the graph propagation
solves for, as a normalized coordinate range of the component: `"abs"` uses
`|y|` divided by the component's semispan, `"extent"` uses `(x - x_min)`
divided by its length. Every other vertex follows the CAD at its baseline
parametric coordinates. Free vertices near the stabilizer root let the stab and
the tail cone share the seam motion when the stabilizer pitches.

**Seams.** Mesh vertices within a tolerance of both components are moved onto
the exact deformed intersection. The default tolerance is 2 mm. This mesh's
wing-root trailing edge sits up to 3.4 mm off the fuselage CAD, so the
wing–fuselage seam uses 4 mm, which selects exactly its 158 junction
vertices; the other seams keep the default.

**Mesh notes.** The C208 surface mesh is a full, mixed triangle/quad mesh, so
the example runs with `symmetry=False`. Three of its cells are inverted in the
source mesh. In 58 places the quad recombination merged two triangles across a
junction, so the junction runs along a quad diagonal rather than an edge.

With `VISUALIZE = True`, the example opens the updated CAD, then a linked
baseline/deformed view of every seam's vertices, with the folded quads in cyan
and any inverted cells in red:

```{image} images/c208_seams_stab_root.png
:alt: Stabilizer-root seam vertices on the baseline and deformed surface
```

At a strongly changed design (wing aspect ratio ×0.7, stabilizer aspect ratio
×0.8, stabilizer pitch −5°):

```{image} images/c208_surface_baseline_deformed.png
:alt: Baseline and deformed C208 surface mesh
```

| Check | Result |
| --- | ---: |
| Surface vertices; triangles / quads | 26,882; 7,486 / 23,141 |
| Polygon folds | 0 |
| Inverted cells, source mesh → final | 3 → 3 (none new) |
| Degenerate cells; projection failures | 0; 0 |
| Largest seam-to-deformed-CAD distance | 9.6×10⁻¹¹ m |
| Largest vertex displacement | 1.29 m |

These results hold for the design points tested here; they are not a certified
design envelope.

## Step 3: panel aerodynamics (optional)

```python
RUN_PANEL_ANALYSIS = False
MACH = 0.27
ALPHA_DEG = 2.0
DENSITY_KG_M3 = 0.9046
SPEED_OF_SOUND_M_S = 328.4
STRUT_WAKE = False
JUNCTION_WAKE_GAP_EDGES = 1
```

Set `RUN_PANEL_ANALYSIS = True` (or call `main(run_panel_analysis=True)`) to
append VortexAD's steady source–doublet panel method to the same graph through
`mm.build_panel_aerodynamics`. The flight condition is an illustrative cruise
at 10,000 ft in the standard atmosphere, not validated C208 performance data.
The reference area is the wing-area target and the reference chord is area
over span. The stage reports CL and the induced drag computed two ways: by
integrating surface pressure, and in the Trefftz plane behind the wake.

### Cost

The panel method is dense over all 30,627 panels. On a 16 GB laptop, the panel
stage peaked at about **48 GB of memory** (most of it in swap) and took 15–25
minutes. Mesh motion is evaluated inline as in steps 1 and 2; the panel graph
is then built with `recorder.inline = False` and run by
`csdl.experimental.JaxSimulator`. Evaluating the panel graph inline with
NumPy needs roughly 70 GB and was stopped by the operating system.

### Wake modeling

VortexAD's fixed wake leaves every trailing edge straight along +x, 30
reference chords long. Two consequences on this configuration shape the
defaults:

- **The struts shed no wake** (`STRUT_WAKE = False`). The strut trailing edge
  climbs from the fuselage to the wing, so its straight wake cuts through the
  stabilizer at |y| ≈ 2.3–2.6 m on the baseline geometry and pins the
  stabilizer pressure there at the solver's cutoff. Without a wake, the struts
  carry no circulation.
- **The trailing-edge panel next to the fuselage sheds no wake**
  (`JUNCTION_WAKE_GAP_EDGES = 1`) on the wing and the stabilizer. Otherwise the
  inboard edge of each wake runs straight back along the fuselage surface and
  scatters the fuselage pressure behind the root.

One strut–fuselage junction edge passes VortexAD's trailing-edge angle test and
is always excluded (`TRAILING_EDGE_EDGES_TO_IGNORE`). With the defaults, the
remaining wakes leave 510 trailing-edge edges on the wing and stabilizer.

### Results at the baseline design

```{image} images/c208_panel_baseline.png
:alt: Doublet strength and pressure coefficient on the baseline C208
```

| Wake settings | CL | CDi, surface pressure | CDi, Trefftz plane |
| --- | ---: | ---: | ---: |
| Strut wake on, no junction gap | 0.431 | 0.0187 | 0.0391 |
| Strut wake off, no junction gap | 0.474 | 0.0157 | 0.0188 |
| **Strut wake off, gap of one edge (default)** | **0.447** | **0.0129** | **0.0150** |

These results come from the pre-consolidation version of this example
(separate geometry, mesh and panel scripts) with the same settings. The
consolidated script reproduces that run's baseline mesh bitwise, excludes the
same trailing-edge edges and uses the same flight condition, so its panel
solve receives identical inputs; its own VortexAD run is still outstanding.

With the defaults, lift is 41.3 kN and the two induced-drag estimates agree
within about 17%. They imply a span efficiency `CL² / (π AR CDi)` of 0.44–0.52,
lower than expected for this wing. Some fuselage pressure scatter remains next
to the wing-root trailing edge, where the inboard wake edge still runs close to
the body. **Treat the induced drag as indicative**; the panel model, not the
mesh motion, limits it.

## Outputs

Everything is written to `c208_results/` (the `output_directory` argument
changes it):

| File | Content |
| --- | --- |
| `c208_report.json` | targets, geometry metrics and checks, mesh quality and seam checks, and the panel results when stage 3 ran |
| `c208_geometry_coefficients.npz` | the solved CAD coefficients per component |
| `c208_surface.npz` | baseline and deformed vertices, cells, and component and seam vertex IDs |
| `c208_panel_distributions.npz` / `.png` | per-panel doublet strength and pressure coefficient (stage 3) |

A failed seam check writes `c208_mesh.invalid.json` and raises instead of
returning a result.

`main()` returns the solved geometry graph, the `MeshMotionResult`, the panel
outputs (`None` without stage 3) and the report. The five targets are
registered as design variables on the `GeometryModel`, so the graph can be the
starting point for an optimization, but the derivatives through the complete
chain are not yet validated (see below).

## Validation status

Checked:

- **CAD stage** (`tests/test_cessna_208_geometry_parameterization.py`): the
  baseline is reproduced, targets and attachments are met at a changed
  design, and reverse-mode derivatives through the lsdo_geo inner solve match
  finite differences.
- **Mesh stage** (`tests/test_cessna_208_geometry_to_panel.py`): stages 1 and
  2 run end to end at a changed design, with no folds, no new inverted cells,
  no projection failures, and every seam on the deformed CAD.
- **Panel stage wiring** (same file): the example's panel stage is built and
  run through the JAX simulator with a stand-in solver, and its trailing-edge
  exclusions and panel settings are checked. GAMMA's VortexAD adapter
  reproduces a native VortexAD run exactly on a small test wing.

Outstanding:

- **A VortexAD run of the consolidated script.** The baseline results above
  come from the pre-consolidation run with identical inputs.
- **Derivatives through the complete chain.** The derivatives of CL and the
  induced drag with respect to the five targets pass through the CAD, mesh
  and panel stages, but have not been checked against finite differences.
  Do not rely on them for optimization until they are.

No real panel solve runs in continuous integration.
