# C208 geometry parameterization: design and validation

## Scope and execution

This milestone changes CAD coefficients only. Run from the repository root with:

```bash
/Users/mariusruh/miniconda3/envs/bsm3_py312_main/bin/python -m examples.cessna_208_geometry_parameterization
```

Edit the five target constants near the top of `examples/cessna_208_geometry_parameterization.py`. The script writes a JSON validation report and a four-array coefficient NPZ under `c208_geometry_results/`, then opens the updated CAD in `lsdo_geo.Geometry.plot()`. No separate visualization file is needed. Set `PLOT_GEOMETRY = False` for a noninteractive run. The baseline targets reproduce the original CAD; the focused changed-design test uses wing area ×1.02, wing aspect ratio ×1.03, stabilizer area ×0.98, stabilizer aspect ratio ×1.02, and angle +3°. The editable targets in your working tree may differ from those test values.

The native plot displays the updated CAD functions. The script neither imports the CFD mesh nor invokes mesh motion or an intersection solver. The next milestone can consume the coefficient arrays, but no mesh quality claim follows from this geometry result.

The source checkout `/Users/mariusruh/Documents/Research/nasa_uli/mesh_movement/packages/GAMMA-c208-example` remains on `release/asset-curation` at `8dcb718` with its pre-existing edit to `examples/cessna_208_strut_attachment_deformation.py` untouched. This implementation is in a separate persistent worktree, `/Users/mariusruh/Documents/Codex/GAMMA-c208-geometry-parameterization`, on `example/c208-geometry-parameterization`. Nothing was pushed or merged. No projection or mesh-motion production code was changed.

## Dependencies and precedent

The C208 strut-motion example and `cessna_208_example.py` helper were read first. The parameterization pattern follows `bsm3/core/boundary_surface_movement/embraer_175_geom_parameterization.py` from BSM3 `dev` at `7b52efd`. The checked-out `lsdo_geo` source is commit `3e9cb88` on `marius_dev`. Initially, `lsdo_geo` was absent from `bsm3_py312_main`; after user authorization, a wheel built from a copy of that checkout was installed into that environment. Import then exposed an undeclared `vedo` dependency, so `vedo==2025.5.4` was installed. Runtime import and signatures for `construct_ffd_block_around_entities`, `VolumeSectionalParameterization`, `GeometricVariables`, `ParameterizationSolver`, and `FFDBlock.evaluate_ffd` were checked before implementation. The original `lsdo_geo` checkout was not modified.

The installed `lsdo_geo` version is `0.1.0`. Its `ParameterizationSolver` calls a Newton inner solve and prints its convergence count; it does not expose a stable iteration-count field. This example gates convergence by all four planform and five strut-attachment equality residuals.

## CAD reference metrics

Input: shipped `cessna_208.stp`, SHA-256 `c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9`. Distances are metres. Each wing and stabilizer side has three spanwise B-spline coefficient rows, at span parameters `u=0, 0.5, 1`. At each row, the trailing edge is at closed-curve parameter `v=0`, and the leading edge is the minimum-x point, found from the baseline curve's derivative roots. These baseline `(u,v)` locations are frozen for differentiable metrics on changed geometry. A chord is the **3D Euclidean** distance from its leading to trailing edge. The full reference area is the sum of the two starboard trapezoids, doubled for the exactly mirrored port side: `A = Σ_i (c_i+c_{i+1})(y_{i+1}-y_i)`. Full span is the port-to-starboard leading-edge-tip separation `b = 2(y_2-y_0)`; aspect ratio is `b²/A`. This is a planform reference metric, not integrated curved CAD surface area. The short final segment is the tip closure and is retained in the metric.

| Quantity | Baseline |
| --- | ---: |
| Wing root, middle, tip chords | 2.082983963, 1.217266925, 1.216456800 m |
| Wing full span | 15.747662216 m |
| Wing reference area | 25.985382406 m² |
| Wing aspect ratio | 9.543398722 |
| Stabilizer root, middle, tip chords | 1.320798231, 0.761999291, 0.762000000 m |
| Stabilizer full span | 6.248400000 m |
| Stabilizer reference area | 6.507076017 m² |
| Stabilizer aspect ratio | 6.000007139 |
| Stabilizer root quarter-chord | (9.33017036, 0, 0.45452811) m |
| Stabilizer root trailing edge | (10.32068684, 0, 0.46728874) m |
| Starboard strut fuselage-end section centroid | (3.42698553, 0.46103121, -0.47492385) m |
| Starboard strut wing-end section centroid | (3.51175621, 3.47287061, 0.99575801) m |
| Starboard strut centroid-to-centroid length | 3.352800000 m |

## Constraint construction

Each starboard wing and stabilizer patch is embedded in a two-section `lsdo_geo` FFD. A uniform chordwise sectional stretch and tip-only spanwise translation are the two inner-solver states per host component. Four equalities enforce the requested areas and aspect ratios. The port coefficient patch is an exact reversed-circumference mirror of starboard, so the solver has no asymmetric state. The fuselage coefficients are constant.

Three separated interior markers are defined in each strut end section as `0.8 ×` the section's 64-point perimeter mean plus `0.2 ×` a perimeter sample at curve parameters 0.10, 0.35, and 0.70. Their baseline closest host projections are smooth interior points: wing patch 0 near `(u,v)≈(0.221,0.23)` and fuselage patch 6 near `(0.512,0.81)`. The full baseline FFD volume coordinates of the wing markers are frozen and reported. The wing-side strut row has its **own two sectional states** in the same wing FFD as the host. Two independent solver equalities match the strut's first marker x and second marker y to their host-material FFD targets. Those two equalities determine the two strut sectional states; because the block and embedded coordinates are shared, all three strut markers then follow the host exactly. The fuselage-side row has a separate XYZ translation state. A three-component solver equality holds its first marker at the baseline fuselage-frame position, fixing the entire row and all three markers. These five attachment equalities are added to `GeometricVariables` before `solver.evaluate`, alongside the four planform equalities. The report records all nine inner-solver residuals and independently validates every marker. The two-row ruled strut changes length and can shear as its wing end moves. At the tested design, centroid length rises from 3.3528 to 3.4310 m. The port strut is mirrored exactly.

A scalar signed distance does not locate an attachment tangentially: two degrees of surface motion remain. The inner solver therefore uses host-material FFD anchor coordinates; signed closest distance is an independent embedding check. **No SDF call is part of the parameterization's CSDL forward solve or inner optimization.** FFD construction projects points into its volume once at setup; this is not a host SDF. `validate_c208_geometry` calls `FunctionSetProjectionModel(..., sdf=True)` only after the solve, on numerical coefficient/marker values. The wing markers' baseline signed distances are approximately −0.11024, −0.11118, and −0.12209 m; fuselage markers are approximately −0.03506, −0.03861, and −0.05094 m. At the smooth changed design, every marker remains inside its host, every closest-point solve converges, and patch/candidate branches remain unchanged. Wing signed distances drift by at most **2.38×10⁻⁶ m** because an affine chord/span deformation changes Euclidean distance; they are deliberately not forced to a scalar constant. The fixed fuselage signed distances do not change. The report records baseline/final signed distances, UV, patch IDs, candidate kinds, convergence, and ambiguity for each host.

For the stabilizer, after the area/aspect-ratio solve, let `q = LE_root + 0.25(TE_root−LE_root)`. Rotate all coefficients about the +y axis through `q` by the angle input, then translate all coefficients by `TE_baseline − TE_rotated`. This holds the root trailing edge fixed even when the root chord has changed. The final translated quarter-chord axis moves with the stabilizer. A rigid rotation about a quarter-chord axis fixed in the fuselage frame cannot simultaneously keep the trailing edge fixed at nonzero angle without further shape change. The geometry-only test verifies the actual final trailing-edge coefficient and the measured root-chord rotation.

## Validation and coefficient handoff

At a smooth changed design (wing area ×1.02, wing AR ×1.03, stabilizer area ×0.98, stabilizer AR ×1.02, rotation +3°), all four target residuals are below `4×10⁻¹⁴`; attachment residuals are at or below `1.8×10⁻¹⁵ m`, root trailing-edge error is zero to output precision, and exact coefficient symmetry holds. CSDL reverse-mode derivatives/adjoints through the inner solve of representative wing/stabilizer coefficients and wing-end attachment coordinates match centered finite differences at `h=1×10⁻⁴` and `3×10⁻⁵` for the relevant controls, with test tolerances `2×10⁻⁷` absolute and `2×10⁻⁵` relative. The fixed root trailing-edge derivative with respect to angle is zero. These checks cover one smooth interior design, not a guaranteed larger design rectangle. The native plotting assembly is checked against all seven solved CAD patches.

`stack_component_coefficients` uses ascending patch IDs and C-order flattening of each patch to `(N,3)`. The NPZ keys are `strut`, `wing`, `fuselage`, and `stab`, matching the existing C208 component registration order:

| NPZ key | Patch IDs in order | Patch shapes | Stacked shape |
| --- | --- | --- | --- |
| `strut` | 4, 5 | each `(2,307,3)` | `(1228,3)` |
| `wing` | 0, 1 | each `(3,385,3)` | `(2310,3)` |
| `fuselage` | 6 | `(34,97,3)` | `(3298,3)` |
| `stab` | 2, 3 | each `(3,235,3)` | `(1410,3)` |

The fuselage output equals its baseline CAD coefficients. Tests run only CAD geometry and projection diagnostics; they do not execute mesh deformation, intersection movement, or mesh-quality validation. The two focused parameterization tests pass. `ruff check` passes on the new files. The STEP importer and installed `lsdo_geo` sectional code emit existing deprecation warnings; these do not affect the reported geometry checks.
