# API reference

`gamma_mdo.mesh_motion` is the intended public namespace and is deliberately
compact. Solver-assembly types, component records, projection callbacks, and
polygon helpers stay internal, because a caller never needs them to deform a
surface mesh.

```python
import gamma_mdo.mesh_motion as mm
```

Everything below is re-exported from that module.

## Public export inventory

This list is the complete supported namespace. A structural test compares it
exactly with `gamma_mdo.mesh_motion.__all__`; semantic descriptions and signatures
are still reviewed against the implementation.

<!-- BEGIN GAMMA PUBLIC EXPORTS -->
- `mm.DerivativeCheck`
- `mm.DistanceWeighting`
- `mm.DistortionPenalty`
- `mm.FuelBurnParameters`
- `mm.GeometryModel`
- `mm.InputFiles`
- `mm.MeshMotion`
- `mm.MeshMotionResult`
- `mm.PanelAerodynamicOutputs`
- `mm.PanelCondition`
- `mm.PolygonRegularization`
- `mm.QualityChecks`
- `mm.SurfaceMotion`
- `mm.SurfaceProjectionStatus`
- `mm.SurfaceVertexClassification`
- `mm.Visualization`
- `mm.VolumeMotion`
- `mm.build_panel_aerodynamics`
- `mm.compute_fuel_burn`
- `mm.run`
- `mm.run_fd_sweep`
- `mm.select_fd_objective`
<!-- END GAMMA PUBLIC EXPORTS -->

## `run`

```python
mm.run(
    *,
    inputs: InputFiles,
    geometry: GeometryModel,
    motion: MeshMotion,
    recorder: csdl.Recorder,
    aerodynamic_analysis=None,
    aerodynamic_volume_method="elasticity",
) -> MeshMotionResult
```

Evaluates the differentiable mesh-motion model. All arguments are
keyword-only.

`recorder` is the active CSDL recorder, **owned by the caller**. `run` never
starts or stops it, so mesh motion composes inside a larger graph. It is
retained on the result for convenience. The recorder must have inline
execution enabled: the pipeline consumes forward values and cached projection
state while constructing the result, and raises `RuntimeError` if that state
is unavailable.

`aerodynamic_analysis` is an optional downstream builder that receives the
selected volume coordinates; `aerodynamic_volume_method` selects which volume
method is handed to it.

## `GeometryModel`

A declaration and binding envelope for the components that move. It does not
constrain how coefficients were parameterized, but any design variables they
depend on must belong to the recorder passed to `run`.

| Method | Purpose |
| --- | --- |
| `add_component(*, name, search_name, deformed_coefficients, free_region=None, projection_name=None, projection_mode="all")` | **The general entry point.** Bind externally produced deformed coefficients. See [External parameterization](external_parameterization.md). |
| `add_lifting_surface(*, name, search_name, pivot_intersection, translation_x=0.0, rotation_y_degrees=0.0, area=None, aspect_ratio=None, reference_area=None, reference_aspect_ratio=None, free_span_fraction=0.3, ...)` | Optional convenience for a wing- or tail-like surface. `free_span_fraction` selects the graph-free root region as a fraction of semispan and must lie in `(0, 1]`; farther-outboard vertices are parametrically reevaluated. |
| `add_body(*, name, search_name, diameter_scale=1.0, free_axial_fraction=(0.05, 0.97), projection_name=None)` | Optional convenience for a fuselage-like body. The interval is the graph-free middle; nose and tail vertices outside it are parametrically reevaluated. |
| `connect(*, name, driving_component, query_component, search_direction="u", solver_name=None)` | Name the closed intersection curve between two components. |
| `design_variable(name, value, *, lower=None, upper=None, scaler=None)` | Register a design variable on the active recorder and return the CSDL variable. |
| `design_variables` | Mapping of registered design-variable name to CSDL variable. |
| `validate()` | Check the declaration for consistency. |

`add_lifting_surface` and `add_body` are conveniences oriented at the E175
example, not the generic mechanism.

## Input files

`InputFiles(geometry_file, surface_mesh_file, volume_mesh_file=None,
volume_wall_map_file=None, cache_directory=None)`

All are local paths you supply. `volume_mesh_file` and `volume_wall_map_file`
are only needed for the optional volume chain.

## Settings

`MeshMotion` is the top-level settings object:

| Field | Meaning |
| --- | --- |
| `surface` | `SurfaceMotion` — the graph-Laplacian solve |
| `volume` | `VolumeMotion` — optional volume propagation |
| `quality` | `QualityChecks` — which diagnostics run |
| `visualization` | `Visualization` — optional plotting |
| `derivative_check` | `DerivativeCheck` — settings consumed by the caller through the public FD helpers |
| `symmetry` | Treat the surface as a half model |
| `symmetry_plane_tolerance` | Tolerance for detecting plane membership |
| `setup_projection_resolution` | Setup-time projection sampling |
| `projection_warm_start_resolution` | Warm-start sampling for reprojection |
| `rebuild_setup_cache` | Ignore and rewrite the cached setup |
| `query_seam_reference` | Use the reference seam for query components |
| `lifting_surface_patch_mode` | Patch selection for lifting surfaces |
| `diagnostic_dump` | Optional diagnostic output path |

The nested types:

- `SurfaceMotion(load_steps, stiffening_exponent, quad_diagonal_weight,
  quad_bracing_mode, distance_weighting, distortion_penalty,
  polygon_regularization)`
- `DistanceWeighting(enabled, beta, length_scale, cap, decay, power,
  seed_intersections)`
- `PolygonRegularization(weight)`
- `DistortionPenalty(weight, mode, area, deviatoric, shear, rotation, normal)`
- `VolumeMotion(mode, load_mode, synchronized_load_steps,
  graph_stiffening_exponent, elasticity_poisson_ratio,
  elasticity_stiffening_exponent, output_directory, write_meshes)`
- `QualityChecks(surface, volume, gmsh_volume_metrics,
  fail_on_surface_inversion, fail_on_volume_inversion)`
- `Visualization(enabled, opacity)`
- `DerivativeCheck(enabled, objective, step_sizes)`

`mm.run` does not launch a finite-difference sweep as a side effect. When
`derivative_check.enabled` is true, it uses `mm.select_fd_objective` to
register the configured scalar on the active recorder. Stop the recorder after
graph construction and then run the public sweep:

```python
result = mm.run(inputs=inputs, geometry=geometry, motion=motion, recorder=recorder)
recorder.stop()
if motion.derivative_check.enabled:
    errors = mm.run_fd_sweep(recorder, motion.derivative_check.step_sizes)
```

Call `mm.select_fd_objective` directly when constructing a custom workflow
that does not use the `MeshMotion.derivative_check` settings. The convenience
path calls `set_as_objective()` whenever `derivative_check.enabled` is true;
inside a larger optimization graph this replaces any objective already
registered with the recorder, so leave the debug flag disabled and manage the
FD objective explicitly in that case.

When `diagnostic_dump` is set, the NPZ uses zero-based complete-input-mesh
indexing. For each declared intersection, `{name}_ids` and
`{name}_vertices` have equal row counts and are aligned one-to-one in global
vertex order; on a symmetric solve, both retained and mirror-expanded rows are
included.

## `MeshMotionResult`

Returned by `run`. Differentiable outputs plus forward diagnostics.

| Attribute | Meaning |
| --- | --- |
| `surface_coordinates` | Final reprojected surface, a CSDL variable |
| `preprojected_surface_coordinates` | Surface before reprojection |
| `initial_surface_coordinates` | Baseline surface |
| `volume_coordinates` | Deformed volume coordinates keyed by method |
| `aerodynamic_outputs` | Optional downstream results |
| `surface_inversion_report` | Inversion diagnostics of the final surface |
| `initial_inversion_report` | Same metric on the untouched input mesh |
| `preprojection_inversion_report` | Same metric before reprojection |
| `surface_quality_report` | Aggregate surface-quality metrics |
| `volume_quality_summary` | Optional volume metrics |
| `surface_fold_count` | Polygons whose area-weighted normal flipped |
| `surface_cell_count` | Cells in the surface |
| `surface_ngon_mode_count` | N-gon hourglass modes present |
| `surface_vertex_classification` | Global IDs for deformation, closest-projection, parametric, graph-free, graph-prescribed, symmetry-plane, component, and exact-intersection roles |
| `surface_projection_status` | Closest-point projection IDs and the non-converged subset |
| `elapsed_seconds` | Wall-clock time of the solve |
| `surface_mesh`, `volume_mesh` | Loaded mesh objects |
| `input_files`, `geometry`, `recorder` | The inputs that produced this result |

All classification and status arrays use zero-based IDs in the complete input
surface mesh, including reconstructed mirror-side vertices. The final surface
is composite: closest-point reprojection is used for graph-moved non-seam
vertices, fixed-parametric reevaluation is used outside the deformation set,
and exact bracketed component-intersection coordinates are retained without a
second projection.

The three inversion reports use the same metric, so any two may be compared
directly. `print_summary()` reports vertex, cell, and n-gon-mode counts;
elapsed time; fold and inversion counts; and, when available,
degenerate-element and minimum-scaled-Jacobian quality fields.

```{note}
Points whose reprojection did not converge are still returned. The reports are
the evidence of solve quality; they are diagnostics, not guarantees.
```

## Panel aerodynamics and fuel burn

`build_panel_aerodynamics(result, condition)` appends the optional pinned
VortexAD steady panel method to the caller's active CSDL graph. It consumes
only the public `MeshMotionResult` surface and returns
`PanelAerodynamicOutputs` with `lift_coefficient`,
`induced_drag_coefficient`, `lift_newton`, and `induced_drag_newton`. The same
variables are registered in `result.aerodynamic_outputs` as `CL`, `CDi`, `L`,
and `Di`, so the existing FD selector can address them.

`PanelCondition(reference_area_m2, reference_chord_m, ...)` holds the
freestream, reference geometry, trailing-edge threshold, excluded root edges,
and projection-convergence policy. Connectivity and trailing edges are derived
from the undeformed triangle/quad mesh; the differentiable final coordinates
are passed to the solve. VortexAD is imported only when the builder is called.

`compute_fuel_burn(lift_coefficient, drag_coefficient, parameters)` evaluates
the classical Breguet jet range relation as a pure CSDL graph.
`FuelBurnParameters` requires the mission range, thrust-specific fuel
consumption, cruise speed, and initial aircraft weight, with units documented
on every field. The drag input is total aircraft drag; callers must add any
parasite or other contributions not supplied by the panel method.
