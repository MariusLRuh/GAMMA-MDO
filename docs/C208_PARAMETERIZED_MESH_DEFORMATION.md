# C208 parameterized CAD to CFD surface mesh

Run from the persistent worktree with the installed `bsm3_py312_main` environment:

```bash
/Users/mariusruh/miniconda3/envs/bsm3_py312_main/bin/python -m examples.cessna_208_parameterized_mesh_deformation
```

Edit the five targets in `examples/cessna_208_geometry_parameterization.py`; the mesh example reads those values at run time. It validates the solved CAD, then passes its differentiable `strut`, `wing`, `fuselage`, and `stab` coefficient arrays directly to `GeometryModel.add_component`. The arrays retain the geometry milestone's ascending-patch-ID/C-order flattening convention. The pipeline recomputes `strut_wing`, `strut_fuselage`, `wing_fuselage`, and `stab_fuselage` intersections, propagates the surface mesh, and reprojects onto the deformed CAD. The same C208 `.stp` and `.msh` assets and hash checks are used as in the strut-only example. No production projection or mesh-motion code changed.

The C208 wall is a full, asymmetric, mixed mesh, so the run keeps `symmetry=False`. It uses one surface load step, the strut-only example's free regions and wing leading-edge hold, graph stiffening exponent 1.5, and polygon regularization weight 3. The built-in mesh viewer opens when `VISUALIZE = True`; pass `visualize=False` for a batch run. The output directory is `c208_parameterized_mesh_results/` by default. It contains a JSON report and a diagnostic NPZ with initial/final vertices, triangles, quads, component IDs, and seam IDs.

The example validates final seams against the **deformed host coefficients**, not the baseline host CAD. The geometric parameterization uses no SDF in its inner solve; the mesh pipeline does use signed-distance/projection operations for its moving intersections and final surface placement. The example's final seam SDF checks are numerical diagnostics after mesh motion. A failed seam check writes `c208_parameterized_mesh.invalid.json` and raises.

| Diagnostic | Baseline geometry | Smooth changed design |
| --- | ---: | ---: |
| Surface vertices | 26,882 | 26,882 |
| Triangles / quads | 7,486 / 23,141 | 7,486 / 23,141 |
| Polygon folds | 0 | 0 |
| Inverted elements, input → final | 3 → 3 | 3 → 3 |
| New inverted elements | 0 | 0 |
| Degenerate elements | 0 | 0 |
| Projection failures | 0 | 0 |
| Largest final seam-to-host SDF | 8.99×10⁻¹¹ m | 9.12×10⁻¹¹ m |
| Maximum mesh-coordinate change | 0.00747 m | 0.19683 m |

The changed design is wing area ×1.02, wing aspect ratio ×1.03, stabilizer area ×0.98, stabilizer aspect ratio ×1.02, and stabilizer rotation +3°. The baseline mesh is not an exact nodewise identity because the existing projection pipeline moves some nodes by up to 7.5 mm even when CAD coefficients are unchanged. The three final inverted cells are the same three present in the input; the run adds none. These results establish the behavior of these two design points, not a certified design envelope or a repaired source mesh.

At the changed design, reverse-mode derivatives of a prescribed wing mesh vertex's x-coordinate with respect to wing area and a prescribed stabilizer vertex's z-coordinate with respect to stabilizer angle match centered finite differences at `h=10⁻⁴`: relative errors `8.1×10⁻¹¹` and `1.1×10⁻¹⁰`, respectively. These smooth points verify differentiation through both the lsdo_geo coefficient graph and the GAMMA surface-coordinate graph. Projection branches elsewhere on the mesh may be nonsmooth, so this does not establish a global mesh-gradient tolerance.

This milestone deforms the surface mesh only. It does not move a volume mesh or run CFD.
