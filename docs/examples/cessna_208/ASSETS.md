---
orphan: true
---

# Cessna 208 example inputs

The project owner approved including the exact Cessna 208 STEP and mixed
triangle/quad surface mesh below in GAMMA. The files were supplied for this
example; their original author and distribution terms are not recorded in
the files. The STEP header names only `outfile.stp`. We therefore make no
further attribution claim.

| File | Size | SHA-256 |
| --- | ---: | --- |
| `cessna208_no_elevator_3.stp` | 637,956 B | `c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9` |
| `cessna208_3_recombine_new.msh` | 2,071,329 B | `321b493a18ba0016274c0797a8754613388f09ade58beef0014480a92c1cbeb8` |

Both live in `bsm3/core/boundary_surface_movement/`. The STEP uses metres.
The Gmsh 4.1 surface mesh has 26,882 vertices, 7,486 triangles, and 23,141
quads. The source already has three quads with a negative oriented corner;
that corner test is distinct from a whole-polygon normal flip.
