---
orphan: true
---

# Cessna 208 example inputs

| File | Created by | Tools | Size | SHA-256 |
| --- | --- | --- | ---: | --- |
| `cessna_208.stp` | Anugrah Joshy | OpenVSP | 637,956 B | `c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9` |
| `cessna_208.msh` | Luca Scotzniovsky | OpenVSP and Gmsh (quad recombination) | 2,071,329 B | `321b493a18ba0016274c0797a8754613388f09ade58beef0014480a92c1cbeb8` |

**License.** Both files are licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
with the written consent of their authors. This license covers these two files only;
GAMMA's source code remains under the GNU LGPL v3.0 or later.

**Attribution.** "Cessna 208 STEP geometry" by Anugrah Joshy and "Cessna 208
mixed triangle/quad surface mesh" by Luca Scotzniovsky, CC BY 4.0.

**Source.** The STEP geometry was built in OpenVSP, starting from a
[Cessna 182 model in the OpenVSP Hangar](https://airshow.openvsp.org/vsp/Ur1FCPtGm8kHaqnVCi2i)
that is dedicated to the public domain (CC0). The surface mesh was generated
from that geometry with OpenVSP and Gmsh.

**Disclaimer.** These files are approximate research models for testing mesh
motion. They are not manufacturer data and are not suitable for engineering,
certification or airworthiness use. "Cessna" and "Caravan" are trademarks of
Textron Aviation Inc.; GAMMA is not affiliated with or endorsed by Textron
Aviation.

Both files live in `gamma_mdo/core/boundary_surface_movement/`, next to a shipped
copy of this notice (`CESSNA_208_ASSETS.md`). The STEP uses meters; its header
names only the exporter's default `outfile.stp`. The Gmsh 4.1 surface mesh has
26,882 vertices, 7,486 triangles and 23,141 quads. The source already has
three quads with a negative oriented corner; that corner test is distinct from
a whole-polygon normal flip.
