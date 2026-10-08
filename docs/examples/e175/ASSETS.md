---
orphan: true
---

# E175 example inputs

All files below were created by Marius Ruh and are licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This license covers these files only; GAMMA's source code remains under the
GNU LGPL v3.0 or later.

**Attribution.** "E175 example geometry and meshes" by Marius Ruh, CC BY 4.0.

| Asset name | File | Tools | Size | Shipped |
| --- | --- | --- | ---: | --- |
| `e175-geometry` | `e175.stp` | OpenVSP | 0.9 MB | in the package |
| `e175-quad-panel` | `e175_quad_panel.msh` | Gmsh (OpenCASCADE, per-face quad recombination) | 1.3 MB | in the package |
| `e175-r1-wall` | `e175_r1_wall.msh` | Ansys Fluent | 2.0 MB | in the package |
| `e175-r1-wall-map` | `e175_r1_wall.volume_map.npz` | GAMMA | 0.6 MB | in the package |
| `e175-r1-volume` | `e175_r1_volume.msh` | Ansys Fluent | 55 MB (19 MB download) | download |
| `e175-r5-wall` | `e175_r5_wall.msh` | Ansys Fluent | 5.7 MB | download |
| `e175-r5-wall-map` | `e175_r5_wall.volume_map.npz` | GAMMA | 1.6 MB | download |
| `e175-r5-volume` | `e175_r5_volume.msh` | Ansys Fluent | 131 MB (44 MB download) | download |

SHA-256 checksums for every file are pinned in `gamma_mdo/assets.py`, which also
downloads and verifies the large files:

```bash
python -m gamma_mdo.assets list
python -m gamma_mdo.assets download e175-r1-volume   # or e175-r5, or --all
```

**Source.** The STEP geometry was modified in OpenVSP from the
[E175 model in the OpenVSP Hangar](https://airshow.openvsp.org/vsp/A45NbHrPeHqdh8x0qMzj),
which is dedicated to the public domain (CC0); it has no winglets. All
coordinates are in meters.

**R1 and R5 meshes.** Two refinement levels of a tetrahedral Euler volume mesh
of the half aircraft inside a 400 m half-sphere, generated in Ansys Fluent
Meshing in millimeters and converted to Gmsh 2.2 ASCII in meters by GAMMA.

| Level | Nodes | Tetrahedra | Wall triangles |
| --- | ---: | ---: | ---: |
| R1 | 170,542 | 955,889 | 32,522 |
| R5 | 392,490 | 2,198,811 | 90,387 |

Each `*_wall.msh` is the aircraft wall extracted from its volume mesh, and
each `*.volume_map.npz` maps every wall vertex to its volume-mesh node.

**Disclaimer.** These files are approximate research models for testing mesh
motion. They are not manufacturer data and are not suitable for engineering,
certification or airworthiness use. "Embraer" and "E175" are trademarks of
Embraer S.A.; GAMMA is not affiliated with or endorsed by Embraer.
