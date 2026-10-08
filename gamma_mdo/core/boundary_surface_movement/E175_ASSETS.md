# E175 example inputs: license notice

The E175 example geometry and meshes are by Marius Ruh and are licensed under
the Creative Commons Attribution 4.0 International license (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

Packaged here: `e175.stp`, `e175_quad_panel.msh`, `e175_r1_wall.msh` and
`e175_r1_wall.volume_map.npz`. The larger R1 and R5 volume-mesh files are
downloaded with `python -m gamma_mdo.assets download` and carry the same license.

- `e175.stp`: modified in OpenVSP from a public-domain (CC0) E175 model in the
  OpenVSP Hangar (https://airshow.openvsp.org/vsp/A45NbHrPeHqdh8x0qMzj).
- `e175_quad_panel.msh`: generated with Gmsh.
- R1 and R5 meshes: generated with Ansys Fluent and converted to Gmsh
  format, in meters.

This license covers these files only. GAMMA's source code is licensed under
the GNU Lesser General Public License v3.0 or later.

These files are approximate research models for testing mesh motion. They are
not manufacturer data and are not suitable for engineering, certification or
airworthiness use. "Embraer" and "E175" are trademarks of Embraer S.A.; GAMMA
is not affiliated with or endorsed by Embraer.
