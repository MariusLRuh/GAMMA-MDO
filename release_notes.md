# GAMMA 0.3.0a1 (alpha)

GAMMA is research software under active development, released as an alpha
(pre-release) version; its interfaces and default settings may still change
between releases. It is provided as is, without warranty.

## New features

- Cessna 208 strut-attachment example on a mixed triangle/quad mesh, with all
  four component seams kept on their intersection curves and an optional
  finite-difference check of the mean-node sensitivities
  (`CHECK_DERIVATIVES`).
- `gamma_mdo.assets`: a registry of the example geometry and meshes with pinned
  SHA-256 checksums, and `python -m gamma_mdo.assets download` for the large
  E175 R1 and R5 meshes attached to the `assets-v1` GitHub release.

## Improvements

- The package is about 3.7 MB; large meshes download on demand.
- The example assets are licensed under CC BY 4.0, with attribution and
  trademark disclaimers shipped next to the files.
- American English throughout the documentation.

## Backwards-incompatible changes

- The import package is renamed from `bsm3` to `gamma_mdo`.
- E175 asset files are renamed: `embraer_175_no_winglets.stp` is now
  `e175.stp`, `embraer_175_panel_quad_dominant_high_quality.msh` is now
  `e175_quad_panel.msh`, and the R1 wall files are `e175_r1_wall.msh` and
  `e175_r1_wall.volume_map.npz`.
- The OpenVSP Euler volume mesh, `E175_w_fairing.*`, `wall_surface.npz`, the
  `wing_fuse_test*` files and the E175 DAFoam driver are no longer distributed.
  The DAFoam coupling library remains.

## Upgrade process

- Replace `import bsm3...` and `from bsm3...` with `gamma_mdo`.
- Use `gamma_mdo.assets.asset_path(name)` instead of hard-coded asset paths,
  and download the large meshes with `python -m gamma_mdo.assets download`.
