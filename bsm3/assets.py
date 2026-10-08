"""Locate GAMMA's example geometry and meshes, and download the large ones.

Small assets ship inside the package. The large E175 volume meshes and the R5
wall set are attached to a GitHub release instead; download the ones you need
once, and they are cached and verified by SHA-256::

    python -m bsm3.assets list
    python -m bsm3.assets download e175-r1-volume
    python -m bsm3.assets download e175-r5
    python -m bsm3.assets download --all

Downloads go to ``~/.cache/gamma/assets`` unless ``GAMMA_ASSET_DIR`` is set.
``GAMMA_ASSET_URL`` overrides the release URL, for example to use a mirror.
See each asset's licence notice (``E175_ASSETS.md`` and ``CESSNA_208_ASSETS.md``
next to the packaged files).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIRECTORY = Path(__file__).resolve().parent / "core" / "boundary_surface_movement"
RELEASE_URL = "https://github.com/MariusLRuh/GAMMA-MDO/releases/download/assets-v1"


@dataclass(frozen=True)
class Asset:
    """One curated input file.

    Attributes
    ----------
    name
        Registry name used by :func:`asset_path` and the command line.
    filename
        File name, both in the package or cache and inside the release.
    size
        Size of the usable (uncompressed) file in bytes.
    sha256
        SHA-256 of the usable (uncompressed) file.
    packaged
        True when the file ships inside the package; False when it must be
        downloaded from the release as ``<filename>.gz``.
    description
        One-line description shown by ``python -m bsm3.assets list``.
    """

    name: str
    filename: str
    size: int
    sha256: str
    packaged: bool
    description: str


ASSETS = {asset.name: asset for asset in (
    Asset("e175-geometry", "e175.stp", 928_759,
          "3a55ddc748116d6c5ca0827a4df250548edb1dc99b643dbe25ec85b60d40bcb2",
          True, "E175 STEP geometry (no winglets), metres"),
    Asset("e175-quad-panel", "e175_quad_panel.msh", 1_300_938,
          "92feeeda05905a13d23a18c863e76b9596773beccb021148cc2d4e7016cd733c",
          True, "E175 quad-dominant surface panel mesh"),
    Asset("e175-r1-wall", "e175_r1_wall.msh", 1_986_453,
          "353b684372cd03648aeda20f38325c1313037ba1371d53cf59703ed70dd2aad9",
          True, "E175 R1 half-aircraft wall triangles"),
    Asset("e175-r1-wall-map", "e175_r1_wall.volume_map.npz", 570_913,
          "881bb6de7c788bc8c9322bcf285497ae5e5524a4d38ab139249754db123e7362",
          True, "E175 R1 wall-to-volume vertex map"),
    Asset("e175-r1-volume", "e175_r1_volume.msh", 54_793_992,
          "7562c040cd4432322a0594e56f9dc7f7de0e2b8259fa3272d3f8f82357bcc3a3",
          False, "E175 R1 tetrahedral Euler volume mesh (19 MB download)"),
    Asset("e175-r5-wall", "e175_r5_wall.msh", 5_676_508,
          "2e7f8bc42e3d7315c71e7a626a057157cf93839dc65c133f1af0d32e3769374f",
          False, "E175 R5 half-aircraft wall triangles (2 MB download)"),
    Asset("e175-r5-wall-map", "e175_r5_wall.volume_map.npz", 1_575_206,
          "7fd1dca9835096d133cb6cb64ec1655e3b5be9d3931e94c7f62df98f396598d3",
          False, "E175 R5 wall-to-volume vertex map (1.5 MB download)"),
    Asset("e175-r5-volume", "e175_r5_volume.msh", 131_125_009,
          "ba446902e0a4bf9a073b95520cc93bbc2395ae94ffcb91e311e3188571bc15a1",
          False, "E175 R5 tetrahedral Euler volume mesh (44 MB download)"),
    Asset("c208-geometry", "cessna_208.stp", 637_956,
          "c22fd7d08f369abf10297dc9ce53727d06c6b771779bd622014723c0831106b9",
          True, "Cessna 208 STEP geometry, metres"),
    Asset("c208-surface-mesh", "cessna_208.msh", 2_071_329,
          "321b493a18ba0016274c0797a8754613388f09ade58beef0014480a92c1cbeb8",
          True, "Cessna 208 mixed triangle/quad surface mesh"),
)}

GROUPS = {
    "e175-r1": ("e175-r1-volume",),
    "e175-r5": ("e175-r5-wall", "e175-r5-wall-map", "e175-r5-volume"),
}


def cache_directory() -> Path:
    """Return the download cache directory (``GAMMA_ASSET_DIR`` if set)."""
    configured = os.environ.get("GAMMA_ASSET_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "gamma" / "assets"


def asset_path(name: str) -> Path:
    """Return where an asset lives, without checking that it exists.

    Packaged assets resolve inside the installed package; downloadable ones
    resolve inside :func:`cache_directory`. Use :func:`require` when the
    file must already be present.
    """
    asset = _lookup(name)
    root = PACKAGE_DIRECTORY if asset.packaged else cache_directory()
    return root / asset.filename


def require(name: str) -> Path:
    """Return an asset's path, or explain how to get it if it is missing.

    Raises
    ------
    FileNotFoundError
        If the file is absent. For a downloadable asset the message gives the
        download command.
    """
    path = asset_path(name)
    if path.is_file():
        return path
    asset = ASSETS[name]
    if asset.packaged:
        raise FileNotFoundError(f"Packaged GAMMA asset {name!r} is missing: {path}")
    raise FileNotFoundError(
        f"GAMMA asset {name!r} ({asset.description}) is not downloaded. Run:\n"
        f"    python -m bsm3.assets download {name}"
    )


def download(names, *, force: bool = False, progress: bool = True) -> list[Path]:
    """Download, decompress and verify downloadable assets.

    Parameters
    ----------
    names
        Asset or group names (see :data:`GROUPS`), or ``"all"``.
    force
        Download again even when a verified copy exists.
    progress
        Print one line per file.

    Returns
    -------
    list of Path
        Paths of the requested downloadable assets.

    Raises
    ------
    ValueError
        If a downloaded file's size or SHA-256 does not match the registry; the
        partial file is removed.
    """
    selected = _expand(names)
    base_url = os.environ.get("GAMMA_ASSET_URL", RELEASE_URL).rstrip("/")
    target_directory = cache_directory()
    target_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in selected:
        asset = ASSETS[name]
        target = target_directory / asset.filename
        if not force and target.is_file() and _matches(target, asset):
            if progress:
                print(f"{name}: already present at {target}")
            paths.append(target)
            continue
        url = f"{base_url}/{asset.filename}.gz"
        if progress:
            print(f"{name}: downloading {url}", flush=True)
        partial = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, \
                    gzip.GzipFile(fileobj=response) as unpacked, \
                    open(partial, "wb") as stream:
                shutil.copyfileobj(unpacked, stream, 1 << 20)
            if not _matches(partial, asset):
                raise ValueError(
                    f"Downloaded {asset.filename} does not match its registered "
                    "size and SHA-256."
                )
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        if progress:
            print(f"{name}: verified {target}")
        paths.append(target)
    return paths


def _lookup(name):
    try:
        return ASSETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown GAMMA asset {name!r}; choose one of {sorted(ASSETS)}."
        ) from None


def _expand(names):
    if isinstance(names, str):
        names = [names]
    selected = []
    for name in names:
        if name == "all":
            members = [n for n, asset in ASSETS.items() if not asset.packaged]
        elif name in GROUPS:
            members = list(GROUPS[name])
        else:
            if _lookup(name).packaged:
                raise ValueError(f"{name!r} ships with the package; nothing to download.")
            members = [name]
        selected.extend(member for member in members if member not in selected)
    return selected


def _matches(path, asset):
    if path.stat().st_size != asset.size:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == asset.sha256


def main(argv=None) -> int:
    """Run the ``python -m bsm3.assets`` command line."""
    parser = argparse.ArgumentParser(
        prog="python -m bsm3.assets",
        description="List or download GAMMA example assets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show every asset and whether it is present")
    fetch = commands.add_parser("download", help="download assets or groups")
    fetch.add_argument("names", nargs="*", help=f"assets or groups {sorted(GROUPS)}")
    fetch.add_argument("--all", action="store_true", help="download every large asset")
    fetch.add_argument("--force", action="store_true", help="download again")
    arguments = parser.parse_args(argv)

    if arguments.command == "list":
        for name, asset in ASSETS.items():
            where = "packaged" if asset.packaged else "download"
            present = "present" if asset_path(name).is_file() else "missing"
            print(f"{name:<18s} {where:<9s} {present:<8s} {asset.description}")
        print(f"\nGroups: {', '.join(f'{g} ({len(m)})' for g, m in GROUPS.items())}")
        print(f"Download cache: {cache_directory()}")
        return 0
    names = ["all"] if arguments.all else arguments.names
    if not names:
        parser.error("name at least one asset or group, or pass --all")
    download(names, force=arguments.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
