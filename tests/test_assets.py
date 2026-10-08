"""Registry, download and verification tests for ``gamma_mdo.assets``."""

import gzip
import hashlib

import pytest

from gamma_mdo import assets


def _fake_asset(monkeypatch, tmp_path, payload, *, registered=None):
    """Publish ``payload`` as a one-file local release and register it."""
    release = tmp_path / "release"
    release.mkdir()
    with gzip.open(release / "fake.msh.gz", "wb") as stream:
        stream.write(payload)
    registered = payload if registered is None else registered
    fake = assets.Asset(
        "fake-volume", "fake.msh", len(registered),
        hashlib.sha256(registered).hexdigest(), False, "test asset",
    )
    monkeypatch.setitem(assets.ASSETS, fake.name, fake)
    monkeypatch.setenv("GAMMA_ASSET_URL", release.as_uri())
    monkeypatch.setenv("GAMMA_ASSET_DIR", str(tmp_path / "cache"))
    return fake


def test_packaged_assets_match_their_registered_size_and_hash():
    """Pin every file that ships inside the package."""
    for name, asset in assets.ASSETS.items():
        if not asset.packaged:
            continue
        path = assets.asset_path(name)
        assert path.is_file(), name
        assert path.stat().st_size == asset.size, name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset.sha256, name


def test_downloadable_assets_resolve_inside_the_cache(monkeypatch, tmp_path):
    """Keep large files out of the package directory."""
    monkeypatch.setenv("GAMMA_ASSET_DIR", str(tmp_path))
    for name, asset in assets.ASSETS.items():
        expected = tmp_path if not asset.packaged else assets.PACKAGE_DIRECTORY
        assert assets.asset_path(name) == expected / asset.filename
    for members in assets.GROUPS.values():
        assert all(not assets.ASSETS[name].packaged for name in members)


def test_require_explains_how_to_download_a_missing_asset(monkeypatch, tmp_path):
    """Missing downloads name the command that fetches them."""
    monkeypatch.setenv("GAMMA_ASSET_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="python -m gamma_mdo.assets download e175-r5-volume"):
        assets.require("e175-r5-volume")
    with pytest.raises(KeyError, match="Unknown GAMMA asset"):
        assets.asset_path("e175-r9-volume")


def test_download_decompresses_verifies_and_reuses(monkeypatch, tmp_path, capsys):
    """Fetch from a local release, then skip the verified copy."""
    payload = b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n" * 100
    _fake_asset(monkeypatch, tmp_path, payload)
    (path,) = assets.download("fake-volume")
    assert path == tmp_path / "cache" / "fake.msh"
    assert path.read_bytes() == payload
    assert assets.require("fake-volume") == path
    assets.download(["fake-volume"])
    assert "already present" in capsys.readouterr().out


def test_download_rejects_a_mismatched_file_and_leaves_nothing(monkeypatch, tmp_path):
    """A corrupted or substituted release file never reaches the cache."""
    _fake_asset(monkeypatch, tmp_path, b"tampered", registered=b"expected")
    with pytest.raises(ValueError, match="does not match"):
        assets.download("fake-volume", progress=False)
    assert list((tmp_path / "cache").iterdir()) == []


def test_groups_and_all_expand_to_downloadable_assets():
    """Expand names without duplicates and refuse packaged assets."""
    assert assets._expand(["e175-r5", "e175-r5-volume"]) == list(assets.GROUPS["e175-r5"])
    assert set(assets._expand("all")) == {
        name for name, asset in assets.ASSETS.items() if not asset.packaged
    }
    with pytest.raises(ValueError, match="ships with the package"):
        assets._expand("e175-geometry")


def test_command_line_lists_assets_and_requires_a_selection(monkeypatch, tmp_path, capsys):
    """List every asset; refuse a download command without names."""
    monkeypatch.setenv("GAMMA_ASSET_DIR", str(tmp_path))
    assert assets.main(["list"]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in assets.ASSETS)
    with pytest.raises(SystemExit):
        assets.main(["download"])
