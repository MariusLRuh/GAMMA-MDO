"""Sphinx configuration for the GAMMA documentation site.

The build is deliberately hermetic: it renders hand-written Markdown only. It
does not import :mod:`gamma_mdo`, execute the E175 example, or reach for DAFoam,
OpenFOAM, VortexAD, mpi4py, or any untracked asset. That keeps the site
buildable from ``docs/requirements.txt`` alone.
"""

from __future__ import annotations

import re
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "GAMMA"
author = "GAMMA contributors"
copyright = "2026, GAMMA contributors"


def _read_version() -> str:
    """Read ``gamma_mdo.__version__`` from source without importing the package.

    Importing ``gamma_mdo`` would pull in the whole geometry stack, which the
    documentation environment deliberately does not install.
    """
    init = Path(__file__).resolve().parents[1] / "gamma_mdo" / "__init__.py"
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init.read_text(), re.MULTILINE
    )
    if match is None:  # pragma: no cover - guarded by tests/test_documentation.py
        raise RuntimeError(f"Could not find __version__ in {init}")
    return match.group(1)


version = _read_version()
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "deflist"]
# Only generate anchors for the heading levels the pages actually cross-link.
myst_heading_anchors = 3

root_doc = "index"
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "README.md", "overhaul", "Thumbs.db", ".DS_Store"]

# Every warning is an error in CI, so keep the reference resolution strict.
nitpicky = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"GAMMA {version}"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#2980B9",
}
