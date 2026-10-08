"""Structural guards for the user documentation contract.

These are fast, standard-library checks of the high-value invariants a reader
depends on: that the inherited template identity is gone, that the navigation
targets exist, that the public API names the site promises are real, and that
the installation instructions still name the validated dependency revisions
and the load-bearing pip flags.

They deliberately do **not** reimplement Sphinx. The build itself, with
warnings as errors, runs in CI and in the documentation environment.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPOSITORY_ROOT / "docs"
DOCS_SOURCE = DOCS / "src"
CONF = DOCS / "conf.py"
INDEX = DOCS / "index.md"
README = REPOSITORY_ROOT / "README.md"
READTHEDOCS = REPOSITORY_ROOT / ".readthedocs.yaml"
DOCS_REQUIREMENTS = DOCS / "requirements.txt"
CI_REQUIREMENTS = REPOSITORY_ROOT / "requirements-ci.txt"
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "actions.yml"

CSDL_REVISION = "73a9efd1033016a835779db10a9b9e81ed2254ce"
LFS_REVISION = "307ad3aabfff31c6fb44ddf51bc0dcc41a60c420"

# Pages the site is expected to ship, beyond the landing page.
EXPECTED_PAGES = (
    "getting_started",
    "examples",
    "external_parameterization",
    "api",
    "background",
    "integrations",
)


def _user_facing_documents() -> list[Path]:
    """Every user-facing documentation file, excluding the overhaul record."""
    paths = [README, READTHEDOCS, CONF]
    paths.extend(
        path
        for path in DOCS.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".py", ".txt", ".bib"}
        and "overhaul" not in path.relative_to(DOCS).parts
    )
    return sorted(set(paths))


def test_documentation_files_exist():
    """Require the landing page, configuration, and every expected page."""
    assert CONF.is_file()
    assert INDEX.is_file()
    assert DOCS_REQUIREMENTS.is_file()
    for name in EXPECTED_PAGES:
        assert (DOCS_SOURCE / f"{name}.md").is_file(), name


@pytest.mark.parametrize("token", ["lsdo_project_template", "quartic"])
def test_no_inherited_template_identity(token):
    """Fail if the inherited template identity survives in user-facing docs.

    ``docs/overhaul/`` is excluded on purpose: it is a historical engineering
    record and is expected to mention both terms.
    """
    offenders = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in _user_facing_documents()
        if token in path.read_text(encoding="utf-8").lower()
    ]
    assert not offenders, f"{token!r} still present in {offenders}"


def test_toctree_targets_resolve_to_local_files():
    """Every toctree entry in the landing page must name a real document."""
    text = INDEX.read_text(encoding="utf-8")
    blocks = re.findall(r"```\{toctree\}(.*?)```", text, re.DOTALL)
    assert blocks, "the landing page must declare a toctree"

    entries = []
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            entries.append(line)

    assert entries, "the toctree must list at least one document"
    for entry in entries:
        assert (DOCS / f"{entry}.md").is_file(), entry

    # Every expected page is actually reachable from the navigation.
    for name in EXPECTED_PAGES:
        assert f"src/{name}" in entries, name


def test_configuration_targets_python_312_and_real_version():
    """Read the Docs must use 3.12 and conf.py must not import the package."""
    readthedocs = READTHEDOCS.read_text(encoding="utf-8")
    assert 'python: "3.12"' in readthedocs
    assert "docs/requirements.txt" in readthedocs
    assert "fail_on_warning: true" in readthedocs

    conf = CONF.read_text(encoding="utf-8")
    assert "lsdo_project_template" not in conf
    assert 'project = "GAMMA"' in conf
    # The version is read from source rather than imported, so the docs build
    # needs none of the geometry stack.
    assert "import gamma_mdo" not in conf
    assert "__version__" in conf

    init = REPOSITORY_ROOT / "gamma_mdo" / "__init__.py"
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        init.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "conf.py's version regex must still match gamma_mdo/__init__.py"


def _literal_public_exports() -> set[str]:
    """Read the literal ``mesh_motion.__all__`` without importing ``gamma_mdo``."""
    module_path = REPOSITORY_ROOT / "gamma_mdo" / "mesh_motion.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), module_path)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, "mesh_motion.py must define one literal __all__"
    value = ast.literal_eval(assignments[0].value)
    assert isinstance(value, (list, tuple))
    assert all(isinstance(name, str) for name in value)
    assert len(value) == len(set(value)), "mesh_motion.__all__ has duplicates"
    return set(value)


def test_public_api_inventory_exactly_matches_exports():
    """Keep the explicit API inventory equal to the module's public exports."""
    api = (DOCS_SOURCE / "api.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- BEGIN GAMMA PUBLIC EXPORTS -->\n(.*?)\n"
        r"<!-- END GAMMA PUBLIC EXPORTS -->",
        api,
        re.DOTALL,
    )
    assert match, "api.md must contain the delimited public export inventory"

    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    documented = []
    for line in lines:
        item = re.fullmatch(r"- `mm\.([A-Za-z_][A-Za-z0-9_]*)`", line)
        assert item, f"malformed public export inventory line: {line!r}"
        documented.append(item.group(1))
    assert len(documented) == len(set(documented)), "duplicate API inventory item"

    assert set(documented) == _literal_public_exports()


def test_installation_pins_the_validated_stack():
    """Installation must bootstrap the full stack before no-deps installs."""
    getting_started = (DOCS_SOURCE / "getting_started.md").read_text(
        encoding="utf-8"
    )
    readme = README.read_text(encoding="utf-8")

    for text in (getting_started, readme):
        assert CSDL_REVISION in text
        assert LFS_REVISION in text
        blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
        install_blocks = [
            block for block in blocks if "requirements-ci.txt" in block
        ]
        assert len(install_blocks) == 1
        install = install_blocks[0]
        assert "conda create -n gamma_mdo python=3.12" in install
        assert "conda activate gamma_mdo" in install
        assert "python -m pip install -r requirements-ci.txt" in install
        assert "lsdo_function_spaces @ git+" in install
        assert "--no-deps" in install
        assert "--no-deps --no-build-isolation -e ." in install
        assert "CSDL_alpha.git@" not in install
        assert install.index("requirements-ci.txt") < install.index(
            "lsdo_function_spaces @ git+"
        )
        assert install.index("lsdo_function_spaces @ git+") < install.index(
            "--no-deps --no-build-isolation -e ."
        )

    requirements = CI_REQUIREMENTS.read_text(encoding="utf-8")
    assert CSDL_REVISION in requirements
    # Official LFS 307ad3a is pure Python. The retired extension pins NumPy
    # 1.26.4 and makes this NumPy-2.0 environment unresolvable from scratch.
    assert "lsdo_b_splines_cython" not in requirements


def test_documentation_avoids_rejected_behavioral_guarantees():
    """Keep the four reviewed overclaims out of user-facing prose."""
    normalized = {
        path: " ".join(path.read_text(encoding="utf-8").split())
        for path in _user_facing_documents()
    }
    for path, text in normalized.items():
        assert "so the mesh stays valid" not in text, path
        assert "back exactly on the deformed" not in text, path
        assert "GAMMA never starts or stops a recorder" not in text, path

    for name in ("examples.md", "api.md"):
        text = normalized[DOCS_SOURCE / name]
        assert not re.search(r"print_summary.{0,120}load stepping", text), name


def test_readme_marks_the_api_call_shape_as_incomplete():
    """Prevent the empty GeometryModel skeleton from posing as runnable code."""
    text = " ".join(README.read_text(encoding="utf-8").split()).lower()
    assert "call shape, not a standalone example" in text
    assert "at least one component" in text


def test_docs_requirements_are_pinned_and_runtime_free():
    """Docs dependencies must be pinned and must not drag in the solver stack."""
    text = DOCS_REQUIREMENTS.read_text(encoding="utf-8")
    requirements = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert requirements, "docs/requirements.txt must pin something"
    for requirement in requirements:
        assert "==" in requirement, f"{requirement} is not pinned"

    # Only the requirement lines matter; the header comment names the stack
    # it deliberately excludes.
    declared = " ".join(requirements).lower()
    for runtime in ("csdl", "lsdo_function_spaces", "jax", "gmsh", "dafoam"):
        assert runtime not in declared, runtime


def test_ci_builds_documentation_with_warnings_as_errors():
    """CI must build the site strictly and outside the repository."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "docs/requirements.txt" in workflow
    assert "-W --keep-going -b html docs" in workflow
    # The build output must not land in the working tree.
    assert "RUNNER_TEMP" in workflow


def test_documentation_never_executes_the_example():
    """The site must not run the E175 example or require optional solvers."""
    conf = CONF.read_text(encoding="utf-8")
    # No notebook/py execution machinery is configured.
    assert "nb_execution_mode" not in conf
    assert "autoapi" not in conf
    assert "sphinxcontrib.collections" not in conf

    readthedocs = READTHEDOCS.read_text(encoding="utf-8")
    # Read the Docs installs docs requirements only, not the package.
    assert "method: pip" not in readthedocs
