"""Geometry component discovery and FunctionSet construction helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lsdo_function_spaces as lfs


DEFAULT_STEP_FILE = (
    Path(__file__).resolve().parents[1]
    / "core"
    / "boundary_surface_movement"
    / "e175.stp"
)

_COMPONENT_ALIASES = {
    "fuselage": ("fuse", "body"),
    "fuse": ("fuselage", "body"),
    "fuselagefairing": ("fairing", "underbelly", "bellyfairing"),
    "fairing": ("fuselagefairing", "underbelly", "bellyfairing"),
    "wing": ("liftingsurface",),
    "liftingsurface": ("wing",),
    "ht": ("horizontaltail", "htail", "horizontalstabilizer"),
    "horizontaltail": ("ht", "htail", "horizontalstabilizer"),
    "htail": ("ht", "horizontaltail", "horizontalstabilizer"),
    "horizontalstabilizer": ("ht", "horizontaltail", "htail"),
    "vt": ("verticaltail", "vtail", "verticalstabilizer"),
    "verticaltail": ("vt", "vtail", "verticalstabilizer"),
    "vtail": ("vt", "verticaltail", "verticalstabilizer"),
    "verticalstabilizer": ("vt", "verticaltail", "vtail"),
}

def create_components(
    keys: Iterable[Iterable[int] | int | None] | None = None,
    search_names: Iterable[str | Iterable[str] | None] | None = None,
    *,
    geometry=None,
    step_file: str | Path | None = None,
    parallelize: bool = False,
):
    """Create component function sets from imported geometry.

    Parameters
    ----------
    keys : iterable, optional
        Explicit surface keys for each requested component. If supplied with
        ``search_names``, the two iterables must have the same length. Individual
        entries may be ``None`` to use name matching for that component.
    search_names : iterable, optional
        Name or alternative names to match for each requested component. A tuple
        such as ``("wing", "lifting_surface")`` matches surfaces containing
        either string.
    geometry : object, optional
        Existing geometry/function-set object. If omitted, ``step_file`` is
        imported with ``lsdo_function_spaces.import_file_patched``.
    step_file : path-like, optional
        STEP file to import. Defaults to the bundled E175-with-fairing geometry.
    parallelize : bool
        Passed to ``import_file_patched`` when importing geometry.

    Returns
    -------
    object or tuple[object, ...]
        A single component function set when one component is requested, or a
        tuple of component function sets when multiple components are requested.
    """
    if keys is None and search_names is None:
        raise ValueError("Provide at least one of keys or search_names.")

    geometry = geometry if geometry is not None else _import_geometry(step_file, parallelize)
    functions = _geometry_functions(geometry)

    key_groups = _normalize_key_groups(keys)
    name_groups = _normalize_search_name_groups(search_names)

    if key_groups is None:
        key_groups = [None] * len(name_groups)
    if name_groups is None:
        name_groups = [None] * len(key_groups)
    if len(key_groups) != len(name_groups):
        raise ValueError(
            "keys and search_names must have the same length. "
            f"Received {len(key_groups)} keys entries and {len(name_groups)} "
            "search_names entries."
        )

    components = []
    for index, (requested_keys, requested_names) in enumerate(zip(key_groups, name_groups)):
        selected_keys = (
            _normalize_requested_keys(requested_keys, functions, index)
            if requested_keys is not None
            else _keys_matching_names(requested_names, functions, index)
        )
        selected_functions = {key: functions[key] for key in selected_keys}
        components.append(_new_function_set_like(geometry, selected_functions))

    if len(components) == 1:
        return components[0]
    return tuple(components)


def available_components(geometry) -> list[dict[str, object]]:
    """Return component-level labels and keys inferred from geometry surfaces."""
    return _available_components(_geometry_functions(geometry))



def _import_geometry(step_file: str | Path | None, parallelize: bool):
    step_path = Path(step_file) if step_file is not None else DEFAULT_STEP_FILE
    try:
        import lsdo_function_spaces as lfs
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "create_components requires lsdo_function_spaces to import STEP files. "
            "Pass an existing geometry object with a .functions mapping, or install "
            "lsdo_function_spaces."
        ) from exc

    return lfs.import_file_patched(step_path, parallelize=parallelize)


def _geometry_functions(geometry) -> dict:
    functions = getattr(geometry, "functions", None)
    if functions is None:
        raise TypeError("geometry must expose a .functions mapping.")
    return dict(functions)


def _normalize_key_groups(keys):
    if keys is None:
        return None
    if np.isscalar(keys):
        return [keys]
    if isinstance(keys, np.ndarray):
        if keys.ndim == 0:
            return [int(keys)]
        return [keys.tolist()]

    key_list = list(keys)
    if not key_list:
        return []
    if all(item is None or np.isscalar(item) for item in key_list):
        return [key_list]
    return key_list


def _normalize_search_name_groups(search_names):
    if search_names is None:
        return None
    if isinstance(search_names, str):
        return [search_names]

    name_list = list(search_names)
    if not name_list:
        return []
    if isinstance(search_names, tuple) and all(isinstance(item, str) for item in name_list):
        return [search_names]
    return name_list


def _normalize_requested_keys(requested_keys, functions: dict, component_index: int) -> list:
    if np.isscalar(requested_keys):
        normalized = [int(requested_keys)]
    else:
        normalized = [int(key) for key in list(requested_keys)]

    missing = [key for key in normalized if key not in functions]
    if missing:
        raise KeyError(
            f"keys[{component_index}] contains unavailable keys {missing}. "
            f"Available keys: {list(functions.keys())}"
        )
    if not normalized:
        raise ValueError(f"keys[{component_index}] is empty.")
    return normalized


def _keys_matching_names(requested_names, functions: dict, component_index: int) -> list:
    terms = _normalize_search_terms(requested_names, component_index)
    selected = []
    for key, function in functions.items():
        if any(_term_matches_function(term, key, function) for term in terms):
            selected.append(key)

    if not selected:
        original = requested_names if requested_names is not None else None
        raise ValueError(
            f"search_names[{component_index}]={original!r} did not match any "
            "geometry components.\n"
            f"Available components:\n{_format_available_components(functions)}\n"
            "Use one of these component labels/aliases or pass explicit keys, "
            "for example keys=np.arange(start, stop)."
        )
    return selected


def _normalize_search_terms(requested_names, component_index: int) -> tuple[str, ...]:
    if requested_names is None:
        raise ValueError(
            f"search_names[{component_index}] is required when keys[{component_index}] is None."
        )

    if isinstance(requested_names, str):
        raw_terms = [requested_names]
    else:
        raw_terms = list(requested_names)

    terms = tuple(str(term).strip() for term in raw_terms if str(term).strip())
    if not terms:
        raise ValueError(f"search_names[{component_index}] has no non-empty strings.")
    return terms


def _searchable_options(key, function) -> tuple[str, ...]:
    raw_options = [str(key)]
    for attr in ("name", "label", "surface_name", "primitive_name"):
        value = getattr(function, attr, None)
        if value is not None:
            raw_options.append(str(value))

    metadata = getattr(function, "metadata", None)
    if isinstance(metadata, dict):
        raw_options.extend(str(value) for value in metadata.values() if value is not None)

    name = getattr(function, "name", None)
    component_label = _component_label_from_surface_name(name)
    if component_label is not None:
        raw_options.append(component_label)

    return tuple(option for option in raw_options if option)


def _component_label_from_surface_name(name) -> str | None:
    if name is None:
        return None
    parts = [part.strip() for part in str(name).split(",")]
    if len(parts) >= 2 and parts[0].lower().startswith("surf"):
        return parts[1]
    return None


def _normalize_name(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _expanded_terms(term: str) -> set[str]:
    normalized = _normalize_name(term)
    return {normalized, *_COMPONENT_ALIASES.get(normalized, ())}


def _term_matches_options(term: str, options: tuple[str, ...]) -> bool:
    terms = _expanded_terms(term)
    normalized_options = {_normalize_name(option) for option in options}
    for expanded_term in terms:
        if any(expanded_term in option or option in terms for option in normalized_options):
            return True
    return False


def _term_matches_function(term: str, key, function) -> bool:
    component_label = _component_label_from_surface_name(getattr(function, "name", None))
    if component_label is not None:
        return bool(_expanded_terms(term) & _label_terms(component_label))
    return _term_matches_options(term, _searchable_options(key, function))


def _label_terms(label: str) -> set[str]:
    normalized = _normalize_name(label)
    return {normalized, *_COMPONENT_ALIASES.get(normalized, ())}


def _available_options(functions: dict) -> list[str]:
    labels = []
    for key, function in functions.items():
        name = getattr(function, "name", None)
        if name is None:
            labels.append(str(key))
        else:
            labels.append(f"{key}: {name}")
    return labels


def _available_components(functions: dict) -> list[dict[str, object]]:
    components: dict[str, list] = {}
    for key, function in functions.items():
        label = _component_label_from_surface_name(getattr(function, "name", None))
        if label is None:
            label = _component_label_from_surface_name(getattr(function, "label", None))
        if label is None:
            label = _component_label_from_surface_name(getattr(function, "surface_name", None))
        if label is None:
            label = getattr(function, "name", None) or getattr(function, "label", None) or str(key)
        components.setdefault(str(label), []).append(key)

    return [
        {
            "label": label,
            "keys": keys,
            "aliases": _aliases_for_component_label(label),
        }
        for label, keys in components.items()
    ]


def _aliases_for_component_label(label: str) -> list[str]:
    normalized = _normalize_name(label)
    aliases = []
    for candidate in _COMPONENT_ALIASES.get(normalized, ()):
        aliases.append(_display_alias(candidate))
    return aliases


def _display_alias(alias: str) -> str:
    display_names = {
        "fuse": "fuse",
        "fuselage": "fuselage",
        "fairing": "fairing",
        "underbelly": "underbelly",
        "liftingsurface": "lifting_surface",
        "horizontaltail": "horizontal_tail",
        "htail": "htail",
        "horizontalstabilizer": "horizontal_stabilizer",
        "verticaltail": "vertical_tail",
        "vtail": "vtail",
        "verticalstabilizer": "vertical_stabilizer",
    }
    return display_names.get(alias, alias)


def _format_key_span(keys: list) -> str:
    sorted_keys = sorted(keys)
    if len(sorted_keys) == 1:
        return str(sorted_keys[0])
    if sorted_keys == list(range(sorted_keys[0], sorted_keys[-1] + 1)):
        return f"{sorted_keys[0]}-{sorted_keys[-1]}"
    return ", ".join(str(key) for key in sorted_keys)


def _format_available_components(functions: dict) -> str:
    lines = []
    for component in _available_components(functions):
        alias_text = ""
        aliases = component["aliases"]
        if aliases:
            alias_text = f"; aliases: {', '.join(aliases)}"
        lines.append(f"  - {component['label']} (keys {_format_key_span(component['keys'])}{alias_text})")
    return "\n".join(lines)


def _new_function_set_like(geometry, functions: dict):
    try:
        import lsdo_function_spaces as lfs

        return lfs.FunctionSet(functions=functions)
    except Exception:
        pass

    geometry_type = type(geometry)
    constructor_attempts = []
    if hasattr(geometry, "space"):
        constructor_attempts.append({"functions": functions, "space": geometry.space})
    constructor_attempts.append({"functions": functions})

    for kwargs in constructor_attempts:
        try:
            return geometry_type(**kwargs)
        except TypeError:
            pass

    return Component(functions=functions)


class Component:
    """Fallback component container used when no FunctionSet class is available."""

    def __init__(self, functions: dict):
        self.functions = dict(functions)
