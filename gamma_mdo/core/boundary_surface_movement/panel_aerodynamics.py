"""Optional differentiable VortexAD panel-method integration.

The adapter consumes only :class:`MeshMotionResult`'s public surface, imports
VortexAD only when the builder is called, and leaves recorder ownership with
the caller. It targets the pinned VortexAD revision documented in
``requirements.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import csdl_alpha as csdl
import numpy as np


VORTEXAD_REPOSITORY = "https://github.com/LSDOlab/VortexAD.git"
VORTEXAD_REVISION = "8c5bc86fda5fa1e359fecde24c6c6e8527c773b5"


@dataclass(frozen=True)
class PanelCondition:
    """Configure one steady VortexAD panel-method evaluation.

    Parameters
    ----------
    reference_area_m2
        Aerodynamic reference area in square meters.
    reference_chord_m
        Aerodynamic reference chord in meters.
    mach
        Freestream Mach number.
    alpha_degrees
        Angle of attack in degrees. A scalar CSDL variable may be supplied.
    density_kg_m3
        Freestream density in kilograms per cubic meter.
    speed_of_sound_m_s
        Freestream speed of sound in meters per second.
    pressure_coefficient_floor
        Lower numerical cutoff applied to the pressure coefficient.
    trailing_edge_angle_degrees
        Dihedral-angle threshold used by VortexAD to identify trailing edges.
    trailing_edge_edges_to_ignore
        Zero-based node-index pairs excluded from trailing-edge detection.
    require_converged_projection
        Whether any non-converged OML projection rejects the panel solve.
    """

    reference_area_m2: float | csdl.Variable
    reference_chord_m: float | csdl.Variable
    mach: float | csdl.Variable = 0.7
    alpha_degrees: float | csdl.Variable = 0.0
    density_kg_m3: float | csdl.Variable = 0.65
    speed_of_sound_m_s: float | csdl.Variable = 310.0
    pressure_coefficient_floor: float = -3.0
    trailing_edge_angle_degrees: float = 125.0
    trailing_edge_edges_to_ignore: tuple[tuple[int, int], ...] = ()
    require_converged_projection: bool = True

    def __post_init__(self) -> None:
        """Validate constant settings without evaluating symbolic inputs."""
        for name in (
            "reference_area_m2",
            "reference_chord_m",
            "mach",
            "density_kg_m3",
            "speed_of_sound_m_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, csdl.Variable) and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.trailing_edge_angle_degrees < 180.0:
            raise ValueError(
                "trailing_edge_angle_degrees must lie strictly between 0 and 180."
            )
        for edge in self.trailing_edge_edges_to_ignore:
            if len(edge) != 2 or edge[0] == edge[1] or min(edge) < 0:
                raise ValueError(
                    "Each ignored trailing-edge entry must contain two distinct "
                    "nonnegative node indices."
                )


@dataclass(frozen=True)
class PanelAerodynamicOutputs:
    """Named differentiable outputs from the steady panel method.

    Attributes
    ----------
    lift_coefficient
        Dimensionless aircraft lift coefficient.
    induced_drag_coefficient
        Dimensionless induced-drag coefficient.
    lift_newton
        Integrated lift in newtons.
    induced_drag_newton
        Integrated induced drag in newtons.
    """

    lift_coefficient: csdl.Variable
    induced_drag_coefficient: csdl.Variable
    lift_newton: csdl.Variable
    induced_drag_newton: csdl.Variable

    def as_dict(self) -> dict[str, csdl.Variable]:
        """Return outputs under the names understood by VortexAD and FD tools."""
        return {
            "CL": self.lift_coefficient,
            "CDi": self.induced_drag_coefficient,
            "L": self.lift_newton,
            "Di": self.induced_drag_newton,
        }


def _load_vortexad() -> tuple[Any, Any, Any]:
    """Import the pinned optional interface only when panel analysis is used."""
    try:
        from VortexAD import PanelMethod, TE_detection, find_cell_adjacency
    except ImportError as error:
        install = (
            f'python -m pip install --no-deps "VortexAD @ git+'
            f'{VORTEXAD_REPOSITORY}@{VORTEXAD_REVISION}"'
        )
        raise ImportError(
            "VortexAD is an optional GAMMA integration and is not installed. "
            f"Install the pinned revision with: {install}"
        ) from error
    return PanelMethod, find_cell_adjacency, TE_detection


def _surface_cells(surface_mesh: Any) -> dict[str, np.ndarray]:
    """Copy triangle and quad connectivity accepted by the pinned solver."""
    if not hasattr(surface_mesh, "cell_blocks"):
        raise TypeError("result.surface_mesh must expose a cell_blocks mapping.")
    blocks = {
        name: np.asarray(cells, dtype=np.int64).copy()
        for name, cells in surface_mesh.cell_blocks.items()
        if name in ("triangle", "quad") and len(cells)
    }
    if not blocks:
        raise ValueError("The panel surface contains no triangles or quads.")
    unsupported = [
        name
        for name, cells in surface_mesh.cell_blocks.items()
        if name not in ("triangle", "quad")
        and np.asarray(cells).ndim == 2
        and np.asarray(cells).shape[1] >= 3
        and len(cells)
    ]
    if unsupported:
        raise ValueError(
            "VortexAD supports triangle and quad panels; unsupported blocks: "
            f"{sorted(unsupported)}."
        )
    return blocks


def _validate_result(result: Any, condition: PanelCondition) -> None:
    """Validate the public mesh-motion contract needed by the panel adapter."""
    coordinates = getattr(result, "surface_coordinates", None)
    if not isinstance(coordinates, csdl.Variable) or coordinates.shape[-1:] != (3,):
        raise TypeError(
            "result.surface_coordinates must be a CSDL variable of shape "
            "(num_vertices, 3)."
        )
    surface_mesh = getattr(result, "surface_mesh", None)
    cells = _surface_cells(surface_mesh)
    num_vertices = coordinates.shape[0]
    initial = np.asarray(
        getattr(result, "initial_surface_coordinates", np.empty((0, 3))),
        dtype=float,
    )
    if initial.shape != (num_vertices, 3):
        raise ValueError(
            "result.initial_surface_coordinates must match the final surface shape."
        )
    for name, block in cells.items():
        if block.ndim != 2 or block.size == 0:
            raise ValueError(f"Surface cell block {name!r} must be a nonempty matrix.")
        if int(block.min()) < 0 or int(block.max()) >= num_vertices:
            raise ValueError(f"Surface cell block {name!r} contains an invalid node ID.")

    classification = getattr(result, "surface_vertex_classification", None)
    projection = getattr(result, "surface_projection_status", None)
    if classification is None or projection is None:
        raise TypeError(
            "The panel adapter requires surface_vertex_classification and "
            "surface_projection_status from mm.run."
        )
    reprojected = np.asarray(projection.reprojected_vertex_ids, dtype=np.int64)
    nonconverged = np.asarray(
        projection.nonconverged_vertex_ids, dtype=np.int64
    )
    if condition.require_converged_projection and nonconverged.size:
        raise ValueError(
            "Panel analysis requires converged OML projection; "
            f"{nonconverged.size} vertices did not converge."
        )
    intersections = tuple(classification.intersection_vertex_ids.values())
    intersection_ids = (
        np.concatenate(intersections).astype(np.int64, copy=False)
        if intersections
        else np.empty(0, dtype=np.int64)
    )
    for label, vertex_ids in (
        ("reprojected", reprojected),
        ("nonconverged", nonconverged),
        ("intersection", intersection_ids),
    ):
        if vertex_ids.size and (
            int(vertex_ids.min()) < 0 or int(vertex_ids.max()) >= num_vertices
        ):
            raise ValueError(f"The {label} vertex IDs fall outside the surface mesh.")
    overlap = np.intersect1d(reprojected, intersection_ids)
    if overlap.size:
        raise ValueError(
            "Exact intersection vertices must not also be closest-point "
            f"reprojected; found {overlap.size} overlapping IDs."
        )
    if not isinstance(getattr(result, "aerodynamic_outputs", None), dict):
        raise TypeError("result.aerodynamic_outputs must be a mutable dictionary.")


def _case_variable(name: str, value: float | csdl.Variable) -> csdl.Variable:
    """Return a one-case CSDL variable without replacing an existing graph input."""
    if isinstance(value, csdl.Variable):
        return value
    return csdl.Variable(name=name, value=np.array([float(value)], dtype=float))


def build_panel_aerodynamics(
    result: Any,
    condition: PanelCondition,
) -> PanelAerodynamicOutputs:
    """Append a steady VortexAD solve to a mesh-motion graph.

    Parameters
    ----------
    result
        Public :class:`MeshMotionResult` returned by :func:`gamma_mdo.mesh_motion.run`.
        The final surface coordinates remain differentiable inputs to the panel
        solve. Exact-intersection and projection-status data are checked before
        constructing it.
    condition
        Freestream, reference-geometry, and trailing-edge settings.

    Returns
    -------
    PanelAerodynamicOutputs
        Lift and induced-drag coefficients and forces. The same four variables
        are registered in ``result.aerodynamic_outputs`` under ``CL``, ``CDi``,
        ``L``, and ``Di``, so :func:`select_fd_objective` can address them.

    Notes
    -----
    Connectivity and trailing edges are derived once from the undeformed
    curated triangle/quad mesh. The pinned VortexAD detector assumes outward
    cell ordering, a freestream aligned with the global x axis, and a sharp
    trailing edge captured by ``trailing_edge_angle_degrees``; ambiguous root
    edges may be excluded explicitly by node pair. The caller must own and
    keep active the same CSDL recorder used by mesh motion. This function never
    starts or stops a recorder.
    """
    _validate_result(result, condition)
    PanelMethod, find_cell_adjacency, detect_trailing_edges = _load_vortexad()

    baseline = np.asarray(result.initial_surface_coordinates, dtype=float)
    cells = _surface_cells(result.surface_mesh)
    adjacency = find_cell_adjacency(points=baseline.copy(), cells=cells)
    topology_points = np.asarray(adjacency[0], dtype=float)
    if topology_points.shape != baseline.shape:
        raise ValueError(
            "VortexAD removed duplicate topology nodes, so its connectivity no "
            "longer indexes the differentiable surface coordinates."
        )
    trailing_edges = detect_trailing_edges(
        points=topology_points,
        cells=adjacency[1],
        edges2cells=adjacency[3],
        points2cells=adjacency[4],
        threshold_theta=condition.trailing_edge_angle_degrees,
        edges2ignore=[list(edge) for edge in condition.trailing_edge_edges_to_ignore],
    )

    panel_method = PanelMethod(
        solver_input_dict={
            "Mach": condition.mach,
            "alpha": _case_variable("panel_alpha_degrees", condition.alpha_degrees),
            "Cp cutoff": condition.pressure_coefficient_floor,
            "ref_area": condition.reference_area_m2,
            "BC": "Dirichlet",
            "compressibility": True,
            "partition_size": 1,
            "reuse_AIC": True,
            "rho": condition.density_kg_m3,
            "sos": condition.speed_of_sound_m_s,
            "ref_chord": condition.reference_chord_m,
        },
        skip_geometry=True,
    )
    panel_method.insert_grid_data(
        mesh=result.surface_coordinates,
        cell_adjacency_data=adjacency,
        TE_properties=trailing_edges,
    )
    output_names = ("CL", "CDi", "L", "Di")
    panel_method.declare_outputs(list(output_names))
    raw_outputs = dict(panel_method.evaluate())
    missing = [name for name in output_names if name not in raw_outputs]
    if missing:
        raise ValueError(f"VortexAD did not return required outputs: {missing}.")
    for name in output_names:
        if not isinstance(raw_outputs[name], csdl.Variable):
            raise TypeError(f"VortexAD output {name!r} must be a CSDL variable.")
        raw_outputs[name].add_name(f"panel_{name}")

    outputs = PanelAerodynamicOutputs(
        lift_coefficient=raw_outputs["CL"],
        induced_drag_coefficient=raw_outputs["CDi"],
        lift_newton=raw_outputs["L"],
        induced_drag_newton=raw_outputs["Di"],
    )
    duplicate = sorted(set(outputs.as_dict()) & set(result.aerodynamic_outputs))
    if duplicate:
        raise ValueError(
            "Aerodynamic outputs are already registered under: "
            f"{duplicate}."
        )
    result.aerodynamic_outputs.update(outputs.as_dict())
    return outputs


__all__ = [
    "PanelAerodynamicOutputs",
    "PanelCondition",
    "VORTEXAD_REPOSITORY",
    "VORTEXAD_REVISION",
    "build_panel_aerodynamics",
]
