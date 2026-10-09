"""Compact public namespace for differentiable surface mesh motion.

This module is the intended entry point. A user writes::

    import gamma_mdo.mesh_motion as mm

and then builds three things: the input files, a :class:`GeometryModel`
describing what moves, and a :class:`MeshMotion` settings object describing how
the mesh follows. :func:`run` evaluates the differentiable model and returns a
:class:`MeshMotionResult`.

:class:`GeometryModel` is a declaration and binding envelope, not a
parameterization. Its :meth:`GeometryModel.add_component` accepts deformed
component coefficients produced by any differentiable CSDL/LFS-compatible
parameterization, including one owned entirely by another package, and places
no constraint on how they were built. Its lifting-surface and body helpers are
optional conveniences for callers who would rather describe a motion than
supply coefficients.

Only high-level names are re-exported here. Solver-assembly types, component
records, projection callbacks, and polygon helpers stay internal, because a
caller never needs them to deform a surface mesh.
"""

from __future__ import annotations

from typing import Any

import csdl_alpha as csdl

from .core.boundary_surface_movement import (
    DerivativeCheck,
    DistanceWeighting,
    DistortionPenalty,
    FuelBurnParameters,
    GeometryModel,
    InputFiles,
    MeshMotion,
    MeshMotionResult,
    PanelAerodynamicOutputs,
    PanelCondition,
    PolygonRegularization,
    QualityChecks,
    SurfaceMotion,
    SurfaceProjectionStatus,
    SurfaceVertexClassification,
    Visualization,
    VolumeMotion,
    build_panel_aerodynamics,
    compute_fuel_burn,
    run_fd_sweep,
    run_mesh_motion,
    select_fd_objective,
)

__all__ = [
    "DerivativeCheck",
    "DistanceWeighting",
    "DistortionPenalty",
    "FuelBurnParameters",
    "GeometryModel",
    "InputFiles",
    "MeshMotion",
    "MeshMotionResult",
    "PanelAerodynamicOutputs",
    "PanelCondition",
    "PolygonRegularization",
    "QualityChecks",
    "SurfaceMotion",
    "SurfaceProjectionStatus",
    "SurfaceVertexClassification",
    "Visualization",
    "VolumeMotion",
    "build_panel_aerodynamics",
    "compute_fuel_burn",
    "run",
    "run_fd_sweep",
    "select_fd_objective",
]


def run(
    *,
    inputs: InputFiles,
    geometry: GeometryModel,
    motion: MeshMotion,
    recorder: csdl.Recorder,
    aerodynamic_analysis: Any = None,
    aerodynamic_volume_method: str = "elasticity",
) -> MeshMotionResult:
    """Evaluate the differentiable mesh-motion model.

    Parameters
    ----------
    inputs
        Geometry, surface-mesh, and optional volume-mesh paths.
    geometry
        Declaration and binding envelope for the components that move. It may
        carry externally produced differentiable coefficients and does not
        constrain how they were parameterized; any design variables they
        depend on must belong to ``recorder``.
    motion
        Surface, volume, quality, visualization, and derivative-check settings.
        When its derivative check is enabled, this function registers the
        configured scalar objective; call :func:`run_fd_sweep` after stopping
        the recorder to execute the numerical comparison.
    recorder
        Active CSDL recorder, owned by the caller. This call never starts or
        stops it, so mesh motion composes inside a larger graph such as the
        DAFoam driver. It is retained on the result for convenience.
    aerodynamic_analysis
        Optional downstream builder receiving the selected volume coordinates.
    aerodynamic_volume_method
        Volume-motion method handed to ``aerodynamic_analysis``.

    Returns
    -------
    MeshMotionResult
        Differentiable outputs plus forward diagnostics. Call
        :meth:`MeshMotionResult.print_summary` for a readable report.
    """
    result = run_mesh_motion(
        recorder=recorder,
        input_files=inputs,
        geometry=geometry,
        config=motion,
        aerodynamic_analysis=aerodynamic_analysis,
        aerodynamic_volume_method=aerodynamic_volume_method,
    )
    result.recorder = recorder
    if motion.derivative_check.enabled:
        select_fd_objective(result, motion.derivative_check.objective)
    return result
