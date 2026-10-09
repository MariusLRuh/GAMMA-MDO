"""Surface-mesh quality and inversion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gamma_mdo.preprocessing.mesh_io import _as_mesh_data


@dataclass(frozen=True)
class ElementInversionReport:
    """Report element orientation relative to the undeformed mesh.

    Parameters
    ----------
    inverted_element_ids
        Elements containing at least one reversed corner Jacobian.
    degenerate_element_ids
        Elements containing a zero-area or degenerate corner.
    cell_types
        Cell type aligned with the stable element ordering.
    """

    inverted_element_ids: np.ndarray
    degenerate_element_ids: np.ndarray
    cell_types: np.ndarray

    @property
    def num_inverted(self) -> int:
        """Return the number of inverted elements.

        Returns
        -------
        int
            Size of ``inverted_element_ids``.
        """
        return int(self.inverted_element_ids.size)

    @property
    def has_inversions(self) -> bool:
        """Return whether any element is inverted.

        Returns
        -------
        bool
            ``True`` when at least one element has a reversed corner.
        """
        return bool(self.num_inverted)

    def __bool__(self) -> bool:
        """Report whether the report is clean, so it reads as a pass/fail flag.

        Note the inversion of sense against :attr:`has_inversions`: a report
        with inversions is falsey.

        Returns
        -------
        bool
            ``True`` when no element is inverted.
        """
        return not self.has_inversions


@dataclass(frozen=True)
class MeshQualityReport:
    """Store aggregate quality metrics for a polygonal surface mesh.

    Attributes
    ----------
    element_count
        Number of evaluated surface cells.
    minimum_angle_degrees
        Minimum corner angle in degrees.
    maximum_aspect_ratio
        Largest edge-length aspect ratio.
    minimum_scaled_jacobian
        Smallest oriented corner scaled Jacobian.
    minimum_area_ratio
        Smallest deformed-to-baseline area ratio.
    inverted_elements
        Number of elements with reversed corners.
    inverted_corners
        Total number of reversed corners.
    degenerate_elements
        Number of elements with degenerate geometry.
    """

    element_count: int
    minimum_angle_degrees: float
    angle_p05_degrees: float
    maximum_aspect_ratio: float
    aspect_ratio_p95: float
    minimum_scaled_jacobian: float
    scaled_jacobian_p05: float
    minimum_area_ratio: float
    area_ratio_p05: float
    maximum_area_ratio: float
    inverted_elements: int
    inverted_corners: int
    degenerate_elements: int


def check_element_inversion(*, mesh, final_mesh_vertices) -> ElementInversionReport:
    """Detect locally reversed corners relative to the baseline orientation.

    A quad can be concave or self-folded while its area-weighted polygon normal
    still points in the baseline direction.  Testing every corner catches that
    failure; an element is inverted when any of its corner Jacobians is
    negative.

    Parameters
    ----------
    mesh
        Baseline mesh and connectivity.
    final_mesh_vertices
        Final numeric or CSDL coordinate array.

    Returns
    -------
    ElementInversionReport
        Inverted and degenerate element IDs in stable cell order.
    """
    mesh_data = _as_mesh_data(mesh)
    baseline = np.asarray(mesh_data.vertices, dtype=float)
    deformed = np.asarray(
        getattr(final_mesh_vertices, "value", final_mesh_vertices),
        dtype=float,
    )
    cells, cell_types = _surface_cells(mesh_data)
    inverted = np.zeros(len(cells), dtype=bool)
    degenerate = np.zeros(len(cells), dtype=bool)
    for index, cell in enumerate(cells):
        baseline_normal = _polygon_normal(baseline[cell])
        normal_magnitude = np.linalg.norm(baseline_normal)
        if normal_magnitude <= 1e-14:
            degenerate[index] = True
            continue
        signed_jacobians, corner_degenerate = _signed_corner_scaled_jacobians(
            deformed[cell],
            baseline_normal / normal_magnitude,
        )
        degenerate[index] = bool(np.any(corner_degenerate))
        inverted[index] = bool(np.any(signed_jacobians < -1e-12))
    return ElementInversionReport(
        inverted_element_ids=np.where(inverted)[0].astype(np.int64),
        degenerate_element_ids=np.where(degenerate)[0].astype(np.int64),
        cell_types=cell_types,
    )


def evaluate_mesh_quality(*, mesh, vertices=None) -> MeshQualityReport:
    """Compute surface quality relative to baseline coordinates.

    Parameters
    ----------
    mesh
        Baseline mesh and polygon connectivity.
    vertices
        Optional deformed coordinates; omit to evaluate the baseline.

    Returns
    -------
    MeshQualityReport
        Aggregate angle, aspect-ratio, Jacobian, area, and inversion metrics.
    """
    mesh_data = _as_mesh_data(mesh)
    baseline = np.asarray(mesh_data.vertices, dtype=float)
    points = (
        baseline
        if vertices is None
        else np.asarray(getattr(vertices, "value", vertices), dtype=float)
    )
    cells, _ = _surface_cells(mesh_data)
    angles = []
    aspect_ratios = []
    scaled_jacobians = []
    area_ratios = []
    degenerate_count = 0
    inverted_count = 0
    inverted_corner_count = 0

    for cell in cells:
        baseline_polygon = baseline[cell]
        polygon = points[cell]
        baseline_normal = _polygon_normal(baseline_polygon)
        normal = _polygon_normal(polygon)
        baseline_area = 0.5 * np.linalg.norm(baseline_normal)
        area = 0.5 * np.linalg.norm(normal)
        if baseline_area <= 1e-14:
            signed_corner_jacobians = np.zeros(cell.size, dtype=float)
            corner_degenerate = np.ones(cell.size, dtype=bool)
        else:
            signed_corner_jacobians, corner_degenerate = (
                _signed_corner_scaled_jacobians(
                    polygon,
                    baseline_normal / (2.0 * baseline_area),
                )
            )
        if (
            baseline_area <= 1e-14
            or area <= 1e-14
            or np.any(corner_degenerate)
        ):
            degenerate_count += 1
        inverted_corners = signed_corner_jacobians < -1e-12
        if np.any(inverted_corners):
            inverted_count += 1
            inverted_corner_count += int(np.count_nonzero(inverted_corners))
        polygon_angles = _corner_angles(polygon)
        angles.append(float(np.min(polygon_angles)))
        edge_lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        aspect_ratios.append(
            float(np.max(edge_lengths) / max(np.min(edge_lengths), 1e-14))
        )
        scaled_jacobians.append(float(np.min(signed_corner_jacobians)))
        area_ratios.append(area / max(baseline_area, 1e-14))

    angle_array = np.asarray(angles, dtype=float)
    aspect_array = np.asarray(aspect_ratios, dtype=float)
    jacobian_array = np.asarray(scaled_jacobians, dtype=float)
    area_ratio_array = np.asarray(area_ratios, dtype=float)
    return MeshQualityReport(
        element_count=len(cells),
        minimum_angle_degrees=float(np.min(angle_array)),
        angle_p05_degrees=float(np.percentile(angle_array, 5.0)),
        maximum_aspect_ratio=float(np.max(aspect_array)),
        aspect_ratio_p95=float(np.percentile(aspect_array, 95.0)),
        minimum_scaled_jacobian=float(np.min(jacobian_array)),
        scaled_jacobian_p05=float(np.percentile(jacobian_array, 5.0)),
        minimum_area_ratio=float(np.min(area_ratio_array)),
        area_ratio_p05=float(np.percentile(area_ratio_array, 5.0)),
        maximum_area_ratio=float(np.max(area_ratio_array)),
        inverted_elements=inverted_count,
        inverted_corners=inverted_corner_count,
        degenerate_elements=degenerate_count,
    )


def compare_mesh_quality(*, mesh, deformed_vertices) -> tuple[MeshQualityReport, MeshQualityReport]:
    """Evaluate baseline and deformed quality with identical connectivity.

    Parameters
    ----------
    mesh
        Baseline mesh and connectivity.
    deformed_vertices
        Coordinates of the deformed surface mesh.

    Returns
    -------
    tuple[MeshQualityReport, MeshQualityReport]
        Baseline report followed by the deformed report.
    """
    return (
        evaluate_mesh_quality(mesh=mesh),
        evaluate_mesh_quality(mesh=mesh, vertices=deformed_vertices),
    )


def _surface_cells(mesh):
    cells = []
    types = []
    # Triangles and quads first (stable ordering for existing tri/quad meshes),
    # then any other polygon blocks (e.g. "polygon6" for a hex-dominant CFD
    # surface mesh).  The per-corner inversion/quality logic is already n-gon
    # general, so this makes hex/pentagon cells first-class without triangulating
    # them (fan-triangulating an n-gon invents sliver cells that report spurious
    # inversions).
    ordered_types = ["triangle", "quad"]
    ordered_types += [t for t in mesh.cell_blocks if t not in ("triangle", "quad")]
    for cell_type in ordered_types:
        block = mesh.cell_blocks.get(cell_type)
        if block is None:
            continue
        block_array = np.asarray(block, dtype=np.int64)
        if block_array.ndim != 2 or block_array.shape[1] < 3:
            continue
        for cell in block_array:
            cells.append(cell)
            types.append(cell_type)
    if not cells:
        raise ValueError("Mesh quality requires surface cells with at least 3 vertices.")
    return cells, np.asarray(types, dtype=object)


def _polygon_normal(points):
    points = np.asarray(points, dtype=float)
    return np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)


def _corner_angles(points):
    previous = np.roll(points, 1, axis=0) - points
    following = np.roll(points, -1, axis=0) - points
    denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    cosine = np.einsum("ij,ij->i", previous, following) / np.maximum(denominator, 1e-14)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _signed_corner_scaled_jacobians(points, reference_unit_normal):
    """Return each corner's oriented, normalized surface Jacobian.

    For corner ``i``, with incoming and outgoing edge vectors

    ``a_i = x_i - x_(i-1)`` and ``b_i = x_(i+1) - x_i``,

    the value is ``n0 dot (a_i cross b_i) / (|a_i| |b_i|)``, where ``n0`` is
    the baseline element unit normal.
    """
    previous = points - np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0) - points
    denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    cross_products = np.cross(previous, following)
    signed = np.einsum(
        "ij,j->i",
        cross_products,
        np.asarray(reference_unit_normal, dtype=float),
    ) / np.maximum(denominator, 1e-14)
    return (
        signed,
        denominator <= 1e-14,
    )


__all__ = [
    "ElementInversionReport",
    "MeshQualityReport",
    "check_element_inversion",
    "compare_mesh_quality",
    "evaluate_mesh_quality",
]
