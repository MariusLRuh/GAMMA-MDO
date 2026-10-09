"""Hybrid volume-mesh motion driven by an exactly matching wall mesh.

The mesh may be purely tetrahedral or a tetrahedron/pyramid hybrid.  Pyramids
appear when the aircraft wall is quad-dominant: Gmsh caps every boundary
quadrangle with a pyramid rather than splitting it into triangles.  Both
propagators consume a *decomposition* of the mesh into tetrahedra
(:attr:`TetraVolumeMesh.assembly_cells`) so a single code path serves both mesh
types; each pyramid contributes two tetrahedra split across its shorter base
diagonal.

Two reference-configuration propagators are provided:

``graph``
    A scalar graph-Laplacian assembled from all six edges of every decomposed
    tetrahedron.  The same matrix is used for each Cartesian component.

``elasticity``
    Coupled three-dimensional linear elasticity assembled from four-node
    constant-strain tetrahedra.

Both one-step paths expose a differentiable CSDL map from wall coordinates to
all volume coordinates.  The matrices are setup-time constants and the
existing :class:`SPDSolveOperation` supplies the forward and reverse solves.
The NumPy paths can rebuild the matrices on a deformed mesh for forward-only
load stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

import csdl_alpha as csdl

from .spd_solve_custom_op import SPDFactor, SPDSolveOperation, factorize_spd


PHYSICAL_AIRCRAFT = 1
PHYSICAL_SYMMETRY = 2
PHYSICAL_INLET = 3
PHYSICAL_OUTLET = 4
PHYSICAL_FARFIELD = 5
PHYSICAL_FLUID = 100

VolumeMethod = Literal["graph", "elasticity"]


def _empty_cells(width: int):
    return lambda: np.zeros((0, width), dtype=np.int64)


def _empty_ids():
    return np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class TetraVolumeMesh:
    """Arrays and patch topology from an ASCII Gmsh 2.2 volume mesh.

    Quadrangle boundary faces and pyramids are optional, so a purely
    tetrahedral mesh constructs exactly as before.

    Attributes
    ----------
    source_path
        File the mesh was read from.
    node_ids
        Gmsh node identifiers, shape ``(n_nodes,)``, in the file's order.
    vertices
        Node coordinates, shape ``(n_nodes, 3)``, aligned with ``node_ids``.
    triangles
        Triangular boundary faces as node indices, shape ``(n_tri, 3)``.
    triangle_physical_ids
        Physical-group id per triangle, shape ``(n_tri,)``.
    tetrahedra
        Tetrahedral cells as node indices, shape ``(n_tet, 4)``.
    tetrahedron_physical_ids
        Physical-group id per tetrahedron, shape ``(n_tet,)``.
    physical_names
        Group names keyed by ``(dimension, physical_id)``.
    quadrangles
        Quadrilateral boundary faces, shape ``(n_quad, 4)``. Empty for a purely
        tetrahedral mesh.
    quadrangle_physical_ids
        Physical-group id per quadrangle, shape ``(n_quad,)``.
    pyramids
        Pyramid cells, shape ``(n_pyr, 5)``, with the four base nodes first.
        Empty for a purely tetrahedral mesh.
    pyramid_physical_ids
        Physical-group id per pyramid, shape ``(n_pyr,)``.
    """

    source_path: Path
    node_ids: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_physical_ids: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_physical_ids: np.ndarray
    physical_names: dict[tuple[int, int], str]
    quadrangles: np.ndarray = dataclass_field(default_factory=_empty_cells(4))
    quadrangle_physical_ids: np.ndarray = dataclass_field(
        default_factory=_empty_ids
    )
    pyramids: np.ndarray = dataclass_field(default_factory=_empty_cells(5))
    pyramid_physical_ids: np.ndarray = dataclass_field(default_factory=_empty_ids)

    @property
    def is_hybrid(self) -> bool:
        """Whether the mesh contains pyramids as well as tetrahedra.

        Returns
        -------
        bool
            ``True`` when at least one pyramid is present.
        """
        return self.pyramids.shape[0] > 0

    @property
    def aircraft_triangles(self) -> np.ndarray:
        """Triangular faces belonging to the aircraft wall patch.

        Returns
        -------
        numpy.ndarray
            Node indices of the selected faces, shape ``(k, 3)``. Empty when
            the wall carries no triangles.
        """
        return self.triangles[
            self.triangle_physical_ids == PHYSICAL_AIRCRAFT
        ]

    @property
    def aircraft_quadrangles(self) -> np.ndarray:
        """Quadrilateral faces belonging to the aircraft wall patch.

        Returns
        -------
        numpy.ndarray
            Node indices of the selected faces, shape ``(k, 4)``. Empty for a
            purely tetrahedral mesh.
        """
        return self.quadrangles[
            self.quadrangle_physical_ids == PHYSICAL_AIRCRAFT
        ]

    @property
    def aircraft_nodes(self) -> np.ndarray:
        """Nodes on the aircraft wall, from both face types.

        Returns
        -------
        numpy.ndarray
            Sorted unique node indices, shape ``(k,)``.
        """
        return _unique_nodes(
            self.aircraft_triangles, self.aircraft_quadrangles
        )

    @property
    def assembly_cells(self) -> np.ndarray:
        """Tetrahedra used to assemble the motion operators.

        Every pyramid is split into two tetrahedra across its shorter base
        diagonal, which is the better-conditioned of the two splits.
        """
        return _decompose_cells(
            self.vertices, self.tetrahedra, self.pyramids, both_diagonals=False
        )

    @property
    def quality_cells(self) -> np.ndarray:
        """Tetrahedra used to test validity.

        Both base diagonals of every pyramid are included, so a pyramid counts
        as valid only when it is non-degenerate either way it is split.
        """
        return _decompose_cells(
            self.vertices, self.tetrahedra, self.pyramids, both_diagonals=True
        )

    @property
    def symmetry_nodes(self) -> np.ndarray:
        """Nodes on the symmetry-plane patch.

        Returns
        -------
        numpy.ndarray
            Sorted unique node indices, shape ``(k,)``. These slide within the
            plane rather than being fully fixed.
        """
        return _nodes_for_physical_patch(
            self, PHYSICAL_SYMMETRY
        )

    @property
    def fixed_outer_nodes(self) -> np.ndarray:
        """Nodes on the inlet, outlet, and farfield patches.

        Returns
        -------
        numpy.ndarray
            Sorted unique node indices, shape ``(k,)``, pooled across the three
            patches. These are held fixed during volume motion.
        """
        return np.unique(
            np.concatenate(
                [
                    _nodes_for_physical_patch(self, physical_id)
                    for physical_id in (
                        PHYSICAL_INLET,
                        PHYSICAL_OUTLET,
                        PHYSICAL_FARFIELD,
                    )
                ]
            )
        )

    @property
    def boundary_nodes(self) -> np.ndarray:
        """Every node appearing on any boundary face.

        Returns
        -------
        numpy.ndarray
            Sorted unique node indices, shape ``(k,)``, across all patches and
            both face types.
        """
        return _unique_nodes(self.triangles, self.quadrangles)


@dataclass(frozen=True)
class VolumeBoundaryPartition:
    """Node and DOF partitions for a half-domain moving-aircraft mesh.

    Node arrays hold node indices; DOF arrays hold indices into the flattened
    ``3 * n_nodes`` displacement vector.

    Attributes
    ----------
    wall_nodes
        Aircraft-wall nodes, whose motion is prescribed.
    outer_nodes
        Inlet, outlet, and farfield nodes, held fixed.
    symmetry_tangent_nodes
        Symmetry-plane nodes, free within the plane but not across it.
    interior_nodes
        Nodes on no constrained patch, free in every direction.
    graph_free_xz
        Nodes solved for in the scalar x and z graph systems, where symmetry
        nodes are free.
    graph_free_y
        Nodes solved for in the scalar y graph system, which excludes symmetry
        nodes because their normal component is pinned.
    elasticity_free_dofs
        Flattened DOF indices solved for in the vector elasticity system, with
        the symmetry-normal components already removed.
    wall_dofs
        Flattened DOF indices of the prescribed wall motion.
    """

    wall_nodes: np.ndarray
    outer_nodes: np.ndarray
    symmetry_tangent_nodes: np.ndarray
    interior_nodes: np.ndarray
    graph_free_xz: np.ndarray
    graph_free_y: np.ndarray
    elasticity_free_dofs: np.ndarray
    wall_dofs: np.ndarray


@dataclass
class GraphVolumeSystem:
    """Factored scalar graph systems for x/z and y motion.

    Volume motion only. It is separate from the retained surface method, which
    is a graph Laplacian with N-gon regularization; this class propagates a
    prescribed wall motion into the volume. The x and z components share one
    scalar system because both leave symmetry nodes free, while y needs its own
    because the symmetry-normal component is pinned.

    Attributes
    ----------
    mesh
        Volume mesh the system was assembled on; supplies the baseline
        vertices.
    partition
        Node and DOF partitions separating prescribed wall motion from the
        free interior.
    factor_xz
        Factored scalar operator for the x and z components.
    factor_y
        Factored scalar operator for the y component.
    coupling_wall_xz
        Block coupling free x/z rows to the prescribed wall columns.
    coupling_wall_y
        Block coupling free y rows to the prescribed wall columns.
    assembly_seconds
        Wall-clock seconds spent assembling the operators.
    factorization_seconds
        Wall-clock seconds spent factorizing them.
    stiffening_exponent
        Exponent making small cells stiffer than large ones.
    volume_floor
        Lower bound on the cell volume used for stiffening, guarding against
        division by a vanishing volume.
    method
        Identifier of this propagator, ``"graph"``.
    """

    mesh: TetraVolumeMesh
    partition: VolumeBoundaryPartition
    factor_xz: SPDFactor
    factor_y: SPDFactor
    coupling_wall_xz: sp.csr_matrix
    coupling_wall_y: sp.csr_matrix
    assembly_seconds: float
    factorization_seconds: float
    stiffening_exponent: float
    volume_floor: float

    method: str = "graph"

    def evaluate(self, wall_positions) -> csdl.Variable:
        """Deform the volume differentiably from prescribed wall positions.

        The factorization is treated as a reference matrix: it is reused as
        assembled, so the result is linear in the wall displacement and the
        reverse pass goes through the same factor.

        Parameters
        ----------
        wall_positions
            Deformed wall-node positions of shape ``(n_wall, 3)``, ordered as
            ``partition.wall_nodes``. The displacement is formed against the
            baseline vertices.

        Returns
        -------
        csdl_alpha.Variable
            Deformed volume coordinates of shape ``(n_nodes, 3)``, in the
            mesh's node order.
        """
        baseline = self.mesh.vertices
        wall = self.partition.wall_nodes
        wall_displacement = wall_positions - baseline[wall]

        rhs_x = -csdl.sparse.matmat(
            self.coupling_wall_xz,
            wall_displacement[csdl.slice[:, 0:1]],
        )
        rhs_z = -csdl.sparse.matmat(
            self.coupling_wall_xz,
            wall_displacement[csdl.slice[:, 2:3]],
        )
        rhs_y = -csdl.sparse.matmat(
            self.coupling_wall_y,
            wall_displacement[csdl.slice[:, 1:2]],
        )
        solved_x = SPDSolveOperation(self.factor_xz).evaluate(rhs_x)
        solved_z = SPDSolveOperation(self.factor_xz).evaluate(rhs_z)
        solved_y = SPDSolveOperation(self.factor_y).evaluate(rhs_y)

        displacement = csdl.Variable(value=np.zeros_like(baseline))
        displacement = displacement.set(_row_slice(wall), wall_displacement)
        displacement = displacement.set(
            _component_slice(self.partition.graph_free_xz, 0), solved_x
        )
        displacement = displacement.set(
            _component_slice(self.partition.graph_free_xz, 2), solved_z
        )
        displacement = displacement.set(
            _component_slice(self.partition.graph_free_y, 1), solved_y
        )
        return csdl.Variable(value=baseline) + displacement

    def solve_numpy(self, wall_displacement: np.ndarray) -> np.ndarray:
        """Solve an incremental displacement with the current factored matrix.

        The eager NumPy counterpart of :meth:`evaluate`, used by the
        forward-only load-stepping path. It returns a displacement, not a
        position.

        Parameters
        ----------
        wall_displacement
            Incremental wall displacement reshaped to ``(n_wall, 3)``, ordered
            as ``partition.wall_nodes``.

        Returns
        -------
        numpy.ndarray
            Displacement of every node, shape ``(n_nodes, 3)``, carrying the
            prescribed values on wall nodes and zeros on constrained outer
            nodes.

        Raises
        ------
        ValueError
            If the row count does not match the wall-node count.
        """
        wall_displacement = np.asarray(wall_displacement, dtype=float).reshape(
            (-1, 3)
        )
        if wall_displacement.shape[0] != self.partition.wall_nodes.size:
            raise ValueError("wall_displacement does not match the wall node count.")
        result = np.zeros_like(self.mesh.vertices)
        result[self.partition.wall_nodes] = wall_displacement
        result[self.partition.graph_free_xz, 0] = self.factor_xz.solve(
            -self.coupling_wall_xz @ wall_displacement[:, 0]
        )
        result[self.partition.graph_free_xz, 2] = self.factor_xz.solve(
            -self.coupling_wall_xz @ wall_displacement[:, 2]
        )
        result[self.partition.graph_free_y, 1] = self.factor_y.solve(
            -self.coupling_wall_y @ wall_displacement[:, 1]
        )
        return result


@dataclass
class ElasticVolumeSystem:
    """Factored coupled linear-tetrahedral elasticity system.

    Volume motion only, and distinct from the surface method. Unlike
    :class:`GraphVolumeSystem`, the three components are coupled, so one vector
    system is assembled over the flattened DOFs.

    Attributes
    ----------
    mesh
        Volume mesh the system was assembled on.
    partition
        Node and DOF partitions separating prescribed wall DOFs from free
        DOFs.
    factor
        Factored stiffness operator restricted to the free DOFs.
    coupling_wall
        Block coupling free DOF rows to the prescribed wall DOF columns.
    free_scatter
        Scatters the solved free-DOF vector back into the full DOF vector.
    wall_scatter
        Scatters the prescribed wall DOFs into the full DOF vector.
    assembly_seconds
        Wall-clock seconds spent assembling the stiffness operator.
    factorization_seconds
        Wall-clock seconds spent factorizing it.
    poisson_ratio
        Poisson's ratio of the fictitious elastic material.
    stiffening_exponent
        Exponent making small cells stiffer than large ones.
    volume_floor
        Lower bound on the cell volume used for stiffening.
    stiffness_nnz
        Number of stored nonzeros in the assembled stiffness matrix.
    method
        Identifier of this propagator, ``"elasticity"``.
    """

    mesh: TetraVolumeMesh
    partition: VolumeBoundaryPartition
    factor: SPDFactor
    coupling_wall: sp.csr_matrix
    free_scatter: sp.csr_matrix
    wall_scatter: sp.csr_matrix
    assembly_seconds: float
    factorization_seconds: float
    poisson_ratio: float
    stiffening_exponent: float
    volume_floor: float
    stiffness_nnz: int

    method: str = "elasticity"

    def evaluate(self, wall_positions) -> csdl.Variable:
        """Deform the volume differentiably from prescribed wall positions.

        The factorization is treated as a reference matrix and reused as
        assembled, so the map is linear in the wall displacement and the
        reverse pass goes through the same factor's transpose solve.

        Parameters
        ----------
        wall_positions
            Deformed wall-node positions of shape ``(n_wall, 3)``, ordered as
            ``partition.wall_nodes``.

        Returns
        -------
        csdl_alpha.Variable
            Deformed volume coordinates of shape ``(n_nodes, 3)``.
        """
        wall = self.partition.wall_nodes
        wall_displacement = wall_positions - self.mesh.vertices[wall]
        wall_vector = wall_displacement.reshape(
            (3 * int(wall.size), 1)
        )
        rhs = -csdl.sparse.matmat(self.coupling_wall, wall_vector)
        solved = SPDSolveOperation(self.factor).evaluate(rhs)
        full_displacement = (
            csdl.sparse.matmat(self.free_scatter, solved)
            + csdl.sparse.matmat(self.wall_scatter, wall_vector)
        )
        return csdl.Variable(value=self.mesh.vertices) + full_displacement.reshape(
            self.mesh.vertices.shape
        )

    def solve_numpy(self, wall_displacement: np.ndarray) -> np.ndarray:
        """Solve an incremental displacement with the current factored matrix.

        The eager NumPy counterpart of :meth:`evaluate`, used by the
        forward-only load-stepping path. It returns a displacement, not a
        position.

        Parameters
        ----------
        wall_displacement
            Incremental wall displacement reshaped to ``(n_wall, 3)``, ordered
            as ``partition.wall_nodes``.

        Returns
        -------
        numpy.ndarray
            Displacement of every node, shape ``(n_nodes, 3)``.

        Raises
        ------
        ValueError
            If the row count does not match the wall-node count.
        """
        wall_displacement = np.asarray(wall_displacement, dtype=float).reshape(
            (-1, 3)
        )
        if wall_displacement.shape[0] != self.partition.wall_nodes.size:
            raise ValueError("wall_displacement does not match the wall node count.")
        wall_vector = wall_displacement.reshape(-1)
        solved = self.factor.solve(-self.coupling_wall @ wall_vector)
        full = self.free_scatter @ solved + self.wall_scatter @ wall_vector
        return np.asarray(full, dtype=float).reshape((-1, 3))


@dataclass(frozen=True)
class VolumeQualityReport:
    """Fast topology-fixed tetrahedral deformation quality.

    The topology is held fixed. Percentile fields describe the **lower tail of
    the metric values**: ``p001`` is the 0.1st-percentile value and ``p01`` the
    1st-percentile value. They say how bad the metric gets at those quantiles,
    not how many cells sit near the worst value.

    Attributes
    ----------
    num_tetrahedra
        Number of tetrahedra tested, counting each decomposed pyramid cell.
    inverted_tetrahedra
        How many have a non-positive **relative** Jacobian, meaning the cell's
        orientation reversed with respect to its own baseline. This is not the
        same as a non-positive raw signed volume: a cell whose baseline
        determinant is already negative counts as inverted only if the
        deformed one no longer shares that sign.
    minimum_relative_jacobian
        Smallest deformed-to-baseline signed determinant ratio. One means
        undeformed; at or below zero means orientation reversal.
    relative_jacobian_p001, relative_jacobian_p01
        The 0.1st- and 1st-percentile values of that ratio.
    minimum_volume_ratio
        Smallest ratio of deformed to baseline cell volume, taken from absolute
        determinants and so insensitive to orientation.
    volume_ratio_p001, volume_ratio_p01
        The 0.1st- and 1st-percentile values of that ratio.
    minimum_mean_ratio
        Smallest mean-ratio shape metric. It is an **absolute** measure of the
        current cell's shape, computed from the deformed cell's own edges and
        volume: a regular tetrahedron scores one and a degenerate cell zero. It
        does not compare against the baseline shape, so a baseline cell that is
        already sliver-like scores low even when undeformed. Its sign follows
        the **raw** deformed determinant, whereas inversion elsewhere in this
        report is the deformed-to-baseline determinant ratio. The two agree only
        under a positive-baseline-orientation convention: given a
        negative-oriented baseline cell, an unchanged cell has a positive
        relative Jacobian yet a negative mean ratio.
    mean_ratio_p001, mean_ratio_p01
        The 0.1st- and 1st-percentile values of that metric.
    """

    num_tetrahedra: int
    inverted_tetrahedra: int
    minimum_relative_jacobian: float
    relative_jacobian_p001: float
    relative_jacobian_p01: float
    minimum_volume_ratio: float
    volume_ratio_p001: float
    volume_ratio_p01: float
    minimum_mean_ratio: float
    mean_ratio_p001: float
    mean_ratio_p01: float

    def as_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable scalars.

        Returns
        -------
        dict
            Every field keyed by its own name, with NumPy scalars converted to
            plain Python numbers.
        """
        return {
            key: _json_scalar(value)
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class LoadStepRecord:
    """Cost and quality for one forward-only volume load increment.

    Attributes
    ----------
    step
        Zero-based index of this increment.
    fraction
        Cumulative fraction of the total wall motion applied after it, where
        ``1.0`` is the full motion.
    assembly_seconds
        Wall-clock seconds spent reassembling the operator for this increment.
    factorization_seconds
        Wall-clock seconds spent refactorizing it.
    solve_seconds
        Wall-clock seconds spent solving it.
    quality
        Volume-quality report of the mesh after this increment.
    """

    step: int
    fraction: float
    assembly_seconds: float
    factorization_seconds: float
    solve_seconds: float
    quality: VolumeQualityReport

    def as_dict(self) -> dict[str, Any]:
        """Return the record as JSON-serializable values.

        Returns
        -------
        dict
            The timing and step fields as plain Python numbers, with
            ``"quality"`` holding the nested report dictionary.
        """
        return {
            "step": int(self.step),
            "fraction": float(self.fraction),
            "assembly_seconds": float(self.assembly_seconds),
            "factorization_seconds": float(self.factorization_seconds),
            "solve_seconds": float(self.solve_seconds),
            "quality": self.quality.as_dict(),
        }


@dataclass(frozen=True)
class LoadSteppedVolumeResult:
    """Final vertices and per-increment records of a load-stepped solve.

    Produced by the forward-only load-stepping path, which reassembles and
    refactorizes at each increment and therefore carries no derivative.

    Attributes
    ----------
    vertices
        Deformed volume coordinates after the final increment, shape
        ``(n_nodes, 3)``.
    records
        One :class:`LoadStepRecord` per increment, in the order applied.
    """

    vertices: np.ndarray
    records: tuple[LoadStepRecord, ...]


@dataclass(frozen=True)
class WallPositionHistory:
    """Validated aircraft-wall targets for replaying volume deformation.

    Attributes
    ----------
    positions
        One deformed wall-position array per recorded increment, each of shape
        ``(n_wall, 3)`` in wall-node order.
    fractions
        Cumulative load fraction of each entry, aligned with ``positions``.
    source_path
        File the history was read from.
    """

    positions: tuple[np.ndarray, ...]
    fractions: tuple[float, ...]
    source_path: Path


def read_gmsh22_volume(
    path: str | Path, *, validate: bool = True
) -> TetraVolumeMesh:
    """Read nodes, supported linear cells and physical tags.

    ``validate=False`` is reserved for diagnosing a rejected mesh that cannot
    pass the normal orientation/topology checks. Production callers should
    retain the default.

    Parameters
    ----------
    path
        ASCII Gmsh 2.2 volume mesh to read.
    validate
        Run the orientation and topology checks. Setting it to ``False`` loads
        a mesh that would otherwise be rejected, for diagnosis only.

    Returns
    -------
    TetraVolumeMesh
        Nodes, tetrahedra, boundary faces, and physical-group names. Quadrangle
        faces and pyramids are present only when the file contains them.

    Raises
    ------
    ValueError
        If the file is not ASCII Gmsh 2.2, lacks a required section, or fails a
        validation check while ``validate`` is set.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    physical_names: dict[tuple[int, int], str] = {}
    node_ids = None
    vertices = None
    node_id_to_index: dict[int, int] = {}
    triangles: list[list[int]] = []
    triangle_physical_ids: list[int] = []
    tetrahedra: list[list[int]] = []
    tetrahedron_physical_ids: list[int] = []
    quadrangles: list[list[int]] = []
    quadrangle_physical_ids: list[int] = []
    pyramids: list[list[int]] = []
    pyramid_physical_ids: list[int] = []

    with source.open("r", encoding="utf8") as stream:
        lines = iter(stream)
        for line in lines:
            marker = line.strip()
            if marker == "$MeshFormat":
                fields = next(lines).split()
                if not fields or fields[0] != "2.2" or fields[1] != "0":
                    raise ValueError(
                        "Volume motion currently requires ASCII Gmsh 2.2."
                    )
            elif marker == "$PhysicalNames":
                for _ in range(int(next(lines))):
                    fields = next(lines).strip().split(maxsplit=2)
                    physical_names[(int(fields[0]), int(fields[1]))] = (
                        fields[2].strip().strip('"')
                    )
            elif marker == "$Nodes":
                count = int(next(lines))
                node_ids = np.empty(count, dtype=np.int64)
                vertices = np.empty((count, 3), dtype=float)
                for row in range(count):
                    fields = next(lines).split()
                    node_id = int(fields[0])
                    node_ids[row] = node_id
                    vertices[row] = [float(value) for value in fields[1:4]]
                    node_id_to_index[node_id] = row
            elif marker == "$Elements":
                if vertices is None:
                    raise ValueError("$Elements appears before $Nodes.")
                for _ in range(int(next(lines))):
                    fields = next(lines).split()
                    element_type = int(fields[1])
                    num_tags = int(fields[2])
                    tags = [int(value) for value in fields[3 : 3 + num_tags]]
                    connectivity = [
                        node_id_to_index[int(value)]
                        for value in fields[3 + num_tags :]
                    ]
                    physical_id = tags[0] if tags else 0
                    if element_type == 2:
                        triangles.append(connectivity)
                        triangle_physical_ids.append(physical_id)
                    elif element_type == 3:
                        quadrangles.append(connectivity)
                        quadrangle_physical_ids.append(physical_id)
                    elif element_type == 4:
                        tetrahedra.append(connectivity)
                        tetrahedron_physical_ids.append(physical_id)
                    elif element_type == 7:
                        pyramids.append(connectivity)
                        pyramid_physical_ids.append(physical_id)
                    elif element_type in (5, 6):
                        raise ValueError(
                            f"{source} contains element type {element_type}; "
                            "only tetrahedra and pyramids are supported."
                        )

    if node_ids is None or vertices is None:
        raise ValueError(f"{source} does not contain nodes.")
    if not tetrahedra:
        raise ValueError(f"{source} does not contain linear tetrahedra.")
    mesh = TetraVolumeMesh(
        source_path=source,
        node_ids=node_ids,
        vertices=vertices,
        triangles=np.asarray(triangles, dtype=np.int64),
        triangle_physical_ids=np.asarray(
            triangle_physical_ids, dtype=np.int64
        ),
        tetrahedra=np.asarray(tetrahedra, dtype=np.int64),
        tetrahedron_physical_ids=np.asarray(
            tetrahedron_physical_ids, dtype=np.int64
        ),
        physical_names=physical_names,
        quadrangles=(
            np.asarray(quadrangles, dtype=np.int64)
            if quadrangles
            else np.zeros((0, 4), dtype=np.int64)
        ),
        quadrangle_physical_ids=np.asarray(
            quadrangle_physical_ids, dtype=np.int64
        ),
        pyramids=(
            np.asarray(pyramids, dtype=np.int64)
            if pyramids
            else np.zeros((0, 5), dtype=np.int64)
        ),
        pyramid_physical_ids=np.asarray(pyramid_physical_ids, dtype=np.int64),
    )
    if validate:
        _validate_volume_mesh(mesh)
    return mesh


def extract_aircraft_wall_mesh(
    mesh: TetraVolumeMesh,
    output_path: str | Path,
    mapping_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the exact aircraft boundary as a compact surface MSH and map.

    The written surface is renumbered compactly, so the companion map is what
    relates its nodes back to the volume mesh.

    Parameters
    ----------
    mesh
        Volume mesh whose aircraft wall patch is extracted.
    output_path
        Destination for the compact surface mesh.
    mapping_path
        Destination for the wall-to-volume node index map.

    Returns
    -------
    tuple
        The resolved ``(output_path, mapping_path)`` actually written.
    """
    output = Path(output_path).expanduser().resolve()
    mapping = (
        output.with_suffix(".volume_map.npz")
        if mapping_path is None
        else Path(mapping_path).expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)

    volume_nodes = mesh.aircraft_nodes
    volume_to_wall = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    volume_to_wall[volume_nodes] = np.arange(volume_nodes.size, dtype=np.int64)
    wall_triangles = volume_to_wall[mesh.aircraft_triangles]
    wall_quadrangles = volume_to_wall[mesh.aircraft_quadrangles]
    wall_vertices = mesh.vertices[volume_nodes]

    with output.open("w", encoding="utf8") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        stream.write('$PhysicalNames\n1\n2 1 "aircraft"\n$EndPhysicalNames\n')
        stream.write(f"$Nodes\n{wall_vertices.shape[0]}\n")
        for index, point in enumerate(wall_vertices, start=1):
            stream.write(
                f"{index} {point[0]:.17g} {point[1]:.17g} "
                f"{point[2]:.17g}\n"
            )
        stream.write("$EndNodes\n")
        total = int(wall_triangles.shape[0] + wall_quadrangles.shape[0])
        stream.write(f"$Elements\n{total}\n")
        # ``gamma_mdo.preprocessing.import_mesh`` expects one contiguous triangle
        # block followed by one contiguous quadrangle block.
        index = 0
        for element_type, block in (
            (2, wall_triangles),
            (3, wall_quadrangles),
        ):
            for cell in block:
                index += 1
                connectivity = " ".join(
                    str(int(node) + 1) for node in cell
                )
                stream.write(f"{index} {element_type} 2 1 1 {connectivity}\n")
        stream.write("$EndElements\n")

    np.savez(
        mapping,
        wall_to_volume=volume_nodes,
        baseline_wall_vertices=wall_vertices,
        surface_triangles=wall_triangles,
        surface_quadrangles=wall_quadrangles,
        volume_mesh_path=str(mesh.source_path),
        wall_mesh_path=str(output),
    )
    return output, mapping


def load_wall_to_volume_map(
    mapping_path: str | Path,
    mesh: TetraVolumeMesh,
    wall_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Load and validate the compact-wall to volume-node index map.

    Parameters
    ----------
    mapping_path
        Map file written by :func:`extract_aircraft_wall_mesh`.
    mesh
        Volume mesh the map must refer to; its wall nodes are the expected
        target.
    wall_vertices
        Compact wall vertices the map is checked against, so a mismatched
        surface is rejected rather than silently misapplied.

    Returns
    -------
    numpy.ndarray
        Volume-node index per compact wall node, shape ``(n_wall,)``.

    Raises
    ------
    ValueError
        If the map does not match this mesh's wall nodes or the supplied wall
        vertices.
    """
    payload = np.load(Path(mapping_path), allow_pickle=False)
    mapping = np.asarray(payload["wall_to_volume"], dtype=np.int64)
    baseline = np.asarray(payload["baseline_wall_vertices"], dtype=float)
    if mapping.size != mesh.aircraft_nodes.size:
        raise ValueError("Wall map and volume mesh have different wall sizes.")
    if not np.array_equal(mapping, mesh.aircraft_nodes):
        raise ValueError("Wall map does not match the volume aircraft nodes.")
    if not np.array_equal(baseline, mesh.vertices[mapping]):
        raise ValueError("Wall map baseline coordinates do not match the volume.")
    if wall_vertices is not None and not np.array_equal(
        np.asarray(wall_vertices, dtype=float), baseline
    ):
        maximum = float(
            np.max(np.abs(np.asarray(wall_vertices, dtype=float) - baseline))
        )
        raise ValueError(
            "Surface test mesh is not the exact volume wall; "
            f"maximum coordinate difference={maximum:.3e}."
        )
    return mapping


def write_wall_position_history(
    path: str | Path,
    mesh: TetraVolumeMesh,
    wall_history: list[np.ndarray] | tuple[np.ndarray, ...],
    fractions: list[float] | tuple[float, ...],
) -> Path:
    """Persist exact surface load-step targets for cheap volume-only replay.

    Parameters
    ----------
    path
        Destination file.
    mesh
        Volume mesh the targets belong to; identifies the baseline the history
        may be replayed against.
    wall_history
        One deformed wall-position array per increment, each ``(n_wall, 3)`` in
        wall-node order.
    fractions
        Cumulative load fraction of each increment, aligned with
        ``wall_history``.

    Returns
    -------
    pathlib.Path
        The resolved file written.
    """
    output = Path(path).expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("Wall-position history must use an .npz suffix.")
    targets, load_fractions = _validate_wall_history(
        mesh, wall_history, fractions
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        wall_positions=np.stack(targets, axis=0),
        load_fractions=np.asarray(load_fractions, dtype=float),
        wall_to_volume=mesh.aircraft_nodes,
        baseline_wall_vertices=mesh.vertices[mesh.aircraft_nodes],
        volume_mesh_path=np.asarray(str(mesh.source_path)),
    )
    return output


def read_wall_position_history(
    path: str | Path,
    mesh: TetraVolumeMesh,
) -> WallPositionHistory:
    """Load a history and reject use with a different baseline volume mesh.

    Parameters
    ----------
    path
        History file written by :func:`write_wall_position_history`.
    mesh
        Volume mesh the history must have been recorded against.

    Returns
    -------
    WallPositionHistory
        The stored positions and load fractions.

    Raises
    ------
    ValueError
        If the history does not correspond to this baseline mesh, so replaying
        it would apply targets to the wrong nodes.
    """
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        positions = np.asarray(payload["wall_positions"], dtype=float)
        fractions = np.asarray(payload["load_fractions"], dtype=float)
        wall_to_volume = np.asarray(payload["wall_to_volume"], dtype=np.int64)
        baseline_wall = np.asarray(
            payload["baseline_wall_vertices"], dtype=float
        )
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("wall_positions must have shape (steps, wall_nodes, 3).")
    if not np.array_equal(wall_to_volume, mesh.aircraft_nodes):
        raise ValueError("Wall history does not match this volume wall mapping.")
    if not np.array_equal(
        baseline_wall, mesh.vertices[mesh.aircraft_nodes]
    ):
        raise ValueError("Wall history baseline does not match the volume mesh.")
    targets, load_fractions = _validate_wall_history(
        mesh,
        tuple(positions[index] for index in range(positions.shape[0])),
        tuple(fractions.tolist()),
    )
    return WallPositionHistory(targets, load_fractions, source)


def write_deformed_gmsh22(
    mesh: TetraVolumeMesh,
    vertices: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Copy the source MSH while replacing only the node coordinates.

    Connectivity, physical groups, and every other section are copied through
    unchanged, so the output stays readable by the same tools.

    Parameters
    ----------
    mesh
        Mesh supplying the source file and its node ordering.
    vertices
        Replacement coordinates of shape ``(n_nodes, 3)``, in that same order.
    output_path
        Destination file.

    Returns
    -------
    pathlib.Path
        The resolved file written.
    """
    coordinates = np.asarray(vertices, dtype=float).reshape(mesh.vertices.shape)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Cannot write nonfinite volume coordinates.")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with mesh.source_path.open("r", encoding="utf8") as source:
        with output.open("w", encoding="utf8") as target:
            lines = iter(source)
            for line in lines:
                target.write(line)
                if line.strip() != "$Nodes":
                    continue
                count_line = next(lines)
                target.write(count_line)
                count = int(count_line)
                if count != coordinates.shape[0]:
                    raise ValueError("Source node count changed unexpectedly.")
                for row in range(count):
                    fields = next(lines).split()
                    point = coordinates[row]
                    target.write(
                        f"{fields[0]} {point[0]:.17g} {point[1]:.17g} "
                        f"{point[2]:.17g}\n"
                    )
    return output


def write_aircraft_wall_gmsh22(
    mesh: TetraVolumeMesh,
    wall_positions: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write one compact aircraft-wall target for inspection or reuse.

    Parameters
    ----------
    mesh
        Volume mesh supplying the wall connectivity.
    wall_positions
        Deformed wall positions of shape ``(n_wall, 3)`` in wall-node order.
    output_path
        Destination file.

    Returns
    -------
    pathlib.Path
        The resolved compact surface mesh written.
    """
    positions = np.asarray(wall_positions, dtype=float).reshape((-1, 3))
    wall_nodes = mesh.aircraft_nodes
    if positions.shape[0] != wall_nodes.size:
        raise ValueError("wall_positions does not match the aircraft wall.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Cannot write nonfinite wall coordinates.")
    volume_to_wall = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    volume_to_wall[wall_nodes] = np.arange(wall_nodes.size, dtype=np.int64)
    triangles = volume_to_wall[mesh.aircraft_triangles]
    quadrangles = volume_to_wall[mesh.aircraft_quadrangles]

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf8") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        stream.write('$PhysicalNames\n1\n2 1 "aircraft"\n$EndPhysicalNames\n')
        stream.write(f"$Nodes\n{positions.shape[0]}\n")
        for node_id, point in enumerate(positions, start=1):
            stream.write(
                f"{node_id} {point[0]:.17g} {point[1]:.17g} "
                f"{point[2]:.17g}\n"
            )
        stream.write("$EndNodes\n")
        total = int(triangles.shape[0] + quadrangles.shape[0])
        stream.write(f"$Elements\n{total}\n")
        element_id = 0
        for element_type, block in ((2, triangles), (3, quadrangles)):
            for cell in block:
                element_id += 1
                connectivity = " ".join(str(int(node) + 1) for node in cell)
                stream.write(
                    f"{element_id} {element_type} 2 1 1 {connectivity}\n"
                )
        stream.write("$EndElements\n")
    return output


def make_boundary_partition(mesh: TetraVolumeMesh) -> VolumeBoundaryPartition:
    """Create fixed/freedom sets with tangential symmetry-plane motion.

    Wall nodes are prescribed and inlet, outlet, and farfield nodes are fixed.
    Symmetry-plane nodes are free to slide within the plane but pinned across
    it, which is why the scalar graph systems use different free sets for the
    in-plane and normal components.

    Parameters
    ----------
    mesh
        Volume mesh whose patches define the partition.

    Returns
    -------
    VolumeBoundaryPartition
        Node sets and the flattened DOF index sets for the elasticity system.
    """
    wall = mesh.aircraft_nodes
    outer = mesh.fixed_outer_nodes
    symmetry = mesh.symmetry_nodes
    overlap_wall_outer = np.intersect1d(wall, outer)
    if overlap_wall_outer.size:
        raise ValueError("Aircraft and outer farfield node sets overlap.")

    symmetry_tangent = np.setdiff1d(symmetry, np.union1d(wall, outer))
    all_nodes = np.arange(mesh.vertices.shape[0], dtype=np.int64)
    interior = np.setdiff1d(all_nodes, mesh.boundary_nodes)
    graph_free_xz = np.sort(np.concatenate((interior, symmetry_tangent)))
    graph_free_y = interior

    elasticity_free_dofs = np.sort(
        np.concatenate(
            (
                _node_dofs(interior),
                3 * symmetry_tangent,
                3 * symmetry_tangent + 2,
            )
        )
    )
    wall_dofs = _node_dofs(wall)
    return VolumeBoundaryPartition(
        wall_nodes=wall,
        outer_nodes=outer,
        symmetry_tangent_nodes=symmetry_tangent,
        interior_nodes=interior,
        graph_free_xz=graph_free_xz,
        graph_free_y=graph_free_y,
        elasticity_free_dofs=elasticity_free_dofs,
        wall_dofs=wall_dofs,
    )


def assemble_graph_volume_system(
    mesh: TetraVolumeMesh,
    *,
    vertices: np.ndarray | None = None,
    stiffening_exponent: float = 0.0,
    volume_floor: float = 1.0e-15,
) -> GraphVolumeSystem:
    """Assemble and factor the tetrahedral all-edge graph Laplacian.

    Uses the assembly cell set, so each pyramid contributes the single
    better-conditioned split rather than both. Two scalar systems are factored:
    one shared by the x and z components, one for y.

    Parameters
    ----------
    mesh
        Volume mesh to assemble on.
    vertices
        Coordinates the operator is assembled about, allowing a load-stepped
        caller to reassemble at the current configuration rather than the
        baseline.
    stiffening_exponent
        Exponent making small cells stiffer than large ones. Zero gives uniform
        weights.
    volume_floor
        Lower bound on cell volume used for that weighting, guarding against a
        vanishing volume.

    Returns
    -------
    GraphVolumeSystem
        The factored systems with their coupling blocks and timings.
    """
    started = time.perf_counter()
    points = (
        mesh.vertices
        if vertices is None
        else np.asarray(vertices, dtype=float).reshape(mesh.vertices.shape)
    )
    exponent = float(stiffening_exponent)
    if exponent < 0.0:
        raise ValueError("stiffening_exponent must be nonnegative.")
    if volume_floor <= 0.0:
        raise ValueError("volume_floor must be positive.")
    partition = make_boundary_partition(mesh)

    tetrahedra = mesh.assembly_cells
    volumes = np.maximum(
        np.abs(_tetrahedron_determinants(points, tetrahedra)) / 6.0,
        float(volume_floor),
    )
    reference_volume = float(np.median(volumes))
    element_weights = (
        np.ones_like(volumes)
        if exponent == 0.0
        else (reference_volume / volumes) ** exponent
    )
    edges = np.concatenate(
        [
            tetrahedra[:, [i, j]]
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ],
        axis=0,
    )
    weights = np.tile(element_weights, 6)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((weights, weights))
    adjacency = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(points.shape[0], points.shape[0]),
    ).tocsr()
    adjacency.sum_duplicates()
    laplacian = (
        sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    ).tocsc()

    free_xz = partition.graph_free_xz
    free_y = partition.graph_free_y
    wall = partition.wall_nodes
    matrix_xz = laplacian[free_xz][:, free_xz].tocsc()
    matrix_y = laplacian[free_y][:, free_y].tocsc()
    coupling_xz = laplacian[free_xz][:, wall].tocsr()
    coupling_y = laplacian[free_y][:, wall].tocsr()
    assembly_seconds = time.perf_counter() - started

    factor_started = time.perf_counter()
    factor_xz = factorize_spd(matrix_xz)
    factor_y = factorize_spd(matrix_y)
    factorization_seconds = time.perf_counter() - factor_started
    return GraphVolumeSystem(
        mesh=_mesh_with_vertices(mesh, points),
        partition=partition,
        factor_xz=factor_xz,
        factor_y=factor_y,
        coupling_wall_xz=coupling_xz,
        coupling_wall_y=coupling_y,
        assembly_seconds=assembly_seconds,
        factorization_seconds=factorization_seconds,
        stiffening_exponent=exponent,
        volume_floor=float(volume_floor),
    )


def assemble_elastic_volume_system(
    mesh: TetraVolumeMesh,
    *,
    vertices: np.ndarray | None = None,
    poisson_ratio: float = 0.3,
    stiffening_exponent: float = 0.0,
    volume_floor: float = 1.0e-15,
) -> ElasticVolumeSystem:
    """Assemble and factor coupled linear elasticity on four-node tets.

    Uses the assembly cell set, so each pyramid contributes one split. Unlike
    the graph systems the three components are coupled, giving one vector
    system over the free DOFs.

    Parameters
    ----------
    mesh
        Volume mesh to assemble on.
    vertices
        Coordinates the stiffness is assembled about, allowing reassembly at
        the current configuration.
    poisson_ratio
        Poisson's ratio of the fictitious elastic material.
    stiffening_exponent
        Exponent making small cells stiffer than large ones.
    volume_floor
        Lower bound on cell volume used for that weighting.

    Returns
    -------
    ElasticVolumeSystem
        The factored system with its coupling and scatter blocks, timings, and
        stiffness sparsity count.
    """
    started = time.perf_counter()
    points = (
        mesh.vertices
        if vertices is None
        else np.asarray(vertices, dtype=float).reshape(mesh.vertices.shape)
    )
    nu = float(poisson_ratio)
    if not (-1.0 < nu < 0.5):
        raise ValueError("poisson_ratio must lie strictly between -1 and 0.5.")
    exponent = float(stiffening_exponent)
    if exponent < 0.0:
        raise ValueError("stiffening_exponent must be nonnegative.")
    partition = make_boundary_partition(mesh)

    stiffness = _assemble_tetrahedral_elasticity_bsr(
        points,
        mesh.assembly_cells,
        poisson_ratio=nu,
        stiffening_exponent=exponent,
        volume_floor=float(volume_floor),
    ).tocsr()
    free = partition.elasticity_free_dofs
    wall_dofs = partition.wall_dofs
    free_matrix = stiffness[free][:, free].tocsc()
    coupling_wall = stiffness[free][:, wall_dofs].tocsr()
    assembly_seconds = time.perf_counter() - started

    factor_started = time.perf_counter()
    factor = factorize_spd(free_matrix)
    factorization_seconds = time.perf_counter() - factor_started

    total_dofs = 3 * points.shape[0]
    free_scatter = sp.csr_matrix(
        (
            np.ones(free.size),
            (free, np.arange(free.size, dtype=np.int64)),
        ),
        shape=(total_dofs, free.size),
    )
    wall_scatter = sp.csr_matrix(
        (
            np.ones(wall_dofs.size),
            (wall_dofs, np.arange(wall_dofs.size, dtype=np.int64)),
        ),
        shape=(total_dofs, wall_dofs.size),
    )
    return ElasticVolumeSystem(
        mesh=_mesh_with_vertices(mesh, points),
        partition=partition,
        factor=factor,
        coupling_wall=coupling_wall,
        free_scatter=free_scatter,
        wall_scatter=wall_scatter,
        assembly_seconds=assembly_seconds,
        factorization_seconds=factorization_seconds,
        poisson_ratio=nu,
        stiffening_exponent=exponent,
        volume_floor=float(volume_floor),
        stiffness_nnz=int(stiffness.nnz),
    )


def run_forward_volume_load_steps(
    mesh: TetraVolumeMesh,
    wall_history: list[np.ndarray] | tuple[np.ndarray, ...],
    fractions: list[float] | tuple[float, ...],
    *,
    method: VolumeMethod,
    graph_stiffening_exponent: float = 0.0,
    elasticity_poisson_ratio: float = 0.3,
    elasticity_stiffening_exponent: float = 0.0,
    volume_floor: float = 1.0e-15,
) -> LoadSteppedVolumeResult:
    """Reassemble/refactor at each prescribed wall-position increment.

    Forward-only: the operator is reassembled and refactorized about the
    current configuration at every increment, so the sequence carries no
    derivative and is not the differentiable path.

    Parameters
    ----------
    mesh
        Volume mesh supplying the baseline configuration and connectivity.
    wall_history
        Deformed wall positions per increment, each ``(n_wall, 3)`` in
        wall-node order.
    fractions
        Cumulative load fraction of each increment, aligned with
        ``wall_history`` and validated against it.
    method
        Which propagator to use at each increment, the graph systems or the
        elasticity system.
    graph_stiffening_exponent
        Cell-size stiffening exponent for the graph method.
    elasticity_poisson_ratio
        Poisson's ratio for the elasticity method.
    elasticity_stiffening_exponent
        Cell-size stiffening exponent for the elasticity method.
    volume_floor
        Lower bound on cell volume used by either stiffening rule.

    Returns
    -------
    LoadSteppedVolumeResult
        Final coordinates and one record per increment, each carrying its
        timings and resulting volume quality.

    Raises
    ------
    ValueError
        If the history and fractions are inconsistent, or do not match this
        mesh's wall nodes.
    """
    targets, load_fractions = _validate_wall_history(
        mesh, wall_history, fractions
    )

    current = mesh.vertices.copy()
    current_wall = current[mesh.aircraft_nodes].copy()
    records: list[LoadStepRecord] = []
    baseline_vertices = mesh.vertices

    for step, (fraction, target) in enumerate(
        zip(load_fractions, targets), start=1
    ):
        current_mesh = _mesh_with_vertices(mesh, current)
        if method == "graph":
            system = assemble_graph_volume_system(
                current_mesh,
                stiffening_exponent=graph_stiffening_exponent,
                volume_floor=volume_floor,
            )
        elif method == "elasticity":
            system = assemble_elastic_volume_system(
                current_mesh,
                poisson_ratio=elasticity_poisson_ratio,
                stiffening_exponent=elasticity_stiffening_exponent,
                volume_floor=volume_floor,
            )
        else:
            raise ValueError(f"Unknown volume method {method!r}.")
        solve_started = time.perf_counter()
        increment = system.solve_numpy(target - current_wall)
        solve_seconds = time.perf_counter() - solve_started
        current = current + increment
        current_wall = target.copy()
        quality = evaluate_volume_quality(
            baseline_vertices, current, mesh.tetrahedra, pyramids=mesh.pyramids
        )
        records.append(
            LoadStepRecord(
                step=step,
                fraction=fraction,
                assembly_seconds=system.assembly_seconds,
                factorization_seconds=system.factorization_seconds,
                solve_seconds=solve_seconds,
                quality=quality,
            )
        )
    return LoadSteppedVolumeResult(current, tuple(records))


def _validate_wall_history(
    mesh: TetraVolumeMesh,
    wall_history: list[np.ndarray] | tuple[np.ndarray, ...],
    fractions: list[float] | tuple[float, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    targets = tuple(
        np.asarray(item, dtype=float).reshape((-1, 3))
        for item in wall_history
    )
    load_fractions = tuple(float(value) for value in fractions)
    if len(targets) != len(load_fractions) or not targets:
        raise ValueError("wall_history and fractions must be nonempty and align.")
    if any(item.shape[0] != mesh.aircraft_nodes.size for item in targets):
        raise ValueError("Every wall-history entry must match aircraft_nodes.")
    if any(not np.all(np.isfinite(item)) for item in targets):
        raise ValueError("Wall-position history contains nonfinite coordinates.")
    if any(
        right <= left
        for left, right in zip(load_fractions[:-1], load_fractions[1:])
    ):
        raise ValueError("Load fractions must be strictly increasing.")
    if load_fractions[0] <= 0.0 or load_fractions[-1] > 1.0:
        raise ValueError("Load fractions must lie in (0, 1].")
    return targets, load_fractions


def evaluate_volume_quality(
    baseline_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    tetrahedra: np.ndarray,
    pyramids: np.ndarray | None = None,
) -> VolumeQualityReport:
    """Evaluate signed-Jacobian, volume-ratio and mean-ratio metrics.

    When ``pyramids`` is given, each pyramid is tested under both base splits,
    so ``inverted_tetrahedra`` counts every decomposed cell whose orientation
    reversed relative to its baseline.

    Parameters
    ----------
    baseline_vertices
        Undeformed coordinates, shape ``(n_nodes, 3)``, forming the denominator
        of every ratio.
    deformed_vertices
        Deformed coordinates in the same node order and shape.
    tetrahedra
        Tetrahedral cells as node indices, shape ``(n_tet, 4)``.
    pyramids
        Pyramid cells, shape ``(n_pyr, 5)``. ``None`` or an empty array tests
        the tetrahedra alone.

    Returns
    -------
    VolumeQualityReport
        Cell count, inverted-cell count, and the minimum and low-percentile
        values of the relative Jacobian, volume ratio, and mean ratio.
    """
    baseline = np.asarray(baseline_vertices, dtype=float)
    deformed = np.asarray(deformed_vertices, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if pyramids is not None and np.asarray(pyramids).size:
        cells = _decompose_cells(
            baseline,
            cells,
            np.asarray(pyramids, dtype=np.int64),
            both_diagonals=True,
        )
    determinant_0 = _tetrahedron_determinants(baseline, cells)
    determinant_1 = _tetrahedron_determinants(deformed, cells)
    scale = np.maximum(np.abs(determinant_0), np.finfo(float).tiny)
    relative_jacobian = determinant_1 / determinant_0
    volume_ratio = np.abs(determinant_1) / scale
    mean_ratio = _tetrahedron_mean_ratio(deformed, cells, determinant_1)
    return VolumeQualityReport(
        num_tetrahedra=int(cells.shape[0]),
        inverted_tetrahedra=int(np.count_nonzero(relative_jacobian <= 0.0)),
        minimum_relative_jacobian=float(np.min(relative_jacobian)),
        relative_jacobian_p001=float(np.quantile(relative_jacobian, 0.001)),
        relative_jacobian_p01=float(np.quantile(relative_jacobian, 0.01)),
        minimum_volume_ratio=float(np.min(volume_ratio)),
        volume_ratio_p001=float(np.quantile(volume_ratio, 0.001)),
        volume_ratio_p01=float(np.quantile(volume_ratio, 0.01)),
        minimum_mean_ratio=float(np.min(mean_ratio)),
        mean_ratio_p001=float(np.quantile(mean_ratio, 0.001)),
        mean_ratio_p01=float(np.quantile(mean_ratio, 0.01)),
    )


def evaluate_gmsh_tetra_quality(
    mesh_path: str | Path,
) -> dict[str, dict[str, float | int]]:
    """Evaluate Gmsh's CFD-oriented metrics on all linear tetrahedra.

    Running this on the written result also verifies that the deformed output
    remains a readable volume mesh.

    Parameters
    ----------
    mesh_path
        Volume mesh file to evaluate. It is read back through Gmsh, so an
        unreadable or invalid file fails here.

    Returns
    -------
    dict
        Gmsh's CFD-oriented metrics over all linear tetrahedra, keyed by metric
        name.

    Raises
    ------
    ImportError
        If Gmsh is unavailable.
    """
    import gmsh

    path = Path(mesh_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    already_initialized = bool(gmsh.isInitialized())
    previous_model = gmsh.model.getCurrent() if already_initialized else ""
    if not already_initialized:
        gmsh.initialize()
    quality_model = f"gamma_mdo_volume_quality_{time.time_ns()}"
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(quality_model)
        gmsh.model.setCurrent(quality_model)
        gmsh.merge(str(path))
        tetrahedron_tags, _ = gmsh.model.mesh.getElementsByType(4)
        pyramid_tags, _ = gmsh.model.mesh.getElementsByType(7)
        tags = np.concatenate(
            (
                np.asarray(tetrahedron_tags, dtype=np.int64),
                np.asarray(pyramid_tags, dtype=np.int64),
            )
        )
        if tags.size == 0:
            raise ValueError(f"{path} contains no linear tetrahedra.")
        report: dict[str, dict[str, float | int]] = {}
        for metric in ("minSICN", "minSIGE", "gamma"):
            values = np.asarray(
                gmsh.model.mesh.getElementQualities(tags, metric),
                dtype=float,
            )
            report[metric] = {
                "minimum": float(np.min(values)),
                "p001": float(np.quantile(values, 0.001)),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "nonpositive": int(np.count_nonzero(values <= 0.0)),
            }
        return report
    finally:
        if already_initialized:
            if quality_model:
                gmsh.model.setCurrent(quality_model)
                gmsh.model.remove()
            if previous_model:
                gmsh.model.setCurrent(previous_model)
        else:
            gmsh.finalize()


def write_comparison_summary(
    path: str | Path, payload: dict[str, Any]
) -> Path:
    """Write a JSON comparison summary, creating parent directories.

    Parameters
    ----------
    path
        Destination file; ``~`` is expanded, the path resolved, and missing
        parent directories created. An existing file is overwritten.
    payload
        Values to serialize. They must already be JSON-serializable; convert
        NumPy scalars first, for example with
        :meth:`VolumeQualityReport.as_dict`.

    Returns
    -------
    pathlib.Path
        The resolved file written.
    """
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf8")
    return output


def _assemble_tetrahedral_elasticity_bsr(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    poisson_ratio: float,
    stiffening_exponent: float,
    volume_floor: float,
) -> sp.bsr_matrix:
    """Assemble isotropic CST stiffness into a 3x3 block sparse matrix."""
    points = np.asarray(vertices, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    num_nodes = points.shape[0]

    determinant = _tetrahedron_determinants(points, cells)
    absolute_volume = np.abs(determinant) / 6.0
    if np.any(absolute_volume <= volume_floor):
        count = int(np.count_nonzero(absolute_volume <= volume_floor))
        raise ValueError(
            f"Cannot assemble elasticity: {count} tetrahedra are degenerate."
        )
    matrix = np.ones((cells.shape[0], 4, 4), dtype=float)
    matrix[:, :, 1:] = points[cells]
    inverse = np.linalg.inv(matrix)
    gradients = np.transpose(inverse[:, 1:, :], (0, 2, 1))

    reference_volume = float(np.median(absolute_volume))
    modulus = (
        np.ones_like(absolute_volume)
        if stiffening_exponent == 0.0
        else (reference_volume / absolute_volume) ** stiffening_exponent
    )
    nu = poisson_ratio
    lame_lambda = modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lame_mu = modulus / (2.0 * (1.0 + nu))

    pair_rows = np.repeat(cells, 4, axis=1).astype(np.int32).ravel()
    pair_cols = np.tile(cells, (1, 4)).astype(np.int32).ravel()
    adjacency = sp.coo_matrix(
        (
            np.ones(pair_rows.size, dtype=np.int8),
            (pair_rows, pair_cols),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    data = np.zeros((adjacency.nnz, 3, 3), dtype=float)

    lookup = adjacency.copy()
    lookup.data = np.arange(adjacency.nnz, dtype=np.int64)
    identity = np.eye(3)
    for local_i in range(4):
        gradient_i = gradients[:, local_i, :]
        for local_j in range(4):
            gradient_j = gradients[:, local_j, :]
            block_indices = np.asarray(
                lookup[cells[:, local_i], cells[:, local_j]]
            ).reshape(-1)
            dot = np.einsum("ij,ij->i", gradient_i, gradient_j)
            blocks = absolute_volume[:, None, None] * (
                lame_lambda[:, None, None]
                * gradient_i[:, :, None]
                * gradient_j[:, None, :]
                + lame_mu[:, None, None]
                * gradient_j[:, :, None]
                * gradient_i[:, None, :]
                + lame_mu[:, None, None] * dot[:, None, None] * identity
            )
            np.add.at(data, block_indices, blocks)

    return sp.bsr_matrix(
        (data, adjacency.indices, adjacency.indptr),
        shape=(3 * num_nodes, 3 * num_nodes),
    )


def _unique_nodes(*blocks: np.ndarray) -> np.ndarray:
    """Sorted unique node indices over any number of connectivity blocks."""
    present = [block.ravel() for block in blocks if block.size]
    if not present:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(present))


# Gmsh orders a pyramid as four base nodes followed by the apex.  Splitting the
# base across 0-2 or across 1-3 gives these tetrahedra.
_PYRAMID_SPLIT_02 = ((0, 1, 2, 4), (0, 2, 3, 4))
_PYRAMID_SPLIT_13 = ((1, 2, 3, 4), (1, 3, 0, 4))


def _decompose_cells(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    pyramids: np.ndarray,
    *,
    both_diagonals: bool,
) -> np.ndarray:
    """Represent a hybrid mesh as tetrahedra only."""
    if pyramids.size == 0:
        return tetrahedra
    if both_diagonals:
        splits = _PYRAMID_SPLIT_02 + _PYRAMID_SPLIT_13
    else:
        points = np.asarray(vertices, dtype=float)
        corners = points[pyramids[:, :4]]
        first = np.linalg.norm(corners[:, 2] - corners[:, 0], axis=1)
        second = np.linalg.norm(corners[:, 3] - corners[:, 1], axis=1)
        use_02 = first <= second
        blocks = [tetrahedra]
        for group, split in ((use_02, _PYRAMID_SPLIT_02), (~use_02, _PYRAMID_SPLIT_13)):
            selected = pyramids[group]
            if selected.size == 0:
                continue
            blocks.extend(selected[:, list(corner)] for corner in split)
        return np.concatenate(blocks, axis=0)
    return np.concatenate(
        [tetrahedra] + [pyramids[:, list(corner)] for corner in splits],
        axis=0,
    )


def _tetrahedron_determinants(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    xyz = np.asarray(vertices, dtype=float)[np.asarray(tetrahedra, dtype=np.int64)]
    return np.einsum(
        "ij,ij->i",
        xyz[:, 1] - xyz[:, 0],
        np.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 0]),
    )


def _tetrahedron_mean_ratio(
    vertices: np.ndarray, tetrahedra: np.ndarray, determinant: np.ndarray
) -> np.ndarray:
    xyz = np.asarray(vertices, dtype=float)[np.asarray(tetrahedra, dtype=np.int64)]
    squared_edges = sum(
        np.einsum("ij,ij->i", xyz[:, i] - xyz[:, j], xyz[:, i] - xyz[:, j])
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    )
    volume = np.abs(determinant) / 6.0
    quality = (
        12.0 * np.power(3.0 * volume, 2.0 / 3.0)
        / np.maximum(squared_edges, np.finfo(float).tiny)
    )
    return np.where(determinant >= 0.0, quality, -quality)


def _nodes_for_physical_patch(
    mesh: TetraVolumeMesh, physical_id: int
) -> np.ndarray:
    physical_id = int(physical_id)
    return _unique_nodes(
        mesh.triangles[mesh.triangle_physical_ids == physical_id],
        mesh.quadrangles[mesh.quadrangle_physical_ids == physical_id],
    )


def _validate_volume_mesh(mesh: TetraVolumeMesh) -> None:
    required_surfaces = {PHYSICAL_AIRCRAFT, PHYSICAL_SYMMETRY}
    # Whether the outer boundary is split into inlet/outlet caps or written as
    # one far-field surface is a meshing choice; ``fixed_outer_nodes`` unions
    # all three either way, so only the union has to be non-empty.
    outer_surfaces = {PHYSICAL_INLET, PHYSICAL_OUTLET, PHYSICAL_FARFIELD}
    present = set(int(value) for value in np.unique(mesh.triangle_physical_ids))
    present.update(int(value) for value in np.unique(mesh.quadrangle_physical_ids))
    missing = required_surfaces - present
    if missing:
        raise ValueError(f"Volume mesh is missing physical surfaces {missing}.")
    if not (outer_surfaces & present):
        raise ValueError(
            "Volume mesh has no outer boundary; expected at least one of the "
            f"physical surfaces {sorted(outer_surfaces)}."
        )
    fluid_ids = set(int(value) for value in np.unique(mesh.tetrahedron_physical_ids))
    fluid_ids.update(int(value) for value in np.unique(mesh.pyramid_physical_ids))
    if PHYSICAL_FLUID not in fluid_ids:
        raise ValueError("Volume mesh is missing physical volume 'fluid'.")

    determinant = _tetrahedron_determinants(mesh.vertices, mesh.tetrahedra)
    if np.any(determinant == 0.0):
        raise ValueError("Volume mesh contains degenerate tetrahedra.")
    signs = np.sign(determinant)
    if np.any(signs != signs[0]):
        raise ValueError("Tetrahedron orientations are inconsistent.")
    if mesh.pyramids.size:
        # Each pyramid must be non-degenerate under either base split, and
        # wound the same way as the tetrahedra.
        pyramid_cells = _decompose_cells(
            mesh.vertices,
            np.zeros((0, 4), dtype=np.int64),
            mesh.pyramids,
            both_diagonals=True,
        )
        pyramid_determinant = _tetrahedron_determinants(
            mesh.vertices, pyramid_cells
        )
        if np.any(pyramid_determinant == 0.0):
            raise ValueError("Volume mesh contains degenerate pyramids.")
        if np.any(np.sign(pyramid_determinant) != signs[0]):
            raise ValueError(
                "Pyramid orientations are inconsistent with the tetrahedra."
            )


def _mesh_with_vertices(
    mesh: TetraVolumeMesh, vertices: np.ndarray
) -> TetraVolumeMesh:
    return TetraVolumeMesh(
        source_path=mesh.source_path,
        node_ids=mesh.node_ids,
        vertices=np.asarray(vertices, dtype=float),
        triangles=mesh.triangles,
        triangle_physical_ids=mesh.triangle_physical_ids,
        tetrahedra=mesh.tetrahedra,
        tetrahedron_physical_ids=mesh.tetrahedron_physical_ids,
        physical_names=mesh.physical_names,
        quadrangles=mesh.quadrangles,
        quadrangle_physical_ids=mesh.quadrangle_physical_ids,
        pyramids=mesh.pyramids,
        pyramid_physical_ids=mesh.pyramid_physical_ids,
    )


def _node_dofs(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64)
    return (3 * nodes[:, None] + np.arange(3, dtype=np.int64)).ravel()


def _row_slice(rows: np.ndarray):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, :]
    return csdl.slice[rows.tolist(), :]


def _component_slice(rows: np.ndarray, component: int):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 1:
        row = int(rows[0])
        return csdl.slice[row : row + 1, component : component + 1]
    return csdl.slice[rows.tolist(), component : component + 1]


def _json_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


__all__ = [
    "ElasticVolumeSystem",
    "GraphVolumeSystem",
    "LoadStepRecord",
    "LoadSteppedVolumeResult",
    "TetraVolumeMesh",
    "VolumeBoundaryPartition",
    "VolumeQualityReport",
    "assemble_elastic_volume_system",
    "assemble_graph_volume_system",
    "evaluate_gmsh_tetra_quality",
    "evaluate_volume_quality",
    "extract_aircraft_wall_mesh",
    "load_wall_to_volume_map",
    "make_boundary_partition",
    "read_gmsh22_volume",
    "run_forward_volume_load_steps",
    "write_comparison_summary",
    "write_deformed_gmsh22",
]
