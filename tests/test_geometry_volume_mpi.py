"""Dependency-free / mock-MPI tests for the rank-0 geometry-to-volume coupling.

These cover the communication protocol, cache behavior, matrix-free VJP, and
the forward-only FD guard without requiring mpi4py, DAFoam, PETSc, or an
OpenFOAM case.  Multi-rank behavior is exercised with small scripted
communicators; the heavy DAFoam-coupled validation ladder runs on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

import csdl_alpha as csdl

from bsm3.core.boundary_surface_movement.geometry_volume_mpi import (
    SerialComm,
    assemble_local_gradient,
    broadcast_array,
    extract_local_coordinates,
    reduce_gradient,
    resolve_comm,
    run_on_root,
    verify_replicated_values,
    verify_seed_ownership,
)
from bsm3.core.boundary_surface_movement.geometry_volume_backend import (
    CSDLRecorderBackend,
)
from bsm3.core.boundary_surface_movement.geometry_volume_operation import (
    GeometryVolumeOperation,
    GeometryVolumeVJP,
)
from bsm3.core.boundary_surface_movement.forward_only_fd_checker import (
    ADJOINT_MARKER,
    DerivativeComparison,
    check_derivatives_forward_first,
    degree_radian_relative_error,
    guard_against_adjoint_output,
)


# ---------------------------------------------------------------------------
# Scripted communicators
# ---------------------------------------------------------------------------
class ScriptedComm:
    """Single-threaded stand-in for one rank of a small MPI world.

    A shared ``world`` dict carries the root's broadcast payloads between the
    per-rank ``compute`` calls, which the test invokes in rank order.
    """

    def __init__(self, rank: int, size: int, world: dict):
        self.rank = rank
        self.size = size
        self.world = world

    def bcast(self, obj, root=0):
        if self.rank == root:
            self.world["bcast"] = obj
            return obj
        return self.world.get("bcast")

    def Bcast(self, buffer, root=0):
        if self.rank == root:
            self.world["buffer"] = np.array(buffer, copy=True)
        else:
            buffer[...] = self.world["buffer"]

    def Barrier(self):
        return None


class CollectiveComm:
    """Mock one rank of a small world for the validation collectives.

    ``bcast`` returns the stored rank-0 reference; ``allgather`` returns this
    rank's contribution in its slot and the caller-specified messages from the
    other ranks, which is exactly what a real ``allgather`` would produce.  This
    lets a single-threaded test prove the *synchronous* raise: a rank whose own
    value is fine still raises when a peer's is not.
    """

    def __init__(self, rank, size, *, reference=None, other_messages=None):
        self.rank = rank
        self.size = size
        self._reference = reference
        self._others = other_messages or {}

    def bcast(self, obj, root=0):
        return self._reference if self.rank != root else obj

    def allgather(self, obj):
        return [obj if r == self.rank else self._others.get(r) for r in range(self.size)]


class SummingComm:
    """Mock whose reductions sum a fixed list of per-rank contributions."""

    def __init__(self, rank: int, size: int, contributions: list[np.ndarray]):
        self.rank = rank
        self.size = size
        self._contributions = contributions

    def Allreduce(self, sendbuf, recvbuf, op=None):
        recvbuf[...] = np.sum(self._contributions, axis=0)

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        if self.rank == root:
            recvbuf[...] = np.sum(self._contributions, axis=0)


# ---------------------------------------------------------------------------
# Backends used by the tests
# ---------------------------------------------------------------------------
def _smooth_model_builder(num_points: int = 6):
    base = (np.linspace(0.0, 1.0, num_points).reshape(num_points, 1)
            * np.ones((1, 3)))

    def build(recorder):
        a = csdl.Variable(name="a", value=0.3)
        b = csdl.Variable(name="b", value=-0.7)
        base_v = csdl.Variable(value=base)
        volume = base_v + a * base_v**2 + b * csdl.sin(base_v)
        return {"a": a, "b": b}, volume

    return build


class CountingBackend(CSDLRecorderBackend):
    """CSDL backend that records how many forward solves it runs."""

    def __init__(self, *args, **kwargs):
        self.forward_calls = 0
        super().__init__(*args, **kwargs)

    def forward(self, design_variables):
        self.forward_calls += 1
        return super().forward(design_variables)


class ScriptedBackend:
    """Analytic backend that also records forward/VJP invocation counts."""

    design_variable_names = ("a", "b")
    output_shape = (4, 3)

    def __init__(self):
        self.forward_calls = 0
        self.vjp_calls = 0

    def forward(self, design_variables):
        self.forward_calls += 1
        a = float(np.asarray(design_variables["a"]).reshape(-1)[0])
        b = float(np.asarray(design_variables["b"]).reshape(-1)[0])
        grid = np.arange(12, dtype=float).reshape(self.output_shape)
        return grid + a * grid**2 + b * np.cos(grid)

    def compute_vjp(self, design_variables, volume_seed):
        self.vjp_calls += 1
        a = float(np.asarray(design_variables["a"]).reshape(-1)[0])
        b = float(np.asarray(design_variables["b"]).reshape(-1)[0])
        grid = np.arange(12, dtype=float).reshape(self.output_shape)
        seed = np.asarray(volume_seed, dtype=float).reshape(self.output_shape)
        return {
            "a": np.array(np.sum(seed * grid**2)),
            "b": np.array(-np.sum(seed * np.sin(grid))),
        }


# ---------------------------------------------------------------------------
# MPI utilities
# ---------------------------------------------------------------------------
def test_serial_comm_is_identity():
    comm = resolve_comm(None)
    assert isinstance(comm, SerialComm)
    assert comm.rank == 0 and comm.size == 1
    array = np.arange(6.0).reshape(2, 3)
    out = broadcast_array(comm, array, shape=(2, 3))
    np.testing.assert_array_equal(out, array)


def test_run_on_root_propagates_exception_symmetrically():
    world = {}
    root = ScriptedComm(0, 2, world)
    nonroot = ScriptedComm(1, 2, world)

    def boom():
        raise ValueError("root failed")

    with pytest.raises(RuntimeError, match="root failed"):
        run_on_root(root, boom)
    # Non-root sees the same broadcast error string and raises identically.
    with pytest.raises(RuntimeError, match="root failed"):
        run_on_root(nonroot, lambda: None)


def test_verify_replicated_values_raises_synchronously_on_mismatch():
    reference = np.array([[1.0, 2.0, 3.0]])
    # Rank 1's local value differs -> it raises.
    comm1 = CollectiveComm(1, 2, reference=reference, other_messages={0: None})
    with pytest.raises(RuntimeError, match="not identical"):
        verify_replicated_values(comm1, reference + 1.0, name="dv")
    # Matching values on all ranks -> no raise.
    comm0 = CollectiveComm(0, 2, reference=reference, other_messages={1: None})
    verify_replicated_values(comm0, reference, name="dv")


def test_verify_replicated_values_is_collective_not_one_sided():
    # The crux of the fix: a rank whose OWN value matches must still raise if a
    # peer reports a mismatch, so no rank is left in a later collective.
    reference = np.zeros((1, 3))
    peer_failure = {1: "rank 1 differs by 1.000e+00 (tolerance 0.000e+00)"}
    comm0 = CollectiveComm(0, 2, reference=reference, other_messages=peer_failure)
    with pytest.raises(RuntimeError, match="rank 1 differs"):
        verify_replicated_values(comm0, reference, name="dv")


def test_seed_ownership_root_mode_rejects_nonzero_nonroot_seed():
    comm = CollectiveComm(1, 2, reference=None, other_messages={0: None})
    with pytest.raises(RuntimeError, match="must be zero on non-root"):
        verify_seed_ownership(comm, np.ones((4, 3)), ownership="root", name="seed")


def test_seed_ownership_root_mode_accepts_zero_nonroot_seed():
    comm = CollectiveComm(1, 2, reference=None, other_messages={0: None})
    verify_seed_ownership(comm, np.zeros((4, 3)), ownership="root", name="seed")


def test_seed_ownership_root_mode_ignores_root_seed():
    # Rank 0 owns the assembled cotangent; its own seed is not checked for zero.
    comm = CollectiveComm(0, 2, reference=None, other_messages={1: None})
    verify_seed_ownership(comm, np.ones((4, 3)), ownership="root", name="seed")


def test_seed_ownership_rejects_invalid_mode():
    with pytest.raises(ValueError, match="replicated"):
        verify_seed_ownership(SerialComm(), np.zeros((4, 3)), ownership="bogus")


def test_geometry_volume_vjp_enforces_root_seed_ownership():
    # A non-root rank whose seed is not zero (root ownership) must fail before
    # the VJP touches any broadcast collective.
    comm = CollectiveComm(1, 2, reference=None, other_messages={0: None})
    vjp = GeometryVolumeVJP(
        None, comm, output_shape=(4, 3),
        design_variable_specs={"a": (1,)}, seed_ownership="root",
    )
    inputs = {"a": np.array([0.3]), "d_volume_coordinates": np.ones((4, 3))}
    with pytest.raises(RuntimeError, match="must be zero on non-root"):
        vjp.compute(inputs, {})


def test_duplicate_processor_boundary_points_sum():
    # Local rows 0 and 2 are a duplicated processor-boundary point -> global 3.
    local_gradient = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.0, 3.0], [0.0, 0.0, 1.0]]
    )
    local_to_global = np.array([3, 1, 3, 0])
    assembled = assemble_local_gradient(local_gradient, local_to_global, 4)
    np.testing.assert_array_equal(assembled[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(assembled[1], [0.0, 2.0, 0.0])
    np.testing.assert_array_equal(assembled[2], [0.0, 0.0, 0.0])
    # Global point 3 receives the sum of local rows 0 and 2.
    np.testing.assert_array_equal(assembled[3], [1.5, 0.0, 3.0])


def test_forward_scatter_and_reverse_add_are_transposes():
    # The forward global->local scatter S (extract_local_coordinates) and the
    # reverse local->global scatter-add S^T (assemble_local_gradient) must satisfy
    # the transpose identity <S x, y> == <x, S^T y>, INCLUDING duplicated
    # processor-boundary points. Overwrite-forward / duplicate-reverse would fail.
    rng = np.random.default_rng(7)
    num_global = 5
    # local_to_global with a duplicated global id (3 appears twice) and an
    # unreferenced global id (2 appears zero times).
    local_to_global = np.array([3, 1, 3, 0, 4])
    x = rng.standard_normal((num_global, 3))
    y = rng.standard_normal((local_to_global.size, 3))

    s_x = extract_local_coordinates(x, local_to_global)
    st_y = assemble_local_gradient(y, local_to_global, num_global)

    left = float(np.sum(s_x * y))          # <S x, y>
    right = float(np.sum(x * st_y))        # <x, S^T y>
    assert abs(left - right) <= 1.0e-12 * (abs(left) + 1.0)


def test_forward_scatter_matches_dafoam_backend_indexing():
    # extract_local_coordinates reproduces the exact indexing PYDAFoamBackend
    # uses internally (global_coordinates[local_to_global]).
    x = np.arange(15.0).reshape(5, 3)
    local_to_global = np.array([4, 0, 2, 2])
    np.testing.assert_array_equal(
        extract_local_coordinates(x, local_to_global),
        x[local_to_global],
    )


def test_allreduce_and_root_reduce_are_equivalent_on_root():
    contributions = [
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([[0.5, 0.0, -1.0], [2.0, -2.0, 1.0]]),
        np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    expected = np.sum(contributions, axis=0)
    for rank in range(3):
        comm = SummingComm(rank, 3, contributions)
        replicated = reduce_gradient(
            comm, contributions[rank], ownership="replicated"
        )
        root = reduce_gradient(comm, contributions[rank], ownership="root")
        np.testing.assert_allclose(replicated, expected)
        if rank == 0:
            np.testing.assert_allclose(root, expected)
        else:
            np.testing.assert_allclose(root, np.zeros_like(expected))


# ---------------------------------------------------------------------------
# Rank-0-only forward + broadcast invariance
# ---------------------------------------------------------------------------
def test_forward_runs_only_on_root_and_broadcasts_identically():
    reference_backend = ScriptedBackend()
    design = {"a": np.array(0.4), "b": np.array(-1.1)}
    expected = reference_backend.forward(design)

    world = {}
    specs = {"a": (), "b": ()}
    root_backend = ScriptedBackend()
    root_op = GeometryVolumeOperation(
        root_backend, ScriptedComm(0, 2, world),
        output_shape=(4, 3), design_variable_specs=specs,
    )
    nonroot_op = GeometryVolumeOperation(
        None, ScriptedComm(1, 2, world),
        output_shape=(4, 3), design_variable_specs=specs,
    )

    root_outputs: dict = {}
    root_op.compute(design, root_outputs)
    nonroot_outputs: dict = {}
    nonroot_op.compute(design, nonroot_outputs)

    np.testing.assert_allclose(root_outputs["volume_coordinates"], expected)
    # Rank invariance: the non-root output is bit-identical to the root's.
    np.testing.assert_array_equal(
        nonroot_outputs["volume_coordinates"],
        root_outputs["volume_coordinates"],
    )
    # The heavy backend ran once, on the root only.
    assert root_backend.forward_calls == 1


def test_geometry_volume_op_vjp_through_full_csdl_graph():
    # Mirror the driver's shape path: scalar design variables are shape (1,), and
    # the GeometryVolumeVJP cotangents must match that through CSDL's reverse.
    backend = CSDLRecorderBackend(_smooth_model_builder(), build_eagerly=True)
    specs = backend.design_variable_shapes()
    assert all(shape == (1,) for shape in specs.values())

    rng = np.random.default_rng(3)
    seed = rng.standard_normal(backend.output_shape)
    design = {"a": np.array([0.3]), "b": np.array([-0.7])}
    backend_vjp = backend.compute_vjp(design, seed)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    variables = {
        "a": csdl.Variable(name="a", value=0.3),
        "b": csdl.Variable(name="b", value=-0.7),
    }
    op = GeometryVolumeOperation(
        backend, SerialComm(), output_shape=backend.output_shape,
        design_variable_specs=specs,
    )
    volume = op.evaluate(variables)
    contraction = csdl.sum(volume * csdl.Variable(value=seed))
    derivatives = csdl.derivative([contraction], [variables["a"], variables["b"]])
    recorder.stop()

    np.testing.assert_allclose(
        np.asarray(derivatives[contraction, variables["a"]].value).reshape(-1),
        np.asarray(backend_vjp["a"]).reshape(-1),
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(derivatives[contraction, variables["b"]].value).reshape(-1),
        np.asarray(backend_vjp["b"]).reshape(-1),
        rtol=1e-9, atol=1e-12,
    )


def test_root_requires_a_backend():
    with pytest.raises(ValueError, match="root rank requires"):
        GeometryVolumeOperation(
            None, SerialComm(), output_shape=(4, 3),
            design_variable_specs={"a": ()},
        )


# ---------------------------------------------------------------------------
# CSDL backend: matrix-free VJP, cache reuse and invalidation
# ---------------------------------------------------------------------------
def test_matrix_free_vjp_matches_central_finite_difference():
    backend = CSDLRecorderBackend(_smooth_model_builder(), build_eagerly=True)
    design = {"a": np.array(0.3), "b": np.array(-0.7)}
    rng = np.random.default_rng(1)
    seed = rng.standard_normal(backend.output_shape)
    delta = {"a": np.array(rng.standard_normal()), "b": np.array(rng.standard_normal())}

    step = 1.0e-6
    plus = backend.forward({k: design[k] + step * delta[k] for k in design})
    minus = backend.forward({k: design[k] - step * delta[k] for k in design})
    directional_fd = float(np.sum(seed * (plus - minus) / (2.0 * step)))

    vjp = backend.compute_vjp(design, seed)
    directional_adjoint = float(sum(np.sum(vjp[k] * delta[k]) for k in design))

    assert abs(directional_fd - directional_adjoint) <= 1.0e-6 * (
        abs(directional_fd) + 1.0e-12
    )


def test_forward_cache_is_reused_and_invalidated():
    backend = CountingBackend(_smooth_model_builder(), build_eagerly=True)
    design = {"a": np.array(0.3), "b": np.array(-0.7)}
    seed = np.ones(backend.output_shape)

    backend.forward(design)
    calls_after_forward = backend.forward_calls
    # VJP at the same design point must not rerun the forward solve.
    backend.compute_vjp(design, seed)
    assert backend.forward_calls == calls_after_forward
    # A different design point must invalidate the cache and rerun the forward.
    backend.compute_vjp({"a": np.array(0.31), "b": np.array(-0.7)}, seed)
    assert backend.forward_calls == calls_after_forward + 1


# ---------------------------------------------------------------------------
# Forward-only FD guard and checker
# ---------------------------------------------------------------------------
def test_guard_raises_when_adjoint_marker_appears():
    # DAFoam/PETSc emit this marker at the C/file-descriptor level, so the test
    # writes directly to fd 1 rather than through Python's print().
    import os

    with pytest.raises(RuntimeError, match="Adjoint solve detected"):
        with guard_against_adjoint_output():
            os.write(1, f"iteration 3 {ADJOINT_MARKER} residual 1e-6\n".encode())


def test_guard_allows_clean_forward_output():
    import os

    with guard_against_adjoint_output():
        os.write(1, b"primal residual 1e-8 converged\n")


def test_forward_first_checker_runs_fd_before_analytical():
    backend = ScriptedBackend()
    baseline = {"a": np.array(0.4), "b": np.array(-1.1)}
    order = []

    def forward_fn(design_point):
        order.append("forward")
        coords = backend.forward(design_point)
        # A single scalar aerodynamic-like output for the check.
        return {"CD": float(np.sum(coords**2))}

    def analytical_fn():
        order.append("analytical")
        # Exact gradient of sum(coords^2) wrt a, b at the baseline.
        coords = backend.forward(baseline)
        grid = np.arange(12, dtype=float).reshape(backend.output_shape)
        d_da = float(np.sum(2.0 * coords * grid**2))
        d_db = float(np.sum(2.0 * coords * np.cos(grid)))
        return {("CD", "a"): np.array(d_da), ("CD", "b"): np.array(d_db)}

    comparison = check_derivatives_forward_first(
        forward_fn,
        analytical_fn,
        baseline,
        output_names=("CD",),
        design_variable_specs={"a": (), "b": ()},
        etas=(1.0e-3, 1.0e-4, 1.0e-5),
        print_results=False,
    )
    # All forward evaluations precede the single analytical evaluation.
    assert order.count("analytical") == 1
    assert order.index("analytical") == len(order) - 1
    best = comparison.best()
    assert best[("CD", "a")][1] < 1.0e-4
    assert best[("CD", "b")][1] < 1.0e-4


def test_degree_radian_convention_is_verified():
    grad_rad = np.array([1.5])
    grad_deg = (np.pi / 180.0) * grad_rad
    assert degree_radian_relative_error(grad_deg, grad_rad) < 1.0e-14
    wrong = grad_rad.copy()
    assert degree_radian_relative_error(wrong, grad_rad) > 1.0e-1


def test_derivative_comparison_best_step_is_none_without_a_recorded_error():
    """Return ``(None, inf)`` when an analytical key has no finite error."""
    import math
    import typing

    key = ("CD", "wing_area")
    comparison = DerivativeComparison(
        analytical={key: np.array([1.25])},
        fd_by_eta={},
    )
    # No step was swept, so nothing populated relative_error_by_eta.
    assert comparison.relative_error_by_eta == {}

    best = comparison.best()
    assert set(best) == {key}
    best_eta, best_error = best[key]
    assert best_eta is None
    assert math.isinf(best_error)

    hints = typing.get_type_hints(DerivativeComparison.best)
    assert hints["return"] == dict[tuple[str, str], tuple[float | None, float]]


def test_derivative_comparison_best_step_is_a_float_when_errors_exist():
    """Keep the populated-comparison selection behavior unchanged."""
    key = ("CD", "wing_area")
    comparison = DerivativeComparison(
        analytical={key: np.array([1.0])},
        fd_by_eta={1e-2: {key: np.array([2.0])}, 1e-4: {key: np.array([1.0])}},
        relative_error_by_eta={1e-2: {key: 0.5}, 1e-4: {key: 1e-9}},
    )

    best_eta, best_error = comparison.best()[key]
    assert best_eta == pytest.approx(1e-4)
    assert best_error == pytest.approx(1e-9)


def test_mesh_motion_volume_backend_calls_the_current_pipeline_api(monkeypatch):
    """Execute the backend build path against a current-signature fake pipeline.

    This guards the API boundary M1.1 broke: the backend must call
    ``run_mesh_motion`` with ``input_files`` and ``geometry``, not the removed
    ``model_files``/``geometry_parameterization`` spellings. It patches the
    pipeline function the backend imports, so it needs no CAD asset, volume
    mesh, MPI, or DAFoam.
    """
    import csdl_alpha as csdl

    from bsm3.core.boundary_surface_movement import mesh_motion_pipeline
    from bsm3.core.boundary_surface_movement.geometry_volume_backend import (
        MeshMotionVolumeBackend,
    )

    seen = {}

    class _FakeResult:
        def __init__(self, volume):
            self.volume_coordinates = {"elasticity": volume}

    def _fake_run_mesh_motion(
        *,
        recorder,
        input_files,
        geometry,
        config,
        aerodynamic_analysis=None,
        aerodynamic_volume_method="elasticity",
    ):
        seen.update(
            recorder=recorder,
            input_files=input_files,
            geometry=geometry,
            config=config,
            aerodynamic_analysis=aerodynamic_analysis,
            aerodynamic_volume_method=aerodynamic_volume_method,
        )
        # A minimal (num_points, 3) volume output driven by the design variable.
        design = geometry["shift"]
        base = csdl.Variable(value=np.zeros((4, 3)))
        return _FakeResult(base + design)

    monkeypatch.setattr(
        mesh_motion_pipeline, "run_mesh_motion", _fake_run_mesh_motion
    )

    sentinel_files = object()
    sentinel_config = object()

    def _factory(variables):
        # The backend hands its own CSDL variables to the factory.
        assert set(variables) == {"shift"}
        return {"shift": variables["shift"]}

    backend = MeshMotionVolumeBackend(
        input_files=sentinel_files,
        geometry_values={"shift": 0.25},
        pipeline_config=sentinel_config,
        parameterization_factory=_factory,
        aerodynamic_volume_method="elasticity",
    )

    coordinates = backend.forward({"shift": 0.5})

    # The fake received the current keyword names and the caller's objects.
    assert seen["input_files"] is sentinel_files
    assert seen["config"] is sentinel_config
    assert seen["geometry"] == {"shift": seen["geometry"]["shift"]}
    assert seen["aerodynamic_analysis"] is None
    assert seen["aerodynamic_volume_method"] == "elasticity"
    assert isinstance(seen["recorder"], csdl.Recorder)

    # The selected volume output came back and tracks the design variable.
    assert coordinates.shape == (4, 3)
    np.testing.assert_allclose(coordinates, np.full((4, 3), 0.5))
    assert backend.output_shape == (4, 3)
    assert backend.design_variable_names == ("shift",)


def test_mesh_motion_volume_backend_rejects_the_removed_keyword():
    """Refuse the pre-M1.1 ``model_files`` spelling; this is a clean break."""
    from bsm3.core.boundary_surface_movement.geometry_volume_backend import (
        MeshMotionVolumeBackend,
    )

    with pytest.raises(TypeError, match="model_files"):
        MeshMotionVolumeBackend(
            model_files=object(),
            geometry_values={"shift": 0.0},
            pipeline_config=object(),
            parameterization_factory=lambda variables: variables,
        )
