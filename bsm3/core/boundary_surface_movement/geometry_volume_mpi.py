"""MPI communication helpers for the rank-0 geometry-to-volume operation.

The geometry, projection, surface-motion, and volume-motion pipeline is
expensive and is executed only on MPI rank 0.  Its deformed global volume
coordinates are then broadcast so that the distributed DAFoam backend on every
rank can extract its local OpenFOAM partition.  This module centralizes the
small set of collectives that the forward and reverse operations need, with two
properties that matter for correctness and testability:

* ``mpi4py`` is imported lazily.  Importing BSM3, running the dependency-free
  unit tests, and single-process execution therefore do not require an MPI
  build.  When ``mpi4py`` is unavailable (or no communicator is supplied), a
  :class:`SerialComm` provides the identity behavior of a one-rank world.
* Every collective is wrapped so that a rank-0 exception is broadcast *before*
  any bulk data transfer.  This prevents the classic deadlock where rank 0
  raises while the other ranks block inside a matching collective.

The typed array collectives (:func:`broadcast_array`, :func:`reduce_gradient`)
use the buffer protocol (``Bcast``/``Reduce``/``Allreduce``) rather than the
pickled object collectives, as required for the large coordinate arrays.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Communicator resolution and serial fallback
# ---------------------------------------------------------------------------
class SerialComm:
    """A minimal single-rank stand-in for an MPI communicator.

    Only the subset of the ``mpi4py`` interface used by the geometry-to-volume
    operation is implemented, with the semantics of a one-process world: every
    broadcast is the identity, every reduction returns its own contribution, and
    every gather/scatter is a one-element list.  It lets the forward/reverse
    machinery and its tests run unchanged without an MPI build.
    """

    def __init__(self, rank: int = 0, size: int = 1):
        if size != 1 or rank != 0:
            raise ValueError(
                "SerialComm models a single-rank world; use a real MPI "
                "communicator for multi-rank execution."
            )
        self.rank = 0
        self.size = 1

    # Pickled object collectives ------------------------------------------------
    def bcast(self, obj: Any, root: int = 0) -> Any:
        """Return ``obj`` unchanged, the identity broadcast of a one-rank world.

        Parameters
        ----------
        obj
            Object the single rank contributes. It is returned by identity, not
            copied or pickled, so callers that mutate the result also mutate
            the input.
        root
            Accepted for signature compatibility with ``mpi4py`` and ignored;
            the only rank is always the root.

        Returns
        -------
        Any
            The same object that was passed in.
        """
        return obj

    def gather(self, obj: Any, root: int = 0) -> list[Any]:
        """Return this rank's contribution as a one-element list.

        Parameters
        ----------
        obj
            Object contributed by the single rank. It is placed in the list by
            identity, not copied.
        root
            Accepted for signature compatibility and ignored.

        Returns
        -------
        list
            A new list of length one holding ``obj``.
        """
        return [obj]

    def scatter(self, sequence: Optional[list[Any]], root: int = 0) -> Any:
        """Return the first element of ``sequence``, this rank's only share.

        Parameters
        ----------
        sequence
            Per-rank shares. In a one-rank world only the first element is
            used; ``None`` yields ``None`` rather than raising, so a non-root
            call pattern still works.
        root
            Accepted for signature compatibility and ignored.

        Returns
        -------
        Any
            ``sequence[0]`` by identity, or ``None`` when ``sequence`` is
            ``None``.
        """
        if sequence is None:
            return None
        return sequence[0]

    def allgather(self, obj: Any) -> list[Any]:
        """Return this rank's contribution as a one-element list.

        Parameters
        ----------
        obj
            Object contributed by the single rank, placed in the list by
            identity rather than copied.

        Returns
        -------
        list
            A new list of length one holding ``obj``.
        """
        return [obj]

    # Typed buffer collectives --------------------------------------------------
    def Bcast(self, buffer: np.ndarray, root: int = 0) -> None:
        """Do nothing: a one-rank broadcast leaves the buffer already correct.

        Parameters
        ----------
        buffer
            Array that would receive the broadcast. It is **not** read or
            written, because the single rank already holds the data.
        root
            Accepted for signature compatibility and ignored.

        Returns
        -------
        None
            Matching the ``mpi4py`` buffer-collective convention.
        """
        return None

    def Reduce(self, sendbuf: np.ndarray, recvbuf: np.ndarray, op=None, root: int = 0) -> None:
        """Copy ``sendbuf`` into ``recvbuf`` in place, the one-rank reduction.

        With a single contribution any associative reduction returns that
        contribution, so no arithmetic is performed.

        Parameters
        ----------
        sendbuf
            This rank's contribution. Read only.
        recvbuf
            Destination, written **in place** via ``recvbuf[...]``. It must
            already have a broadcast-compatible shape.
        op
            Reduction operation, accepted for signature compatibility and
            ignored because a single contribution needs none.
        root
            Accepted for signature compatibility and ignored.

        Returns
        -------
        None
            The result is written into ``recvbuf``.
        """
        recvbuf[...] = np.asarray(sendbuf)

    def Allreduce(self, sendbuf: np.ndarray, recvbuf: np.ndarray, op=None) -> None:
        """Copy ``sendbuf`` into ``recvbuf`` in place, the one-rank all-reduce.

        Identical to :meth:`Reduce` here, since the only rank is also the root
        and therefore already owns the result.

        Parameters
        ----------
        sendbuf
            This rank's contribution. Read only.
        recvbuf
            Destination, written **in place** via ``recvbuf[...]``.
        op
            Reduction operation, accepted for signature compatibility and
            ignored.

        Returns
        -------
        None
            The result is written into ``recvbuf``.
        """
        recvbuf[...] = np.asarray(sendbuf)

    def Barrier(self) -> None:
        """Do nothing: a single rank is always already synchronized.

        Returns
        -------
        None
        """
        return None


def resolve_comm(comm: Any = None) -> Any:
    """Return a usable communicator.

    If ``comm`` is provided it is returned unchanged.  Otherwise the real
    ``MPI.COMM_WORLD`` is used when ``mpi4py`` is importable, and a
    :class:`SerialComm` is returned when it is not.

    Parameters
    ----------
    comm
        Communicator to use. Any non-``None`` value is returned as given and is
        not validated, so a mock communicator is accepted.

    Returns
    -------
    Any
        ``comm`` when supplied, otherwise ``MPI.COMM_WORLD`` or a fresh
        :class:`SerialComm`.
    """
    if comm is not None:
        return comm
    try:
        from mpi4py import MPI
    except ImportError:
        return SerialComm()
    return MPI.COMM_WORLD


def comm_rank(comm: Any) -> int:
    """Return the rank of ``comm``, defaulting to 0.

    Parameters
    ----------
    comm
        Communicator. An object without a ``rank`` attribute is treated as a
        one-rank world, so mock communicators need not define one.

    Returns
    -------
    int
        ``comm.rank`` as an ``int``, or ``0`` when the attribute is absent.
    """
    return int(getattr(comm, "rank", 0))


def comm_size(comm: Any) -> int:
    """Return the size of ``comm``, defaulting to 1.

    Parameters
    ----------
    comm
        Communicator. An object without a ``size`` attribute is treated as a
        one-rank world.

    Returns
    -------
    int
        ``comm.size`` as an ``int``, or ``1`` when the attribute is absent.
    """
    return int(getattr(comm, "size", 1))


def is_root(comm: Any, root: int = 0) -> bool:
    """Report whether this rank is the designated root.

    Parameters
    ----------
    comm
        Communicator whose rank is compared.
    root
        Rank treated as the root.

    Returns
    -------
    bool
        ``True`` when this rank equals ``root``. Always ``True`` for a
        communicator that reports no rank.
    """
    return comm_rank(comm) == root


# ---------------------------------------------------------------------------
# Collective exception propagation
# ---------------------------------------------------------------------------
def run_on_root(
    comm: Any,
    function: Callable[[], Any],
    *,
    root: int = 0,
) -> Any:
    """Run ``function`` on ``root`` and propagate any exception to all ranks.

    The root rank runs ``function`` inside a try/except and reduces the outcome
    to an error string.  That string is broadcast (pickled ``bcast`` of a small
    object) to every rank *before* any bulk data movement, so a root failure
    raises :class:`RuntimeError` symmetrically on all ranks instead of leaving
    non-root ranks blocked in a later collective.  Non-root ranks always return
    ``None``.

    This is collective: every rank must call it, because every rank
    participates in the error ``bcast``.

    Parameters
    ----------
    comm
        Communicator all ranks participate in.
    function
        Zero-argument callable executed **only on the root rank**. Its
        exceptions are caught and reduced to a ``"TypeName: message"`` string.
    root
        Rank that runs ``function``.

    Returns
    -------
    Any
        ``function``'s return value on the root rank, and ``None`` on every
        other rank.

    Raises
    ------
    RuntimeError
        On **every** rank when the root call raised, carrying the root's
        exception type and message. The original traceback is not preserved on
        non-root ranks.
    """
    error_message: Optional[str] = None
    result: Any = None
    if is_root(comm, root):
        try:
            result = function()
        except Exception as error:  # noqa: BLE001 - re-raised on all ranks
            error_message = f"{type(error).__name__}: {error}"
    error_message = comm.bcast(error_message, root=root)
    if error_message is not None:
        raise RuntimeError(error_message)
    return result


# ---------------------------------------------------------------------------
# Typed array collectives
# ---------------------------------------------------------------------------
def broadcast_array(
    comm: Any,
    array: Optional[np.ndarray],
    *,
    shape: tuple[int, ...],
    dtype=np.float64,
    root: int = 0,
) -> np.ndarray:
    """Broadcast a dense, contiguous array from ``root`` to every rank.

    On the root rank ``array`` must already hold the data with the declared
    ``shape``/``dtype``; on non-root ranks it may be ``None`` and is allocated
    here.  A typed ``MPI.Bcast`` is used (not the pickled ``bcast``) so the
    large coordinate array moves through the buffer protocol.

    This is collective: every rank must call it. The transfer is skipped
    entirely in a one-rank world.

    Parameters
    ----------
    comm
        Communicator all ranks participate in.
    array
        Source data on the root rank, required there and validated against
        ``shape``. Ignored on non-root ranks, where ``None`` is expected and a
        receive buffer is allocated instead.
    shape
        Declared shape of the broadcast buffer, identical on every rank.
    dtype
        Element type of the buffer, identical on every rank.
    root
        Rank holding the source data.

    Returns
    -------
    numpy.ndarray
        A contiguous array of ``shape`` and ``dtype``. On the root rank this is
        a contiguous copy of ``array`` rather than ``array`` itself.

    Raises
    ------
    ValueError
        On the root rank when ``array`` is ``None``, or when its shape does not
        match ``shape``.
    """
    if is_root(comm, root):
        if array is None:
            raise ValueError("Root rank must supply the array to broadcast.")
        buffer = np.ascontiguousarray(np.asarray(array, dtype=dtype))
        if buffer.shape != tuple(shape):
            raise ValueError(
                f"Root array shape {buffer.shape} does not match the declared "
                f"broadcast shape {tuple(shape)}."
            )
    else:
        buffer = np.empty(shape, dtype=dtype)
    if comm_size(comm) > 1:
        comm.Bcast(buffer, root=root)
    return buffer


def extract_local_coordinates(
    global_coordinates: np.ndarray,
    local_to_global: np.ndarray,
) -> np.ndarray:
    """Forward scatter: pick each rank's local points from the global array.

    ``local = global[local_to_global]``.  Processor-boundary points may appear
    on several ranks; this is the forward operator ``S`` whose exact transpose
    ``S^T`` is :func:`assemble_local_gradient` (scatter-add).  The pair must
    satisfy ``<S x, ybar> == <x, S^T ybar>``; see
    ``test_forward_scatter_and_reverse_add_are_transposes``.

    Purely local: it performs no communication and may be called on one rank.

    Parameters
    ----------
    global_coordinates
        Dense global-order points of shape ``(n_global, 3)``.
    local_to_global
        Index of each local point in the global array. Entries may repeat,
        which is how processor-boundary points appear on several ranks.

    Returns
    -------
    numpy.ndarray
        Local points of shape ``(n_local, 3)``, where ``n_local`` is the size
        of ``local_to_global``. A new array, not a view.

    Raises
    ------
    ValueError
        If ``global_coordinates`` is not two-dimensional with three columns.
    """
    global_array = np.asarray(global_coordinates, dtype=np.float64)
    if global_array.ndim != 2 or global_array.shape[1] != 3:
        raise ValueError("global_coordinates must have shape (n_global, 3).")
    return global_array[np.asarray(local_to_global)]


def assemble_local_gradient(
    local_gradient: np.ndarray,
    local_to_global: np.ndarray,
    num_global_points: int,
) -> np.ndarray:
    """Scatter-add a local per-point gradient into a dense global-order buffer.

    This is the exact transpose ``S^T`` of :func:`extract_local_coordinates`.
    ``np.add.at`` is required (not fancy-index assignment) because
    processor-boundary points are duplicated across partitions and legitimately
    map several local rows to the same global point; their contributions must
    add rather than overwrite.  Forward overwrite and reverse duplication would
    *not* be a transpose pair.

    Purely local: it performs no communication. Summing the per-rank results is
    the separate job of :func:`reduce_gradient`.

    Parameters
    ----------
    local_gradient
        Per-local-point cotangent of shape ``(n_local, 3)``.
    local_to_global
        Global index of each local row, the same mapping used in the forward
        direction. Repeated entries accumulate.
    num_global_points
        Number of rows in the returned global buffer.

    Returns
    -------
    numpy.ndarray
        Dense global-order gradient of shape ``(num_global_points, 3)``,
        zero-filled where this rank owns no contribution.

    Raises
    ------
    ValueError
        If ``local_gradient`` is not two-dimensional with three columns, or its
        row count does not match ``local_to_global``.
    """
    local = np.asarray(local_gradient, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError("local_gradient must have shape (n_local, 3).")
    if local.shape[0] != np.asarray(local_to_global).size:
        raise ValueError(
            "local_gradient rows must match the length of local_to_global."
        )
    global_gradient = np.zeros((int(num_global_points), 3), dtype=np.float64)
    np.add.at(global_gradient, np.asarray(local_to_global), local)
    return global_gradient


def reduce_gradient(
    comm: Any,
    local_global_gradient: np.ndarray,
    *,
    ownership: str = "root",
    root: int = 0,
) -> np.ndarray:
    """Sum per-rank dense global-order gradients across the communicator.

    ``ownership='root'`` uses ``MPI.Reduce`` so only ``root`` owns the assembled
    result (non-root ranks receive zeros); ``ownership='replicated'`` uses
    ``MPI.Allreduce`` so every rank holds an identical copy.  The two are
    numerically equivalent on the root rank and are validated against each other
    during the first MPI validation stage.

    This is collective: every rank must call it with the same ``ownership``. In
    a one-rank world no collective runs and a copy of the contribution is
    returned regardless of ``ownership``.

    Parameters
    ----------
    comm
        Communicator all ranks participate in.
    local_global_gradient
        This rank's dense global-order contribution, shape ``(n_global, 3)``,
        typically from :func:`assemble_local_gradient`. Read only; a contiguous
        copy is made.
    ownership
        ``"root"`` reduces to ``root`` alone; ``"replicated"`` gives every rank
        an identical copy.
    root
        Destination rank when ``ownership="root"``. Ignored for
        ``"replicated"``.

    Returns
    -------
    numpy.ndarray
        The summed gradient with the shape of the contribution. Under
        ``"root"`` this is the assembled sum on the root rank and an array of
        zeros on every other rank.

    Raises
    ------
    ValueError
        If ``ownership`` is neither ``"root"`` nor ``"replicated"``.

    Notes
    -----
    ``mpi4py`` is imported lazily to obtain ``MPI.SUM``. When it is absent the
    operation is passed as ``None``, which the mock communicators used by the
    dependency-free tests ignore.
    """
    if ownership not in ("root", "replicated"):
        raise ValueError("ownership must be 'root' or 'replicated'.")
    contribution = np.ascontiguousarray(
        np.asarray(local_global_gradient, dtype=np.float64)
    )
    if comm_size(comm) == 1:
        return contribution.copy()

    # Real MPI needs a concrete reduction op; mock communicators used in the
    # dependency-free tests ignore it, so tolerate mpi4py being absent.
    try:
        from mpi4py import MPI

        sum_op = MPI.SUM
    except ImportError:
        sum_op = None

    reduced = np.zeros_like(contribution)
    if ownership == "replicated":
        comm.Allreduce(contribution, reduced, op=sum_op)
        return reduced
    comm.Reduce(contribution, reduced, op=sum_op, root=root)
    if is_root(comm, root):
        return reduced
    return np.zeros_like(contribution)


def _raise_if_any_rank_failed(
    comm: Any,
    message: Optional[str],
    *,
    context: str,
) -> None:
    """Raise on every rank, collectively, if *any* rank reports a message.

    Each rank contributes ``message`` (``None`` on success); an ``allgather``
    makes the combined outcome known to everyone, so validation never raises on
    a single rank while the others block in a later collective.
    """
    if comm_size(comm) == 1:
        if message is not None:
            raise RuntimeError(f"{context}: {message}")
        return
    messages = comm.allgather(message)
    offenders = [m for m in messages if m is not None]
    if offenders:
        raise RuntimeError(f"{context}: " + "; ".join(offenders))


def verify_replicated_values(
    comm: Any,
    array: np.ndarray,
    *,
    name: str,
    absolute_tolerance: float = 0.0,
) -> None:
    """Assert, collectively, that ``array`` is identical on every rank.

    Used in debug mode to catch silent divergence of the replicated design
    variables before rank 0 uses its own copy.  Every rank computes its own
    mismatch against the rank-0 reference and the outcome is reduced with
    :func:`_raise_if_any_rank_failed`, so the error is raised synchronously on
    all ranks.  A no-op in a one-rank world with matching values.

    This is collective: every rank must call it, because every rank
    participates in the reference broadcast and the outcome ``allgather``.

    Parameters
    ----------
    comm
        Communicator all ranks participate in.
    array
        This rank's copy of the value that is meant to be replicated. Compared
        against the rank-0 copy by shape and then by largest absolute
        difference.
    name
        Label for the array, used in the error message only.
    absolute_tolerance
        Largest absolute difference accepted. The default of ``0.0`` demands a
        zero numeric difference between finite values, which is not the same as
        bit-for-bit identity: ``+0.0`` and ``-0.0`` compare equal. The check
        uses the largest absolute difference, and because a comparison against
        ``NaN`` is false, an array containing ``NaN`` is **not** rejected here.

    Returns
    -------
    None
        Returns normally when every rank agrees.

    Raises
    ------
    RuntimeError
        On **every** rank when any rank's shape or values differ, naming the
        offending ranks.
    """
    if comm_size(comm) == 1:
        return
    reference = comm.bcast(
        np.asarray(array, dtype=np.float64).copy() if is_root(comm) else None,
        root=0,
    )
    local = np.asarray(array, dtype=np.float64)
    message: Optional[str] = None
    if local.shape != reference.shape:
        message = (
            f"rank {comm_rank(comm)} shape {local.shape} != {reference.shape}"
        )
    else:
        worst = float(np.max(np.abs(local - reference))) if local.size else 0.0
        if worst > absolute_tolerance:
            message = (
                f"rank {comm_rank(comm)} differs by {worst:.3e} "
                f"(tolerance {absolute_tolerance:.3e})"
            )
    _raise_if_any_rank_failed(
        comm, message, context=f"Replicated array {name!r} is not identical across ranks"
    )


def verify_seed_ownership(
    comm: Any,
    seed: np.ndarray,
    *,
    ownership: str,
    name: str = "seed",
    absolute_tolerance: float = 0.0,
) -> None:
    """Enforce the cotangent-seed ownership contract collectively.

    ``replicated`` requires the seed to be identical on every rank;
    ``root`` requires the seed to be exactly zero on every non-root rank (only
    rank 0 owns the assembled cotangent).  Both checks are reduced across ranks
    so a violation raises synchronously everywhere.

    This is collective: every rank must call it with the same ``ownership``. It
    returns immediately in a one-rank world, after validating ``ownership``.

    Parameters
    ----------
    comm
        Communicator all ranks participate in.
    seed
        This rank's cotangent seed, checked against the declared ownership.
    ownership
        ``"replicated"`` defers to :func:`verify_replicated_values`;
        ``"root"`` requires the seed to vanish on every non-root rank.
    name
        Label for the seed, used in the error message only.
    absolute_tolerance
        Largest absolute deviation accepted, whether from the rank-0 reference
        or from zero. The default of ``0.0`` demands a zero numeric difference
        between finite values rather than bit-for-bit identity, and neither
        this check nor the replicated one rejects a seed containing ``NaN``.

    Returns
    -------
    None
        Returns normally when the ownership contract holds.

    Raises
    ------
    ValueError
        If ``ownership`` is neither ``"replicated"`` nor ``"root"``. Raised
        before any collective, so it is safe in a one-rank world.
    RuntimeError
        On **every** rank when the contract is violated.
    """
    if ownership not in ("replicated", "root"):
        raise ValueError("ownership must be 'replicated' or 'root'.")
    if comm_size(comm) == 1:
        return
    if ownership == "replicated":
        verify_replicated_values(
            comm, seed, name=name, absolute_tolerance=absolute_tolerance
        )
        return
    message: Optional[str] = None
    if not is_root(comm):
        local = np.asarray(seed, dtype=np.float64)
        worst = float(np.max(np.abs(local))) if local.size else 0.0
        if worst > absolute_tolerance:
            message = f"rank {comm_rank(comm)} nonzero seed (max {worst:.3e})"
    _raise_if_any_rank_failed(
        comm, message, context=f"Root-owned {name!r} must be zero on non-root ranks"
    )


__all__ = [
    "SerialComm",
    "resolve_comm",
    "comm_rank",
    "comm_size",
    "is_root",
    "run_on_root",
    "broadcast_array",
    "extract_local_coordinates",
    "assemble_local_gradient",
    "reduce_gradient",
    "verify_replicated_values",
    "verify_seed_ownership",
]
