"""CSDL custom operation exposing the rank-0 geometry-to-volume map.

``GeometryVolumeOperation`` presents the whole geometry parameterization,
surface deformation, and volume deformation to the outer CSDL graph as one
explicit operation ``d -> X`` whose forward runs only on MPI rank 0 and whose
result is broadcast to every rank.  ``GeometryVolumeVJP`` is its matrix-free
reverse: rank 0 evaluates ``dbar = (dX/dd)^T Xbar`` and broadcasts the small
design-variable cotangent vector.  The distributed DAFoam operation downstream
continues to own the partitioned OpenFOAM mesh and extracts its local volume
coordinates from the broadcast global array.

The operation follows the same custom-VJP contract as ``DAFoamAnalysisOperation``
(``CustomExplicitOperationBeta`` + ``declare_vjp_function``), so the two compose
cleanly into ``d -> X -> (CL, CD)`` in a single recorder.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import csdl_alpha as csdl

from gamma_mdo.core.boundary_surface_movement.geometry_volume_mpi import (
    broadcast_array,
    comm_rank,
    comm_size,
    is_root,
    resolve_comm,
    run_on_root,
    verify_replicated_values,
    verify_seed_ownership,
)


def _ordered_specs(
    design_variable_specs: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (str(name), tuple(int(dim) for dim in shape))
        for name, shape in design_variable_specs.items()
    )


class GeometryVolumeOperation(csdl.experimental.CustomExplicitOperationBeta):
    """Deform the volume mesh on rank 0 and broadcast the global coordinates.

    Parameters
    ----------
    backend
        Live :class:`GeometryVolumeBackend` on the root rank; ``None`` on
        non-root ranks (which only participate in the collectives).
    comm
        MPI communicator; a serial communicator is resolved when omitted.
    output_shape
        Global ``(num_points, 3)`` coordinate shape, known on every rank.
    design_variable_specs
        Ordered mapping of design-variable name to shape, identical on every
        rank so all ranks declare the same inputs in the same order.
    debug
        Verify that the replicated design variables agree across ranks before
        rank 0 uses its own copy.
    seed_ownership
        Contract the incoming volume-coordinate cotangent must satisfy in the
        reverse pass: ``replicated`` (every rank holds the identical global
        cotangent, matching a DAFoam ``Allreduce``) or ``root`` (only rank 0
        holds it, non-root seeds are zero, matching a DAFoam ``Reduce``).  Must
        agree with the DAFoam ``volume_gradient_ownership``.
    """

    def __init__(
        self,
        backend: Any,
        comm: Any = None,
        *,
        output_shape: tuple[int, int],
        design_variable_specs: Mapping[str, tuple[int, ...]],
        debug: bool = False,
        seed_ownership: str = "replicated",
    ):
        super().__init__()
        self.comm = resolve_comm(comm)
        self.backend = backend
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.specs = _ordered_specs(design_variable_specs)
        self.design_variable_names = tuple(name for name, _ in self.specs)
        self.debug = bool(debug)
        if seed_ownership not in ("replicated", "root"):
            raise ValueError(
                "seed_ownership must be 'replicated' or 'root'; got "
                f"{seed_ownership!r}."
            )
        self.seed_ownership = str(seed_ownership)
        if is_root(self.comm) and backend is None:
            raise ValueError("The root rank requires a live backend.")

    def evaluate(self, design_variables: Mapping[str, csdl.Variable]):
        """Declare the design-variable inputs and the volume-coordinate output.

        Also registers :class:`GeometryVolumeVJP` as the reverse function,
        forwarding the backend, communicator, shapes, and ownership settings.

        Parameters
        ----------
        design_variables
            Mapping of design-variable name to CSDL variable. It must contain
            every name declared at construction; extra entries are ignored.
            Inputs are declared in the constructor's order, not the mapping's,
            so every rank declares them identically.

        Returns
        -------
        csdl_alpha.Variable
            A single ``"volume_coordinates"`` variable of the configured
            ``(num_points, 3)`` global shape. This is one variable, not a
            mapping.

        Raises
        ------
        KeyError
            If any declared design variable is absent from
            ``design_variables``, naming the missing entries.
        """
        missing = set(self.design_variable_names).difference(design_variables)
        if missing:
            raise KeyError(
                "GeometryVolumeOperation is missing design variables: "
                + ", ".join(sorted(missing))
            )
        for name, _shape in self.specs:
            self.declare_input(name, design_variables[name])
        volume_coordinates = self.create_output(
            "volume_coordinates", self.output_shape
        )
        self.declare_vjp_function(
            GeometryVolumeVJP,
            backend=self.backend,
            comm=self.comm,
            output_shape=self.output_shape,
            design_variable_specs=dict(self.specs),
            debug=self.debug,
            seed_ownership=self.seed_ownership,
        )
        return volume_coordinates

    def compute(self, inputs, outputs):
        """Deform the mesh on the root rank and broadcast the coordinates.

        Collective: every rank must reach this call, because all ranks
        participate in the root-error broadcast and the coordinate broadcast.

        Parameters
        ----------
        inputs
            Mapping holding every declared design variable, read as float
            arrays. Under ``debug`` they are first checked to be identical
            across ranks.
        outputs
            Output buffer written in place with ``"volume_coordinates"``, the
            broadcast global array, identical on every rank.

        Returns
        -------
        None
            The result is written into ``outputs``.

        Raises
        ------
        RuntimeError
            On every rank when the root-rank forward evaluation raised.
        """
        design_variables = {
            name: np.asarray(inputs[name], dtype=float)
            for name, _shape in self.specs
        }
        if self.debug:
            for name in self.design_variable_names:
                verify_replicated_values(
                    self.comm, design_variables[name], name=name
                )

        coordinates = run_on_root(
            self.comm,
            lambda: self._root_forward(design_variables),
        )
        coordinates = broadcast_array(
            self.comm,
            coordinates if is_root(self.comm) else None,
            shape=self.output_shape,
            dtype=np.float64,
        )
        outputs["volume_coordinates"] = coordinates

    def _root_forward(
        self, design_variables: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        coordinates = np.asarray(
            self.backend.forward(design_variables), dtype=np.float64
        )
        if coordinates.shape != self.output_shape:
            raise ValueError(
                f"Backend returned coordinates with shape {coordinates.shape}; "
                f"expected {self.output_shape}."
            )
        if not np.all(np.isfinite(coordinates)):
            raise FloatingPointError(
                "Backend produced nonfinite volume coordinates."
            )
        return coordinates


class GeometryVolumeVJP(csdl.experimental.CustomExplicitOperationBeta):
    """Matrix-free reverse of :class:`GeometryVolumeOperation`.

    Rank 0 receives the assembled global volume-coordinate cotangent, evaluates
    the mesh-motion VJP, and broadcasts the small concatenated design-variable
    cotangent vector.  Every rank then writes the identical per-design-variable
    cotangent, so the replicated outer graph stays consistent across ranks.
    """

    def __init__(
        self,
        backend: Any,
        comm: Any = None,
        *,
        output_shape: tuple[int, int],
        design_variable_specs: Mapping[str, tuple[int, ...]],
        debug: bool = False,
        seed_ownership: str = "replicated",
    ):
        super().__init__()
        self.comm = resolve_comm(comm)
        self.backend = backend
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.specs = _ordered_specs(design_variable_specs)
        self.design_variable_names = tuple(name for name, _ in self.specs)
        self.debug = bool(debug)
        if seed_ownership not in ("replicated", "root"):
            raise ValueError(
                "seed_ownership must be 'replicated' or 'root'."
            )
        self.seed_ownership = str(seed_ownership)
        self._total_size = int(
            sum(int(np.prod(shape)) for _name, shape in self.specs)
        )

    def evaluate(self, inputs, d_outputs):
        """Declare the reverse inputs and one cotangent output per design variable.

        Parameters
        ----------
        inputs
            Mapping of the forward operation's inputs, holding every declared
            design variable.
        d_outputs
            Mapping of reverse seeds, holding ``"volume_coordinates"`` with the
            forward output's global shape.

        Returns
        -------
        dict of str to csdl_alpha.Variable
            Cotangents keyed by **design-variable name**, one entry per
            declared variable, each with that variable's own shape. The
            underlying CSDL outputs are named ``d_<name>``, but the mapping
            keys are the plain names. The mapping is keyed, not ordered.
        """
        for name, _shape in self.specs:
            self.declare_input(name, inputs[name])
        self.declare_input(
            "d_volume_coordinates", d_outputs["volume_coordinates"]
        )
        derivatives = {}
        for name, shape in self.specs:
            derivatives[name] = self.create_output(f"d_{name}", shape)
        return derivatives

    def compute(self, inputs, outputs):
        """Evaluate the reverse product on the root rank and broadcast it.

        The per-variable cotangents are concatenated into one flat vector on
        the root rank, broadcast once, then split back out so every rank writes
        identical values.

        Collective: every rank must reach this call. The seed-ownership
        contract is enforced collectively *before* the broadcast, so a
        violation cannot strand a rank.

        Parameters
        ----------
        inputs
            Mapping holding every declared design variable and the seed
            ``"d_volume_coordinates"``, reshaped to the global output shape.
        outputs
            Output buffer written in place with one ``d_<name>`` entry per
            design variable, each reshaped to that variable's shape and
            identical on every rank.

        Returns
        -------
        None
            Results are written into ``outputs``.

        Raises
        ------
        RuntimeError
            On every rank when the seed violates the configured ownership
            contract, or when the root-rank reverse evaluation raised.
        """
        design_variables = {
            name: np.asarray(inputs[name], dtype=float)
            for name, _shape in self.specs
        }
        seed = np.asarray(
            inputs["d_volume_coordinates"], dtype=float
        ).reshape(self.output_shape)

        # Enforce the cotangent-ownership contract collectively before rank 0
        # uses its own seed: 'replicated' requires identical seeds on all ranks,
        # 'root' requires zero on non-root ranks. Raises synchronously on a
        # violation so no rank is stranded in the broadcast below.
        verify_seed_ownership(
            self.comm,
            seed,
            ownership=self.seed_ownership,
            name="d_volume_coordinates",
        )

        flat = run_on_root(
            self.comm,
            lambda: self._root_vjp(design_variables, seed),
        )
        flat = broadcast_array(
            self.comm,
            flat if is_root(self.comm) else None,
            shape=(self._total_size,),
            dtype=np.float64,
        )

        offset = 0
        for name, shape in self.specs:
            size = int(np.prod(shape))
            outputs[f"d_{name}"] = flat[offset : offset + size].reshape(shape)
            offset += size

    def _root_vjp(
        self,
        design_variables: Mapping[str, np.ndarray],
        seed: np.ndarray,
    ) -> np.ndarray:
        vjps = self.backend.compute_vjp(design_variables, seed)
        missing = set(self.design_variable_names).difference(vjps)
        if missing:
            raise KeyError(
                "Backend VJP is missing design variables: "
                + ", ".join(sorted(missing))
            )
        pieces = []
        for name, shape in self.specs:
            value = np.asarray(vjps[name], dtype=np.float64).reshape(shape)
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"Backend VJP for {name!r} contains nonfinite values."
                )
            pieces.append(value.reshape(-1))
        return np.concatenate(pieces) if pieces else np.zeros(0)


def build_geometry_volume_operation(
    backend: Any,
    comm: Any = None,
    *,
    output_shape: tuple[int, int],
    design_variable_specs: Mapping[str, tuple[int, ...]],
    debug: bool = False,
) -> GeometryVolumeOperation:
    """Construct a :class:`GeometryVolumeOperation` with a resolved comm.

    A thin convenience wrapper: it forwards its arguments unchanged and leaves
    communicator resolution to the constructor. It does not expose
    ``seed_ownership``, so the operation's default of ``"replicated"`` applies.

    Parameters
    ----------
    backend
        Live geometry-to-volume backend on the root rank; ``None`` on non-root
        ranks.
    comm
        MPI communicator. ``None`` resolves to ``MPI.COMM_WORLD`` when
        ``mpi4py`` is importable and to a serial communicator otherwise.
    output_shape
        Global ``(num_points, 3)`` coordinate shape, identical on every rank.
    design_variable_specs
        Ordered mapping of design-variable name to shape, identical on every
        rank.
    debug
        Verify that replicated design variables agree across ranks before the
        root rank uses its own copy.

    Returns
    -------
    GeometryVolumeOperation
        The constructed operation.
    """
    return GeometryVolumeOperation(
        backend,
        comm,
        output_shape=output_shape,
        design_variable_specs=design_variable_specs,
        debug=debug,
    )


__all__ = [
    "GeometryVolumeOperation",
    "GeometryVolumeVJP",
    "build_geometry_volume_operation",
]
