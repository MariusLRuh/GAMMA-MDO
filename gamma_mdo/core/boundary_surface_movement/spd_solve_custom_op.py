"""Differentiable SPD linear solve as a CSDL custom operation.

The physics-based mesh-motion propagator (see ``elasticity.py`` and
``motion.py``) reduces to a symmetric positive-definite system
``L_ff u_f = rhs`` that is factored once on the reference mesh and
back-substituted every design iteration.  Per the project's requirement that
specialized linear algebra go through CSDL custom operations, only this solve
is wrapped here; the right-hand-side assembly ``rhs = -L_fp u_p`` stays in
plain CSDL (``csdl.sparse.matmat``) so its adjoint is automatic.

The forward is ``u_f = factor.solve(rhs)`` (per column).  Because ``L_ff`` is
symmetric, the adjoint of ``x = A^{-1} b`` is ``bar_b = A^{-T} bar_x =
factor.solve(bar_x)`` and reuses the *same* factor -- no extra setup.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import csdl_alpha as csdl


def factorize_spd(matrix) -> "SPDFactor":
    """Factor a symmetric positive-definite sparse matrix once.

    Prefers a sparse Cholesky (scikit-sparse CHOLMOD) and falls back to
    ``scipy.sparse.linalg.splu`` when CHOLMOD is unavailable.  The returned
    object exposes a single ``solve`` method that accepts a ``(n,)`` or
    ``(n, k)`` right-hand side and shares the factorization across the forward
    and adjoint solves.

    Parameters
    ----------
    matrix
        Square sparse symmetric positive-definite matrix.

    Returns
    -------
    SPDFactor
        Reusable factorization wrapper.

    Raises
    ------
    ValueError
        If ``matrix`` is not square.
    """
    return SPDFactor(matrix)


class SPDFactor:
    """Cache an SPD factorization with a multi-column solve.

    Parameters
    ----------
    matrix
        Square sparse symmetric positive-definite matrix.
    """

    def __init__(self, matrix):
        csc = sp.csc_matrix(matrix)
        if csc.shape[0] != csc.shape[1]:
            raise ValueError("SPD factorization requires a square matrix.")
        self.shape = (int(csc.shape[0]), int(csc.shape[1]))
        self._backend, self._factor = _build_factor(csc)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ``A x = rhs`` for one or more right-hand sides.

        Parameters
        ----------
        rhs
            Array with shape ``(n,)`` or ``(n, k)``.

        Returns
        -------
        numpy.ndarray
            Solution with the same shape as ``rhs``.

        Raises
        ------
        ValueError
            If ``rhs`` is not one- or two-dimensional or has the wrong row
            count.
        """
        rhs_array = np.asarray(rhs, dtype=float)
        if rhs_array.ndim == 1:
            return np.asarray(self._solve_2d(rhs_array.reshape(-1, 1)), dtype=float).reshape(-1)
        if rhs_array.ndim != 2:
            raise ValueError("SPDFactor.solve expects a 1D or 2D right-hand side.")
        return np.asarray(self._solve_2d(rhs_array), dtype=float)

    def _solve_2d(self, rhs_array: np.ndarray) -> np.ndarray:
        if rhs_array.shape[0] != self.shape[0]:
            raise ValueError(
                "Right-hand side rows must match the factored matrix dimension."
            )
        if self._backend == "cholmod":
            # CHOLMOD solves every column of a dense 2D right-hand side at once.
            return np.asarray(self._factor(np.asfortranarray(rhs_array)), dtype=float)
        # scipy splu solves a single column at a time.
        solution = np.empty_like(rhs_array)
        for column in range(rhs_array.shape[1]):
            solution[:, column] = self._factor.solve(rhs_array[:, column])
        return solution


def _build_factor(csc):
    try:
        from sksparse.cholmod import cholesky

        return "cholmod", cholesky(csc)
    except Exception:
        # scipy's sparse LU is robust for the SPD systems here even though it
        # does not exploit symmetry; it is the documented fallback backend.
        from scipy.sparse.linalg import splu

        return "splu", splu(csc.tocsc())


class SPDSolveOperation(csdl.experimental.CustomExplicitOperationBeta):
    """Wrap ``u_f = L_ff^{-1} rhs`` for a precomputed factorization.

    Parameters
    ----------
    factor
        Cached factorization shared by the forward and reverse operations.
    """

    def __init__(self, factor: SPDFactor):
        super().__init__()
        self.factor = factor

    def evaluate(self, rhs):
        """Declare and return the differentiable SPD solution.

        Parameters
        ----------
        rhs
            CSDL right-hand-side variable.

        Returns
        -------
        csdl.Variable
            Solution variable with the same shape as ``rhs``.
        """
        self.declare_input("rhs", rhs)
        u_f = self.create_output("u_f", rhs.shape)
        self.declare_vjp_function(
            SPDSolveVJP,
            factor=self.factor,
            output_name="u_f",
        )
        return u_f

    def compute(self, inputs, outputs):
        """Evaluate the numeric forward solve.

        Parameters
        ----------
        inputs
            Custom-operation inputs containing ``rhs``.
        outputs
            Mutable custom-operation outputs receiving ``u_f``.
        """
        outputs["u_f"] = self.factor.solve(np.asarray(inputs["rhs"], dtype=float))


class SPDSolveVJP(csdl.experimental.CustomExplicitOperationBeta):
    """Compute the SPD-solve adjoint using the symmetric factor.

    Parameters
    ----------
    factor
        Cached forward factorization.
    output_name
        Name used to retrieve the output cotangent.
    """

    def __init__(self, factor: SPDFactor, output_name: str = "u_f"):
        super().__init__()
        self.factor = factor
        self.output_name = str(output_name)

    def evaluate(self, inputs, d_outputs):
        """Declare the reverse solve and its input cotangent.

        Parameters
        ----------
        inputs
            Forward inputs containing ``rhs``.
        d_outputs
            Output cotangents keyed by ``output_name``.

        Returns
        -------
        dict[str, csdl.Variable]
            Cotangent variable for ``rhs``.
        """
        rhs = inputs["rhs"]
        d_u_f = d_outputs[self.output_name]

        self.declare_input("d_u_f", d_u_f)
        d_rhs = self.create_output("d_rhs", rhs.shape)

        return {"rhs": d_rhs}

    def compute(self, inputs, outputs):
        """Evaluate the numeric transpose solve.

        Parameters
        ----------
        inputs
            Reverse inputs containing the solution cotangent.
        outputs
            Mutable outputs receiving the right-hand-side cotangent.
        """
        outputs["d_rhs"] = self.factor.solve(np.asarray(inputs["d_u_f"], dtype=float))


__all__ = [
    "SPDFactor",
    "SPDSolveOperation",
    "SPDSolveVJP",
    "factorize_spd",
]
