# This file is part of PyCI.
#
# PyCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# PyCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCI. If not, see <http://www.gnu.org/licenses/>.

r"""PyCI additional routines module."""

from functools import partial
from typing import Any

import numpy as np
import scipy.sparse.linalg as sp

from . import pyci


__all__ = [
    "solve",
    "solve_cepa0",
]


def solve(
    *args: Any,
    n: int = 1,
    c0: np.ndarray = None,
    ncv: int = None,
    maxiter: int = None,
    tol: float = 1.0e-12,
    method: str = "spectra",
):
    r"""
    Solve a CI eigenproblem.

    Parameters
    ----------
    args : (pyci.sparse_op,) or (pyci.hamiltonian, pyci.wavefunction)
        System to solve.
    n : int, default=1
        Number of lowest eigenpairs to find.
    c0 : np.ndarray, optional
        Initial guess for lowest eigenvector.
    ncv : int, default=min(nrow, max(2 * n + 1, 20))
        Number of Lanczos vectors to use.
    maxiter : int, default=nrow * n * 10
        Maximum number of iterations to perform.
    tol : float, default=1.0e-12
        Convergence tolerance.
    method : ("spectra" | "arpack"), default="spectra"
        Whether to use the C++ solver (Spectra) or the SciPy ARPACK solver.

    Returns
    -------
    es : np.ndarray
        Energies.
    cs : np.ndarray
        Coefficient vectors.

    """
    # Handle inputs
    if len(args) == 1:
        op = args[0]
    elif len(args) == 2:
        op = pyci.sparse_op(*args, symmetric=True)
    else:
        raise ValueError("must pass `ham, wfn` or `op`")
    # Handle default parameters
    if ncv is None:
        ncv = min(op.shape[0], max(2 * n + 1, 20))
    if maxiter is None:
        maxiter = n * op.shape[0] * 10
    if c0 is None:
        c0 = np.zeros(op.shape[0], dtype=pyci.c_double)
        c0[0] = 1
    elif c0.size < op.shape[0]:
        c0 = np.concatenate((c0, np.zeros(op.shape[0] - c0.shape[0], dtype=pyci.c_double)))
    else:
        c0 = np.asarray(c0, dtype=pyci.c_double)
    # Check which method to use
    if method == "spectra":
        return op._solve(c0, n=n, ncv=ncv, maxiter=maxiter, tol=tol)
    elif method != "arpack":
        raise ValueError('`method` must be one of "spectra" or "arpack"')
    # Solve using SciPy's ARPACK interface
    if op.shape[0] == 1:
        return (
            np.full(1, op.get_element(0, 0) + op.ecore, dtype=pyci.c_double),
            np.ones((1, 1), dtype=pyci.c_double),
        )
    es, cs = sp.eigsh(
        sp.LinearOperator(matvec=op.matvec, shape=op.shape, dtype=pyci.c_double),
        v0=c0,
        k=n,
        ncv=ncv,
        maxiter=maxiter,
        tol=tol,
        which="SA",
    )
    es += op.ecore
    return es, cs.transpose()


def solve_cepa0(*args, e0=None, c0=None, refind=0, maxiter=5000, tol=1.0e-12, lstsq=False):
    r"""
    Solve a CEPA0 problem.

    Parameters
    ----------
    args : (pyci.sparse_op,) or (pyci.hamiltonian, pyci.wavefunction)
        System to solve.
    c0 : np.ndarray, optional
        Initial guess for lowest eigenvector.
    refind : int, optional
        Index of determinant to use as reference.
    maxiter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence tolerance.
    lstsq : bool, optional
        Whether to find the least-squares solution.

    Returns
    -------
    e : float
        Energy.
    c : np.ndarray
        Coefficient vector.

    """
    # Handle inputs
    if len(args) == 1:
        op = args[0]
    elif len(args) == 2:
        op = pyci.sparse_op(*args)
    else:
        raise ValueError("must pass `ham, wfn` or `op`")
    # Prepare initial guess
    c0 = np.zeros(op.shape[1], dtype=pyci.c_double) if c0 is None else c0 / c0[refind]
    c0[refind] = op.get_element(refind, refind) if e0 is None else e0
    # Prepare left-hand side matrix
    matvec = partial(op._matvec_cepa0, refind=refind)
    rmatvec = partial(op._rmatvec_cepa0, refind=refind)
    lhs = sp.LinearOperator(matvec=matvec, rmatvec=rmatvec, shape=op.shape, dtype=pyci.c_double)
    # Prepare right-hand side vector
    rhs = op._rhs_cepa0(refind=refind)
    rhs -= op._matvec_cepa0(c0, refind=refind)
    # Solve equations
    if lstsq:
        result = sp.lsqr(lhs, rhs, iter_lime=maxiter, btol=tol, atol=tol)
    else:
        result = sp.lgmres(lhs, rhs, maxiter=maxiter, tol=tol, atol=tol)
    # Return result
    c = result[0]
    c += c0
    e = np.full(1, c[refind] + op.ecore, dtype=pyci.c_double)
    c[refind] = 1
    return e, c[None, :]
