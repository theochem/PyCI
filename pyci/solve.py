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

r"""PyCI solver routines module."""

from typing import Union, Tuple

import numpy as np

from . import pyci


__all__ = [
    "solve",
]


def solve(
    *args: Union[Tuple[pyci.wavefunction, pyci.hamiltonian], pyci.sparse_op],
    n: int = 1,
    c0: np.ndarray = None,
    ncv: int = -1,
    maxiter: int = -1,
    tol: float = 1.0e-12,
):
    r"""
    Solve a CI eigenproblem.

    Parameters
    ----------
    args : (pyci.sparse_op,) or (pyci.hamiltonian, pyci.wavefunction)
        System to solve.
    n : int, default=1
        Number of lowest eigenpairs to find.
    c0 : numpy.ndarray, default=[1,0,...,0]
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
    es : numpy.ndarray
        Energies.
    cs : numpy.ndarray
        Coefficient vectors.

    """
    if len(args) == 1:
        op = args[0]
    elif len(args) == 2:
        op = pyci.sparse_op(*args, symmetric=True)
    else:
        raise ValueError("must pass `ham, wfn` or `op`")
    return op.solve(n=n, c0=c0, ncv=ncv, maxiter=maxiter, tol=tol)
