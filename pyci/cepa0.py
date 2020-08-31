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

r"""PyCI CEPA0 module."""

from typing import Union, Tuple

import numpy as np

from scipy.sparse.linalg import lgmres, lsqr

from . import pyci


__all__ = [
    "CEPA0",
]


class CEPA0:
    r"""CEPA0 solver class."""

    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        Shape of the linear operator.

        Returns
        -------
        nrow : int
            Number of rows.
        ncol : int
            Number of columns.

        """
        return self.op.shape

    @property
    def dtype(self) -> np.dtype:
        r"""
        Data type of the linear operator.

        Returns
        -------
        dtype : numpy.dtype
            Data type of the linear operator.

        """
        return self.op.dtype

    def __init__(
        self,
        *args: Union[Tuple[pyci.wavefunction, pyci.hamiltonian], pyci.sparse_op],
        ref: int = 0,
        damp: float = np.inf,
    ) -> None:
        r"""
        Initialize the CEPA0 solver.

        Parameters
        ----------
        args : (pyci.sparse_op,) or (pyci.hamiltonian, pyci.wavefunction)
            System to solve.
        ref : int, default=0
            Index of reference determinant.
        damp : float, default=numpy.inf
            Damping parameter.

        """
        if len(args) == 1:
            self.op = args[0]
            if self.op.symmetric:
                raise TypeError("Cannot run CEPA0 on symmetric matrix operator")
            elif self.op.shape[0] < self.op.shape[1]:
                raise ValueError("Cannot solve underdetermined system (nrow < ncol)")
        elif len(args) == 2:
            self.op = pyci.sparse_op(*args, symmetric=False)
        else:
            raise ValueError("must pass `ham, wfn` or `op`")
        self.ref = ref
        self.h_ref = self.op.get_element(ref, ref)
        self.damp = -1.0 / damp ** 2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r"""
        Apply the linear operator to a vector ``x``.

        Parameters
        ----------
        x : numpy.ndarray
            Vector to which linear operator will be applied.

        Returns
        -------
        y : numpy.ndarray
            Result of applying linear operator to ``x``.

        """
        return self.matvec(x)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        r"""
        Apply the linear operator to a vector ``x``.

        Parameters
        ----------
        x : numpy.ndarray
            Vector to which linear operator will be applied.

        Returns
        -------
        y : numpy.ndarray
            Result of applying linear operator to ``x``.

        """
        op = self.op
        ref = self.ref
        h_ref = self.h_ref
        damp = self.damp
        y = np.zeros(op.shape[0], dtype=pyci.c_double)
        for i in range(op.shape[0]):
            h_ref_damp = np.exp(damp * x[i] ** 2) * h_ref
            if i == ref:
                for j in range(self.op.shape[1]):
                    if j == ref:
                        y[i] += x[j]
                    else:
                        y[i] -= op.get_element(i, j) * x[j]
            else:
                for j in range(self.op.shape[1]):
                    if i == j:
                        y[i] += (h_ref_damp - op.get_element(i, j)) * x[j]
                    elif j != ref:
                        y[i] -= op.get_element(i, j) * x[j]
        return y

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        r"""
        Apply the transpose of the linear operator to a vector ``x``.

        Parameters
        ----------
        x : numpy.ndarray
            Vector to which linear operator will be applied.

        Returns
        -------
        y : numpy.ndarray
            Result of applying linear operator to ``x``.

        """
        op = self.op
        h_ref = self.h_ref
        damp = self.damp
        y = np.zeros(op.shape[1], dtype=pyci.c_double)
        for i in range(op.shape[0]):
            h_ref_damp = np.exp(damp * x[i] ** 2) * h_ref
            for j in range(self.op.shape[1]):
                if j == i:
                    y[j] += (h_ref_damp - op.get_element(i, j)) * x[i]
                else:
                    y[j] -= op.get_element(i, j) * x[i]
        y[self.ref] = x[self.ref]
        return y

    def rhs(self, x: np.ndarray) -> np.ndarray:
        r"""
        Construct the right-hand side vector for the linear system of equations.

        Parameters
        ----------
        x : numpy.ndarray
            Initial guess.

        Returns
        -------
        rhs : numpy.ndarray
            Right-hand side vector.

        """
        op = self.op
        ref = self.ref
        y = np.zeros(op.shape[0], dtype=pyci.c_double)
        for i in range(op.shape[0]):
            y[i] = op.get_element(i, ref)
        y -= self.matvec(x)
        return y

    def solve(
        self, c0: np.ndarray = None, e0: float = None, maxiter: int = 1000, tol: float = 1.0e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Solve the CEPA0 problem.

        Parameters
        ----------
        c0 : numpy.ndarray, optional
            Initial guess for lowest eigenvector.
        e0 : float, optional
            Initial guess for energy.
        maxiter : int, default=1000
            Maximum number of iterations to perform.
        tol : float, default=1.0e-12
            Convergence tolerance.

        Returns
        -------
        es : numpy.ndarray
            Energy.
        cs : numpy.ndarray
            Coefficient vector.

        """
        c0 = np.zeros(self.op.shape[1], dtype=pyci.c_double) if c0 is None else c0 / c0[self.ref]
        c0[self.ref] = self.op.get_element(self.ref, self.ref) if e0 is None else e0 - self.op.ecore
        if self.op.shape[0] == self.op.shape[1]:
            cs = lgmres(self, self.rhs(c0), maxiter=maxiter, tol=tol, atol=tol)[0]
        else:
            cs = lsqr(self, self.rhs(c0), maxiter=maxiter, atol=tol, btol=tol)[0]
        cs += c0
        es = np.full(1, cs[self.ref] + self.op.ecore, dtype=pyci.c_double)
        cs[self.ref] = 1
        return es, cs[None, :]
