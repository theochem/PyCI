r"""
FanCI Placeholder module.

"""

from typing import Any, Union

import numpy as np

import pyci

from ..pyci import PlaceholderObjective
from .fanci import FanCI


__all___ = [
    "Placeholder",
]


class Placeholder(FanCI):
    r"""
    DOC
    """

    def __init__(
        self,
        ham: pyci.hamiltonian,
        nocc: int,
        nproj: int = None,
        wfn: pyci.fullci_wfn = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        nocc : int
            Number of occupied orbitals.
        nproj : int, optional
            Number of determinants in projection ("P") space.
        wfn : pyci.fullci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        # SEE OTHER FANCI WFNS

        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

        self._cext = PlaceholderObjective(
            self._ci_op, self._wfn,
            idx_det_cons=idx_det_cons, det_cons=det_cons,
            idx_param_cons=idx_param_cons, param_cons=param_cons,
        )

    def compute_overlap(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].

        Returns
        -------
        ovlp : np.ndarray
            Overlap array.

        """
        return self._cext.overlap(x)

    def compute_overlap_deriv(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative array.

        """
        return self._cext.d_overlap(x)

    def compute_objective(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI objective function.

            f : x[k] -> y[n]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        obj : np.ndarray
            Objective vector.

        """
        return self._cext.objective(self._ci_op, x)

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the Jacobian of the FanCI objective function.

            j : x[k] -> y[n, k]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        jac : np.ndarray
            Jacobian matrix.

        """
        return self._cext.jacobian(self._ci_op, x)
