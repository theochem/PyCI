r"""
FanCI AP1roG module.

"""

from itertools import permutations
from typing import Any, Union

import numpy as np

import pyci

from pyci._pyci import AP1roGObjective
from .fanci import FanCI


__all___ = [
    "AP1roG",
]


class AP1roG(FanCI):
    r"""
    AP1roG FanCI class.

    The AP1roG wave function is the *Antisymmetrized Product of 1-reference-orbital
    Geminals* [AP1roG1]_, which has the form

    .. math::

        \left|\Psi\right> = \prod_{p=1}^{N_\text{elec}/2}{\left(
            a^\dagger_{\alpha;p} a^\dagger_{\beta;p} +
            \sum_{k=N/2+1}^K{
                c_{pk} a^\dagger_{\alpha;k} a^\dagger_{\beta;k}
            }
        \right)} \left|\psi_0\right> \,.

    .. [AP1roG1] Limacher, Peter A., et al. "A new mean-field method suitable for strongly
                 correlated electrons: Computationally facile antisymmetric products of
                 nonorthogonal geminals." *Journal of chemical theory and computation*
                 9.3 (2013): 1394-1401.

    """

    def __init__(
        self,
        ham: pyci.hamiltonian,
        nocc: int,
        nproj: int = None,
        wfn: pyci.doci_wfn = None,
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
        wfn : pyci.doci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        # Compute number of parameters (c_kl + energy)
        nparam = nocc * (ham.nbasis - nocc) + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn (P space == single pair excitations)
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc, nocc)
            wfn.add_excited_dets(1)
        elif not isinstance(wfn, pyci.doci_wfn):
            raise TypeError(f"Invalid `wfn` type `{type(wfn)}`; must be `pyci.doci_wfn`")
        elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
            raise ValueError(f"wfn.nocc_{{up,dn}} does not match `nocc={nocc}` parameter")

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

        # Initialize C extension
        try:
            norm_det = kwargs["norm_det"]
            idx_det_cons = np.asarray([elem[0] for elem in norm_det], dtype=int)
            det_cons = np.asarray([elem[1] for elem in norm_det], dtype=float)
        except KeyError:
            idx_det_cons = None
            det_cons = None
        try:
            norm_param = kwargs["norm_param"]
            idx_param_cons = np.asarray([elem[0] for elem in norm_param], dtype=int)
            param_cons = np.asarray([elem[1] for elem in norm_param], dtype=float)
        except KeyError:
            idx_param_cons = None
            param_cons = None
        self._cext = AP1roGObjective(
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
