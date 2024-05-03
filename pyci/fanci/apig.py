r"""
FanCI APIG module.

"""

from typing import Any, Union

import numpy as np

import pyci

from ..pyci import APIGObjective
from .fanci import FanCI


__all___ = [
    "APIG",
]


class APIG(FanCI):
    r"""
    APIG FanCI class.

    The APIG wave function is the *Antisymmetrized Product of Interacting Geminals* [APIG1]_,
    which has the form

    .. math::

        \left|\Psi\right> = \prod_{p=1}^{N_\text{elec}/2}{
            \sum_{k=1}^K{
                \left( c_{pk} a^\dagger_{\alpha;k} a^\dagger_{\beta;k} \right)
            }
        } \left|\psi_0\right> \,.

    .. [APIG1] Silver, David M. "Natural orbital expansion of interacting geminals."
           *The Journal of Chemical Physics* 50.12 (1969): 5108-5116.

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
        nparam = ham.nbasis * nocc + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc, nocc)
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
        self._cext = APIGObjective(
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
