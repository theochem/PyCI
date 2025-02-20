r"""
FanCI AP1roGeneralizedseno module for AP1roGSDGeneralized_sen-o wavefunction.

"""

from typing import Any, Union

import numpy as np

import pyci

from ..pyci import AP1roGeneralizedSenoObjective
from .fanci import FanCI
import pdb


__all___ = [
    "AP1roGeneralizedSeno",
]


class AP1roGeneralizedSeno(FanCI):
    r"""
    DOC
    """

    def __init__(
        self,
        ham: pyci.hamiltonian,
        nocc: int,
        nproj: int = None,
        wfn: pyci.nonsingletci_wfn = None,
        fill: str = 'excitation',
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
        wfn : pyci.nonsingletci_wfn
            If specified, this PyCI wave function defines the projection ("P") space.
        fill : 
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        # SEE OTHER FANCI WFNS

        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        nparam = nocc * (ham.nbasis - nocc) + (2 * nocc) * (2 * (ham.nbasis - nocc)) + 1 #less params considering we added singles as well
        nproj = nparam if nproj is None else nproj

        # if nproj > nparam:
        #     raise ValueError("nproj cannot be greater than the size of the space")

        if wfn is None:
            # wfn = pyci.doci_wfn(ham.nbasis, nocc, nocc)
            # wfn.add_excited_dets(1) # add pair excited determinants
            wfn = pyci.fullci_wfn(ham.nbasis, nocc, nocc)
            # exc=0 ensures addint HF determinannt first
            pyci.add_excitations(wfn,1,2,3,4)
            print("Printing FCI wfn dets: ")
            for i, sd in enumerate(wfn.to_occ_array()):
                sd = np.array(sd)
                print(wfn.to_det_array()[i],np.concatenate((sd[0],sd[1]+ham.nbasis)))
            
            wfn = pyci.nonsingletci_wfn(wfn)
                   
        elif not isinstance(wfn, pyci.nonsingletci_wfn):
            raise TypeError(f"Invalid `wfn` type `{type(wfn)}`; must be `pyci.nonsingletci_wfn`")
        elif wfn.nocc != nocc:
            raise ValueError(f"wfn.nocc does not match `nocc={nocc}` parameter")


        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)
        print("Nonsingletci nbasis: ",wfn.nbasis)
        print("\n\nPrinting nonsingletci wfn dets: ")
        for sd in (self._sspace):
            print(sd)
            # sd = np.array(sd)
            # print(np.concatenate((sd[0],sd[1]+ham.nbasis)))
        print("Done printing\n\n")

        # Assign reference occupations
        #ref_occs_up = np.arange(nocc_up, dtype=pyci.c_long)
        #ref_occs_dn = np.arange(nocc_dn, dtype=pyci.c_long)
 
        # Save sub-class-specific attributes
        #self._ref_occs = [ref_occs_up, ref_occs_dn]

        # Initiazlize C++ extension
        try:
            norm_det = kwargs["norm_det"]
            idx_det_cons = np.asarray([elem[0] for elem in norm_det], dtype=int)
            det_cons = np.asarray([elem[1] for elem in norm_det], dtype=int)
        except KeyError:
            idx_det_cons = None
            det_cons = None
         
        try:
            norm_param = kwargs["norm_param"]
            idx_param_cons = np.asarray([elem[0] for elem in norm_param], dtype=int)
            param_cons = np.asarray([elem[1] for elem in norm_param], dtype=int)
        except KeyError:
            idx_param_cons = None
            param_cons = None

        self._cext = AP1roGeneralizedSenoObjective(
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

