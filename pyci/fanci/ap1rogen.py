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
        wfn: pyci.genci_wfn = None,
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
        wfn : pyci.genci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        # SEE OTHER FANCI WFNS

        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        nparam = nocc * (ham.nbasis - nocc) + (2 * nocc) * (2 * (ham.nbasis - nocc)) + 1 #less params considering we added singles as well
        nproj = nparam if nproj is None else nproj
        print("\n nparam, nproj", nparam, nproj)

        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc, nocc)
            print("\nCreated DOCI wfn instance for AP1roGSDGeneralized_sen-o.")
            wfn.add_excited_dets(1) # add pair excited determinants

            print("\nCreating fci wfn")
            wfn = pyci.fullci_wfn(wfn)
            print("wfn.nocc, wfn.nocc_up, wfn.nocc_dn: ", wfn.nocc, wfn.nocc_up, wfn.nocc_dn)
            occ_array = wfn.to_occ_array()
            det_array = wfn.to_det_array()
            print(f"\nFor FCI {len(occ_array)} det & corresponding occ_array:")
            for i, j in zip(det_array, occ_array):
                print(i, j)

            print("\nCreating GenCI wfn")
            wfn = pyci.genci_wfn(wfn)
            print("wfn.nocc, wfn.nocc_up, wfn.nocc_dn: ", wfn.nocc, wfn.nocc_up, wfn.nocc_dn)
            occ_array = wfn.to_occ_array()
            det_array = wfn.to_det_array()
            print("\nFor GenCI wfn det_array: ", len(det_array), "\n", det_array)
            #for det, occ in zip(det_array, occ_array):
            #    print(det, bin(int(det)), occ)

            # Generate the bitstring
            bitstring = ((1 << wfn.nocc //2) - 1) << ham.nbasis | (1 << wfn.nocc//2) - 1
            # Convert to binary string with leading zeros
            nb = ham.nbasis
            bit_str = format(bitstring, f'0{2 * nb}b')
            unocc = [i for i in range(len(bit_str)) if bit_str[nb-i-1] == '0']
            occ = [i for i in range(len(bit_str)) if bit_str[nb-i-1] == '1']
            # Adding non-spin-preserving alpha -> beta singles
            for i in occ:
               for a in unocc:
                   exc_str = _excite_det(i, a, int(bit_str,2))
                   print("i, a, exc_str", i, a, exc_str, format(exc_str, f'0{2*nb}b'))
                   wfn.add_det(np.array(exc_str, dtype=pyci.c_ulong))
                   
            det_array = wfn.to_det_array()
            print("\nFor GenCI wfn det_array after adding Gen_sen-o S:", len(det_array), "\n", det_array)
                   
        elif not isinstance(wfn, pyci.genci_wfn):
            raise TypeError(f"Invalid `wfn` type `{type(wfn)}`; must be `pyci.genci_wfn`")
        #elif wfn.nocc_up != nocc or wfn.nocc != nocc:
        elif wfn.nocc != nocc:
            raise ValueError(f"wfn.nocc does not match `nocc={nocc}` parameter")


        # Initialize base class
        print("\nInitializing base class")
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

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


def _excite_det(i: int, a: int, bitstring: int):
    bitstring &= ~(1 << i)
    bitstring |= (1 << a)
    #pdb.set_trace()
    return bitstring
