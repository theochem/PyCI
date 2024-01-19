r"""
FanCI AP1roG module.

"""

from itertools import permutations
from typing import Any, Union

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    "AP1roG",
]


class AP1roG(FanCI):
    r"""
    AP1roG FanCI class.

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

        # Assign reference occupations
        ref_occs = np.arange(nocc, dtype=pyci.c_long)

        # Use set differences to get hole/particle indices
        hlist = [np.setdiff1d(ref_occs, occs, assume_unique=1) for occs in self._sspace]
        plist = [np.setdiff1d(occs, ref_occs, assume_unique=1) - nocc for occs in self._sspace]

        # Save sub-class -specific attributes
        self._ref_occs = ref_occs
        self._sspace_data = hlist, plist
        self._pspace_data = hlist[:nproj], plist[:nproj]

    def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap array.

        """
        # Check if we can use our pre-computed {p,s}space_data
        if isinstance(occs_array, np.ndarray):
            # Use set differences to get hole/particle indices
            nocc = self._wfn.nocc_up
            ref_occs = self._ref_occs
            hlist = [np.setdiff1d(ref_occs, occs, assume_unique=1) for occs in occs_array]
            plist = [np.setdiff1d(occs, ref_occs, assume_unique=1) - nocc for occs in occs_array]
        elif occs_array == "P":
            occs_array = self._pspace
            hlist, plist = self._pspace_data
        elif occs_array == "S":
            occs_array = self._sspace
            hlist, plist = self._sspace_data
        else:
            raise ValueError("invalid `occs_array` argument")

        # Reshape parameter array to AP1roG matrix
        x_mat = x.reshape(self._wfn.nocc_up, self._wfn.nvir_up)

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, (occs, holes, parts) in enumerate(zip(occs_array, hlist, plist)):
            # Overlap is equal to one for the reference determinant
            y[i] = permanent(x_mat[holes, :][:, parts]) if holes.size else 1
        return y

    def compute_overlap_deriv(
        self, x: np.ndarray, occs_array: Union[np.ndarray, str]
    ) -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative array.

        """
        # Check if we can use our precomputed {p,s}space_{exc,pos}_data
        if isinstance(occs_array, np.ndarray):
            # Use set differences to get hole/particle indices
            nocc = self._wfn.nocc_up
            ref_occs = self._ref_occs
            hlist = [np.setdiff1d(ref_occs, occs, assume_unique=1) for occs in occs_array]
            plist = [np.setdiff1d(occs, ref_occs, assume_unique=1) - nocc for occs in occs_array]
        elif occs_array == "P":
            occs_array = self._pspace
            hlist, plist = self._pspace_data
        elif occs_array == "S":
            occs_array = self._sspace
            hlist, plist = self._sspace_data
        else:
            raise ValueError("invalid `occs_array` argument")

        # Reshape parameter array to AP1roG matrix
        x_mat = x.reshape(self._wfn.nocc_up, self._wfn.nvir_up)

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        # Iterate over occupation vectors
        for y_row, holes, parts in zip(y, hlist, plist):

            # Iterate over all parameters (i) and active parameters (j)
            j = -1
            for i, m in enumerate(self._mask[:-1]):

                # Check if element is active
                if not m:
                    continue
                j += 1

                # Check for reference determinant
                if not holes.size:
                    continue

                # Cut out the rows and columns corresponding to the element wrt which the permanent
                # is derivatized
                rows = holes[holes != (i // self.wfn.nvir_up)]
                cols = parts[parts != (i % self.wfn.nvir_up)]
                if rows.size == cols.size == 0:
                    y_row[j] = 1.0
                elif rows.size != holes.size and cols.size != parts.size:
                    y_row[j] = permanent(x_mat[rows, :][:, cols])

        # Return overlap derivative matrix
        return y


def permanent(matrix: np.ndarray) -> float:
    r"""
    Compute the permanent of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    result : matrix.dtype
        Permanent of the matrix.

    """
    rows = np.arange(matrix.shape[0])
    return sum(np.prod(matrix[rows, cols]) for cols in permutations(rows))
