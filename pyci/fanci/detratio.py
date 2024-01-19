r"""
FanCI Determinant ratio module.

Adapted from:
https://github.com/QuantumElephant/fanpy/blob/master/wfns/wfn/quasiparticle/det_ratio.py

"""

from typing import Any, Sequence, Union

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    "DetRatio",
]


class DetRatio(FanCI):
    r"""
    Determinant ratio FanCI class.

    """

    def __init__(
        self,
        ham: pyci.hamiltonian,
        nocc: int,
        numerator: int,
        denominator: int,
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
        numerator : int
            Number of matrices in the numerator.
        denominator : int
            Number of matrices in the denominator.
        nproj : int, optional
            Number of determinants in projection ("P") space.
        wfn : pyci.doci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        # Check number of matrices
        nmatrices = numerator + denominator
        if nmatrices % 2:
            raise ValueError(f"Number of matrices `nmatrices={nmatrices}` cannot be odd")

        # Compute number of parameters (c_{i;kl} + energy)
        nparam = nmatrices * ham.nbasis * nocc + 1

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

        # Get results of 'searchsorted(i)' from i=0 to i=nbasis for each det. in "S" space
        arange = np.arange(self._wfn.nbasis, dtype=pyci.c_long)
        sspace_data = [occs.searchsorted(arange) for occs in self._sspace]
        pspace_data = sspace_data[: self._nproj]

        # Save sub-class -specific attributes
        self._nmatrices = nmatrices
        self._numerator = numerator
        self._denominator = denominator
        self._matrix_mask = self._mask[:-1].reshape(nmatrices, ham.nbasis, nocc)
        self._sspace_data = sspace_data
        self._pspace_data = pspace_data

    def freeze_matrix(self, *matrices: Sequence[int]) -> None:
        r"""
        Set a matrix to be frozen during optimization.

        Parameters
        ----------
        matrices : Sequence[int]
            Indices of matrices to freeze.

        """
        for matrix in matrices:
            self._matrix_mask[matrix] = False
        # Update nactive
        self._nactive = self._mask.sum()

    def unfreeze_matrix(self, *matrices: Sequence[int]) -> None:
        r"""
        Set a matrix to be active during optimization.

        Parameters
        ----------
        matrices : Sequence[int]
            Indices of matrices to unfreeze.

        """
        for matrix in matrices:
            self._matrix_mask[matrix] = True
        # Update nactive
        self._nactive = self._mask.sum()

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
        if isinstance(occs_array, np.ndarray):
            pass
        elif occs_array == "P":
            occs_array = self._pspace
        elif occs_array == "S":
            occs_array = self._sspace
        else:
            raise ValueError("invalid `occs_array` argument")

        # Reshape parameter array to numerator and denominator matrices
        x_mats = x.reshape(self._nmatrices, self._wfn.nbasis, self._wfn.nocc_up)
        n_mats = x_mats[: self._numerator]
        d_mats = x_mats[self._numerator :]

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, occs in enumerate(occs_array):
            y[i] = np.prod([np.linalg.det(n_mat[occs]) for n_mat in n_mats]) / np.prod(
                [np.linalg.det(d_mat[occs]) for d_mat in d_mats]
            )
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
        if isinstance(occs_array, np.ndarray):
            pass
        elif occs_array == "P":
            occs_array = self._pspace
        elif occs_array == "S":
            occs_array = self._sspace
        else:
            raise ValueError("invalid `occs_array` argument")

        # Reshape parameter array to numerator and denominator matrices
        x_mats = x.reshape(self._nmatrices, self._wfn.nbasis, self._wfn.nocc_up)
        mat_size = self._wfn.nbasis * self._wfn.nocc_up

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        # Iterate over occupation vectors
        col_inds = np.arange(self._wfn.nocc_up, dtype=pyci.c_long)
        for y_row, occs in zip(y, occs_array):

            # Compute determinants of numerator and denominator matrices
            dets = [np.linalg.det(mat[occs]) for mat in x_mats]
            n_det_prod = np.prod(dets[: self._numerator])
            d_det_prod = np.prod(dets[self._numerator :])

            # Iterate over all parameters (i) and active parameters (j)
            i = -1
            j = -1

            # Iterate over numerator matrices
            for mask in self._mask[: self._numerator * mat_size]:
                i += 1

                # Check if element is active
                if not mask:
                    continue
                j += 1

                # Compute derivative of overlap function
                m = i // mat_size
                n = i % mat_size
                r = n // self._wfn.nocc_up
                c = n % self._wfn.nocc_up
                rows = occs[occs != r]
                cols = col_inds[col_inds != c]
                if rows.size == cols.size == 0:
                    y_row[j] = 1.0
                elif rows.size != occs.size and cols.size != col_inds.size:
                    val = -1 if np.searchsorted(occs, r) % 2 else +1
                    val *= -1 if np.searchsorted(col_inds, c) % 2 else +1
                    val *= n_det_prod * np.linalg.det(x_mats[m][rows, :][:, cols])
                    val /= d_det_prod * dets[m]
                    y_row[j] = val

            # Iterate over denominator matrices
            for mask in self._mask[self._numerator * mat_size : -1]:
                i += 1

                # Check if element is active
                if not mask:
                    continue
                j += 1

                # Compute derivative of overlap function
                m = i // mat_size
                n = i % mat_size
                r = n // self._wfn.nocc_up
                c = n % self._wfn.nocc_up
                rows = occs[occs != r]
                cols = col_inds[col_inds != c]
                if rows.size == cols.size == 0:
                    y_row[j] = 1.0
                elif rows.size != occs.size and cols.size != col_inds.size:
                    val = -1 if np.searchsorted(occs, r) % 2 else +1
                    val *= +1 if np.searchsorted(col_inds, c) % 2 else -1
                    val *= n_det_prod * np.linalg.det(x_mats[m][rows, :][:, cols])
                    val /= d_det_prod * dets[m]
                    y_row[j] = val

        # Return overlap derivative matrix
        return y
