r"""
FanCI APIG module.

"""

from typing import Any, Union

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    "APIG",
]


class APIG(FanCI):
    r"""
    APIG FanCI class.

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

        # Reshape parameter array to APIG matrix
        x_mat = x.reshape(self._wfn.nbasis, self._wfn.nocc_up)

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, occs in enumerate(occs_array):
            y[i] = permanent(x_mat[occs])
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

        # Reshape parameter array to APIG matrix
        x_mat = x.reshape(self._wfn.nbasis, self._wfn.nocc_up)

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        col_inds = np.arange(self._wfn.nocc_up, dtype=pyci.c_long)

        # Iterate over occupation vectors
        for y_row, occs in zip(y, occs_array):

            # Iterate over all parameters (i) and active parameters (j)
            j = -1
            for i, m in enumerate(self._mask[:-1]):

                # Check if element is active
                if not m:
                    continue
                j += 1

                # Compute derivative of overlap
                rows = occs[occs != (i // self._wfn.nocc_up)]
                cols = col_inds[col_inds != (i % self._wfn.nocc_up)]
                if rows.size == cols.size == 0:
                    y_row[j] = 1.0
                elif rows.size != occs.size and cols.size != col_inds.size:
                    y_row[j] = permanent(x_mat[rows, :][:, cols])

        # Return overlap derivative matrix
        return y


def permanent(matrix: np.ndarray) -> float:
    r"""
    Compute the permanent of a square matrix using Glynn's algorithm.

    Gray code generation from Knuth, D. E. (2005). The Art of Computer Programming,
    Volume 4, Fascicle 2: Generating All Tuples and Permutations.

    Glynn's algorithm from Glynn, D. G. (2010). The permanent of a square matrix.
    European Journal of Combinatorics, 31(7), 1887-1891.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    result : matrix.dtype
        Permanent of the matrix.

    """
    # Permanent of zero-by-zero matrix is 1
    n = matrix.shape[0]
    if not n:
        return 1

    # Initialize gray code
    pos = 0
    sign = 1
    bound = n - 1
    delta = np.ones(n, dtype=np.int)
    graycode = np.arange(n, dtype=np.int)

    # Iterate over every delta
    result = np.prod(np.sum(matrix, axis=0))
    while pos < bound:
        # Update delta and add term to permanent
        sign *= -1
        delta[bound - pos] *= -1
        result += sign * np.prod(delta.dot(matrix))
        # Update gray code and position
        graycode[0] = 0
        graycode[pos] = graycode[pos + 1]
        graycode[pos + 1] = pos + 1
        pos = graycode[0]

    # Divide by constant factor
    return result / (2 ** bound)
