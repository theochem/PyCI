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

r"""PyCI integrals module."""

from typing import Tuple

import numpy as np


__all__ = [
    "make_senzero_integrals",
    "reduce_senzero_integrals",
    "make_rdms",
]


def make_senzero_integrals(
    one_mo: np.ndarray, two_mo: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Return the non-zero chunks for seniority-zero of the full one- and two- electron integrals.

    Parameters
    ----------
    one_mo : np.ndarray
        Full one-electron integral array.
    two_mo : np.ndarray
        Full two-electron integral array.

    Returns
    -------
    h : np.ndarray
        Seniority-zero one-electron integrals.
    v : np.ndarray
        Seniority-zero two-electron integrals.
    w : np.ndarray
        Seniority-two two-electron integrals.

    Notes
    -----
    Currently only works for restricted/generalized integrals.

    """
    h = np.copy(np.diagonal(one_mo))
    v = np.copy(np.diagonal(np.diagonal(two_mo)))
    w = np.copy(np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 2, 3, 1)))))
    w *= 2
    w -= np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 3, 2, 1))))
    return h, v, w


def reduce_senzero_integrals(
    h: np.ndarray, v: np.ndarray, w: np.ndarray, nocc: int
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Reduce the non-zero chunks for seniority-zero of the one- and two- electron integrals.

    Parameters
    ----------
    h : np.ndarray
        Seniority-zero one-electron integrals.
    v : np.ndarray
        Seniority-zero two-electron integrals.
    w : np.ndarray
        Seniority-two two-electron integrals.
    nocc : int
        Number of pair-occupied orbitals.

    Returns
    -------
    rv : np.ndarray
        Reduced seniority-zero two-electron integrals.
    rw : np.ndarray
        Reduced seniority-two two-electron integrals.

    Notes
    -----
    Currently only works for restricted/generalized integrals.

    """
    factor = 2.0 / (nocc * 2 - 1)
    rv = np.diag(h)
    rv *= factor
    rv += v
    rw = np.zeros_like(w)
    for h_row, rw_row, rw_col in zip(h, rw, np.transpose(rw)):
        rw_row += h_row
        rw_col += h_row
    rw *= factor
    rw += w
    return rv, rw


def make_rdms(d1: np.ndarray, d2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Convert the DOCI matrices or FullCI RDM spin-blocks to full, generalized RDMs.

    Parameters
    ----------
    d1 : np.ndarray
        :math:`D_0` matrix or FullCI 1-RDM spin-blocks.
    d2 : np.ndarray
        :math:`D_2` matrix or FullCI 2-RDM spin-blocks.

    Returns
    -------
    rdm1 : np.ndarray
        Generalized one-electron RDM.
    rdm2 : np.ndarray
        Generalized two-electron RDM.

    """
    nbasis = d1.shape[1]
    nspin = nbasis * 2
    rdm1 = np.zeros((nspin, nspin), dtype=np.double)
    rdm2 = np.zeros((nspin, nspin, nspin, nspin), dtype=np.double)
    aa = rdm1[:nbasis, :nbasis]
    bb = rdm1[nbasis:, nbasis:]
    aaaa = rdm2[:nbasis, :nbasis, :nbasis, :nbasis]
    bbbb = rdm2[nbasis:, nbasis:, nbasis:, nbasis:]
    abab = rdm2[:nbasis, nbasis:, :nbasis, nbasis:]
    baba = rdm2[nbasis:, :nbasis, nbasis:, :nbasis]
    abba = rdm2[:nbasis, nbasis:, nbasis:, :nbasis]
    baab = rdm2[nbasis:, :nbasis, :nbasis, nbasis:]
    if d1.ndim == 2:
        # DOCI matrices
        for p in range(nbasis):
            aa[p, p] = d1[p, p]
            bb[p, p] = d1[p, p]
            for q in range(nbasis):
                abab[p, p, q, q] += d1[p, q]
                baba[p, p, q, q] += d1[p, q]
                aaaa[p, q, p, q] += d2[p, q]
                bbbb[p, q, p, q] += d2[p, q]
                abab[p, q, p, q] += d2[p, q]
                baba[p, q, p, q] += d2[p, q]
        rdm2 -= np.transpose(rdm2, axes=(1, 0, 2, 3))
        rdm2 -= np.transpose(rdm2, axes=(0, 1, 3, 2))
        rdm2 *= 0.5
    else:
        # FullCI RDM spin-blocks
        aa += d1[0]  # +aa
        bb += d1[1]  # +bb
        aaaa += d2[0]  # +aaaa
        bbbb += d2[1]  # +bbbb
        abab += d2[2]  # +abab
        baba += np.swapaxes(np.swapaxes(d2[2], 0, 1), 2, 3)  # +abab
        abba -= np.swapaxes(d2[2], 2, 3)  # -abab
        baab -= np.swapaxes(d2[2], 0, 1)  # -abab
    return rdm1, rdm2
