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

r"""PyCI utility module."""

import numpy as np
import pyci._pyci as pyci

__all__ = [
    "make_senzero_integrals",
    "reduce_senzero_integrals",
    "spinize_rdms",
    "odometer_one_spin",
    "odometer_two_spin",
]


def make_senzero_integrals(one_mo, two_mo):
    r"""
    Return the non-zero chunks for seniority-zero of the full one- and two- particle integrals.

    Parameters
    ----------
    one_mo : numpy.ndarray
        Full one-particle integral array.
    two_mo : numpy.ndarray
        Full two-particle integral array.

    Returns
    -------
    h : numpy.ndarray
        Seniority-zero one-particle integrals.
    v : numpy.ndarray
        Seniority-zero two-particle integrals.
    w : numpy.ndarray
        Seniority-two two-particle integrals.

    """
    h = np.copy(np.diagonal(one_mo))
    v = np.copy(np.diagonal(np.diagonal(two_mo)))
    w = np.copy(np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 2, 3, 1)))))
    w *= 2
    w -= np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 3, 2, 1))))
    return h, v, w


def reduce_senzero_integrals(h, v, w, nocc):
    r"""
    Reduce the non-zero chunks for seniority-zero of the one- and two- particle integrals.

    Parameters
    ----------
    h : numpy.ndarray
        Seniority-zero one-particle integrals.
    v : numpy.ndarray
        Seniority-zero two-particle integrals.
    w : numpy.ndarray
        Seniority-two two-particle integrals.
    nocc : int
        Number of pair-occupied orbitals.

    Returns
    -------
    rv : numpy.ndarray
        Reduced seniority-zero two-particle integrals.
    rw : numpy.ndarray
        Reduced seniority-two two-particle integrals.

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

def spinize_rdms(d1, d2):
    r"""
    Convert the DOCI matrices or FullCI RDM spin-blocks to full, generalized RDMs.

    Parameters
    ----------
    d1 : numpy.ndarray
        :math:`D_0` matrix or FullCI 1-RDM spin-blocks.
    d2 : numpy.ndarray
        :math:`D_2` matrix or FullCI 2-RDM spin-blocks.

    Returns
    -------
    rdm1 : numpy.ndarray
        Generalized one-particle RDM.
    rdm2 : numpy.ndarray
        Generalized two-particle RDM.

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


def spinize_rdms_1234(d1, d2, d3, d4):
    r"""
    Convert the DOCI matrices or FullCI RDM spin-blocks to full, generalized RDMs.

    Parameters
    ----------
    .. math::
        d_0 = \left<pp|qq\right>
    .. math::
        d_2 = \left<pq|pq\right>
    .. math::
        d_3 = \left<pqr|pqr\right>
    .. math::
        d_4 = \left<pqq|prr\right>
    Returns
    -------
    rdm1 : numpy.ndarray
        Generalized one-particle RDM.
    rdm2 : numpy.ndarray
        Generalized two-particle RDM.
    rdm3 : numpy.ndarray
        Generalized three-particle RDM.
    """
    if d1.ndim != 2:
        raise TypeError('wfn must be a DOCI')
    nbasis = d1.shape[1]
    nspin = nbasis * 2
    rdm1 = np.zeros((nspin, nspin), dtype=np.double)
    rdm2 = np.zeros((nspin, nspin, nspin, nspin), dtype=np.double)
    rdm3 = np.zeros((nspin, nspin, nspin, nspin,nspin,nspin), dtype=np.double)
    aa = rdm1[:nbasis, :nbasis]
    bb = rdm1[nbasis:, nbasis:]
    aaaa = rdm2[:nbasis, :nbasis, :nbasis, :nbasis]
    bbbb = rdm2[nbasis:, nbasis:, nbasis:, nbasis:]
    abab = rdm2[:nbasis, nbasis:, :nbasis, nbasis:]
    baba = rdm2[nbasis:, :nbasis, nbasis:, :nbasis]
    abba = rdm2[:nbasis, nbasis:, nbasis:, :nbasis]
    baab = rdm2[nbasis:, :nbasis, :nbasis, nbasis:]
    aaaaaa = rdm3[:nbasis,:nbasis,:nbasis,:nbasis,:nbasis,:nbasis]
    bbbbbb = rdm3[nbasis:,nbasis:,nbasis:,nbasis:,nbasis:,nbasis:]
    bbabba = rdm3[nbasis:,nbasis:,:nbasis,nbasis:,nbasis:,:nbasis]
    aabaab = rdm3[:nbasis,:nbasis,nbasis:,:nbasis,:nbasis,nbasis:]
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
            bbabba[p, q, q, p, q, q] += 2.0 * d2[p, q]
            aabaab[p, q, q, p, q, q] += 2.0 * d2[p, q]
            for r in range(nbasis):               
                bbabba[p, q, q, p, r, r] += 2.0 * d4[p, q, r]
                aabaab[p, q, q, p, r, r] += 2.0 * d4[p, q, r] 
                aaaaaa[p, q, r, p, q, r] += d3[p, q, r]
                bbbbbb[p, q, r, p, q, r] += d3[p, q, r]
                bbabba[p, q, r, p, q, r] += d3[p, q, r]
                aabaab[p, q, r, p, q, r] += d3[p, q, r]
    rdm2 -= np.transpose(rdm2, axes=(1, 0, 2, 3))
    rdm2 -= np.transpose(rdm2, axes=(0, 1, 3, 2))
    rdm2 *= 0.5
    rdm3 += np.einsum('ijklmn -> ijknlm', rdm3)+\
            np.einsum('ijklmn -> ijkmnl', rdm3) 
    rdm3 *= 1
    rdm3 -= np.einsum('ijklmn -> ijkmln', rdm3)
    rdm3 += np.einsum('ijklmn -> kijlmn', rdm3)+\
            np.einsum('ijklmn -> jkilmn', rdm3) 
    aaaaaa *= 1/3
    bbbbbb *= 1/3
    rdm3 -= np.einsum('ijklmn -> jiklmn', rdm3)
    rdm3 *= 0.5
    return rdm1, rdm2, rdm3

def spin_free_rdms(d1, d2, d3=None, d4=None):
    r"""

    Wrapper of spinze_rdms function that sums over the spin degree of freedom 
    to obtain spinless rdms.

    Parameters
    ----------
    d1 : numpy.ndarray
        :math:`D_0` matrix or FullCI 1-RDM spin-blocks.
    d2 : numpy.ndarray
        :math:`D_2` matrix or FullCI 2-RDM spin-blocks.
    d3 : numpy.ndarray
        :math `D_3` matrix for 3-RDM spin-blocks    
    d4 : numpy.ndarray
        :math `D_4` matrix for 3-RDM spin-blocks   

    Returns
    -------
    rdm1 : numpy.ndarray
        Spin traced one-particle RDM.
    rdm2 : numpy.ndarray
        Spin traced two-particle RDM.
    rdm3 : numpy.ndarray
        Spin traced three-particle RDM.
    """
    nbasis = d1.shape[1]
    rdm1, rdm2, rdm3 = spinize_rdms_1234(d1, d2, d3, d4)
    rdm1_sf = np.zeros((nbasis, nbasis), dtype=np.double)
    rdm2_sf = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.double)
    rdm3_sf = np.zeros((nbasis, nbasis, nbasis, nbasis, nbasis, nbasis), dtype=np.double)
    aa = rdm1[:nbasis, :nbasis]
    bb = rdm1[nbasis:, nbasis:]
    aaaa = rdm2[:nbasis, :nbasis, :nbasis, :nbasis]
    bbbb = rdm2[nbasis:, nbasis:, nbasis:, nbasis:]
    abab = rdm2[:nbasis, nbasis:, :nbasis, nbasis:]
    baba = rdm2[nbasis:, :nbasis, nbasis:, :nbasis]
    if d1.ndim == 2:
    # DOCI matrices
        rdm3_sf=np.zeros((nbasis, nbasis, nbasis, nbasis,nbasis,nbasis), dtype=np.double)
        aaaaaa= rdm3[:nbasis, :nbasis, :nbasis, :nbasis, :nbasis, :nbasis]
        bbbbbb= rdm3[nbasis:, nbasis:, nbasis:, nbasis:, nbasis:, nbasis:]
        bbabba= rdm3[nbasis:, nbasis:, :nbasis, nbasis:, nbasis:, :nbasis]
        abbabb= rdm3[:nbasis, nbasis:, nbasis:, :nbasis, nbasis:, nbasis:]
        babbab = rdm3[nbasis:, :nbasis, nbasis:, nbasis:, :nbasis, nbasis:]
        aabaab = rdm3[:nbasis, :nbasis, nbasis:, :nbasis, :nbasis, nbasis:]
        abaaba = rdm3[:nbasis, nbasis:, :nbasis, :nbasis, nbasis:, :nbasis]
        baabaa = rdm3[nbasis:, :nbasis, :nbasis, nbasis:, :nbasis, :nbasis]
        rdm1_sf = aa + bb
        rdm2_sf = aaaa + abab+ baba+ bbbb
        rdm3_sf = aaaaaa + bbbbbb + aabaab + abaaba + baabaa + bbabba + babbab + abbabb
        return (rdm1_sf, rdm2_sf, rdm3_sf)
    else:
        # FullCI RDM spin-blocks
        rdm1_sf = aa + bb
        rdm2_sf = aaaa + abab + baba + bbbb
        return (rdm1_sf, rdm2_sf)

def odometer_one_spin(wfn, cost, t, qmax):
    r"""
    Iterates over determinants of a one-spin wave function, adding those that satisfy
    the given condition to the wave function in place.

    Parameters
    ----------
    wfn : pyci.wavefunction
        The wave function to be modified.
    cost : np.ndarray
        An array where each element represents the weight or significance of the
        corresponding node in the determinant. These weights are used in evaluating
        whether a determinant satisfies a condition based on its total weighted sum.
    t : float
        A scaling factor for the last node's weight in the condition check, adjusting
        the influence of the last particle's node weight on the overall determinant evaluation.
    qmax : float
        Cost of the most important neglected determinant.

    """
    old = np.arange(wfn.nocc_up, dtype=pyci.c_long)
    new = np.copy(old)
    # Index of last particle
    j = wfn.nocc_up - 1
    # Select determinants
    while True:
        if new[-1] < wfn.nbasis and (np.sum(cost[new]) + t * cost[new[-1]]) < qmax:
            # Accept determinant and go back to last particle
            wfn.add_occs(new)
            j = wfn.nocc_up - 1
        else:
            # Reject determinant and cycle j
            new[:] = old
            j -= 1
        # Check termination condition
        if j < 0:
            break
        # Generate next determinant
        old[:] = new
        new[j:] = np.arange(new[j] + 1, new[j] + wfn.nocc_up - j + 1)


def odometer_two_spin(wfn, cost, t, qmax):
    r"""
    Processes a two-spin wave function, applying the odometer algorithm to both up-spin
    and down-spin components and combining the accepted determinants into the wave function.

    Parameters
    ----------
    wfn : pyci.wavefunction
        The wave function to be modified.
    cost : np.ndarray
        An array where each element represents the weight or significance of the
        corresponding node in the determinant. These weights are used in evaluating
        whether a determinant satisfies a condition based on its total weighted sum.
    t : float
        A scaling factor for the last node's weight in the condition check, adjusting
        the influence of the last particle's node weight on the overall determinant evaluation.
    qmax : float
        Cost of the most important neglected determinant.

    """

    wfn_up = pyci.doci_wfn(wfn.nbasis, wfn.nocc_up, wfn.nocc_up)
    odometer_one_spin(wfn_up, cost, t, qmax)
    if not len(wfn_up):
        return
    if wfn.nocc_dn:
        wfn_dn = pyci.doci_wfn(wfn.nbasis, wfn.nocc_dn, wfn.nocc_dn)
        odometer_one_spin(wfn_dn, cost, t, qmax)
        if not len(wfn_dn):
            return
        for i in range(len(wfn_up)):
            det_up = wfn_up[i]
            for j in range(len(wfn_dn)):
                wfn.add_det(np.vstack((det_up, wfn_dn[j])))
    else:
        det_dn = np.zeros_like(wfn_up[0])
        for i in range(len(wfn_up)):
            wfn.add_det(np.vstack((wfn_up[i], det_dn)))
