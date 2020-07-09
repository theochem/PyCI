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

r"""
PyCI additional routines module.

"""

from itertools import combinations

from typing import List, Sequence

import numpy as np

from . import pyci


__all__ = [
        'add_excitations',
        'add_seniorities',
        'add_gkci',
        ]


def add_excitations(wfn: pyci.wavefunction, *excitations: Sequence[int], ref=None) -> None:
    r"""
    Convenience function for adding multiple excitation levels of determinants to a wave function.

    Parameters
    ----------
    wfn : pyci.wavefunction
        Wave function.
    excitations : Sequence[int]
        List of excitation levels of determinants to add.
    ref : np.ndarray, optional
        Reference determinant by which to determine excitation levels.
        Default is the Hartree-Fock determinant.

    """
    for e in excitations:
        wfn.add_excited_dets(e, ref=ref)


def add_seniorities(wfn: pyci.fullci_wfn, *seniorities: Sequence[int]) -> None:
    r"""
    Add determinants of the specified seniority/ies to the wave function.

    Adapted from Gaby's original code.

    Parameters
    ----------
    wfn : pyci.fullci_wfn
        FullCI wave function.
    seniorities : Sequence[int]
        List of seniorities of determinants to add.

    """
    # Check wave function
    if not isinstance(wfn, pyci.fullci_wfn):
        raise TypeError('wfn must be a pyci.fullci_wfn')

    # Check specified seniorities
    smin = wfn.nocc_up - wfn.nocc_dn
    smax = min(wfn.nocc_up, wfn.nvir_up)
    if any(s < smin or s > smax or s % 2 != smin % 2 for s in seniorities):
        raise ValueError('invalid seniority number specified')

    # Make seniority-zero occupation vectors
    sz_wfn = pyci.doci_wfn(wfn.nbasis, wfn.nocc_up)
    sz_wfn.add_all_dets()
    occ_up_array = sz_wfn.to_occ_array()
    del sz_wfn

    # Make working arrays
    brange = np.arange(wfn.nbasis, dtype=pyci.c_int)
    occs = np.empty((2, wfn.nocc_up), dtype=pyci.c_int)

    # Add determinants of specified seniorities
    for s in seniorities:
        if not s:
            # Seniority-zero
            for occs_up in occ_up_array:
                occs[0, :] = occs_up
                occs[1, :] = occs_up
                wfn.add_occs(occs)
        else:
            # Seniority-nonzero
            pairs = (wfn.nocc - s) // 2
            if pairs == wfn.nocc_dn:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    for occs_dn in combinations(occs_up, wfn.nocc_dn):
                        occs[1, :wfn.nocc_dn] = occs_dn
                        wfn.add_occs(occs)
            elif not pairs:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    virs_up = np.setdiff1d(brange, occs_up, assume_unique=True)
                    for occs_dn in combinations(virs_up, wfn.nocc_dn):
                        occs[1, :wfn.nocc_dn] = occs_dn
                        wfn.add_occs(occs)
            else:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    virs_up = np.setdiff1d(brange, occs_up, assume_unique=True)
                    for occs_i_dn in combinations(occs_up, pairs):
                        occs[1, :pairs] = occs_i_dn
                        for occs_a_dn in combinations(virs_up, wfn.nocc_dn - pairs):
                            occs[1, pairs:wfn.nocc_dn] = occs_a_dn
                            wfn.add_occs(occs)


def add_gkci(wfn: pyci.wavefunction, t: float = -0.5, p: float = 1.0, mode: str = 'cntsp',
        node_dim: int = None) -> None:
    r"""
    Add determinants to the wave function according to the odometer algorithm (Griebel-Knapeck CI).

    Adapted from Gaby and Farnaz's original code.

    Parameters
    ----------
    wfn : pyci.wavefunction
        Wave function.
    t : float, default=-0.5
        ???
    p : float, default=1.0
        ???
    mode : ('cntsp' | 'gamma' | 'cubic'), default='cntsp'
        Node pattern.
    node_dim : int, default=None
        Number of nodes (if specified).

    """
    # Check arguments
    if mode != 'gamma' and node_dim is not None:
        raise ValueError('node_dim must not be specified with mode ' + mode)
    elif mode == 'gamma' and node_dim is None:
        raise ValueError('node_dim must be specified with mode \'gamma\'')

    # Construct nodes
    if mode == 'cntsp':
        nodes = list()
        shell = 1
        while len(nodes) < wfn.nbasis:
            nodes.extend([shell - 1.0] * shell ** 2)
            shell += 1
    elif mode == 'gamma':
        raise NotImplementedError
    elif mode == 'cubic':
        raise NotImplementedError
    else:
        raise ValueError('mode must be one of (\'cntsp\', \'gamma\', \'cubic\'')
    nodes = np.array(nodes[:wfn.nbasis])

    # Run odometer algorithm
    if isinstance(wfn, (pyci.doci_wfn, pyci.genci_wfn)):
        _odometer_one_spin(wfn, nodes, t, p)
    elif isinstance(wfn, pyci.fullci_wfn):
        _odometer_two_spin(wfn, nodes, t, p)
    else:
        raise TypeError('wfn must be a pyci.{doci,fullci,genci}_wfn')


def _odometer_one_spin(wfn: pyci.one_spin_wfn, nodes: List[int], t: float, p: float) -> None:
    r"""
    Run the odometer algorithm for a one-spin wave function.

    """
    aufbau_occs = np.arange(wfn.nocc_up, dtype=pyci.c_int)
    new_occs = np.copy(aufbau_occs)
    old_occs = np.copy(aufbau_occs)
    # Index of last electron
    j_electron = wfn.nocc_up  - 1
    # Compute cost of the most important neglected determinant
    nodes_s = nodes[new_occs]
    qs_neg = np.sum(nodes_s[:-1]) * p + (t + 1) * nodes[-1] * p
    # Select determinants
    while True:
        if new_occs[wfn.nocc_up - 1] >= wfn.nbasis:
            # Reject determinant b/c of occupying an inactive or non-existant orbital;
            # go back to last-accepted determinant and excite the previous electron
            new_occs[:] = old_occs
            j_electron -= 1
        else:
            # Compute nodes and cost of occupied orbitals
            nodes_s = nodes[new_occs]
            qs = np.sum(nodes_s) + t * np.max(nodes_s)
            if qs < qs_neg:
                # Accept determinant and excite the last electron again
                wfn.add_occs(new_occs)
                j_electron = wfn.nocc_up - 1
            else:
                # Reject determinant because of high cost; go back to last-accepted
                # determinant and excite the previous electron
                new_occs[:] = old_occs
                j_electron -= 1
        if j_electron < 0:
            # Done
            break
        # Record last-accepted determinant and excite j_electron
        old_occs[:] = new_occs
        new_occs[j_electron] += 1
        if j_electron != wfn.nocc_up - 1:
            for k in range(j_electron + 1, wfn.nocc_up):
                new_occs[k] = new_occs[j_electron] + k - j_electron


def _odometer_two_spin(wfn: pyci.two_spin_wfn, nodes: List[int], t: float, p: float) -> None:
    r"""
    Run the odometer algorithm for a two-spin wave function.

    """
    raise NotImplementedError
