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

r"""PyCI Griebel-Knapek CI module."""

import numpy as np

from . import pyci


__all__ = [
    "add_energy",
]


def add_energy(wfn, t=-0.5, p=1.0, energies=None):
    r"""
    Add determinants to the wave function according to the odometer algorithm
    (Griebel-Knapeck CI) [GKCI1]_.

    .. [GKCI1] Anderson, James SM, Farnaz Heidar-Zadeh, and Paul W. Ayers. "Breaking the curse of
               dimension for the electronic Schrödinger equation with functional analysis."
               *Computational and Theoretical Chemistry* 1142 (2018): 66-77.

    Parameters
    ----------
    wfn : pyci.wavefunction
        Wave function.
    t : float, default=-0.5
        Smoothness factor.
    p : float, default=1.0
        Cost factor.
    energies : np.ndarray
        Orbital energies 

    """
    # Run odometer algorithm
    if isinstance(wfn, (pyci.doci_wfn, pyci.genci_wfn)):
        odometer_one_spin(wfn, energies, t, p)
    elif isinstance(wfn, pyci.fullci_wfn):
        odometer_two_spin(wfn, energies, t, p)
    else:
        raise TypeError(f"invalid `wfn` type `{type(wfn)}`; must be `pyci.wavefunction`")


def odometer_one_spin(wfn, nodes, t, p):
    r"""Run the odometer algorithm for a one-spin wave function."""
    old = np.arange(wfn.nocc_up, dtype=pyci.c_long)
    new = np.copy(old)
    # Index of last particle
    j = wfn.nocc_up - 1
    # Compute cost of the most important neglected determinant
    q_max = (np.sum(nodes[: wfn.nocc_up - 1]) + (t + 1) * nodes[-1]) * p
    # Select determinants
    while True:
        if new[-1] < wfn.nbasis and (np.sum(nodes[new]) + t * nodes[new[-1]]) < q_max:
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


def odometer_two_spin(wfn, nodes, t, p):
    r"""Run the odometer algorithm for a two-spin wave function."""
    wfn_up = pyci.doci_wfn(wfn.nbasis, wfn.nocc_up, wfn.nocc_up)
    odometer_one_spin(wfn_up, nodes, t, p)
    if not len(wfn_up):
        return
    if wfn.nocc_dn:
        wfn_dn = pyci.doci_wfn(wfn.nbasis, wfn.nocc_dn, wfn.nocc_dn)
        odometer_one_spin(wfn_dn, nodes, t, p)
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

