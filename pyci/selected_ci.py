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

r"""PyCI selected CI routines module."""

from itertools import combinations
from typing import List, Sequence, Union

import numpy as np

from scipy.special import gammaln, polygamma

from . import pyci


__all__ = [
    "add_excitations",
    "add_seniorities",
    "add_gkci",
]


def add_excitations(wfn: pyci.wavefunction, *excitations: Sequence[int], ref=None) -> None:
    r"""
    Add multiple excitation levels of determinants to a wave function.

    Convenience function.

    Parameters
    ----------
    wfn : pyci.wavefunction
        Wave function.
    excitations : Sequence[int]
        List of excitation levels of determinants to add.
    ref : numpy.ndarray, optional
        Reference determinant by which to determine excitation levels.
        Default is the Hartree-Fock determinant.

    """
    for e in excitations:
        wfn.add_excited_dets(e, ref=ref)


def add_seniorities(wfn: pyci.fullci_wfn, *seniorities: Sequence[int]) -> None:
    r"""
    Add determinants of the specified seniority/ies to the wave function.

    Parameters
    ----------
    wfn : pyci.fullci_wfn
        FullCI wave function.
    seniorities : Sequence[int]
        List of seniorities of determinants to add.

    """
    # Check wave function
    if not isinstance(wfn, pyci.fullci_wfn):
        raise TypeError(f"invalid `wfn` type `{type(wfn)}`; must be `pyci.fullci_wfn`")

    # Check specified seniorities
    smin = wfn.nocc_up - wfn.nocc_dn
    smax = min(wfn.nocc_up, wfn.nvir_up)
    if any(s < smin or s > smax or s % 2 != smin % 2 for s in seniorities):
        raise ValueError(f"invalid seniority number in `seniorities = {seniorities}`")

    # Make seniority-zero occupation vectors
    sz_wfn = pyci.doci_wfn(wfn.nbasis, wfn.nocc_up, wfn.nocc_up)
    sz_wfn.add_all_dets()
    occ_up_array = sz_wfn.to_occ_array()
    del sz_wfn

    # Make working arrays
    brange = np.arange(wfn.nbasis, dtype=pyci.c_long)
    occs = np.empty((2, wfn.nocc_up), dtype=pyci.c_long)

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
                        occs[1, : wfn.nocc_dn] = occs_dn
                        wfn.add_occs(occs)
            elif not pairs:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    virs_up = np.setdiff1d(brange, occs_up, assume_unique=True)
                    for occs_dn in combinations(virs_up, wfn.nocc_dn):
                        occs[1, : wfn.nocc_dn] = occs_dn
                        wfn.add_occs(occs)
            else:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    virs_up = np.setdiff1d(brange, occs_up, assume_unique=True)
                    for occs_i_dn in combinations(occs_up, pairs):
                        occs[1, :pairs] = occs_i_dn
                        for occs_a_dn in combinations(virs_up, wfn.nocc_dn - pairs):
                            occs[1, pairs : wfn.nocc_dn] = occs_a_dn
                            wfn.add_occs(occs)


def add_gkci(
    wfn: pyci.wavefunction,
    t: float = -0.5,
    p: float = 1.0,
    mode: Union[str, List[int]] = "cntsp",
    dim: int = 3,
) -> None:
    r"""
    Add determinants to the wave function according to the odometer algorithm (Griebel-Knapeck CI).

    Adapted from Gaby and Farnaz's original code.

    Parameters
    ----------
    wfn : pyci.wavefunction
        Wave function.
    t : float, default=-0.5
        Smoothness factor.
    p : float, default=1.0
        Cost factor.
    mode : List[int] or ('cntsp' | 'gamma'), default='cntsp'
        Node pattern.
    dim : int, default=3
        Number of nodes (for 'gamma' mode).

    """
    # Check arguments
    if isinstance(mode, str):
        if mode == "cntsp":
            nodes = compute_nodes_cntsp(wfn.nbasis)
        elif mode == "gamma":
            nodes = compute_nodes_gamma(wfn.nbasis, dim)
        else:
            raise ValueError(f"invalid `mode` value `{mode}`; must be one of ('cntsp', 'gamma')")
    else:
        nodes = np.asarray(mode)

    # Run odometer algorithm
    if isinstance(wfn, (pyci.doci_wfn, pyci.genci_wfn)):
        odometer_one_spin(wfn, nodes[: wfn.nbasis], t, p)
    elif isinstance(wfn, pyci.fullci_wfn):
        odometer_two_spin(wfn, nodes[: wfn.nbasis], t, p)
    else:
        raise TypeError(f"invalid `wfn` type `{type(wfn)}`; must be `pyci.wavefunction`")


def compute_nodes_cntsp(nbasis: int) -> None:
    r"""
    Approximate the number of nodes for each function in a basis set as a sphere.

    Parameters
    ----------
    nbasis : int
        Number of basis functions.

    Returns
    -------
    nodes : np.ndarray
        Number of nodes for each basis function.

    """
    nodes = list()
    shell = 1
    while len(nodes) < nbasis:
        nodes.extend([shell - 1.0] * shell ** 2)
        shell += 1
    return np.asarray(nodes)


def compute_nodes_gamma(nbasis: int, d: int, maxiter: int = 100, tol: float = 1.0e-9):
    r"""
    Approximate the number of nodes for each function in a basis set as a polynomial.

    Approximate the number of nodes by solving the following equations for ``n`` (for each basis
    function ``k``) via the Halley method:

    .. math::

        f(n) = \frac{\Gamma(n + d + 1)}{\Gamma(d + 1)\Gamma(n + 1)} - k - 1 = 0

    .. math::

        \delta n_{i + 1} = n_i - \frac{2 f(n_i) f'(n_i)}{2 {[f'(n_i)]}^2 - f(n_i) f"(n_i)}

    Parameters
    ----------
    nbasis : int
        Number of basis functions.
    d : int
        Dimension of polynomial.
    maxiter : int, default=100
        Maximum number of iterations to perform.
    tol : float, default=1.0e-9
        Convergence tolerance.

    Returns
    -------
    nodes : np.ndarray
        Number of nodes for each basis function.

    """
    nodes = np.zeros(nbasis)
    # Iterate over basis functions; n = nodes[0] is always equal to zero
    n = 0
    for k in range(1, nbasis):
        # Optimize n using the Halley method
        for _ in range(maxiter):
            # Compute components of function f(n) = Gamma(d + n + 1) / Gamma(n + 1) * Gamma(d + 1)
            # and its first and second derivatives with respect to the number of nodes n
            gn, gd, gnd = gammaln((n + 1.0, d + 1.0, n + d + 1.0))
            dgn, dgnd = polygamma(0, (n + 1.0, n + d + 1.0))
            pgn, pgnd = polygamma(1, (n + 1.0, n + d + 1.0))
            # Compute Halley step
            t = np.exp(gnd - gd - gn)
            f = t - k - 1.0
            fp = (dgnd - dgn) * t
            fpp = (dgnd * dgnd + dgn * dgn - 2.0 * dgnd * dgn + pgnd - pgn) * t
            dn = 2.0 * f * (fp / (2.0 * fp * fp - f * fpp))
            # If we've converged, we're done
            if np.abs(dn) < tol:
                break
            # Update n using Halley step
            n -= dn
        else:
            raise RuntimeError(f"Did not converge in {maxiter} iterations")
        # Update nodes array
        nodes[k] = n
        # Set initial guess for next basis function
        n += n - nodes[k - 1]
    return nodes


def odometer_one_spin(wfn: pyci.one_spin_wfn, nodes: List[int], t: float, p: float) -> None:
    r"""Run the odometer algorithm for a one-spin wave function."""
    aufbau_occs = np.arange(wfn.nocc_up, dtype=pyci.c_long)
    new_occs = np.copy(aufbau_occs)
    old_occs = np.copy(aufbau_occs)
    # Index of last particle
    j_particle = wfn.nocc_up - 1
    # Compute cost of the most important neglected determinant
    nodes_s = nodes[new_occs]
    qs_neg = np.sum(nodes_s[:-1]) * p + (t + 1) * nodes[-1] * p
    # Select determinants
    while True:
        if new_occs[wfn.nocc_up - 1] >= wfn.nbasis:
            # Reject determinant b/c of occupying an inactive or non-existant orbital;
            # go back to last-accepted determinant and excite the previous particle
            new_occs[:] = old_occs
            j_particle -= 1
        else:
            # Compute nodes and cost of occupied orbitals
            nodes_s = nodes[new_occs]
            qs = np.sum(nodes_s) + t * np.max(nodes_s)
            if qs < qs_neg:
                # Accept determinant and excite the last particle again
                wfn.add_occs(new_occs)
                j_particle = wfn.nocc_up - 1
            else:
                # Reject determinant because of high cost; go back to last-accepted
                # determinant and excite the previous particle
                new_occs[:] = old_occs
                j_particle -= 1
        if j_particle < 0:
            # Done
            break
        # Record last-accepted determinant and excite j_particle
        old_occs[:] = new_occs
        new_occs[j_particle] += 1
        if j_particle != wfn.nocc_up - 1:
            for k in range(j_particle + 1, wfn.nocc_up):
                new_occs[k] = new_occs[j_particle] + k - j_particle


def odometer_two_spin(wfn: pyci.two_spin_wfn, nodes: List[int], t: float, p: float) -> None:
    r"""Run the odometer algorithm for a two-spin wave function."""
    aufbau_occs = np.arange(wfn.nocc, dtype=pyci.c_long)
    aufbau_occs[wfn.nocc_up :] -= wfn.nocc_up
    new_occs = np.copy(aufbau_occs)
    old_occs = np.copy(aufbau_occs)
    # Index of last particle
    j_particle = wfn.nocc - 1
    # Compute cost of the most important neglected determinant
    nodes_up = nodes[new_occs[: wfn.nocc_up]]
    nodes_dn = nodes[new_occs[wfn.nocc_up :]]
    q_up_neg = np.sum(nodes_up[:-1]) * p + (t + 1) * nodes[-1] * p
    q_dn_neg = np.sum(nodes_dn[:-1]) * p + (t + 1) * nodes[-1] * p
    # Select determinants
    while True:
        if max(new_occs[wfn.nocc_up - 1], new_occs[wfn.nocc - 1]) >= wfn.nbasis:
            # Reject determinant b/c of occupying an inactive or non-existant orbital;
            # go back to last-accepted determinant and excite the previous particle
            new_occs[:] = old_occs
            j_particle -= 1
        else:
            # Compute nodes and cost of occupied orbitals
            nodes_up = nodes[new_occs[: wfn.nocc_up]]
            nodes_dn = nodes[new_occs[wfn.nocc_up :]]
            q_up = np.sum(nodes_up) + t * np.max(nodes_up)
            q_dn = np.sum(nodes_dn) + t * np.max(nodes_dn)
            if q_up < q_up_neg and q_dn < q_dn_neg:
                # Accept determinant and excite the last particle again
                wfn.add_occs(new_occs.reshape(2, -1))
                j_particle = wfn.nocc - 1
            else:
                # Reject determinant because of high cost; go back to last-accepted
                # determinant and excite the previous particle
                new_occs[:] = old_occs
                j_particle -= 1
        if j_particle < 0:
            # Done
            break
        # Record last-accepted determinant and excite j_particle
        old_occs[:] = new_occs
        new_occs[j_particle] += 1
        if j_particle < wfn.nocc_up:
            # excite spin-up particle
            for k in range(j_particle + 1, wfn.nocc_up):
                new_occs[k] = new_occs[j_particle] + k - j_particle
        elif j_particle < wfn.nocc - 1:
            # excite spin-down particle
            for k in range(j_particle + 1, wfn.nocc):
                new_occs[k] = new_occs[j_particle] + k - j_particle
