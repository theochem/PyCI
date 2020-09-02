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

r"""PyCI Griebel-Knapek CI routines module."""

from typing import Sequence, Union

import numpy as np

from scipy.special import gammaln, polygamma

from . import pyci


__all__ = [
    "add_gkci",
]


def add_gkci(
    wfn: pyci.wavefunction,
    t: float = -0.5,
    p: float = 1.0,
    mode: Union[str, Sequence[int]] = "cntsp",
    dim: int = 3,
    energies: np.ndarray = None,
    width: float = None,
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
    mode : Sequence[int] or ('cntsp' | 'gamma' | 'interval'), default='cntsp'
        Node pattern.
    dim : int, default=3
        Number of nodes (for 'gamma' mode).
    es : np.ndarray, optional
        Orbital energies (required for 'interval' mode).
    width : float, optional
        Width of one interval (required for 'interval' mode).

    """
    # Check arguments
    if isinstance(mode, str):
        if mode == "cntsp":
            nodes = compute_nodes_cntsp(wfn.nbasis)
        elif mode == "gamma":
            nodes = compute_nodes_gamma(wfn.nbasis, dim)
        elif mode == "interval":
            nodes = compute_nodes_interval(wfn.nbasis, energies, width)
        else:
            raise ValueError(
                f"invalid `mode` value `{mode}`; must be one of ('cntsp', 'gamma', 'interval')"
            )
    else:
        nodes = np.asarray(mode)

    # Run odometer algorithm
    if isinstance(wfn, (pyci.doci_wfn, pyci.genci_wfn)):
        odometer_one_spin(wfn, nodes, t, p)
    elif isinstance(wfn, pyci.fullci_wfn):
        odometer_two_spin(wfn, nodes, t, p)
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
    nodes = np.zeros(nbasis)
    i = 0
    shell = 1
    while i != nbasis:
        j = min(nbasis, i + shell ** 2)
        nodes[i:j] = shell - 1
        i = j
        shell += 1
    return nodes


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
    d += 1.0
    gd = gammaln(d)
    n = 0.0
    for k in range(1, nbasis):
        # Optimize n using the Halley method
        for _ in range(maxiter):
            # Compute components of function f(n) = Gamma(d + n + 1) / Gamma(n + 1) * Gamma(d + 1)
            # and its first and second derivatives with respect to the number of nodes n
            args = n + 1, n + d
            gn, gnd = gammaln(args)
            pn, pnd, p2n, p2nd = polygamma((0, 0, 1, 1), args * 2)
            # Compute Halley step in direction n
            a = np.exp(gnd - gn - gd)
            b = a - k - 1
            c = pn - pnd
            dn = (2 * b * c) / (2 * a * c * c + b * (p2nd - p2n - c * (pn + pnd)))
            # Update n
            n += dn
            # Check for convergence
            if np.abs(dn) < tol:
                break
        else:
            raise RuntimeError(f"Did not converge in {maxiter} iterations")
        # Update nodes array
        nodes[k] = n
        # Set initial guess for next basis function
        n += n - nodes[k - 1]
    return nodes


def compute_nodes_interval(nbasis: int, es: np.ndarray, width: float):
    r"""
    Approximate the number of nodes for each function via intervals.

    Parameters
    ----------
    nbasis : int
        Number of basis functions.
    es : np.ndarray
        Orbital energies.
    width : float
        Width of one interval.

    Returns
    -------
    nodes : np.ndarray
        Number of nodes for each basis function.

    """
    # Compute initial intervals
    w = width / 2
    lower = es - w
    upper = es + w
    # Compute union of intervals
    nint = 0
    for k in range(1, nbasis):
        if es[k] - w < es[k - 1] + w:
            upper[nint] = es[k] + w
        else:
            nint += 1
            lower[nint] = es[k] - w
            upper[nint] = es[k] + w
    nint += 1
    # Compute nodes
    nodes = np.zeros(nbasis)
    for k in range(nbasis):
        for n in range(nint):
            if es[k] > upper[n]:
                nodes[k] += (upper[n] - lower[n]) / width
            else:
                nodes[k] += (es[k] - lower[n]) / width + 0.5
                break
    return nodes


def odometer_one_spin(wfn: pyci.one_spin_wfn, nodes: np.ndarray, t: float, p: float) -> None:
    r"""Run the odometer algorithm for a one-spin wave function."""
    old_occs = np.arange(wfn.nocc_up, dtype=pyci.c_long)
    new_occs = np.copy(old_occs)
    # Index of last particle
    j = wfn.nocc_up - 1
    # Compute cost of the most important neglected determinant
    nodes_up = nodes[new_occs]
    q_max = max(1.0, np.sum(nodes_up) + t * nodes_up[-1]) * p
    # Select determinants
    while True:
        if new_occs[wfn.nocc_up - 1] >= wfn.nbasis:
            # Reject determinant and cycle j
            new_occs[:] = old_occs
            j -= 1
        else:
            # Compute nodes and cost of occupied orbitals
            nodes_up[:] = nodes[new_occs]
            # Add or reject determinant and cycle j
            if np.sum(nodes_up) + t * nodes_up[-1] < q_max:
                wfn.add_occs(new_occs)
                j = wfn.nocc_up - 1
            else:
                new_occs[:] = old_occs
                j -= 1
        # Check termination condition
        if j < 0:
            break
        # Generate next determinant
        old_occs[:] = new_occs
        new_occs[j] += 1
        if j < wfn.nocc_up - 1:
            # Excite spin-up particle
            for k in range(j + 1, wfn.nocc_up):
                new_occs[k] = new_occs[j] + k - j


def odometer_two_spin(wfn: pyci.two_spin_wfn, nodes: np.ndarray, t: float, p: float) -> None:
    r"""Run the odometer algorithm for a two-spin wave function."""
    old_occs = np.arange(wfn.nocc, dtype=pyci.c_long)
    old_occs[wfn.nocc_up :] -= wfn.nocc_up
    new_occs = np.copy(old_occs)
    # Index of last particle
    j = wfn.nocc - 1
    # Compute cost of the most important neglected determinant
    nodes_up = nodes[new_occs[: wfn.nocc_up]]
    nodes_dn = nodes[new_occs[wfn.nocc_up :]]
    q_up_max = max(1.0, np.sum(nodes_up) + t * nodes_up[-1]) * p
    q_dn_max = max(1.0, np.sum(nodes_dn) + t * nodes_dn[-1]) * p
    # Select determinants
    while True:
        if max(new_occs[[wfn.nocc_up - 1, wfn.nocc - 1]]) >= wfn.nbasis:
            # Reject determinant and cycle j
            new_occs[:] = old_occs
            j -= 1
        else:
            # Compute nodes and cost of occupied orbitals
            nodes_up = nodes[new_occs[: wfn.nocc_up]]
            nodes_dn = nodes[new_occs[wfn.nocc_up :]]
            q_up = np.sum(nodes_up) + t * nodes_up[-1]
            q_dn = np.sum(nodes_dn) + t * nodes_dn[-1]
            # Add or reject determinant and cycle j
            if q_up < q_up_max and q_dn < q_dn_max:
                wfn.add_occs(new_occs.reshape(2, -1))
                j = wfn.nocc - 1
            else:
                new_occs[:] = old_occs
                j -= 1
        # Check termination condition
        if j < 0:
            break
        # Generate next determinant
        old_occs[:] = new_occs
        new_occs[j] += 1
        if j < wfn.nocc_up:
            # Excite spin-up particle
            for k in range(j + 1, wfn.nocc_up):
                new_occs[k] = new_occs[j] + k - j
        elif j < wfn.nocc - 1:
            # Excite spin-down particle
            for k in range(j + 1, wfn.nocc):
                new_occs[k] = new_occs[j] + k - j
