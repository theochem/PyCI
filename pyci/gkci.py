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

from scipy.special import gammaln, polygamma

from . import pyci


__all__ = [
    "add_gkci",
]


def add_gkci(wfn, t=-0.5, p=1.0, mode="cntsp", dim=3, energies=None, width=None):
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
    mode : Sequence[int] or ('cntsp' | 'gamma' | 'interval'), default='cntsp'
        Node pattern.
    dim : int, default=3
        Number of nodes (for 'gamma' mode).
    energies : np.ndarray, optional
        Orbital energies (required for 'interval' mode).
    width : float, optional
        Width of one interval (required for 'interval' mode).

    """
    # Check arguments
    if isinstance(mode, str):
        if mode == "cntsp":
            nodes = compute_nodes_cntsp(wfn.nbasis + 1)
        elif mode == "gamma":
            nodes = compute_nodes_gamma(wfn.nbasis + 1, dim)
        elif mode == "interval":
            nodes = compute_nodes_interval(wfn.nbasis + 1, energies, width)
        else:
            raise ValueError(
                f"invalid `mode` value `{mode}`; must be one of ('cntsp', 'gamma', 'interval')"
            )
    else:
        nodes = np.asarray(mode)

    # Compute cost of the most important neglected determinant
    q_max = (np.sum(nodes[: wfn.nocc_up - 1]) + (t + 1) * nodes[-1]) * p

    # Run odometer algorithm
    if isinstance(wfn, (pyci.doci_wfn, pyci.genci_wfn)):
        pyci.odometer_one_spin(wfn, cost=nodes, q_max=q_max, t=t)
    elif isinstance(wfn, pyci.fullci_wfn):
        pyci.odometer_two_spin(wfn, cost=nodes, q_max=q_max, t=t)
    else:
        raise TypeError(f"invalid `wfn` type `{type(wfn)}`; must be `pyci.wavefunction`")


def compute_nodes_cntsp(nbasis):
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


def compute_nodes_gamma(nbasis, d, maxiter=100, tol=1.0e-9):
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


def compute_nodes_interval(nbasis, es, width):
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
    w = width * 0.5
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
