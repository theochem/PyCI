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

r"""PyCI module."""

import numpy as np

from .pyci import __version__, c_long, c_ulong, c_double, sparse_op

from .pyci import hamiltonian, wavefunction, one_spin_wfn, two_spin_wfn
from .pyci import doci_wfn, fullci_wfn, genci_wfn, sparse_op
from .pyci import get_num_threads, set_num_threads, popcnt, ctz
from .pyci import compute_overlap, compute_rdms, add_hci, compute_enpt2

from .seniority_ci import add_seniorities
from .gkci import add_gkci


__all__ = [
    "__version__",
    "c_long",
    "c_ulong",
    "c_double",
    "hamiltonian",
    "wavefunction",
    "one_spin_wfn",
    "two_spin_wfn",
    "doci_wfn",
    "fullci_wfn",
    "genci_wfn",
    "sparse_op",
    "get_num_threads",
    "set_num_threads",
    "popcnt",
    "ctz",
    "add_hci",
    "compute_overlap",
    "compute_rdms",
    "compute_enpt2",
    "solve",
    "add_excitations",
    "make_senzero_integrals",
    "reduce_senzero_integrals",
    "expand_rdms",
    "add_seniorities",
    "add_gkci",
]


__version__ = __version__
r"""
PyCI version string.

"""


c_long = c_long
r"""
Signed integer C++ dtype.

"""


c_ulong = c_ulong
r"""
Unsigned integer C++ dtype.

"""


c_double = c_double
r"""
Floating point C++ dtype.

"""


def solve(*args, n=1, c0=None, ncv=-1, maxiter=-1, tol=1.0e-12):
    r"""
    Solve a CI eigenproblem.

    Parameters
    ----------
    args : (pyci.sparse_op,) or (pyci.hamiltonian, pyci.wavefunction)
        System to solve.
    n : int, default=1
        Number of lowest eigenpairs to find.
    c0 : numpy.ndarray, default=[1,0,...,0]
        Initial guess for lowest eigenvector.
    ncv : int, default=min(nrow, max(2 * n + 1, 20))
        Number of Lanczos vectors to use.
    maxiter : int, default=nrow * n * 10
        Maximum number of iterations to perform.
    tol : float, default=1.0e-12
        Convergence tolerance.
    method : ("spectra" | "arpack"), default="spectra"
        Whether to use the C++ solver (Spectra) or the SciPy ARPACK solver.

    Returns
    -------
    es : numpy.ndarray
        Energies.
    cs : numpy.ndarray
        Coefficient vectors.

    """
    if len(args) == 1:
        op = args[0]
    elif len(args) == 2:
        op = sparse_op(*args, symmetric=True)
    else:
        raise ValueError("must pass `ham, wfn` or `op`")
    return op.solve(n=n, c0=c0, ncv=ncv, maxiter=maxiter, tol=tol)


def add_excitations(wfn, *excitations, ref=None):
    r"""
    Add multiple excitation levels of determinants to a wave function.

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


def expand_rdms(d1, d2):
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
