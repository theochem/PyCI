# cython: language_level=3, wraparound=False, binding=False
# cython: initializedcheck=False, nonecheck=False, boundscheck=False
#
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
PyCI utilities module.

"""

import numpy as np


__all__ = [
        'read_fcidump',
        'write_fcidump',
        'make_senzero_integrals',
        'reduce_senzero_integrals',
        'make_rdms',
        ]


def read_fcidump(filename):
    r"""
    Read an FCIDUMP file.

    Parameters
    ----------
    filename : str
        Name of FCIDUMP file to read.

    Returns
    -------
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.

    """
    ecore = 0.
    with open(filename, 'r', encoding='utf-8') as f:
        # check header
        line = next(f)
        if not line.startswith(' &FCI NORB='):
            raise IOError('Error in FCIDUMP file header')
        # read nbasis from header
        nbasis = int(line[11:line.find(',')])
        # skip rest of header
        for line in f:
            field = line.split()[0]
            if field == '&END' or field == '/END' or field == '/':
                break
        # read integrals
        one_mo = np.zeros((nbasis, nbasis), dtype=np.double)
        two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.double)
        for line in f:
            fields = line.split()
            if len(fields) != 5:
                raise IOError('Expecting 5 fields on each data line in FCIDUMP')
            val = float(fields[0])
            if fields[3] != '0':
                i = int(fields[1]) - 1
                j = int(fields[2]) - 1
                k = int(fields[3]) - 1
                l = int(fields[4]) - 1
                two_mo[i, k, j, l] = val
                two_mo[k, i, l, j] = val
                two_mo[j, k, i, l] = val
                two_mo[i, l, j, k] = val
                two_mo[j, l, i, k] = val
                two_mo[l, j, k, i] = val
                two_mo[k, j, l, i] = val
                two_mo[l, i, k, j] = val
            elif fields[1] != '0':
                i = int(fields[1]) - 1
                j = int(fields[2]) - 1
                one_mo[i, j] = val
                one_mo[j, i] = val
            else:
                ecore = val
    return ecore, one_mo, two_mo


def write_fcidump(filename, ecore, one_mo, two_mo, nelec=0, ms2=0, tol=1.0e-18):
    r"""
    Write a Hamiltonian instance to an FCIDUMP file.

    Parameters
    ----------
    filename : str
        Name of FCIDUMP file to write.
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.
    nelec : int, default=0
        Electron number to write to FCIDUMP file.
    ms2 : int, default=0
        Spin number to write to FCIDUMP file.
    tol : float, default=1.0e-18
        Write elements with magnitude larger than this value.

    """
    nbasis = one_mo.shape[0]
    with open(filename, 'w', encoding='utf-8') as f:
        # write header
        f.write(f' &FCI NORB={nbasis},NELEC={nelec},MS2={ms2},\n')
        f.write(f'  ORBSYM={"1," * nbasis}\n  ISYM=1\n &END\n')
        # write two-electron integrals
        for i in range(nbasis):
            for j in range(i + 1):
                for k in range(nbasis):
                    for l in range(k + 1):
                        if (i * (i + 1)) // 2 + j >= (k * (k + 1)) // 2 + l:
                            val = two_mo[i, k, j, l]
                            if abs(val) > tol:
                                f.write(f'{val:23.16E} {i + 1:4d} {j + 1:4d} {k + 1:4d} {l + 1:4d}\n')
        # write one-electron integrals
        for i in range(nbasis):
            for j in range(i + 1):
                val = one_mo[i, j]
                if abs(val) > tol:
                    f.write(f'{val:23.16E} {i + 1:4d} {j + 1:4d}    0    0\n')
        # write zero-energy integrals
        f.write(f'{ecore if abs(ecore) > tol else 0:23.16E}    0    0    0    0\n')


def make_senzero_integrals(one_mo, two_mo):
    r"""
    Return the seniority-zero chunks of the full one- and two- electron integrals.

    Parameters
    ----------
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.

    Returns
    -------
    h : np.ndarray(c_double(nbasis))
        Seniority-zero one-electron integrals.
    v : np.ndarray(c_double(nbasis, nbasis))
        Seniority-zero two-electron integrals.
    w : np.ndarray(c_double(nbasis, nbasis))
        Seniority-two two-electron integrals.

    """
    h = np.copy(np.diagonal(one_mo))
    v = np.copy(np.diagonal(np.diagonal(two_mo)))
    w = np.copy(np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 2, 3, 1)))))
    w *= 2
    w -= np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 3, 2, 1))))
    return h, v, w


def reduce_senzero_integrals(h, v, w, nocc):
    r"""
    Reduce the reduced seniority-zero one- and two- electron integrals.

    Parameters
    ----------
    h : np.ndarray(c_double(nbasis))
        Seniority-zero one-electron integrals.
    v : np.ndarray(c_double(nbasis, nbasis))
        Seniority-zero two-electron integrals.
    w : np.ndarray(c_double(nbasis, nbasis))
        Seniority-two two-electron integrals.

    Returns
    -------
    rv : np.ndarray(c_double(nbasis, nbasis))
        Reduced seniority-zero two-electron integrals.
    rw : np.ndarray(c_double(nbasis, nbasis))
        Reduced seniority-two two-electron integrals.

    """
    nbasis = h.shape[0]
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


def make_rdms(d1, d2):
    r"""
    Convert the DOCI matrices :math:`D0` and :math:`D2` or the FullCI RDM spin-blocks to full,
    generalized RDMs.

    Parameters
    ----------
    d1 : np.ndarray
        :math:`D_0` matrix or FullCI 1-RDM spin-blocks.
    d2 : np.ndarray
        :math:`D_2` matrix or FullCI 2-RDM spin-blocks.

    Returns
    -------
    rdm1 : np.ndarray(c_double(nbasis * 2, nbasis * 2))
        Generalized one-electron RDM.
    rdm2 : np.ndarray(c_double(nbasis * 2, nbasis * 2, nbasis * 2, nbasis * 2))
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
        aa += d1[0]   #+aa
        bb += d1[1]   #+bb
        aaaa += d2[0] #+aaaa
        bbbb += d2[1] #+bbbb
        abab += d2[2] #+abab
        baba += d2[2] #+abab
        abba -= d2[2] #-abab
        baab -= d2[2] #-abab
    return rdm1, rdm2
