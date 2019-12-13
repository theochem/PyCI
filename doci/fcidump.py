# This file is part of DOCI.
#
# DOCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# DOCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with DOCI. If not, see <http://www.gnu.org/licenses/>.

r"""
DOCI FCIDUMP module.

"""

from __future__ import absolute_import, division, unicode_literals

from ctypes import c_double
from io import open
from sys import version_info

import numpy as np


__all__ = [
    'read',
    'write',
    ]


# Python 2 compatibility hack
if version_info.major == 2:
    range = xrange


def read(filename):
    r"""
    Read an FCIDUMP file.

    Parameters
    ----------
    filename : str
        FCIDUMP file from which to load integrals.

    Returns
    -------
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.
    nelec : int
        Number of electrons.
    ms2 : int
        Spin number.

    """
    with open(filename, 'r', encoding='utf-8') as f:
        # check header
        line = next(f)
        if not line.startswith(' &FCI NORB='):
            raise IOError('Error in FCIDUMP file header')
        # read info from header
        fields = line[5:].split(',')
        header_info = dict()
        for field in fields:
            if field.count('=') == 1:
                key, val = field.split('=')
                header_info[key.strip()] = val.strip()
        nbasis = int(header_info['NORB'])
        nelec = int(header_info.get('NELEC', '0'))
        ms2 = int(header_info.get('MS2', '0'))
        # skip rest of header
        for line in f:
            fields = line.split()
            if fields[0] == '&END' or fields[0] == '/END' or fields[0] == '/':
                break
        # read integrals
        ecore = 0.0
        one_mo = np.zeros((nbasis, nbasis), dtype=np.dtype(c_double))
        two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.dtype(c_double))
        for line in f:
            fields = line.split()
            if len(fields) != 5:
                raise IOError('Expecting 5 fields on each data line in FCIDUMP')
            val = float(fields[0].strip())
            if fields[3] != '0':
                i = int(fields[1].strip()) - 1
                j = int(fields[2].strip()) - 1
                k = int(fields[3].strip()) - 1
                l = int(fields[4].strip()) - 1
                two_mo[i, k, j, l] = val
                two_mo[k, i, l, j] = val
                two_mo[j, k, i, l] = val
                two_mo[i, l, j, k] = val
                two_mo[j, l, i, k] = val
                two_mo[l, j, k, i] = val
                two_mo[k, j, l, i] = val
                two_mo[l, i, k, j] = val
            elif fields[1] != '0':
                i = int(fields[1].strip()) - 1
                j = int(fields[2].strip()) - 1
                one_mo[i, j] = val
                one_mo[j, i] = val
            else:
                ecore = val
    return ecore, one_mo, two_mo, nelec, ms2


def write(filename, ecore, one_mo, two_mo, nelec=0, ms2=0, tol=1.0e-18):
    r"""
    Write an FCIDUMP file.

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
        Number of electrons.
    ms2 : int, default=0
        Spin number.
    tol : float, default=1.0e-18
        Write elements with magnitude larger than this value.

    """
    nbasis = one_mo.shape[0]
    fmt = ' {0:> .16E} {1:>Xd} {2:>Xd} {3:>Xd} {4:>Xd}\n'.replace('X', str(len(str(nbasis))))
    with open(filename, 'w', encoding='utf-8') as f:
        # write header
        f.write(' &FCI NORB={0:d},NELEC={1:d},MS2={2:d},\n'.format(nbasis, nelec, ms2))
        f.write('  ORBSYM={0:s}\n  ISYM=1\n &END\n'.format('1,' * nbasis))
        # write two-electron integrals
        for i in range(nbasis):
            for j in range(i + 1):
                for k in range(nbasis):
                    for l in range(k + 1):
                        if (i * (i + 1)) // 2 + j >= (k * (k + 1)) // 2 + l:
                            val = two_mo[i, k, j, l]
                            if abs(val) > tol:
                                f.write(fmt.format(val, i + 1, j + 1, k + 1, l + 1))
        # write one-electron integrals
        for i in range(nbasis):
            for j in range(i + 1):
                val = one_mo[i, j]
                if abs(val) > tol:
                    f.write(fmt.format(val, i + 1, j + 1, 0, 0))
        # write zero-energy integrals
        f.write(fmt.format((ecore if abs(ecore) > tol else 0.0), 0, 0, 0, 0))
