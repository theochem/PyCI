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

r"""PyCI FCIDUMP module."""

from typing import TextIO, Tuple, Union

import numpy as np

from .integrals import make_senzero_integrals


__all__ = [
    "read_fcidump",
    "write_fcidump",
]


def _load_ham(
    *args: Union[Tuple[float, np.ndarray, np.ndarray], Tuple[str]]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Load the arguments to a PyCI Hamiltonian constructor (C++)."""
    ecore, one_mo, two_mo = read_fcidump(*args) if len(args) == 1 else args
    h, v, w = make_senzero_integrals(one_mo, two_mo)
    return ecore, one_mo, two_mo, h, v, w


def read_fcidump(filename: TextIO) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""
    Read an FCIDUMP file.

    Parameters
    ----------
    filename : TextIO
        FCIDUMP file to read.

    Returns
    -------
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray
        Full one-electron integral array.
    two_mo : np.ndarray
        Full two-electron integral array.

    Notes
    -----
    Currently only works for restricted/generalized integrals.

    """
    ecore = 0.0
    with open(filename, "r", encoding="utf-8") as f:
        # check header
        line = next(f)
        if not line.startswith(" &FCI NORB="):
            raise IOError("Error in FCIDUMP file header")
        # read nbasis from header
        nbasis = int(line[11 : line.find(",")])
        # skip rest of header
        for line in f:
            field = line.split()[0]
            if field == "&END" or field == "/END" or field == "/":
                break
        # read integrals
        one_mo = np.zeros((nbasis, nbasis), dtype=np.double)
        two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.double)
        for line in f:
            fields = line.split()
            if len(fields) != 5:
                raise IOError("Expecting 5 fields on each data line in FCIDUMP")
            val = float(fields[0])
            if fields[3] != "0":
                ii = int(fields[1]) - 1
                jj = int(fields[2]) - 1
                kk = int(fields[3]) - 1
                ll = int(fields[4]) - 1
                two_mo[ii, kk, jj, ll] = val
                two_mo[kk, ii, ll, jj] = val
                two_mo[jj, kk, ii, ll] = val
                two_mo[ii, ll, jj, kk] = val
                two_mo[jj, ll, ii, kk] = val
                two_mo[ll, jj, kk, ii] = val
                two_mo[kk, jj, ll, ii] = val
                two_mo[ll, ii, kk, jj] = val
            elif fields[1] != "0":
                ii = int(fields[1]) - 1
                jj = int(fields[2]) - 1
                one_mo[ii, jj] = val
                one_mo[jj, ii] = val
            else:
                ecore = val
    return ecore, one_mo, two_mo


def write_fcidump(
    filename: TextIO,
    ecore: float,
    one_mo: np.ndarray,
    two_mo: np.ndarray,
    nelec: int = 0,
    ms2: int = 0,
    tol: float = 1.0e-18,
) -> None:
    r"""
    Write a Hamiltonian instance to an FCIDUMP file.

    Parameters
    ----------
    filename : TextIO
        FCIDUMP file to write.
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray
        Full one-electron integral array.
    two_mo : np.ndarray
        Full two-electron integral array.
    nelec : int, default=0
        Electron number to write to FCIDUMP file.
    ms2 : int, default=0
        Spin number to write to FCIDUMP file.
    tol : float, default=1.0e-18
        Write elements with magnitude larger than this value.

    Notes
    -----
    Currently only works for restricted/generalized integrals.

    """
    nbasis = one_mo.shape[0]
    with open(filename, "w", encoding="utf-8") as f:
        # write header
        f.write(f" &FCI NORB={nbasis},NELEC={nelec},MS2={ms2},\n")
        f.write(f'  ORBSYM={"1," * nbasis}\n  ISYM=1\n &END\n')
        # write two-electron integrals
        for ii in range(nbasis):
            for jj in range(ii + 1):
                for kk in range(nbasis):
                    for ll in range(kk + 1):
                        if (ii * (ii + 1)) // 2 + jj >= (kk * (kk + 1)) // 2 + ll:
                            val = two_mo[ii, kk, jj, ll]
                            if abs(val) > tol:
                                print(
                                    f"{val:23.16E} {ii + 1:4d} {jj + 1:4d} {kk + 1:4d} {ll + 1:4d}",
                                    file=f,
                                )
        # write one-electron integrals
        for ii in range(nbasis):
            for jj in range(ii + 1):
                val = one_mo[ii, jj]
                if abs(val) > tol:
                    print(f"{val:23.16E} {ii + 1:4d} {jj + 1:4d}    0    0", file=f)
        # write zero-energy integrals
        print(f"{ecore if abs(ecore) > tol else 0:23.16E}    0    0    0    0", file=f)
