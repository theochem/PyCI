# cython : language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
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
PyCI C extension module.

"""

from libc.stdint cimport int64_t, uint64_t

from libcpp.vector cimport vector

from io import open

cimport numpy as np
import numpy as np

from pyci.cext cimport *


__all__ = [
    '_get_version',
    '_get_c_int',
    '_get_c_uint',
    '_get_c_double',
    'SpinLabel',
    'SPIN_UP',
    'SPIN_DN',
    'hamiltonian',
    'restricted_ham',
    'unrestricted_ham',
    'generalized_ham',
    'wavefunction',
    'one_spin_wfn',
    'two_spin_wfn',
    'doci_wfn',
    'genci_wfn',
    'fullci_wfn',
    'sparse_op',
    ]


cdef np.dtype c_int = np.dtype(np.int64)


cdef np.dtype c_uint = np.dtype(np.uint64)


cdef np.dtype c_double = np.dtype(np.double)


cpdef enum SpinLabel:
    SPIN_UP = 0
    SPIN_DN = 1


def _get_version():
    r"""
    Return the version number string from the C extension.

    Returns
    -------
    version : str
        Version number string.

    """
    return PYCI_VERSION


def _get_c_int():
    r"""
    Return the C extension signed int dtype.

    """
    return c_int


def _get_c_uint():
    r"""
    Return the C extension unsigned int dtype.

    """
    return c_uint


def _get_c_double():
    r"""
    Return the C extension double-precision float dtype.

    """
    return c_double


include 'hamiltonian.pxi'


include 'wavefunction.pxi'


include 'sparse_op.pxi'
