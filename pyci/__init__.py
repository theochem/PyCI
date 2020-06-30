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
PyCI module.

"""

from pyci.cext import *


__all__ = [
    '__version__',
    'c_int',
    'c_uint',
    'c_double',
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
    'fullci_wfn',
    'genci_wfn',
    'sparse_op',
    ]


__version__ = _get_version()
r"""
PyCI version number.

"""


c_int = _get_c_int()
r"""
C extension signed int dtype.

"""


c_uint = _get_c_uint()
r"""
C extension unsigned int dtype.

"""


c_double = _get_c_double()
r"""
C extension double-precision float dtype.

"""


SpinLabel = SpinLabel
r"""
Spin label enumeration.

"""


SPIN_UP = SPIN_UP
r"""
Spin-up label.

"""


SPIN_DN = SPIN_DN
r"""
Spin-down label.

"""
