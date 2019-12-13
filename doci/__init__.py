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
DOCI module.

"""

from __future__ import absolute_import, unicode_literals

from ctypes import c_int64, c_uint64, c_double as c_double_

from numpy import dtype

from doci.cext import comb, dociham, dociwfn, solve_ci, compute_rdms, compute_energy, run_hci


__all__ = [
    '__version__',
    'c_int',
    'c_uint',
    'c_double',
    'comb',
    'dociham',
    'dociwfn',
    'solve_ci',
    'compute_rdms',
    'compute_energy',
    'run_hci',
    ]


__version__ = '0.1.0'
r"""
DOCI version number.

"""


c_int = dtype(c_int64)
r"""
C signed int dtype.

"""


c_uint = dtype(c_uint64)
r"""
C unsigned int dtype.

"""


c_double = dtype(c_double_)
r"""
C double-precision floating point dtype.

"""
