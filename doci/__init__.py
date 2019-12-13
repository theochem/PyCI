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

import numpy as np

from doci.cext import get_version, comb, dociham, dociwfn
from doci.cext import solve_ci, compute_rdms, compute_energy, run_hci


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


__version__ = get_version()
r"""
DOCI version number.

"""


c_int = np.dtype(np.int64)
r"""
C extension signed 64-bit int dtype.

"""


c_uint = np.dtype(np.uint64)
r"""
C extension unsigned 64-bit int dtype.

"""


c_double = np.dtype(np.double)
r"""
C extension double-precision floating point dtype.

"""
