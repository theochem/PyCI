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

import numpy as np

from pyci.cext import _get_version
from pyci.cext import comb


__all__ = [
    '__version__',
    'c_int',
    'c_uint',
    'c_double',
    'comb',
    ]


__version__ = _get_version()
r"""
PyCI version number.

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
