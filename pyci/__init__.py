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

from .pyci import __version__, c_long, c_ulong, c_double
from .pyci import *

from .fcidump import *
from .integrals import *
from .selected_ci import *
from .gkci import *
from .solve import *
from .cepa0 import *


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
