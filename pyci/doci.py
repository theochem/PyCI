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
PyCI DOCI module.

"""

import numpy as np

from pyci.cext import doci_ham as ham
from pyci.cext import doci_wfn as wfn
from pyci.cext import doci_compute_rdms as compute_rdms
from pyci.cext import doci_compute_energy as compute_energy
from pyci.cext import doci_run_hci as run_hci
from pyci.cext import doci_generate_rdms as generate_rdms


__all__ = [
    'ham',
    'wfn',
    'compute_rdms',
    'compute_energy',
    'run_hci',
    'generate_rdms',
    ]
