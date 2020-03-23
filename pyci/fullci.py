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
PyCI FullCI module.

"""

import numpy as np

from pyci.cext import SpinLabel as _SpinLabel, SPIN_UP as _SPIN_UP, SPIN_DN as _SPIN_DN
from pyci.cext import fullci_ham as ham
from pyci.cext import fullci_wfn as wfn
#from pyci.cext import fullci_solve_ci as solve_ci
#from pyci.cext import fullci_compute_rdms as compute_rdms
#from pyci.cext import fullci_compute_energy as compute_energy
#from pyci.cext import fullci_run_hci as run_hci


__all__ = [
    'SpinLabel',
    'SPIN_UP',
    'SPIN_DN',
    'ham',
    'wfn',
    #'solve_ci',
    #'compute_rdms',
    #'compute_energy',
    #'run_hci',
    ]


SpinLabel = _SpinLabel
r"""
Spin label enum.

"""


SPIN_UP = _SPIN_UP
r"""
Spin-up label.

"""


SPIN_DN = _SPIN_DN
r"""
Spin-down label.

"""
