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
from .pyci import hamiltonian
from .pyci import wavefunction, one_spin_wfn, two_spin_wfn
from .pyci import doci_wfn, fullci_wfn, genci_wfn, sparse_op
from .pyci import get_num_threads, set_num_threads, popcnt, ctz, add_hci
from .pyci import compute_overlap, compute_rdms, compute_enpt2

from .integrals import make_senzero_integrals, reduce_senzero_integrals
from .integrals import make_rdms, transform_integrals, natural_orbitals
from .selected_ci import add_excitations, add_seniorities
from .gkci import add_gkci
from .solve import solve


__all__ = [
    "__version__",
    "c_long",
    "c_ulong",
    "c_double",
    "hamiltonian",
    "wavefunction",
    "one_spin_wfn",
    "two_spin_wfn",
    "doci_wfn",
    "fullci_wfn",
    "genci_wfn",
    "sparse_op",
    "get_num_threads",
    "set_num_threads",
    "popcnt",
    "ctz",
    "add_hci",
    "compute_overlap",
    "compute_rdms",
    "compute_enpt2",
    "make_senzero_integrals",
    "reduce_senzero_integrals",
    "make_rdms",
    "transform_integrals",
    "natural_orbitals",
    "add_excitations",
    "add_seniorities",
    "add_gkci",
    "solve",
]


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
