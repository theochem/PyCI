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

from pyci._pyci import __version__, c_long, c_ulong, c_double
from pyci._pyci import secondquant_op, wavefunction, one_spin_wfn, two_spin_wfn
from pyci._pyci import doci_wfn, fullci_wfn, genci_wfn, sparse_op
from pyci._pyci import get_num_threads, set_num_threads, popcnt, ctz
from pyci._pyci import compute_overlap, compute_rdms, compute_transition_rdms,compute_rdms_1234
from pyci._pyci import add_hci, compute_enpt2

from pyci.utility import make_senzero_integrals, reduce_senzero_integrals, spinize_rdms,spinize_rdms_1234,spin_free_rdms
from pyci.utility import odometer_one_spin, odometer_two_spin

from pyci.excitation_ci import add_excitations
from pyci.seniority_ci import add_seniorities
from pyci.gkci import add_gkci
from pyci.cost_ci import add_cost


__all__ = [
    "__version__",
    "c_long",
    "c_ulong",
    "c_double",
    "secondquant_op",
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
    "compute_transition_rdms",
    "compute_enpt2",
    "make_senzero_integrals",
    "reduce_senzero_integrals",
    "spinize_rdms",
    "odometer_one_spin",
    "odometer_two_spin",
    "add_excitations",
    "add_seniorities",
    "add_gkci",
    "add_cost",
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


# Alias `pyci.secondquant_op` to `pyci.hamiltonian` for compatibility with old versions
hamiltonian = secondquant_op
