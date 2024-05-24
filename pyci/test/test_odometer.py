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

from tempfile import NamedTemporaryFile

import pytest

import numpy as np
import numpy.testing as npt

import pyci
from pyci.utility import odometer_one_spin, odometer_two_spin
from pyci.gkci import compute_nodes_cntsp


def get_wfn_ham(fn, occs):
    ham = pyci.hamiltonian(fn)
    wfn = pyci.fullci_wfn(ham.nbasis, *occs)
    return wfn, ham


def get_cost(wfn, ham, n):
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    e_vals, e_vecs = op.solve(n=n, tol=1.0e-9)
    return e_vals * -1


wfn1, ham1 = get_wfn_ham("data/h2.fcidump", (1, 1))
wfnt, hamt = get_wfn_ham("data/h2.fcidump", (1, 1))
cost1 = get_cost(wfnt, hamt, 2)

wfn2, ham2 = get_wfn_ham("data/h4.fcidump", (2, 2))
wfnt, hamt = get_wfn_ham("data/h4.fcidump", (2, 2))
cost2 = get_cost(wfnt, hamt, 4)


@pytest.mark.parametrize(
    "wfn, cost, t, q_max",
    [
        (wfn1, cost1, 0, 0.7),
        (wfn1, cost1, 0, 1.0),
        (wfn1, cost1, 0, 1.2),
    ],
)
def test_odometer_one_spin(wfn, cost, t, q_max):
    odometer_one_spin(wfn, cost, t, q_max)
    assert 1==1


odometer_one_spin(wfn1, cost1, 0, 1.2)
print(wfn1.to_occ_array())