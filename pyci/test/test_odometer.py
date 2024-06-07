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

import pytest

import numpy as np

import pyci
from pyci.utility import odometer_one_spin, odometer_two_spin
from pyci.test import datafile


def get_wfn(fn, wfntype, occs):
    ham = pyci.hamiltonian(fn)
    if wfntype == "fullci":
        wfn = pyci.fullci_wfn(ham.nbasis, *occs)
    elif wfntype == "doci":
        wfn = pyci.doci_wfn(ham.nbasis, *occs)
    elif wfntype == "genci":
        wfn = pyci.genci_wfn(ham.nbasis, *occs)
    else:
        raise ValueError
    return wfn, ham


def get_cost(fn, wfntype, occs):
    wfn, ham = get_wfn(fn, wfntype, occs)
    wfn.add_all_dets()
    n = len(wfn.to_occ_array()) - 1
    op = pyci.sparse_op(ham, wfn)
    e_vals, e_vecs = op.solve(n=n, tol=1.0e-9)
    return e_vals * -1


@pytest.mark.parametrize(
    "fn, wfntype, occs, expected",
    [
        (datafile("h4_sto3g.fcidump"), "doci", (2, 2), {0.15: np.array([]).reshape(0, 2),
                                                        0.16: np.array([[0, 1]]),
                                                        0.74: np.array([[0, 1], [0, 2]])}),
    ],
)
def test_odometer_one_spin(fn, wfntype, occs, expected):
    cost = get_cost(fn, wfntype, occs)
    for q_max in expected:
        wfn, ham = get_wfn(fn, wfntype, occs)
        odometer_one_spin(wfn, cost, 0, q_max)
        assert (wfn.to_occ_array() == expected[q_max]).all() == True


@pytest.mark.parametrize(
    "fn, wfntype, occs, expected",
    [
        (datafile("h2_sto3g.fcidump"), "fullci", (1, 1), {0.34: np.array([]),
                                                          0.35: np.array([[0], [0]]),
                                                          0.75: np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])}),
        (datafile("h4_sto3g.fcidump"), "fullci", (2, 2), {-1.175: np.array([]).reshape(0, 2, 2),
                                                          -1.174: np.array([[[0, 1], [0, 1]]]),
                                                          -0.9038: np.array([[[0, 1], [0, 1]], [[0, 1], [0, 2]], [[0, 2], [0, 1]], [[0, 2], [0, 2]]])}),
    ],
)
def test_odometer_two_spin(fn, wfntype, occs, expected):
    cost = get_cost(fn, wfntype, occs)
    for q_max in expected:
        wfn, ham = get_wfn(fn, wfntype, occs)
        odometer_two_spin(wfn, cost, 0, q_max)
        assert (wfn.to_occ_array() == expected[q_max]).all() == True
