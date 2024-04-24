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
from pyci import secondquant_op
from pyci.utility import odometer_one_spin, odometer_two_spin
from pyci.gkci import compute_nodes_cntsp
from pyci.test import datafile

from pyscf import gto, scf, tools

def build_wavefunction(mol, occs):
    mf = scf.RHF(mol)
    mf.kernel()

    with NamedTemporaryFile(mode='w+', delete=True) as tmpfile:
        tools.fcidump.from_scf(mf, tmpfile.name)
        ham = pyci.hamiltonian(tmpfile.name)

    wfn = pyci.fullci_wfn(ham.nbasis, *occs)
    return wfn, ham

def get_cost(wfn, ham):
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    e_vals, e_vecs = op.solve(n=3, tol=1.0e-9)
    return e_vals * -1

def get_qmax(wfn, ham, t, p):
    nodes = compute_nodes_cntsp(ham.nbasis)
    q_max = (np.sum(nodes[: wfn.nocc_up - 1]) + (t + 1) * nodes[-1]) * p
    return q_max


mol = gto.Mole()
mol.build(atom = "H 0 0 0; H 0 1 0", basis = 'sto-3g')
wfn1, ham1 = build_wavefunction(mol, (1, 1))
cost1 = get_cost(wfn1, ham1)
q_max1 = get_qmax(wfn1, ham1, 0, 1)
# print(wfn1, cost1, q_max1)
# print(np.arange(wfn1.nocc_up, dtype=pyci.c_long))

mol = gto.Mole()
mol.build(atom = "H 0 0 0; H 0 1 0; H 0 2 0; H 0 3 0", basis = 'sto-3g')
wfn2, ham2 = build_wavefunction(mol, (2, 2))
cost2 = get_cost(wfn2, ham2)
q_max2 = get_qmax(wfn2, ham2, 0, 1)
# print(wfn2, cost2, q_max2)
# print(np.arange(wfn2.nocc_up, dtype=pyci.c_long))



@pytest.mark.parametrize(
    "wfn, cost, t, q_max",
    [
        (wfn1, cost1, 0, q_max1),
        (wfn2, cost2, 0, q_max2),
    ],
)
def test_odometer_one_spin(wfn, cost, t, q_max):
    odometer_one_spin(wfn, cost, t, q_max)
    assert 1==1


@pytest.mark.parametrize(
    "wfn, cost, t, q_max",
    [
        (wfn1, cost1, 0, q_max1),
        (wfn2, cost2, 0, q_max2),
    ],
)
def test_odometer_two_spin(wfn, cost, t, q_max):
    odometer_two_spin(wfn, cost, t, q_max)
    assert 1==1
