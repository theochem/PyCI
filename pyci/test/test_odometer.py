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

import numpy.testing as npt

import pyci
from pyci import secondquant_op
from pyci.utility import odometer_one_spin, odometer_two_spin
from pyci.test import datafile

from pyscf import gto, scf, tools

def build_wavefunction(mol, occs):
    mf = scf.RHF(mol)
    mf.kernel()

    with NamedTemporaryFile(mode='w+', delete=True) as tmpfile:
        tools.fcidump.from_scf(mf, tmpfile.name)
        ham = pyci.hamiltonian(tmpfile.name)

    wfn = pyci.fullci_wfn(ham.nbasis, *occs)


mol = gto.Mole()
mol.build(atom = "H 0 0 0; H 0 1 0", basis = 'sto-3g')
wf1 = build_wavefunction(mol, (1, 1))

mol = gto.Mole()
mol.build(atom = "H 0 0 0; H 0 1 0; H 0 2 0; H 0 3 0", basis = 'sto-3g')
wf2 = build_wavefunction(mol, (2, 2))

@pytest.mark.parametrize(
    "wfn, cost, t, qmax",
    [
        (wfnx0, , , ),
        (, , , ),
    ],
)
def test_odometer_one_spin():
    pass


@pytest.mark.parametrize(
    "wfn, cost, t, qmax",
    [
        (wfnx0, , , ),
        (, , , ),
    ],
)
def test_odometer_two_spin():
    pass
