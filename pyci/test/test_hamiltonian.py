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

from filecmp import cmp as compare
from tempfile import NamedTemporaryFile

from nose.tools import assert_raises

import numpy as np
import numpy.testing as npt

from pyci import hamiltonian
from pyci.test import datafile


class TestHamiltonian:

    CASES = ['he_ccpvqz', 'be_ccpvdz', 'h2o_ccpvdz', 'li2_ccpvdz']

    def test_raises(self):
        assert_raises(ValueError, hamiltonian, 1.0, np.zeros((10, 11)), np.zeros((10, 10, 10, 10)))
        assert_raises(ValueError, hamiltonian, 1.0, np.zeros((10, 10)), np.zeros((10, 10, 10, 11)))
        ham = hamiltonian(1.0, np.zeros((10, 10)), np.zeros((10,) * 4), keep_mo=False)
        assert_raises(AttributeError, lambda: ham.one_mo)
        assert_raises(AttributeError, lambda: ham.two_mo)
        ham = hamiltonian(1.0, np.zeros((10, 10)), np.zeros((10,) * 4), doci=False)
        assert_raises(AttributeError, lambda: ham.h)
        assert_raises(AttributeError, lambda: ham.v)
        assert_raises(AttributeError, lambda: ham.w)

    def test_to_from_file(self):
        for filename in self.CASES:
            yield self.run_to_from_file, filename

    def run_to_from_file(self, filename):
        file1 = NamedTemporaryFile()
        file2 = NamedTemporaryFile()
        ham1 = hamiltonian.from_file(datafile('{0:s}.fcidump'.format(filename)))
        ham1.to_file(file1.name)
        ham2 = hamiltonian.from_file(file1.name)
        ham2.to_file(file2.name)
        assert compare(file1.name, file2.name, shallow=False)
        npt.assert_allclose(ham2.ecore, ham1.ecore, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.h, ham1.h, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.h, ham1.h, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.h, ham1.h, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.v, ham1.v, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.w, ham1.w, rtol=0.0, atol=1.0e-12)