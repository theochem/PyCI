# This file is part of DOCI.
#
# DOCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# DOCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with DOCI. If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, unicode_literals

import filecmp
from tempfile import NamedTemporaryFile

from nose.tools import assert_raises

import numpy as np
import numpy.testing as npt

from doci import dociham
from doci.test import unicode_str, datafile


class TestDOCIHam:

    CASES = ['he_ccpvqz', 'be_ccpvdz', 'h2o_ccpvdz', 'li2_ccpvdz']

    def test_raises(self):
        assert_raises(ValueError, dociham.from_mo_arrays, 0.0, np.zeros((10, 11)), np.zeros((10, 10, 10, 10)))
        assert_raises(ValueError, dociham.from_mo_arrays, -1.0, np.zeros((10, 10)), np.zeros((10, 10, 10, 11)))
        ham = dociham.from_file(unicode_str(datafile('be_ccpvdz.fcidump')))
        ham = dociham(ham.ecore, ham.h, ham.v, ham.w)
        assert_raises(AttributeError, lambda: ham.one_mo)
        assert_raises(AttributeError, lambda: ham.two_mo)

    def test_to_from_file(self):
        for filename in self.CASES:
            yield self.run_to_from_file, filename

    def run_to_from_file(self, filename):
        file1 = NamedTemporaryFile()
        file2 = NamedTemporaryFile()
        ham1 = dociham.from_file(unicode_str(datafile('{0:s}.fcidump'.format(filename))))
        ham1.to_file(unicode_str(file1.name))
        ham2 = dociham.from_file(unicode_str(file1.name))
        ham2.to_file(unicode_str(file2.name))
        assert filecmp.cmp(file1.name, file2.name, shallow=False)
        npt.assert_allclose(ham2.ecore, ham1.ecore, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.h, ham1.h, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.v, ham1.v, rtol=0.0, atol=1.0e-12)
        npt.assert_allclose(ham2.w, ham1.w, rtol=0.0, atol=1.0e-12)
