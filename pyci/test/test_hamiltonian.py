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

from pyci import secondquant_op
from pyci.test import datafile


@pytest.mark.parametrize("filename", ["he_ccpvqz", "be_ccpvdz", "h2o_ccpvdz", "li2_ccpvdz"])
def test_to_from_file(filename):
    file1 = NamedTemporaryFile()
    file2 = NamedTemporaryFile()
    ham1 = secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    ham1.to_file(file1.name)
    ham2 = secondquant_op(file1.name)

    npt.assert_allclose(ham2.ecore, ham1.ecore, rtol=0.0, atol=1.0e-12)
    npt.assert_allclose(ham2.one_mo, ham1.one_mo, rtol=0.0, atol=1.0e-12)
    npt.assert_allclose(ham2.two_mo, ham1.two_mo, rtol=0.0, atol=1.0e-12)
    npt.assert_allclose(ham2.h, ham1.h, rtol=0.0, atol=1.0e-12)
    npt.assert_allclose(ham2.v, ham1.v, rtol=0.0, atol=1.0e-12)
    npt.assert_allclose(ham2.w, ham1.w, rtol=0.0, atol=1.0e-12)
