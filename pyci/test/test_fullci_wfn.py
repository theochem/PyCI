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

from scipy.special import comb

import pyci
from pyci.test import datafile


class TestFullCIWfn:

    CASES = [
        (8, 3, 3),
        (64, 1, 1),
        (64, 2, 1),
        #(65, 2, 1),
        #(129, 2, 1),
        ]

    def test_raises(self):
        assert_raises(ValueError, pyci.fullci_wfn, 10, 11, 1)
        assert_raises(ValueError, pyci.fullci_wfn, 10, 11, 11)
        assert_raises(ValueError, pyci.fullci_wfn, 10, 0, 0)
        assert_raises(ValueError, pyci.fullci_wfn, 10, 0, 1)
        assert_raises(ValueError, pyci.fullci_wfn, 10, 2, 3)
        assert_raises(RuntimeError, pyci.fullci_wfn, 100000, 10000, 10000)

    def test_to_from_file(self):
        for nbasis, nocc_up, nocc_dn in self.CASES:
            yield self.run_to_from_file, nbasis, nocc_up, nocc_dn

    def test_to_from_det_array(self):
        for nbasis, nocc_up, nocc_dn in self.CASES:
            yield self.run_to_from_det_array, nbasis, nocc_up, nocc_dn

    def test_to_from_occs_array(self):
        for nbasis, nocc_up, nocc_dn in self.CASES:
            yield self.run_to_from_occs_array, nbasis, nocc_up, nocc_dn

    def test_copy(self):
        for nbasis, nocc_up, nocc_dn in self.CASES:
            yield self.run_copy, nbasis, nocc_up, nocc_dn

    def test_add_all_dets(self):
        for nbasis, nocc_up, nocc_dn in self.CASES:
            yield self.run_add_all_dets, nbasis, nocc_up, nocc_dn

    def test_add_excited_dets(self):
        for nbasis, nocc_up, nocc_dn in self.CASES:
            yield self.run_add_excited_dets, nbasis, nocc_up, nocc_dn

    def run_to_from_file(self, nbasis, nocc_up, nocc_dn):
        file1 = NamedTemporaryFile()
        file2 = NamedTemporaryFile()
        wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
        wfn1.add_all_dets()
        wfn1.to_file(file1.name)
        wfn2 = pyci.fullci_wfn.from_file(file1.name)
        wfn2.to_file(file2.name)
        assert compare(file1.name, file2.name, shallow=False)

    def run_to_from_det_array(self, nbasis, nocc_up, nocc_dn):
        wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
        wfn1.add_all_dets()
        det1 = wfn1.to_det_array()
        wfn2 = pyci.fullci_wfn.from_det_array(nbasis, nocc_up, nocc_dn, det1)
        det2 = wfn2.to_det_array()
        npt.assert_allclose(det1, det2)

    def run_to_from_occs_array(self, nbasis, nocc_up, nocc_dn):
        wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
        wfn1.add_all_dets()
        occs1 = wfn1.to_occs_array()
        assert occs1.shape[2] == nocc_up
        wfn2 = pyci.fullci_wfn.from_occs_array(nbasis, nocc_up, nocc_dn, occs1)
        occs2 = wfn2.to_occs_array()
        npt.assert_allclose(occs1, occs2)

    def run_copy(self, nbasis, nocc_up, nocc_dn):
        wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
        wfn1.add_all_dets()
        wfn2 = wfn1.copy()
        det1 = wfn1.to_det_array()
        det2 = wfn2.to_det_array()
        npt.assert_allclose(det1, det2)

    def run_add_all_dets(self, nbasis, nocc_up, nocc_dn):
        ndet = comb(nbasis, nocc_up, exact=True) * comb(nbasis, nocc_dn, exact=True)
        wfn = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
        wfn.add_all_dets()
        for det in wfn:
            assert wfn.popcnt_det(det, pyci.SPIN_UP) == wfn.nocc_up
            assert wfn.popcnt_det(det, pyci.SPIN_DN) == wfn.nocc_dn
        assert len(wfn) == ndet

    def run_add_excited_dets(self, nbasis, nocc_up, nocc_dn):
        ndet = comb(nbasis, nocc_up, exact=True) * comb(nbasis, nocc_dn, exact=True)
        wfn = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
        wfn.reserve(ndet)
        assert_raises(ValueError, wfn.add_excited_dets, -1)
        assert_raises(ValueError, wfn.add_excited_dets, 100)
        for i in range(wfn.nocc_up + wfn.nocc_dn + 1):
            wfn.add_excited_dets(i)
        assert len(wfn) == ndet
