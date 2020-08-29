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

import pytest

import numpy.testing as npt

from scipy.special import comb

import pyci


def test_doci_raises():
    npt.assert_raises(ValueError, pyci.doci_wfn, 10, 11, 11)
    wfn = pyci.doci_wfn(5, 3, 3)
    wfn.add_hartreefock_det()
    assert wfn[0].flags["OWNDATA"]


def test_fullci_raises():
    npt.assert_raises(ValueError, pyci.fullci_wfn, 10, 11, 1)
    npt.assert_raises(ValueError, pyci.fullci_wfn, 10, 11, 11)
    npt.assert_raises(ValueError, pyci.fullci_wfn, 10, 0, 1)
    npt.assert_raises(ValueError, pyci.fullci_wfn, 10, 2, 3)
    wfn = pyci.fullci_wfn(5, 3, 3)
    wfn.add_hartreefock_det()
    assert wfn[0].flags["OWNDATA"]


@pytest.mark.parametrize("nbasis, nocc", [(16, 8), (64, 1), (64, 4), (65, 1), (65, 4), (129, 3)])
def test_doci_to_from_file(nbasis, nocc):
    file1 = NamedTemporaryFile()
    file2 = NamedTemporaryFile()
    wfn1 = pyci.doci_wfn(nbasis, nocc, nocc)
    wfn1.add_all_dets()
    wfn1.to_file(file1.name)
    wfn2 = pyci.doci_wfn(file1.name)
    wfn2.to_file(file2.name)
    assert compare(file1.name, file2.name, shallow=False)


@pytest.mark.parametrize("nbasis, nocc", [(16, 8), (64, 1), (64, 4), (65, 1), (65, 4), (129, 3)])
def test_doci_to_from_det_array(nbasis, nocc):
    wfn1 = pyci.doci_wfn(nbasis, nocc, nocc)
    wfn1.add_all_dets()
    det1 = wfn1.to_det_array()
    wfn2 = pyci.doci_wfn(nbasis, nocc, nocc, det1)
    det2 = wfn2.to_det_array()
    npt.assert_allclose(det1, det2)


@pytest.mark.parametrize("nbasis, nocc", [(16, 8), (64, 1), (64, 4), (65, 1), (65, 4), (129, 3)])
def test_doci_to_from_occs_array(nbasis, nocc):
    wfn1 = pyci.doci_wfn(nbasis, nocc, nocc)
    wfn1.add_all_dets()
    occs1 = wfn1.to_occ_array()
    wfn2 = pyci.doci_wfn(nbasis, nocc, nocc, occs1)
    occs2 = wfn2.to_occ_array()
    npt.assert_allclose(occs1, occs2)


@pytest.mark.parametrize("nbasis, nocc", [(16, 8), (64, 1), (64, 4), (65, 1), (65, 4), (129, 3)])
def test_doci_copy(nbasis, nocc):
    wfn1 = pyci.doci_wfn(nbasis, nocc, nocc)
    wfn1.add_all_dets()
    wfn2 = wfn1.__class__(wfn1)
    det1 = wfn1.to_det_array()
    det2 = wfn2.to_det_array()
    npt.assert_allclose(det1, det2)


@pytest.mark.parametrize("nbasis, nocc", [(16, 8), (64, 1), (64, 4), (65, 1), (65, 4), (129, 3)])
def test_doci_add_all_dets(nbasis, nocc):
    wfn = pyci.doci_wfn(nbasis, nocc, nocc)
    wfn.add_all_dets()
    for det in wfn.to_det_array():
        assert pyci.popcnt(det) == wfn.nocc_up == wfn.nocc_dn == wfn.nocc // 2
    assert len(wfn) == comb(wfn.nbasis, wfn.nocc_up, exact=True)


@pytest.mark.parametrize("nbasis, nocc", [(16, 8), (64, 1), (64, 4), (65, 1), (65, 4), (129, 3)])
def test_doci_add_excited_dets(nbasis, nocc):
    wfn = pyci.doci_wfn(nbasis, nocc, nocc)
    wfn.reserve(comb(wfn.nbasis, wfn.nocc_up, exact=True))
    length = 0
    for i in range(wfn.nocc_up + 1):
        length += comb(wfn.nocc_up, i, exact=True) * comb(wfn.nvir_up, i, exact=True)
        wfn.add_excited_dets(i)
        assert len(wfn) == length
    assert len(wfn) == comb(wfn.nbasis, wfn.nocc_up, exact=True)


@pytest.mark.parametrize(
    "nbasis, nocc_up, nocc_dn", [(8, 3, 3), (64, 1, 1), (64, 2, 1), (65, 2, 1), (129, 2, 1)]
)
def test_fullci_to_from_file(nbasis, nocc_up, nocc_dn):
    file1 = NamedTemporaryFile()
    file2 = NamedTemporaryFile()
    wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
    wfn1.add_all_dets()
    wfn1.to_file(file1.name)
    wfn2 = pyci.fullci_wfn(file1.name)
    wfn2.to_file(file2.name)
    assert compare(file1.name, file2.name, shallow=False)


@pytest.mark.parametrize(
    "nbasis, nocc_up, nocc_dn", [(8, 3, 3), (64, 1, 1), (64, 2, 1), (65, 2, 1), (129, 2, 1)]
)
def test_fullci_to_from_det_array(nbasis, nocc_up, nocc_dn):
    wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
    wfn1.add_all_dets()
    det1 = wfn1.to_det_array()
    wfn2 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn, det1)
    det2 = wfn2.to_det_array()
    npt.assert_allclose(det1, det2)


@pytest.mark.parametrize(
    "nbasis, nocc_up, nocc_dn", [(8, 3, 3), (64, 1, 1), (64, 2, 1), (65, 2, 1), (129, 2, 1)]
)
def test_fullci_to_from_occs_array(nbasis, nocc_up, nocc_dn):
    wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
    wfn1.add_all_dets()
    occs1 = wfn1.to_occ_array()
    assert occs1.shape[2] == nocc_up
    wfn2 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn, occs1)
    occs2 = wfn2.to_occ_array()
    npt.assert_allclose(occs1[:, 0, :nocc_up], occs2[:, 0, :nocc_up])
    npt.assert_allclose(occs1[:, 1, :nocc_dn], occs2[:, 1, :nocc_dn])


@pytest.mark.parametrize(
    "nbasis, nocc_up, nocc_dn", [(8, 3, 3), (64, 1, 1), (64, 2, 1), (65, 2, 1), (129, 2, 1)]
)
def test_fullci_copy(nbasis, nocc_up, nocc_dn):
    wfn1 = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
    wfn1.add_all_dets()
    wfn2 = wfn1.__class__(wfn1)
    det1 = wfn1.to_det_array()
    det2 = wfn2.to_det_array()
    npt.assert_allclose(det1, det2)


@pytest.mark.parametrize(
    "nbasis, nocc_up, nocc_dn", [(8, 3, 3), (64, 1, 1), (64, 2, 1), (65, 2, 1), (129, 2, 1)]
)
def test_fullci_add_all_dets(nbasis, nocc_up, nocc_dn):
    ndet = comb(nbasis, nocc_up, exact=True) * comb(nbasis, nocc_dn, exact=True)
    wfn = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
    wfn.add_all_dets()
    for det in wfn.to_det_array():
        assert pyci.popcnt(det[0]) == wfn.nocc_up
        assert pyci.popcnt(det[1]) == wfn.nocc_dn
    assert len(wfn) == ndet


@pytest.mark.parametrize(
    "nbasis, nocc_up, nocc_dn", [(8, 3, 3), (64, 1, 1), (64, 2, 1), (65, 2, 1), (129, 2, 1)]
)
def test_fullci_add_excited_dets(nbasis, nocc_up, nocc_dn):
    ndet = comb(nbasis, nocc_up, exact=True) * comb(nbasis, nocc_dn, exact=True)
    wfn = pyci.fullci_wfn(nbasis, nocc_up, nocc_dn)
    wfn.reserve(ndet)
    for i in range(wfn.nocc_up + wfn.nocc_dn + 1):
        wfn.add_excited_dets(i)
    assert len(wfn) == ndet
