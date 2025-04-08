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
import numpy.testing as npt

from scipy.special import comb

import pyci
from pyci.test import datafile


def parity2(p):
    return (
        0 == sum(1 for (x, px) in enumerate(p) for (y, py) in enumerate(p) if x < y and px > py) % 2
    )


def parity(p):
    return 1.0 if parity2(p) else -1.0


@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("he_ccpvqz", pyci.fullci_wfn, (1, 1), -2.902410878),
        ("li2_ccpvdz", pyci.doci_wfn, (3, 3), -14.878455349),
        ("be_ccpvdz", pyci.doci_wfn, (2, 2), -14.600556994),
        ("he_ccpvqz", pyci.doci_wfn, (1, 1), -2.886809116),
        ("be_ccpvdz", pyci.fullci_wfn, (2, 2), -14.617409507),
        ("h2o_ccpvdz", pyci.doci_wfn, (5, 5), -75.634588422),
    ],
)
def test_solve_sparse(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
    npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)


@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("li2_ccpvdz", pyci.doci_wfn, (3, 3), -14.878455349),
        ("h2o_ccpvdz", pyci.doci_wfn, (5, 5), -75.634588422),
        ("be_ccpvdz", pyci.fullci_wfn, (2, 2), -14.600556994),
        ("he_ccpvqz", pyci.fullci_wfn, (1, 1), -2.886809116),
    ],
)
def test_sparse_rectangular(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    pyci.add_excitations(wfn, 0, 1, 2)
    nrow = len(wfn) - 10
    op = pyci.sparse_op(ham, wfn, nrow, symmetric=False)
    assert op.shape == (nrow, len(wfn))
    y = op(np.ones(op.shape[1], dtype=pyci.c_double))
    assert y.ndim == 1
    assert y.shape[0] == op.shape[0]
    z = np.zeros_like(y)
    for i in range(op.shape[0]):
        for j in range(op.shape[1]):
            z[i] += op.get_element(i, j)
    npt.assert_allclose(y, z)


@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("he_ccpvqz", pyci.fullci_wfn, (1, 1), -2.886809116),
        ("li2_ccpvdz", pyci.doci_wfn, (3, 3), -14.878455349),
        ("be_ccpvdz", pyci.doci_wfn, (2, 2), -14.600556994),
        ("he_ccpvqz", pyci.doci_wfn, (1, 1), -2.886809116),
        ("be_ccpvdz", pyci.fullci_wfn, (2, 2), -14.600556994),
        ("h2o_ccpvdz", pyci.doci_wfn, (5, 5), -75.634588422),
    ],
)
def test_compute_rdms(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
    if isinstance(wfn, pyci.doci_wfn):
        d0, d2 = pyci.compute_rdms(wfn, cs[0])
        npt.assert_allclose(np.trace(d0), wfn.nocc_up, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.sum(d2), wfn.nocc_up * (wfn.nocc_up - 1), rtol=0, atol=1.0e-9)
        k0, k2 = pyci.reduce_senzero_integrals(ham.h, ham.v, ham.w, wfn.nocc_up)
        energy = ham.ecore
        energy += np.einsum("ij,ij", k0, d0)
        energy += np.einsum("ij,ij", k2, d2)
        npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)
        rdm1, rdm2 = pyci.spinize_rdms(d0, d2)
    elif isinstance(wfn, pyci.fullci_wfn):
        d1, d2 = pyci.compute_rdms(wfn, cs[0])
        rdm1, rdm2 = pyci.spinize_rdms(d1, d2)
    else:
        rdm1, rdm2 = pyci.compute_rdms(wfn, cs[0])
    with np.load(datafile("{0:s}_spinres.npz".format(filename))) as f:
        one_mo = f["one_mo"]
        two_mo = f["two_mo"]
    assert np.all(np.abs(rdm1 - rdm1.T) < 1e-5)
    # # Test RDM2 is antisymmetric
    # for i in range(0, wfn.nbasis * 2):
    #     for j in range(0, wfn.nbasis * 2):
    #         assert np.all(rdm2[i, j, :, :] + rdm2[i, j, :, :].T) < 1e-5
    #         for k in range(0, wfn.nbasis * 2):
    #             for l in range(0, wfn.nbasis * 2):
    #                 assert np.abs(rdm2[i, j, k, l] - rdm2[k, l, i, j]) < 1e-5
    # "Testing that non Antiysmmetric parts are all zeros."
    for i in range(0, wfn.nbasis * 2):
        assert np.all(np.abs(rdm2[i, i, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, i, i]) < 1e-5)
    energy = ham.ecore
    energy += np.einsum("ij,ij", one_mo, rdm1)
    energy += 0.25 * np.einsum("ijkl,ijkl", two_mo, rdm2)
    npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)

@pytest.mark.bigmem
@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("BH3", pyci.doci_wfn, (4, 4), -26.121994681435808),
        ("h8_fcidump", pyci.doci_wfn, (4, 4),-4.307571602003291 ), 
        ("h6_sto_3g", pyci.doci_wfn, (3, 3), -5.877285606582455),
    ],
)
def test_compute_rdms_1234(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    es, cs = op.solve(n=1, tol=1.0e-6)
    if not isinstance(wfn, pyci.doci_wfn):
        raise TypeError('Wfn must be DOCI')
    d0, d2, d3, d4, d5, d6, d7 = pyci.compute_rdms_1234(wfn, cs[0])
    npt.assert_allclose(np.trace(d0), wfn.nocc_up, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.sum(d2), wfn.nocc_up * (wfn.nocc_up - 1), rtol=0, atol=1.0e-9)
    k0, k2 = pyci.reduce_senzero_integrals(ham.h, ham.v, ham.w, wfn.nocc_up)
    energy = ham.ecore
    energy += np.einsum("ij,ij", k0, d0)
    energy += np.einsum("ij,ij", k2, d2)
    npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)
    rdm1, rdm2, rdm3, rdm4 = pyci.spinize_rdms_1234(d0, d2, d3, d4, d5, d6, d7, flag='34RDM')
    assert np.all(np.abs(rdm1 - rdm1.T) < 1e-5)
    # # Test RDM2 is antisymmetric
    # for i in range(0, wfn.nbasis * 2):
    #     for j in range(0, wfn.nbasis * 2):
    #         assert np.all(rdm2[i, j, :, :] + rdm2[i, j, :, :].T) < 1e-5
    #         for k in range(0, wfn.nbasis * 2):
    #             for l in range(0, wfn.nbasis * 2):
    #                 assert np.abs(rdm2[i, j, k, l] - rdm2[k, l, i, j]) < 1e-5
    # "Testing that non Antiysmmetric parts are all zeros."
    for i in range(0, wfn.nbasis * 2):
        assert np.all(np.abs(rdm2[i, i, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, i, i]) < 1e-5)

    # # Test RDM3 is antisymmetric
    # "Testing that non Antiysmmetric parts are all zeros."
    for i in range(0, wfn.nbasis * 2):
        assert np.all(np.abs(rdm3[i, i, i, :, :, :]) < 1e-5)
        assert np.all(np.abs(rdm3[:, :, :,  i, i, i]) < 1e-5)
        assert np.all(np.abs(rdm3[i, i, :, :, :, :]) < 1e-5)
        assert np.all(np.abs(rdm3[:, :, :,  :, i, i]) < 1e-5)
        assert np.all(np.abs(rdm3[i, :, i, :, :, :]) < 1e-5)
        assert np.all(np.abs(rdm3[:, :, :,  i, :, i]) < 1e-5)
    #TEST COMPLETE 3RDM AND BLOCKS TRACES 
    aabaab = rdm3[:ham.nbasis, :ham.nbasis, ham.nbasis:, :ham.nbasis, :ham.nbasis, ham.nbasis:]
    bbabba = rdm3[ham.nbasis:, ham.nbasis:, :ham.nbasis, ham.nbasis:, ham.nbasis:, :ham.nbasis]
    aaaaaa = rdm3[:ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis]
    bbbbbb = rdm3[ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:]
    abbabb = rdm3[:ham.nbasis, ham.nbasis:, ham.nbasis:, :ham.nbasis, ham.nbasis:, ham.nbasis:]
    baabaa = rdm3[ham.nbasis:, :ham.nbasis, :ham.nbasis, ham.nbasis:, :ham.nbasis, :ham.nbasis]
    npt.assert_allclose(np.einsum('ijkijk -> ', rdm3),(wfn.nocc_up * 2)*(wfn.nocc_up * 2 - 1) * (wfn.nocc_up * 2 - 2) ,rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijkijk -> ', aaaaaa),(wfn.nocc_up) * (wfn.nocc_up - 1) * (wfn.nocc_up - 2) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijkijk -> ', aabaab),(wfn.nocc_up) * (wfn.nocc_dn) * (wfn.nocc_up - 1) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijkijk -> ', bbbbbb),(wfn.nocc_dn) * (wfn.nocc_dn - 1) * (wfn.nocc_dn - 2) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijkijk -> ', bbabba),(wfn.nocc_up) * (wfn.nocc_dn) * (wfn.nocc_dn - 1) , rtol=0, atol=1.0e-9)
    #TEST TRACING OVER TWO INDICES IN 3RDM REDUCES TO 2RDM
    d2_block_aaaa = rdm2[:ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis]
    d2_block_bbbb = rdm2[ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:]
    d2_block_abab = rdm2[:ham.nbasis, ham.nbasis:, :ham.nbasis, ham.nbasis:]
    d2_block_baba = rdm2[ham.nbasis:, :ham.nbasis, ham.nbasis:, :ham.nbasis]
    # All-alpha block
    fac=(1.0 / (wfn.nocc_up - 2.0)) 
    npt.assert_allclose(np.einsum('ijmklm->ijkl ', aaaaaa) * fac,d2_block_aaaa, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('imjkml->ijkl ', aaaaaa) * fac,d2_block_aaaa, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('mijmkl->ijkl ', aaaaaa) * fac,d2_block_aaaa, rtol=0, atol=1.0e-9)

    fac=(1.0 / (wfn.nocc_up)) 
    npt.assert_allclose(np.einsum('ijmklm->ijkl ', aabaab) * fac,d2_block_aaaa, rtol=0, atol=1.0e-9)

    # All-beta block 
    fac=(1.0 / (wfn.nocc_dn - 2.0)) 
    npt.assert_allclose(np.einsum('ijmklm->ijkl ', bbbbbb) * fac,d2_block_bbbb, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('imjkml->ijkl ', bbbbbb) * fac,d2_block_bbbb, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('mijmkl->ijkl ', bbbbbb) * fac,d2_block_bbbb, rtol=0, atol=1.0e-9)

    fac=(1.0 / (wfn.nocc_dn)) 
    npt.assert_allclose(np.einsum('ijmklm->ijkl ', bbabba) * fac,d2_block_bbbb, rtol=0, atol=1.0e-9)

    #Mixed-spin blocks
    fac=(1.0 / (wfn.nocc_up - 1.0))
    npt.assert_allclose(np.einsum('mijmkl->ijkl ', aabaab) * fac,d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('imjkml->ijkl ', aabaab) * fac,d2_block_abab, rtol=0, atol=1.0e-9)

    fac=(1.0 / (wfn.nocc_dn - 1.0))
    npt.assert_allclose(np.einsum('mijmkl->ijkl ', bbabba) * fac,d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('imjkml->ijkl ', bbabba) * fac,d2_block_baba, rtol=0, atol=1.0e-9)
    
    # # Test RDM4 is antisymmetric
    # "Testing that non Antiysmmetric parts are all zeros."
    for i in range(0, wfn.nbasis * 2):
        assert np.all(np.abs(rdm4[i, i, i, i, :, :, :]) < 1e-5)
        assert np.all(np.abs(rdm4[:, :, :, i, i, i, i]) < 1e-5)
        assert np.all(np.abs(rdm4[i, i, :, :, :, :, :, :]) < 1e-5)
        assert np.all(np.abs(rdm4[:, :, :, :, :,  :, i, i]) < 1e-5)
        assert np.all(np.abs(rdm4[:, :, i, i, :, :, :, :]) < 1e-5)
        assert np.all(np.abs(rdm4[:, :, :, :, i, i, :, :]) < 1e-5)
        assert np.all(np.abs(rdm4[i, :, i, :, :, :, :, :]) < 1e-5) 
        assert np.all(np.abs(rdm4[:, :, :, :, :, i, :, i]) < 1e-5)
        assert np.all(np.abs(rdm4[:, i, :, i, :, :, :, :]) < 1e-5) 
        assert np.all(np.abs(rdm4[:, :, :, :, i, :, i, :]) < 1e-5) 
    #TEST COMPLETE 4RDM AND BLOCKS TRACES
    abbbabbb = rdm4[:ham.nbasis, ham.nbasis:, ham.nbasis:, ham.nbasis:, :ham.nbasis, ham.nbasis:, ham.nbasis:, ham.nbasis:]
    baaabaaa = rdm4[ham.nbasis:, :ham.nbasis, :ham.nbasis, :ham.nbasis, ham.nbasis:, :ham.nbasis, :ham.nbasis, :ham.nbasis]
    aaabaaab = rdm4[:ham.nbasis, :ham.nbasis, :ham.nbasis, ham.nbasis:, :ham.nbasis, :ham.nbasis, :ham.nbasis, ham.nbasis:]
    bbbabbba = rdm4[ham.nbasis:, ham.nbasis:, ham.nbasis:, :ham.nbasis, ham.nbasis:, ham.nbasis:, ham.nbasis:, :ham.nbasis]
    abababab = rdm4[:ham.nbasis, ham.nbasis:, :ham.nbasis, ham.nbasis:, :ham.nbasis, ham.nbasis:, :ham.nbasis, ham.nbasis:]
    babababa = rdm4[ham.nbasis:, :ham.nbasis, ham.nbasis:, :ham.nbasis, ham.nbasis:, :ham.nbasis, ham.nbasis:, :ham.nbasis]
    aaaaaaaa = rdm4[:ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis, :ham.nbasis]
    bbbbbbbb = rdm4[ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:, ham.nbasis:] 
    npt.assert_allclose(np.einsum('ijklijkl -> ', rdm4),(wfn.nocc_up * 2)*(wfn.nocc_up * 2 - 1) * (wfn.nocc_up * 2 - 2) * (wfn.nocc_up * 2 - 3) ,rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', abbbabbb),(wfn.nocc_up) * (wfn.nocc_dn) * (wfn.nocc_dn - 1) * (wfn.nocc_dn - 2) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', baaabaaa),(wfn.nocc_dn) * (wfn.nocc_up) * (wfn.nocc_up - 1) * (wfn.nocc_up - 2), rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', aaabaaab),(wfn.nocc_dn) * (wfn.nocc_up) * (wfn.nocc_up - 1) * (wfn.nocc_up - 2) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', bbbabbba),(wfn.nocc_up) * (wfn.nocc_dn) * (wfn.nocc_dn - 1) * (wfn.nocc_dn - 2) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', abababab),(wfn.nocc_up) * (wfn.nocc_dn) * (wfn.nocc_up - 1) * (wfn.nocc_dn - 1) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', babababa),(wfn.nocc_up) * (wfn.nocc_dn) * (wfn.nocc_up - 1) * (wfn.nocc_dn - 1) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', aaaaaaaa),(wfn.nocc_up)*(wfn.nocc_up - 1) * (wfn.nocc_up - 2) * (wfn.nocc_up - 3) , rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijklijkl -> ', bbbbbbbb),(wfn.nocc_dn)*(wfn.nocc_dn - 1) * (wfn.nocc_dn - 2) * (wfn.nocc_dn - 3), rtol=0, atol=1.0e-9)
    
    # # # BLOCK TRACES TESTS
    # #ALL-ALPHA/BETA BLOCKS (only if  wfn.nocc_up/wfn.nocc_up > 3 )
    # With the 3RDM
    if(wfn.nocc_up  > 3):
        fac=1 / (wfn.nocc_up - 3)
        npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',aaaaaaaa)*fac, aaaaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',aaaaaaaa)*fac, aaaaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',aaaaaaaa)*fac, aaaaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',aaaaaaaa)*fac, aaaaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ijkplmnp ->ijklmn',aaaaaaaa)*fac, aaaaaa, rtol=0, atol=1.0e-9)
        fac=1 / (wfn.nocc_dn - 3)
        npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',bbbbbbbb)*fac, bbbbbb, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',bbbbbbbb)*fac, bbbbbb, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',bbbbbbbb)*fac, bbbbbb, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',bbbbbbbb)*fac, bbbbbb, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ijkplmnp ->ijklmn',bbbbbbbb)*fac, bbbbbb, rtol=0, atol=1.0e-9)
        #With 2RDMs
        fac=1 / ((wfn.nocc_up - 3) * (wfn.nocc_up - 2))
        npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',aaaaaaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('piqjpkql ->ijkl',aaaaaaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',aaaaaaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ipjqkplq ->ijkl',aaaaaaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('pijqpklq ->ijkl',aaaaaaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',aaaaaaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        fac=1 / ((wfn.nocc_dn - 3) * (wfn.nocc_dn - 2))
        npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',bbbbbbbb)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('piqjpkql ->ijkl',bbbbbbbb)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('piqjpkql ->ijkl',bbbbbbbb)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',bbbbbbbb)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ipjqkplq ->ijkl',bbbbbbbb)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('pijqpklq ->ijkl',bbbbbbbb)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',bbbbbbbb)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    
    # #ABABABAB/BABABABA BLOCKS
    #With the 3rdms
    fac=1 / (wfn.nocc_dn - 1)
    npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',abababab)*fac, aabaab, rtol=0, atol=1.0e-9)
    fac=1 / (wfn.nocc_up -1)
    npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',abababab)*fac, abbabb, rtol=0, atol=1.0e-9)
    fac=1 / (wfn.nocc_up -1)
    npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',babababa)*fac, bbabba, rtol=0, atol=1.0e-9)
    fac=1 / (wfn.nocc_dn - 1)
    npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',babababa)*fac, baabaa, rtol=0, atol=1.0e-9)
    #With the 2rdms
    fac=1 / ((wfn.nocc_up - 1) * (wfn.nocc_up - 1))
    npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',abababab)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',abababab)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',abababab)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',babababa)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',babababa)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',babababa)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_up) * (wfn.nocc_up - 1))
    npt.assert_allclose(np.einsum('piqjpkql ->ijkl',abababab)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_up - 1) * (wfn.nocc_dn - 1))
    npt.assert_allclose(np.einsum('pijqpklq ->ijkl',abababab)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_dn) * (wfn.nocc_dn - 1))
    npt.assert_allclose(np.einsum('ipjqkplq ->ijkl',abababab)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_dn) * (wfn.nocc_dn - 1))
    npt.assert_allclose(np.einsum('piqjpkql ->ijkl',babababa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_up - 1) * (wfn.nocc_dn - 1))
    npt.assert_allclose(np.einsum('pijqpklq ->ijkl',babababa)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_up) * (wfn.nocc_up - 1))
    npt.assert_allclose(np.einsum('ipjqkplq ->ijkl',babababa)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)

    # AAABAAAB/BBBABBBA BLOCKS
    #With the 3rdms
    fac=1/(wfn.nocc_up-2)
    npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',aaabaaab)*fac, aabaab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',aaabaaab)*fac, aabaab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',aaabaaab)*fac, aabaab, rtol=0, atol=1.0e-9)
    fac=1/(wfn.nocc_dn)
    npt.assert_allclose(np.einsum('ijkplmnp ->ijklmn',aaabaaab)*fac, aaaaaa, rtol=0, atol=1.0e-9)
    fac=1/(wfn.nocc_dn-2)
    npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',bbbabbba)*fac, bbabba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',bbbabbba)*fac, bbabba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',bbbabbba)*fac, bbabba, rtol=0, atol=1.0e-9)
    fac=1/(wfn.nocc_up)
    npt.assert_allclose(np.einsum('ijkplmnp ->ijklmn',bbbabbba)*fac, bbbbbb, rtol=0, atol=1.0e-9)

    #With the 2rdms
    fac=1/((wfn.nocc_up-2)*(wfn.nocc_up-1))
    npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',aaabaaab)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('piqjpkql ->ijkl',aaabaaab)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',aaabaaab)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    fac=1/((wfn.nocc_dn-2)*(wfn.nocc_dn-1))
    npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',bbbabbba)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('piqjpkql ->ijkl',bbbabbba)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',bbbabbba)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    fac=1/((wfn.nocc_dn)*(wfn.nocc_up-2))
    npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',aaabaaab)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    fac=1/((wfn.nocc_up)*(wfn.nocc_dn-2))
    npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',bbbabbba)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)

    # ABBBABBB/BAAABAAA BLOCKS
    #With the 3rdms
    fac=1 / (wfn.nocc_dn-2)
    npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',abbbabbb)*fac, abbabb, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',abbbabbb)*fac, abbabb, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijkplmnp ->ijklmn',abbbabbb)*fac, abbabb, rtol=0, atol=1.0e-9)
    fac=1 / (wfn.nocc_up)
    npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',abbbabbb)*fac, bbbbbb, rtol=0, atol=1.0e-9)
    fac=1 / (wfn.nocc_up-2)
    npt.assert_allclose(np.einsum('ipjklpmn ->ijklmn',baaabaaa)*fac, baabaa, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpklmpn ->ijklmn',baaabaaa)*fac, baabaa, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijkplmnp ->ijklmn',baaabaaa)*fac, baabaa, rtol=0, atol=1.0e-9)
    fac=1 / (wfn.nocc_dn)
    npt.assert_allclose(np.einsum('pijkplmn ->ijklmn',baaabaaa)*fac, aaaaaa, rtol=0, atol=1.0e-9)

    #With the 2rdms
    fac=1 / ((wfn.nocc_dn - 2) * (wfn.nocc_up))
    npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',abbbabbb)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('piqjpkql ->ijkl',abbbabbb)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('pijqpklq ->ijkl',abbbabbb)*fac, d2_block_bbbb, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_dn - 2) * (wfn.nocc_dn - 1))
    npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',abbbabbb)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipjqkplq ->ijkl',abbbabbb)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',abbbabbb)*fac, d2_block_abab, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_up - 2) * (wfn.nocc_dn))
    npt.assert_allclose(np.einsum('pqijpqkl ->ijkl',baaabaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('piqjpkql ->ijkl',baaabaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('pijqpklq ->ijkl',baaabaaa)*fac, d2_block_aaaa, rtol=0, atol=1.0e-9)
    fac=1 / ((wfn.nocc_up - 2) * (wfn.nocc_up - 1))
    npt.assert_allclose(np.einsum('ipqjkpql ->ijkl',baaabaaa)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ipjqkplq ->ijkl',baaabaaa)*fac, d2_block_baba, rtol=0, atol=1.0e-9)
    npt.assert_allclose(np.einsum('ijpqklpq ->ijkl',baaabaaa)*fac, d2_block_baba, rtol=0, atol=1.0e-9)


@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("he_ccpvqz", pyci.fullci_wfn, (1, 1), -2.886809116),
        ("li2_ccpvdz", pyci.doci_wfn, (3, 3), -14.878455349),
        ("be_ccpvdz", pyci.doci_wfn, (2, 2), -14.600556994),
        ("he_ccpvqz", pyci.doci_wfn, (1, 1), -2.886809116),
        ("be_ccpvdz", pyci.fullci_wfn, (2, 2), -14.600556994),
        ("h2o_ccpvdz", pyci.doci_wfn, (5, 5), -75.634588422),
    ],
)
def test_compute_transition_rdms(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn1 = wfn_type(ham.nbasis, *occs)
    wfn1.add_all_dets()
    op = pyci.sparse_op(ham, wfn1)
    es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
    if isinstance(wfn1, pyci.doci_wfn):
        d0, d2 = pyci.compute_transition_rdms(wfn1, wfn1, cs[0], cs[0])
        npt.assert_allclose(np.trace(d0), wfn1.nocc_up, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.sum(d2), wfn1.nocc_up * (wfn1.nocc_up - 1), rtol=0, atol=1.0e-9)
        k0, k2 = pyci.reduce_senzero_integrals(ham.h, ham.v, ham.w, wfn1.nocc_up)
        energy = ham.ecore
        energy += np.einsum("ij,ij", k0, d0)
        energy += np.einsum("ij,ij", k2, d2)
        npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)
        rdm1, rdm2 = pyci.spinize_rdms(d0, d2)
    elif isinstance(wfn1, pyci.fullci_wfn):
        d1, d2 = pyci.compute_transition_rdms(wfn1, wfn1, cs[0], cs[0])
        rdm1, rdm2 = pyci.spinize_rdms(d1, d2)
    else:
        rdm1, rdm2 = pyci.compute_transition_rdms(wfn1, wfn1, cs[0], cs[0])
    with np.load(datafile("{0:s}_spinres.npz".format(filename))) as f:
        one_mo = f["one_mo"]
        two_mo = f["two_mo"]
    assert np.all(np.abs(rdm1 - rdm1.T) < 1e-5)
    for i in range(0, wfn1.nbasis * 2):
        assert np.all(np.abs(rdm2[i, i, :, :]) < 1e-5)
        assert np.all(np.abs(rdm2[:, :, i, i]) < 1e-5)
    energy = ham.ecore
    energy += np.einsum("ij,ij", one_mo, rdm1)
    energy += 0.25 * np.einsum("ijkl,ijkl", two_mo, rdm2)
    npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)


@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("he_ccpvqz", pyci.fullci_wfn, (1, 1), -2.902410878),
        ("li2_ccpvdz", pyci.doci_wfn, (3, 3), -14.878455349),
        ("be_ccpvdz", pyci.doci_wfn, (2, 2), -14.600556994),
        ("he_ccpvqz", pyci.doci_wfn, (1, 1), -2.886809116),
        ("be_ccpvdz", pyci.fullci_wfn, (2, 2), -14.617409507),
        ("h2o_ccpvdz", pyci.doci_wfn, (5, 5), -75.634588422),
    ],
)
def test_run_hci(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    wfn.add_hartreefock_det()
    op = pyci.sparse_op(ham, wfn)
    es, cs = op.solve(n=1, tol=1.0e-6)
    dets_added = 1
    niter = 0
    while dets_added:
        dets_added = pyci.add_hci(ham, wfn, cs[0], eps=1.0e-4)
        op.update(ham, wfn)
        es, cs = op.solve(n=1, tol=1.0e-6)
        niter += 1
    assert niter > 1
    if isinstance(wfn, pyci.fullci_wfn):
        assert len(wfn) <= np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
    else:
        assert len(wfn) <= comb(wfn.nbasis, occs[0], exact=True)
    # npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-6)
    dets_added = 1
    while dets_added:
        dets_added = pyci.add_hci(ham, wfn, cs[0], eps=0.0)
        op.update(ham, wfn)
        es, cs = op.solve(n=1, tol=1.0e-6)
    if isinstance(wfn, pyci.fullci_wfn):
        assert len(wfn) == np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
    else:
        assert len(wfn) == comb(wfn.nbasis, occs[0], exact=True)
    npt.assert_allclose(es[0], energy, rtol=0.0, atol=2.0e-9)


@pytest.mark.parametrize(
    "filename, wfn_type, occs, energy",
    [
        ("he_ccpvqz", pyci.fullci_wfn, (1, 1), -2.964248588),
        ("li2_ccpvdz", pyci.doci_wfn, (3, 3), -14.881173703),
        ("be_ccpvdz", pyci.doci_wfn, (2, 2), -14.603138756),
        ("he_ccpvqz", pyci.doci_wfn, (1, 1), -2.964248588),
        ("be_ccpvdz", pyci.fullci_wfn, (2, 2), -14.617423859),
        ("h2o_ccpvdz", pyci.doci_wfn, (5, 5), -75.784506748),
    ],
)
def test_enpt2(filename, wfn_type, occs, energy):
    ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    pyci.add_excitations(wfn, *range(0, max(wfn.nocc - 1, 1)))
    op = pyci.sparse_op(ham, wfn)
    es, cs = op.solve()
    e = pyci.compute_enpt2(ham, wfn, cs[0], es[0], 1.0e-4)
    npt.assert_allclose(e, energy)


def test_compute_rdm_two_particles_one_up_one_dn():
    wfn = pyci.fullci_wfn(2, 1, 1)
    wfn.add_all_dets()

    coeffs = np.sqrt(np.array([1.0, 2.0, 3.0, 4.0]))
    coeffs /= np.linalg.norm(coeffs)

    d0, d1 = pyci.compute_rdms(wfn, coeffs)

    # Test diagonal of abab.
    assert np.abs(d1[2, 0, 0, 0, 0] - coeffs[0] ** 2.0) < 1e-5
    assert np.abs(d1[2, 0, 1, 0, 1] - coeffs[1] ** 2.0) < 1e-5
    assert np.abs(d1[2, 1, 0, 1, 0] - coeffs[2] ** 2.0) < 1e-5
    assert np.abs(d1[2, 1, 1, 1, 1] - coeffs[3] ** 2.0) < 1e-5

    # "Test Spin-Up off-diagonal of abab.
    assert np.abs(d1[2, 0, 0, 0, 1] - coeffs[0] * coeffs[1]) < 1e-5
    assert np.abs(d1[2, 0, 0, 1, 0] - coeffs[0] * coeffs[2]) < 1e-5
    assert np.abs(d1[2, 0, 0, 1, 1] - coeffs[0] * coeffs[3]) < 1e-5

    assert np.abs(d1[2, 0, 1, 0, 0] - coeffs[1] * coeffs[0]) < 1e-5
    assert np.abs(d1[2, 0, 1, 1, 0] - coeffs[1] * coeffs[2]) < 1e-5
    assert np.abs(d1[2, 0, 1, 1, 1] - coeffs[1] * coeffs[3]) < 1e-5

    assert np.abs(d1[2, 1, 0, 0, 0] - coeffs[2] * coeffs[0]) < 1e-5
    assert np.abs(d1[2, 1, 0, 0, 1] - coeffs[2] * coeffs[1]) < 1e-5
    assert np.abs(d1[2, 1, 0, 1, 1] - coeffs[2] * coeffs[3]) < 1e-5

    assert np.abs(d1[2, 1, 1, 0, 0] - coeffs[3] * coeffs[0]) < 1e-5
    assert np.abs(d1[2, 1, 1, 0, 1] - coeffs[3] * coeffs[1]) < 1e-5
    assert np.abs(d1[2, 1, 1, 1, 0] - coeffs[3] * coeffs[2]) < 1e-5

    # Testing that aaaa is all zeros.
    assert np.all(np.abs(d1[0, :, :, :, :]) < 1e-5)

    # Testing that bbbb is all zeros.
    assert np.all(np.abs(d1[1, :, :, :, :]) < 1e-5)


def test_make_rdm_rdm2_two_particles_one_up_one_dn():
    wfn = pyci.fullci_wfn(2, 1, 1)
    wfn.add_all_dets()

    coeffs = np.sqrt(np.array([1.0, 2.0, 3.0, 4.0]))
    coeffs /= np.linalg.norm(coeffs)

    d0, d1 = pyci.compute_rdms(wfn, coeffs)
    _, rdm2 = pyci.spinize_rdms(d0, d1)

    # "Test out the diagonal RDM2"
    assert np.abs(rdm2[0, 0, 0, 0]) < 1e-5
    assert np.abs(rdm2[0, 1, 0, 1]) < 1e-5  # Since no spin up spin up.and
    assert np.abs(rdm2[0, 2, 0, 2] - coeffs[0] ** 2.0) < 1e-5
    assert np.abs(rdm2[0, 3, 0, 3] - coeffs[1] ** 2.0) < 1e-5
    assert np.abs(rdm2[1, 0, 1, 0]) < 1e-5
    assert np.abs(rdm2[1, 1, 1, 1]) < 1e-5
    assert np.abs(rdm2[1, 2, 1, 2] - coeffs[2] ** 2.0) < 1e-5
    assert np.abs(rdm2[1, 3, 1, 3] - coeffs[3] ** 2.0) < 1e-5
    assert np.abs(rdm2[2, 0, 2, 0] - coeffs[0] ** 2.0) < 1e-5
    assert np.abs(rdm2[2, 1, 2, 1] - coeffs[2] ** 2.0) < 1e-5
    assert np.abs(rdm2[2, 2, 2, 2]) < 1e-5
    assert np.abs(rdm2[2, 3, 2, 3]) < 1e-5
    assert np.abs(rdm2[3, 0, 3, 0] - coeffs[1] ** 2.0) < 1e-5
    assert np.abs(rdm2[3, 1, 3, 1] - coeffs[3] ** 2.0) < 1e-5
    assert np.abs(rdm2[3, 2, 3, 2]) < 1e-5
    assert np.abs(rdm2[3, 3, 3, 3]) < 1e-5

    # "Testing that non Antiysmmetric parts are all zeros."
    for i in range(0, 4):
        assert np.all(rdm2[i, i, :, :] == 0)
        assert np.all(rdm2[:, :, i, i] == 0)

    # "Testing that One has to be Occupied Up and hte other Down."
    assert np.all(np.abs(rdm2[0, 1, :, :]) < 1e-5)
    assert np.all(np.abs(rdm2[:, :, 0, 1]) < 1e-5)
    assert np.all(np.abs(rdm2[1, 0, :, :]) < 1e-5)
    assert np.all(np.abs(rdm2[:, :, 1, 0]) < 1e-5)
    assert np.all(np.abs(rdm2[2, 3, :, :]) < 1e-5)
    assert np.all(np.abs(rdm2[3, 2, :, :]) < 1e-5)
    assert np.all(np.abs(rdm2[:, :, 2, 3]) < 1e-5)
    assert np.all(np.abs(rdm2[:, :, 3, 2]) < 1e-5)

    # Test out off-diagonal.
    assert np.abs(rdm2[0, 2, 0, 3] - coeffs[0] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[0, 2, 1, 2] - coeffs[0] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[0, 2, 1, 3] - coeffs[0] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[0, 2, 2, 0] + coeffs[0] ** 2.0) < 1e-5
    assert np.abs(rdm2[0, 2, 2, 1] + coeffs[0] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[0, 2, 2, 3]) < 1e-5
    assert np.abs(rdm2[0, 2, 3, 0] + coeffs[0] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[0, 2, 3, 1] + coeffs[0] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[0, 2, 3, 2]) < 1e-5

    assert np.abs(rdm2[2, 0, 0, 3] + coeffs[0] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[2, 0, 1, 2] + coeffs[0] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[2, 0, 1, 3] + coeffs[0] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[2, 0, 2, 0] - coeffs[0] ** 2.0) < 1e-5
    assert np.abs(rdm2[2, 0, 2, 1] - coeffs[0] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[2, 0, 2, 3]) < 1e-5
    assert np.abs(rdm2[2, 0, 3, 0] - coeffs[0] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[2, 0, 3, 1] - coeffs[0] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[2, 0, 3, 2]) < 1e-5

    assert np.abs(rdm2[0, 3, 0, 2] - coeffs[1] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[0, 3, 1, 2] - coeffs[1] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[0, 3, 1, 3] - coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[0, 3, 2, 0] + coeffs[1] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[0, 3, 2, 1] + coeffs[1] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[0, 3, 2, 3]) < 1e-5
    assert np.abs(rdm2[0, 3, 3, 0] + coeffs[1] ** 2.0) < 1e-5
    assert np.abs(rdm2[0, 3, 3, 1] + coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[0, 3, 3, 2]) < 1e-5

    assert np.abs(rdm2[3, 0, 0, 2] + coeffs[1] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[3, 0, 1, 2] + coeffs[1] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[3, 0, 1, 3] + coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[3, 0, 2, 0] - coeffs[1] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[3, 0, 2, 1] - coeffs[1] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[3, 0, 2, 3]) < 1e-5
    assert np.abs(rdm2[3, 0, 3, 0] - coeffs[1] ** 2.0) < 1e-5
    assert np.abs(rdm2[3, 0, 3, 1] - coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[3, 0, 3, 2]) < 1e-5

    assert np.abs(rdm2[1, 2, 0, 2] - coeffs[2] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[1, 2, 0, 3] - coeffs[2] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[1, 2, 1, 0]) < 1e-5
    assert np.abs(rdm2[1, 2, 1, 3] - coeffs[2] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[1, 2, 2, 0] + coeffs[2] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[1, 2, 2, 1] + coeffs[2] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[1, 2, 2, 3]) < 1e-5
    assert np.abs(rdm2[1, 2, 3, 0] + coeffs[2] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[1, 2, 3, 1] + coeffs[2] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[1, 2, 3, 2]) < 1e-5
    assert np.abs(rdm2[1, 2, 3, 3]) < 1e-5

    assert np.abs(rdm2[2, 1, 0, 2] + coeffs[2] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[2, 1, 0, 3] + coeffs[2] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[2, 1, 1, 0]) < 1e-5
    assert np.abs(rdm2[2, 1, 1, 3] + coeffs[2] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[2, 1, 2, 0] - coeffs[2] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[2, 1, 2, 1] - coeffs[2] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[2, 1, 2, 3]) < 1e-5
    assert np.abs(rdm2[2, 1, 3, 0] - coeffs[2] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[2, 1, 3, 1] - coeffs[2] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[2, 1, 3, 2]) < 1e-5
    assert np.abs(rdm2[2, 1, 3, 3]) < 1e-5

    assert np.abs(rdm2[1, 3, 0, 1]) < 1e-5
    assert np.abs(rdm2[1, 3, 0, 2] - coeffs[3] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[1, 3, 0, 3] - coeffs[3] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[1, 3, 1, 0]) < 1e-5
    assert np.abs(rdm2[1, 3, 1, 2] - coeffs[3] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[1, 3, 2, 0] + coeffs[3] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[1, 3, 2, 1] + coeffs[3] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[1, 3, 2, 3]) < 1e-5
    assert np.abs(rdm2[1, 3, 3, 0] + coeffs[3] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[1, 3, 3, 1] + coeffs[3] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[1, 3, 3, 2]) < 1e-5

    assert np.abs(rdm2[3, 1, 0, 1]) < 1e-5
    assert np.abs(rdm2[3, 1, 0, 2] + coeffs[3] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[3, 1, 0, 3] + coeffs[3] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[3, 1, 1, 0]) < 1e-5
    assert np.abs(rdm2[3, 1, 1, 2] + coeffs[3] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[3, 1, 2, 0] - coeffs[3] * coeffs[0]) < 1e-5
    assert np.abs(rdm2[3, 1, 2, 1] - coeffs[3] * coeffs[2]) < 1e-5
    assert np.abs(rdm2[3, 1, 2, 3]) < 1e-5
    assert np.abs(rdm2[3, 1, 3, 0] - coeffs[3] * coeffs[1]) < 1e-5
    assert np.abs(rdm2[3, 1, 3, 1] - coeffs[3] * coeffs[3]) < 1e-5
    assert np.abs(rdm2[3, 1, 3, 2]) < 1e-5


def test_make_rdm_rdm1_two_particles_one_up_one_dn():
    wfn = pyci.fullci_wfn(2, 1, 1)
    wfn.add_all_dets()

    coeffs = np.sqrt(np.array([1.0, 2.0, 3.0, 4.0]))
    coeffs /= np.linalg.norm(coeffs)

    d0, d1 = pyci.compute_rdms(wfn, coeffs)
    rdm1, _ = pyci.spinize_rdms(d0, d1)

    assert np.abs(rdm1[0, 0] - coeffs[0] ** 2.0 - coeffs[1] ** 2.0) < 1e-5
    assert np.abs(rdm1[0, 1] - coeffs[0] * coeffs[2] - coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(rdm1[0, 2]) < 1e-5
    assert np.abs(rdm1[0, 3]) < 1e-5

    assert np.abs(rdm1[1, 0] - coeffs[0] * coeffs[2] - coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(rdm1[1, 1] - coeffs[3] ** 2.0 - coeffs[2] ** 2.0) < 1e-5
    assert np.abs(rdm1[1, 2]) < 1e-5
    assert np.abs(rdm1[1, 3]) < 1e-5

    assert np.abs(rdm1[2, 0]) < 1e-5
    assert np.abs(rdm1[2, 1]) < 1e-5
    assert np.abs(rdm1[2, 2] - coeffs[0] ** 2.0 - coeffs[2] ** 2.0) < 1e-5
    assert np.abs(rdm1[2, 3] - coeffs[2] * coeffs[3] - coeffs[0] * coeffs[1]) < 1e-5

    assert np.abs(rdm1[3, 0]) < 1e-5
    assert np.abs(rdm1[3, 1]) < 1e-5
    assert np.abs(rdm1[3, 2] - coeffs[2] * coeffs[3] - coeffs[0] * coeffs[1]) < 1e-5
    assert np.abs(rdm1[3, 3] - coeffs[3] ** 2.0 - coeffs[1] ** 2.0) < 1e-5


def test_make_rdm_rdm2_two_up_one_dn():
    wfn = pyci.fullci_wfn(3, 2, 1)
    wfn.add_all_dets()

    coeffs = np.sqrt(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    coeffs /= np.linalg.norm(coeffs)

    d0, d1 = pyci.compute_rdms(wfn, coeffs)

    # assert antisymmetry aspec of aaaa is all zeros.
    for i in range(0, 3):
        assert np.all(np.abs(d1[0, i, i, :, :]) < 1e-5)
        assert np.all(np.abs(d1[0, :, :, i, i]) < 1e-5)

        for j in range(0, 3):
            assert np.all(np.abs(d1[0, i, j, i, j] + d1[0, i, j, j, i]) < 1e-5)
            assert np.all(np.abs(d1[0, j, i, i, j] - d1[0, i, j, j, i]) < 1e-5)
            assert np.all(np.abs(d1[0, i, j, i, j] + d1[0, j, i, i, j]) < 1e-5)

    aaaa = d1[0]
    # assert Diagonal elements of aaaa.
    assert np.abs(aaaa[0, 1, 0, 1] - coeffs[0] ** 2.0 - coeffs[1] ** 2.0 - coeffs[2] ** 2.0) < 1e-5
    assert np.abs(aaaa[0, 2, 0, 2] - coeffs[3] ** 2.0 - coeffs[4] ** 2.0 - coeffs[5] ** 2.0) < 1e-5
    assert np.abs(aaaa[1, 2, 1, 2] - coeffs[6] ** 2.0 - coeffs[7] ** 2.0 - coeffs[8] ** 2.0) < 1e-5

    # assert non-diagonal elements of aaaa
    elem = aaaa[0, 1, 0, 2] - coeffs[0] * coeffs[3] - coeffs[1] * coeffs[4] - coeffs[2] * coeffs[5]
    assert np.abs(elem) < 1e-5
    elem = aaaa[0, 1, 1, 2] - coeffs[0] * coeffs[6] - coeffs[1] * coeffs[7] - coeffs[2] * coeffs[8]
    assert np.abs(elem) < 1e-5
    elem = aaaa[0, 2, 1, 2] - coeffs[3] * coeffs[6] - coeffs[4] * coeffs[7] - coeffs[5] * coeffs[8]
    assert np.abs(elem) < 1e-5

    # Assert that bbbb is all zeros.
    assert np.all(d1[1] < 1e-5)

    abab = d1[2]

    # Test antisymmetry of abab
    assert np.abs(abab[0, 0, 0, 0] - coeffs[0] ** 2.0 - coeffs[3] ** 2.0) < 1e-5
    assert np.abs(abab[0, 0, 0, 1] - coeffs[0] * coeffs[1] - coeffs[3] * coeffs[4]) < 1e-5
    assert np.abs(abab[0, 0, 0, 2] - coeffs[0] * coeffs[2] - coeffs[3] * coeffs[5]) < 1e-5
    assert np.abs(abab[0, 0, 1, 0] - coeffs[3] * coeffs[6]) < 1e-5
    assert np.abs(abab[0, 0, 1, 1] - coeffs[3] * coeffs[7]) < 1e-5
    assert np.abs(abab[0, 0, 1, 2] - coeffs[3] * coeffs[8]) < 1e-5
    assert np.abs(abab[0, 0, 2, 0] + coeffs[6] * coeffs[0]) < 1e-5
    assert np.abs(abab[0, 0, 2, 1] + coeffs[0] * coeffs[7]) < 1e-5
    assert np.abs(abab[0, 0, 2, 2] + coeffs[0] * coeffs[8]) < 1e-5

    assert np.abs(abab[0, 1, 0, 0] - coeffs[1] * coeffs[0] - coeffs[3] * coeffs[4]) < 1e-5
    assert np.abs(abab[0, 1, 0, 1] - coeffs[1] ** 2.0 - coeffs[4] ** 2.0) < 1e-5
    assert np.abs(abab[0, 1, 0, 2] - coeffs[4] * coeffs[5] - coeffs[1] * coeffs[2]) < 1e-5
    assert np.abs(abab[0, 1, 1, 0] - coeffs[4] * coeffs[6]) < 1e-5
    assert np.abs(abab[0, 1, 1, 1] - coeffs[4] * coeffs[7]) < 1e-5
    assert np.abs(abab[0, 1, 1, 2] - coeffs[4] * coeffs[8]) < 1e-5
    assert np.abs(abab[0, 1, 2, 0] + coeffs[1] * coeffs[6]) < 1e-5
    assert np.abs(abab[0, 1, 2, 1] + coeffs[1] * coeffs[7]) < 1e-5
    assert np.abs(abab[0, 1, 2, 2] + coeffs[1] * coeffs[8]) < 1e-5

    assert np.abs(abab[0, 2, 0, 0] - coeffs[2] * coeffs[0] - coeffs[3] * coeffs[5]) < 1e-5
    assert np.abs(abab[0, 2, 0, 1] - coeffs[2] * coeffs[1] - coeffs[5] * coeffs[4]) < 1e-5
    assert np.abs(abab[0, 2, 1, 0] - coeffs[5] * coeffs[6]) < 1e-5
    assert np.abs(abab[0, 2, 1, 1] - coeffs[5] * coeffs[7]) < 1e-5
    assert np.abs(abab[0, 2, 1, 2] - coeffs[5] * coeffs[8]) < 1e-5
    assert np.abs(abab[0, 2, 2, 0] + coeffs[2] * coeffs[6]) < 1e-5
    assert np.abs(abab[0, 2, 2, 1] + coeffs[2] * coeffs[7]) < 1e-5
    assert np.abs(abab[0, 2, 2, 2] + coeffs[2] * coeffs[8]) < 1e-5

    assert np.abs(abab[1, 0, 0, 0] - coeffs[6] * coeffs[3]) < 1e-5
    assert np.abs(abab[1, 0, 0, 1] - coeffs[6] * coeffs[4]) < 1e-5
    assert np.abs(abab[1, 0, 0, 2] - coeffs[6] * coeffs[5]) < 1e-5
    assert np.abs(abab[1, 0, 1, 1] - coeffs[0] * coeffs[1] - coeffs[6] * coeffs[7]) < 1e-5
    assert np.abs(abab[1, 0, 1, 2] - coeffs[0] * coeffs[2] - coeffs[6] * coeffs[8]) < 1e-5
    assert np.abs(abab[1, 0, 2, 0] - coeffs[0] * coeffs[3]) < 1e-5
    assert np.abs(abab[1, 0, 2, 1] - coeffs[0] * coeffs[4]) < 1e-5
    assert np.abs(abab[1, 0, 2, 2] - coeffs[0] * coeffs[5]) < 1e-5

    assert np.abs(abab[1, 1, 0, 0] - coeffs[3] * coeffs[7]) < 1e-5
    assert np.abs(abab[1, 1, 0, 1] - coeffs[4] * coeffs[7]) < 1e-5
    assert np.abs(abab[1, 1, 0, 2] - coeffs[5] * coeffs[7]) < 1e-5
    assert np.abs(abab[1, 1, 1, 0] - coeffs[0] * coeffs[1] - coeffs[7] * coeffs[6]) < 1e-5
    assert np.abs(abab[1, 1, 1, 2] - coeffs[1] * coeffs[2] - coeffs[7] * coeffs[8]) < 1e-5
    assert np.abs(abab[1, 1, 2, 0] - coeffs[1] * coeffs[3]) < 1e-5
    assert np.abs(abab[1, 1, 2, 1] - coeffs[1] * coeffs[4]) < 1e-5
    assert np.abs(abab[1, 1, 2, 2] - coeffs[1] * coeffs[5]) < 1e-5

    assert np.abs(abab[1, 2, 0, 0] - coeffs[8] * coeffs[3]) < 1e-5
    assert np.abs(abab[1, 2, 0, 1] - coeffs[8] * coeffs[4]) < 1e-5
    assert np.abs(abab[1, 2, 0, 2] - coeffs[8] * coeffs[5]) < 1e-5
    assert np.abs(abab[1, 2, 1, 0] - coeffs[0] * coeffs[2] - coeffs[6] * coeffs[8]) < 1e-5
    assert np.abs(abab[1, 2, 1, 1] - coeffs[8] * coeffs[7] - coeffs[2] * coeffs[1]) < 1e-5
    assert np.abs(abab[1, 2, 2, 0] - coeffs[2] * coeffs[3]) < 1e-5
    assert np.abs(abab[1, 2, 2, 1] - coeffs[2] * coeffs[4]) < 1e-5
    assert np.abs(abab[1, 2, 2, 2] - coeffs[2] * coeffs[5]) < 1e-5

    assert np.abs(abab[2, 2, 0, 0] + coeffs[0] * coeffs[8]) < 1e-5
    assert np.abs(abab[2, 2, 0, 1] + coeffs[8] * coeffs[1]) < 1e-5
    assert np.abs(abab[2, 2, 0, 2] + coeffs[8] * coeffs[2]) < 1e-5
    assert np.abs(abab[2, 2, 1, 0] - coeffs[0] * coeffs[5]) < 1e-5
    assert np.abs(abab[2, 2, 1, 1] - coeffs[1] * coeffs[5]) < 1e-5
    assert np.abs(abab[2, 2, 1, 2] - coeffs[2] * coeffs[5]) < 1e-5
    assert np.abs(abab[2, 2, 2, 0] - coeffs[5] * coeffs[3] - coeffs[6] * coeffs[8]) < 1e-5
    assert np.abs(abab[2, 2, 2, 1] - coeffs[5] * coeffs[4] - coeffs[8] * coeffs[7]) < 1e-5
    assert np.abs(abab[2, 2, 2, 2] - coeffs[8] ** 2.0 - coeffs[5] ** 2.0) < 1e-5
