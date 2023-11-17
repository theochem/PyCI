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


@pytest.mark.xfail
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
def test_solve_sparse(filename, wfn_type, occs, energy):
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
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
    ],
)
def test_sparse_rectangular(filename, wfn_type, occs, energy):
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
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
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
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
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
    wfn1 = wfn_type(ham.nbasis, *occs)
    wfn1.add_all_dets()
    op = pyci.sparse_op(ham, wfn1)
    es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
    if isinstance(wfn1, pyci.doci_wfn):
        d0, d2 = pyci.compute_transition_rdms(wfn1, wfn1, cs[0], cs[0])
        print(d0)
        print(d2)
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


@pytest.mark.xfail
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
def test_run_hci(filename, wfn_type, occs, energy):
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
    wfn = wfn_type(ham.nbasis, *occs)
    wfn.add_hartreefock_det()
    op = pyci.sparse_op(ham, wfn)
    es, cs = op.solve(n=1, tol=1.0e-6)
    dets_added = 1
    niter = 0
    while dets_added:
        dets_added = pyci.add_hci(ham, wfn, cs[0], eps=1.0e-5)
        op.update(ham, wfn)
        es, cs = op.solve(n=1, tol=1.0e-6)
        niter += 1
    print(es[0])
    assert niter > 1
    if isinstance(wfn, pyci.fullci_wfn):
        assert len(wfn) < np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
    else:
        assert len(wfn) < comb(wfn.nbasis, occs[0], exact=True)
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
    npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)


@pytest.mark.xfail
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
def test_enpt2(filename, wfn_type, occs, energy):
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
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
