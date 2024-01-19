""" Test DetRatio"""

import numpy as np

import pytest

import pyci

from pyci.fanci import DetRatio
from pyci.test_fanci import find_datafile, assert_deriv


@pytest.fixture
def dummy_system():
    nocc = 2
    ham = pyci.hamiltonian(find_datafile("be_ccpvdz.fcidump"))
    params = np.zeros(2 * ham.nbasis * nocc + 1, dtype=pyci.c_double)
    params[-1] = -14.6
    params[:-1].reshape(-1, ham.nbasis, nocc)[0, :, :] = np.eye(ham.nbasis, nocc)
    params[:-1].reshape(-1, ham.nbasis, nocc)[1, :, :] = np.eye(ham.nbasis, nocc)
    rows = [i for i in range(ham.nbasis) if i not in range(nocc)]
    np.random.seed(2)
    params[:-1].reshape(-1, ham.nbasis, nocc)[1, rows, :] = 0.01 * np.random.rand(len(rows), nocc)
    params[:-1] += 0.0001 * np.random.rand(*params[:-1].shape)
    return (ham, nocc, params)


def init_errors():
    """
    """
    # Define dummy hamiltonian
    nbasis = 10
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)
    for p in [
        (ValueError, [ham, nocc, 2, 1], {}),
    ]:
        yield p


@pytest.mark.parametrize("expecting, args, kwargs", init_errors())
def test_detratio_init_errors(expecting, args, kwargs):
    """
    """
    with pytest.raises(expecting):
        DetRatio(*args, **kwargs)


def test_detratio_init_defaults(dummy_system):
    """
    """
    ham, nocc, params = dummy_system
    test = DetRatio(ham, nocc, 1, 1)

    assert test.nparam == 2 * ham.nbasis * nocc + 1
    assert test.nproj == 2 * ham.nbasis * nocc + 1
    assert test.nactive == 2 * ham.nbasis * nocc + 1
    assert test.nequation == 2 * ham.nbasis * nocc + 1
    assert np.all(test.mask)

    assert isinstance(test.wfn, pyci.doci_wfn)
    assert test.nbasis == ham.nbasis
    assert test.nocc_up == nocc
    assert test.nocc_dn == nocc
    assert test.nvir_up == ham.nbasis - nocc
    assert test.nvir_dn == ham.nbasis - nocc
    assert test.pspace.shape[0] == 57


def test_detratio_freeze_matrix(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)
    detratio.freeze_matrix(0)

    expected = np.ones_like(params, dtype=np.bool)
    expected[: (ham.nbasis * nocc)] = False
    assert np.allclose(detratio.mask, expected)


def test_detratio_compute_overlap(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    matrices = params[:-1].reshape(2, ham.nbasis, nocc)
    matrix1 = matrices[0]
    matrix2 = matrices[1]
    detratio = DetRatio(ham, nocc, numerator, denominator)

    f = lambda x, y: x / y
    for occ in detratio.sspace:
        expected = f(np.linalg.det(matrix1[occ, :]), np.linalg.det(matrix2[occ, :]))
        answer = detratio.compute_overlap(params[:-1], np.array([occ]))
        assert np.allclose(answer, expected)


def test_detratio_compute_overlap_deriv(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)

    f = lambda x: detratio.compute_overlap(x, detratio.sspace)
    j = lambda x: detratio.compute_overlap_deriv(x, detratio.sspace)
    assert_deriv(f, j, params[:-1], widths=1.0e-5)


def test_detratio_compute_objective(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    nmatrices = numerator + denominator
    detratio = DetRatio(ham, nocc, numerator, denominator)

    nproj = nmatrices * ham.nbasis * nocc + 1
    objective = detratio.compute_objective(params)
    op = pyci.sparse_op(detratio.ham, detratio.wfn, nproj, symmetric=False)
    ovlp = detratio.compute_overlap(params[:-1], detratio.sspace)
    answer = op(ovlp) - params[-1] * ovlp[:nproj]
    assert np.allclose(objective, answer)


def test_detratio_compute_jacobian(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)
    f = detratio.compute_objective
    j = detratio.compute_jacobian
    assert_deriv(f, j, params, widths=1.0e-5)


def systems_ground():
    # nocc, system, E_nucnuc, E_hf(electr), normdet, E_ci(electr)
    options_list = [
        (2, "be_ccpvdz", 0.0, -14.57233, [(0, 1.0)], -14.600556994),
        (2, "lih_631g", 0.0, -8.97458, [(0, 1.0)], -8.983534447),
        (3, "li2_631g", 1.73944, -16.60559, [(0, 1.0)], -16.620149348),
    ]
    for p in options_list:
        yield p


# @pytest.mark.slow
@pytest.mark.parametrize("nocc, system, nucnuc, e_hf, normdet, expected", systems_ground())
def test_detratio_systems_ground(nocc, system, nucnuc, e_hf, normdet, expected):
    """Test cases adapted from FanCI's test_wfn_geminal_apig.

    """
    ham = pyci.hamiltonian(find_datafile("{0:s}.fcidump".format(system)))
    numerator = 1
    denominator = 1
    nmtrx = numerator + denominator
    nproj = nocc * ham.nbasis * nmtrx + 1
    detratio = DetRatio(ham, nocc, numerator, denominator, nproj=nproj, norm_det=normdet)

    # Make initial guess.
    # Based on FanPy's DeterminantRatio template_params()
    # Add random numbers on virtual orbital matrix positions.
    params = np.zeros(detratio.nparam, dtype=pyci.c_double)
    params[-1] = e_hf
    params[:-1].reshape(-1, ham.nbasis, nocc)[0, :, :] = np.eye(ham.nbasis, nocc)
    params[:-1].reshape(-1, ham.nbasis, nocc)[1, :, :] = np.eye(ham.nbasis, nocc)
    rows = [i for i in range(ham.nbasis) if i not in range(nocc)]
    np.random.seed(2)
    params[:-1].reshape(-1, ham.nbasis, nocc)[1, rows, :] = np.random.rand(len(rows), nocc)
    params[:-1] += 0.001 * np.random.rand(*params[:-1].shape)

    results = detratio.optimize(params, use_jac=True)
    detratio_energy = results.x[-1]
    assert np.allclose(detratio_energy, expected)
