""" Test APIG"""

import numpy as np

import pytest

import pyci

from pyci.fanci import APIG
from pyci.fanci.apig import permanent
from pyci.test_fanci import find_datafile, assert_deriv


@pytest.fixture
def dummy_system():
    nbasis = 6
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)
    params = np.arange(nbasis * nocc + 1, dtype=pyci.c_double) + 1
    return (ham, nocc, params)


def init_errors():
    """
    """
    # Define dummy hamiltonian
    nbasis = 3
    nocc = 1
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)
    # Define raise error input options
    nproj_valueerror = 4  # P space > S space
    wfn_valueerror = pyci.doci_wfn(nbasis, 2, 2)  # number of electrons don't match

    for p in [
        (TypeError, [nocc, ham], {}),
        (TypeError, [ham, nocc], {"nproj": "3"}),
        (ValueError, [ham, nocc], {"nproj": nproj_valueerror}),
        (TypeError, [ham, nocc], {"wfn": ["0b001001"]}),
        (ValueError, [ham, nocc], {"wfn": wfn_valueerror}),
    ]:
        yield p


def systems_ground():
    options_list = [
        (1, "h2_hf_sto6g", 0.71317683129, -1.83843, [(0, 1.0)], 2, -1.1459130128588935,),
        (1, "h2_hf_631gdp", 0.71317683129, -1.84444, [(0, 1.0)], 10, -1.1565060295404896,),
        (2, "lih_hf_sto6g", 0.995317634356, -8.94728, [(0, 1.0)], 15, -7.968213475280394,),
    ]
    for p in options_list:
        yield p


@pytest.mark.parametrize("expecting, args, kwargs", init_errors())
def test_apig_init_errors(expecting, args, kwargs):
    """
    """
    with pytest.raises(expecting):
        APIG(*args, **kwargs)


def test_apig_compute_overlap_deriv(dummy_system):
    ham, nocc, params = dummy_system
    apig = APIG(ham, nocc, nproj=None)

    f = lambda x: apig.compute_overlap(x, apig.sspace)
    j = lambda x: apig.compute_overlap_deriv(x, apig.sspace)
    origin = np.random.rand(params[:-1].shape[0])
    assert_deriv(f, j, origin)


def test_apig_compute_objective(dummy_system):
    ham, nocc, params = dummy_system
    nproj = nocc * ham.nbasis + 1
    apig = APIG(ham, nocc, nproj=None)

    objective = apig.compute_objective(params)
    op = pyci.sparse_op(apig.ham, apig.wfn, nproj, symmetric=False)
    ovlp = apig.compute_overlap(params[:-1], apig.sspace)
    answer = op(ovlp) - params[-1] * ovlp[:nproj]
    assert np.allclose(objective, answer)


def test_apig_compute_jacobian(dummy_system):
    ham, nocc, params = dummy_system
    apig = APIG(ham, nocc, norm_det=[(0, 1.0)])

    f = apig.compute_objective
    j = apig.compute_jacobian
    origin = np.random.rand(params.shape[0])
    assert_deriv(f, j, origin)


@pytest.mark.parametrize("nocc, system, nucnuc, e_hf, normdet, nproj, expected", systems_ground())
def test_apig_systems_ground(nocc, system, nucnuc, e_hf, normdet, nproj, expected):
    """Test cases adapted from FanCI's test_wfn_geminal_apig.

    """
    ham = pyci.hamiltonian(find_datafile("{0:s}.fcidump".format(system)))
    apig = APIG(ham, nocc, nproj=nproj, norm_det=normdet)

    params_guess = np.zeros(apig.nparam, dtype=pyci.c_double)
    params_guess[:-1].reshape(ham.nbasis, nocc)[:, :] = np.eye(ham.nbasis, nocc)
    params_guess[-1] = e_hf
    results = apig.optimize(params_guess, use_jac=True)
    apig_energy = results.x[-1] + nucnuc
    assert np.allclose(apig_energy, expected)


def test_apig_h2_sto6g_excited():
    """Test excited state APIG wavefunction using H2 with HF/STO-6G orbital.
    Test adapted from FanCI's test_wfn_geminal_apig.

    Answers obtained from answer_apig_h2_sto6g

    APIG (Electronic) Energy : -0.2416648697421632

    """
    ham = pyci.hamiltonian(find_datafile("h2_hf_sto6g.fcidump"))
    params_guess = np.array([0.0, 0.9, -1.83843])
    apig = APIG(ham, 1, nproj=2, norm_det=[(1, 1.0)])

    results = apig.optimize(params_guess)
    apig_energy = results.x[-1]
    assert np.allclose(apig_energy, -0.2416648697421632)


def test_apig_init_defaults():
    """
    """
    # Define dummy hamiltonian
    nbasis = 6
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)

    test = APIG(ham, nocc)

    assert test.nparam == nbasis * nocc + 1
    assert test.nproj == nbasis * nocc + 1
    assert test.nactive == nbasis * nocc + 1
    assert test.nequation == nbasis * nocc + 1
    assert np.all(test.mask)

    assert isinstance(test.wfn, pyci.doci_wfn)
    assert test.nbasis == nbasis
    assert test.nocc_up == nocc
    assert test.nocc_dn == nocc
    assert test.nvir_up == nbasis - nocc
    assert test.nvir_dn == nbasis - nocc
    assert test.pspace.shape[0] == 13


def test_apig_permanent():
    matrix = np.arange(1, 65, dtype=float)
    answers = [
        1.0,
        1.0,
        10.0,
        450.0,
        55456.0,
        14480700.0,
        6878394720.0,
        5373548250000.0,
        6427291156586496.0,
    ]
    for i, answer in enumerate(answers):
        assert permanent(matrix[: i ** 2].reshape(i, i)) == answer
