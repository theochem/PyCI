""" Test APIG"""

import numpy as np

import pytest

import pyci

from pyci.fanci import AP1roG
from pyci.fanci.apig import permanent
from pyci.test_fanci import find_datafile, assert_deriv


@pytest.fixture
def dummy_system():
    nbasis = 6
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)
    params = np.arange((nbasis - nocc) * nocc + 1, dtype=pyci.c_double) + 1
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
        (1, "h2_hf_sto6g", 0.71317683129, -1.83843, 2, -1.1459130128588935,),
        (1, "h2_hf_631gdp", 0.71317683129, -1.84444, 10, -1.1565060295404896,),
        (2, "lih_hf_sto6g", 0.995317634356, -8.94728, 15, -7.968213475280394,),
    ]
    for p in options_list:
        yield p


@pytest.mark.parametrize("expecting, args, kwargs", init_errors())
def test_ap1rog_init_errors(expecting, args, kwargs):
    """
    """
    with pytest.raises(expecting):
        AP1roG(*args, **kwargs)


def test_ap1rog_compute_overlap_deriv(dummy_system):
    ham, nocc, params = dummy_system
    apig = AP1roG(ham, nocc, nproj=None)

    f = lambda x: apig.compute_overlap(x, "S")
    j = lambda x: apig.compute_overlap_deriv(x, "S")
    origin = np.random.rand(params[:-1].shape[0])
    assert_deriv(f, j, origin)


def test_ap1rog_compute_objective(dummy_system):
    ham, nocc, params = dummy_system
    nproj = nocc * (ham.nbasis - nocc) + 1
    ap1rog = AP1roG(ham, nocc, nproj=None)

    objective = ap1rog.compute_objective(params)
    op = pyci.sparse_op(ap1rog.ham, ap1rog.wfn, nproj, symmetric=False)
    ovlp = ap1rog.compute_overlap(params[:-1], "S")
    answer = op(ovlp) - params[-1] * ovlp[:nproj]
    assert np.allclose(objective, answer)


def test_ap1rog_compute_jacobian(dummy_system):
    ham, nocc, params = dummy_system
    ap1rog = AP1roG(ham, nocc)

    f = ap1rog.compute_objective
    j = ap1rog.compute_jacobian
    origin = np.random.rand(params.shape[0])
    assert_deriv(f, j, origin)


@pytest.mark.parametrize("nocc, system, nucnuc, e_hf, nproj, expected", systems_ground())
def test_ap1rog_systems_ground(nocc, system, nucnuc, e_hf, nproj, expected):
    """Test cases adapted from FanCI's test_wfn_geminal_apig.

    """
    ham = pyci.hamiltonian(find_datafile("{0:s}.fcidump".format(system)))
    ap1rog = AP1roG(ham, nocc, nproj=nproj)

    params_guess = np.zeros(ap1rog.nparam, dtype=pyci.c_double)
    params_guess[-1] = e_hf
    results = ap1rog.optimize(params_guess, use_jac=True)
    ap1rog_energy = results.x[-1] + nucnuc
    assert np.allclose(ap1rog_energy, expected)


def test_ap1rog_init_defaults():
    """
    """
    # Define dummy hamiltonian
    nbasis = 6
    nocc = 2
    nvir = nbasis - nocc
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)

    test = AP1roG(ham, nocc)

    assert test.nparam == nvir * nocc + 1
    assert test.nproj == nvir * nocc + 1
    assert test.nactive == nvir * nocc + 1
    assert test.nequation == nvir * nocc + 1
    assert np.all(test.mask)

    assert isinstance(test.wfn, pyci.doci_wfn)
    assert test.nbasis == nbasis
    assert test.nocc_up == nocc
    assert test.nocc_dn == nocc
    assert test.nvir_up == nbasis - nocc
    assert test.nvir_dn == nbasis - nocc
    assert test.pspace.shape[0] == 9


def test_ap1rog_permanent():
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
