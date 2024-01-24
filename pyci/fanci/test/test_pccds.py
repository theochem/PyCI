""" Test pCCDS"""

import numpy as np

import pytest

import pyci

from pyci.fanci import pCCDS, AP1roG
from pyci.fanci.apig import permanent
from pyci.test import datafile


@pytest.fixture
def dummy_ham():
    nbasis = 5
    one_mo = np.arange(nbasis ** 2, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.hamiltonian(0.0, one_mo, two_mo)
    return ham


def test_4e_5mos_pccds_overlap(dummy_ham):
    pccsd = pCCDS(dummy_ham, 2, 2, nproj=None)
    nparams = pccsd.nocc_up * pccsd.nvir_up + pccsd.wfn.nocc * pccsd.wfn.nvir
    params = np.arange(nparams, dtype=pyci.c_double) + 1

    mat_s = params[pccsd.nocc_up * pccsd.nvir_up:].reshape(pccsd.wfn.nocc, pccsd.wfn.nvir)
    mat_p = params[:pccsd.nocc_up * pccsd.nvir_up].reshape(pccsd.wfn.nocc_up, pccsd.wfn.nvir_up)

    # Reference occs
    # alphas: [11000]
    # betas:  [11000]
    occsv = np.array([[[0,1],[0,1]]])
    ovl = pccsd.compute_overlap(params, occsv)
    assert np.allclose(ovl[0], 1.)
    # Exc=1, seniority=2
    # alphas: [11000]
    # betas:  [10100]
    occsv = np.array([[[0,1],[0,2]]])
    ovl = pccsd.compute_overlap(params, occsv)
    assert np.allclose(ovl[0], mat_s[3,3])
    # Exc=2, seniority=0
    # alphas: [10100]
    # betas:  [10100]
    occsv = np.array([[[0,2],[0,2]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = mat_p[1,0] + (mat_s[1,0]*mat_s[3,3] + mat_s[1,3]*mat_s[3,0])
    assert np.allclose(ovl[0], expected)
    # Exc=2, seniority=4
    # alphas: [10100]
    # betas:  [01010]
    occsv = np.array([[[0,2],[1,3]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = mat_s[1,0]*mat_s[2,4] + mat_s[1,4]*mat_s[2,0]
    assert np.allclose(ovl[0], expected)
    # Exc=3, seniority=2
    # alphas: [00110]
    # betas:  [10100]
    occsv = np.array([[[2,3],[0,2]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = mat_p[1,0]*mat_s[0,1] + permanent(mat_s[[0,1,3], :][:, [0,1,3]])
    assert np.allclose(ovl[0], expected)

    # Exc=3, seniority=4
    # alphas: [00110]
    # betas:  [10001]
    # This Slater determinant does not belong to the pCCSDSpin_sen-o space
    # and must have zero ovelap. If the wfn were pCCSDSpin_sen-free the it is
    # allowed and its ovlp = permanent(mat_s[[0,1,3], :][:, [0,1,5]])
    occsv = np.array([[[2,3],[0,4]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = 0.0
    assert np.allclose(ovl[0], expected)

    # Exc=4, seniority=0
    # alphas: [00110]
    # betas:  [00110]
    occsv = np.array([[[2,3],[2,3]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = mat_p[0,0]*mat_p[1,1] + mat_p[0,1]*mat_p[1,0]
    expected += permanent(mat_s[[0,1,2,3], :][:, [0,1,3,4]])
    expected += mat_p[0,0]*permanent(mat_s[[1,3], :][:, [1,4]]) + mat_p[1,1]*permanent(mat_s[[0,2], :][:, [0,3]]) + mat_p[0,1]*permanent(mat_s[[1,3], :][:, [0,3]]) + mat_p[1,0]*permanent(mat_s[[0,2], :][:, [1,4]])
    assert np.allclose(ovl[0], expected)


def test_2e_5mos_pccds_overlap(dummy_ham):
    pccsd = pCCDS(dummy_ham, 1, 1, nproj=None)
    nparams = pccsd.nocc_up * pccsd.nvir_up + pccsd.wfn.nocc * pccsd.wfn.nvir
    params = np.arange(nparams, dtype=pyci.c_double) + 1

    mat_s = params[pccsd.nocc_up * pccsd.nvir_up:].reshape(pccsd.wfn.nocc, pccsd.wfn.nvir)
    mat_p = params[:pccsd.nocc_up * pccsd.nvir_up].reshape(pccsd.wfn.nocc_up, pccsd.wfn.nvir_up)

    # Reference occs
    # alphas: [10000]
    # betas:  [10000]
    occsv = np.array([[[0],[0]]])
    ovl = pccsd.compute_overlap(params, occsv)
    print(ovl)
    assert np.allclose(ovl[0], 1.)
    # Exc=1, seniority=2
    # alphas: [10000]
    # betas:  [01000]
    occsv = np.array([[[0],[1]]])
    ovl = pccsd.compute_overlap(params, occsv)
    print(mat_s[1,4], ovl[0])
    assert np.allclose(ovl[0], mat_s[1,4])
    # Exc=1, seniority=2
    # alphas: [01000]
    # betas:  [10000]
    occsv = np.array([[[1],[0]]])
    ovl = pccsd.compute_overlap(params, occsv)
    print(mat_s[0,0], ovl[0])
    assert np.allclose(ovl[0], mat_s[0,0])
    # Exc=1, seniority=2
    # alphas: [01000]
    # betas:  [00100]
    # This Slater determinant does not belong to the pCCSDSpin_sen-o space
    # and must have zero ovelap. If the wfn were pCCSDSpin_sen-free the it is
    # allowed and its ovlp = mat_s[0,0]*mat_s[1,5] + mat_s[0,5]*mat_s[1,0]
    occsv = np.array([[[1],[2]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = 0.0 
    print(expected, ovl[0])
    assert np.allclose(ovl[0], expected)
    # Exc=2, seniority=0
    # alphas: [01000]
    # betas:  [01000]
    occsv = np.array([[[1],[1]]])
    ovl = pccsd.compute_overlap(params, occsv)
    expected = mat_p[0,0] + (mat_s[0,0]*mat_s[1,4] + mat_s[0,4]*mat_s[1,0])
    print(expected, ovl[0])
    assert np.allclose(ovl[0], expected)


# def systems_ground():
#     options_list = [
#         ((1,1), "h2_hf_631gdp", 0.71317683129, -1.84444, None, -1.1651487544545007,),
#         ((2,2), "lih_hf_sto6g", 0.995317634356, -8.94728, None, -7.972335583974739,),
#     ]
#     for p in options_list:
#         yield p


@pytest.mark.skip(reason="No reference data to compare against")
@pytest.mark.parametrize(
    "nocc, system, nucnuc, e_hf, nproj, expected", 
    [
        ((1,1), "h2_631gdp", 0.71317683129, -1.84444, None, -1.1651487544545007,),
        ((2,2), "lih_sto6g", 0.995317634356, -8.94728, None, -7.972335583974739,),
    ]
    )
def test_pccds(nocc, system, nucnuc, e_hf, nproj, expected):
    #
    # Use AP1roG as initial guess for pair excitations wfn params
    #
    ham = pyci.hamiltonian(datafile("{0:s}.fcidump".format(system)))
    ap1rog = AP1roG(ham, nocc[0], nproj=None)
    ap1_guess = np.zeros(ap1rog.nparam, dtype=pyci.c_double)
    ap1_guess[-1] = e_hf
    ap1_params = ap1rog.optimize(ap1_guess, use_jac=True)

    #
    # Set up FanCI pair-CCD+S
    #
    wfn = pyci.fullci_wfn(ap1rog.wfn)
    pyci.add_excitations(wfn, 1)
    pccds = pCCDS(ham, *nocc, nproj=nproj, wfn=wfn)
    params_guess = np.zeros(pccds.nparam, dtype=pyci.c_double)
    params_guess[:pccds.wfn.nocc_up * pccds.wfn.nvir_up] = ap1_params.x[:-1]
    params_guess[-1] = ap1_params.x[-1]
    # Solve FanCI
    results = pccds.optimize(params_guess, use_jac=False)

    energy = results.x[-1] + nucnuc
    print('E_sol', energy)
    print('E_ref', expected)
    # assert np.allclose(energy, expected)
