""" Test APIG"""

import os

import numpy as np

import pytest

import numpy.testing as npt

from scipy.optimize import least_squares, minimize, fsolve

import pyci

from pyci.fanci import FanCI

from pyci.fanci.test import find_datafile

from pyci.fanci import AP1roG


def test_fanci_init():
    """ """
    pass


def test_fanci_add_constraint():
    """ """
    pass


def test_fanci_remove_constraint():
    """ """
    pass


def test_fanci_add_constraint():
    """ """
    pass


def test_fanci_freeze_parameter():
    """ """
    pass


def test_fanci_unfreeze_parameter():
    """ """
    pass


def test_fanci_make_param_constraint():
    """ """
    pass


def test_fanci_make_det_constraint():
    """ """
    pass


def test_fanci_optimize():
    filename = find_datafile("lih_hf_sto6g.fcidump")
    ham = pyci.hamiltonian(filename)
    e_dict = {}
    # First compute the hartree-Fock reference energy
    hf_wfn = pyci.doci_wfn(ham.nbasis, 2, 2)
    hf_wfn.add_hartreefock_det()
    hf_op = pyci.sparse_op(ham, hf_wfn)
    e_dict["HF"] = hf_op.solve(n=1)[0][0] - ham.ecore

    # Initialize AP1roG instance
    ap1rog = AP1roG(ham, 2)

    # Make initial guess
    ap1_params = np.zeros(ap1rog.nparam, dtype=pyci.c_double)
    ap1_params[-1] = e_dict["HF"]

    # Testing optimization with least-squares
    ap1rog_results = ap1rog.optimize(ap1_params, use_jac=True)
    e_dict["AP1roG"] = ap1rog_results.x[-1]
    npt.assert_allclose(e_dict["AP1roG"], -8.963531034, rtol=0.0, atol=1.0e-7)

    # Testing optimization with with root scypy function
    ap1rog_results = ap1rog.optimize(ap1_params, use_jac=True, mode="root")
    e_dict["AP1roG"] = ap1rog_results.x[-1]
    npt.assert_allclose(e_dict["AP1roG"], -8.963531034, rtol=0.0, atol=1.0e-7)

    # Testing optimization with with L-BFGS-B
    ap1rog_results = ap1rog.optimize(
        ap1_params,
        use_jac=True,
        mode="custom_scalar",
        custom_optimizer=minimize,
        method="L-BFGS-B",
    )
    e_dict["AP1roG"] = ap1rog_results.x[-1]
    npt.assert_allclose(e_dict["AP1roG"], -8.963531034, rtol=0.0, atol=1.0e-6)

    # Testing optimization with with Newton-CG
    ap1rog_results = ap1rog.optimize(
        ap1_params,
        use_jac=True,
        mode="custom_scalar",
        custom_optimizer=minimize,
        method="Newton-CG",
    )
    e_dict["AP1roG"] = ap1rog_results.x[-1]
    npt.assert_allclose(e_dict["AP1roG"], -8.963531034, rtol=0.0, atol=1.0e-7)

    # Testing optimization with with fsolve
    ap1rog_results = ap1rog.optimize(
        ap1_params, use_jac=False, mode="custom_vector", custom_optimizer=fsolve
    )
    e_dict["AP1roG"] = ap1rog_results[-1]
    npt.assert_allclose(e_dict["AP1roG"], -8.963531034, rtol=0.0, atol=1.0e-7)
