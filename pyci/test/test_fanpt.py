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

import numpy as np
import pyci
from scipy.special import comb

import pytest
from pyci.test import datafile
from pyci.fanci import AP1roG

from pyci.fanci.fanpt_wrapper import reduce_to_fock
from pyci.fanpt import FANPTUpdater, FANPTContainerEParam, FANPTContainerEFree

# Tests for FanPT
@pytest.mark.parametrize("filename, nocc, expected", [("he_ccpvqz", 2, -1.4828564433468483), ("be_ccpvdz", 4, -12.956853565213162), ("li2_ccpvdz", 4, -16.710930177695687), ("lih_sto6g", 4, -6.790870061240047), ("h2_631gdp", 2, -0.8746939753865841)])
def test_fanpt_e_param(filename, nocc, expected):
    nsteps = 10
    order = 2

    # Define ham0 and ham1
    ham1 = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
    ham0 = pyci.hamiltonian(ham1.ecore, ham1.one_mo, reduce_to_fock(ham1.two_mo))

    # contruct empty fci wave function class instance from # of basis functions and occupation
    wfn0 = pyci.fullci_wfn(ham0.nbasis, nocc, nocc)
    wfn0.add_hartreefock_det()

    # initialize sparse matrix operator (hamiltonian into wave function)
    op = pyci.sparse_op(ham0, wfn0)

    # solve for the lowest eigenvalue and eigenvector
    e_hf, e_vecs0 = op.solve(n=1, tol=1.0e-9)

    # Get params as the solution of the fanci wfn with ham0 (last element will be the energy of the "ideal" system).
    nproj = int(comb(ham1.nbasis, nocc))
    pyci_wfn = AP1roG(ham1, nocc, nproj=nproj, norm_det=[(0, 1.0)], wfn=None)

    params_guess = np.zeros(pyci_wfn.nparam, dtype=pyci.c_double)
    params_guess[-1] = e_hf[0]
    params_guess[:-1] = np.eye(ham1.nbasis, nocc).reshape(-1)[len(params_guess)]

    results = pyci_wfn.optimize(params_guess, use_jac=True)

    # Set the initial variables:
    params = results.x
    ham0 = ham0
    ham1 = ham1
    ref_sd = 0
    ham_ci_op = None
    f_pot_ci_op = None
    ovlp_s = None
    d_ovlp_s = None

    # steps = int (number of steps that will be taken along the path)
    steps = nsteps

    # final_order = int (order up to which we'll solve the fanpt equations)
    final_order = order

    # inorm = bool (whether intermediate normalization is applied or not)
    inorm = True

    for l in np.linspace(0.0, 1.0, steps, endpoint=False):
        fanpt_container = FANPTContainerEParam(
            fanci_wfn=pyci_wfn,
            params=params,
            ham0=ham0,
            ham1=ham1,
            l=l,
            ref_sd=ref_sd,
            inorm=inorm,
            ham_ci_op=ham_ci_op,
            f_pot_ci_op=f_pot_ci_op,
            ovlp_s=ovlp_s,
            d_ovlp_s=d_ovlp_s,
        )

        final_l = l + 1 / steps
        fanpt_updater = FANPTUpdater(
            fanpt_container=fanpt_container,
            final_order=final_order,
            final_l=final_l,
            solver=None,
            resum=False,
        )
        new_wfn_params = fanpt_updater.new_wfn_params
        new_energy = fanpt_updater.new_energy

        # These params serve as initial guess to solve the FanCI equations for the given lambda.
        fanpt_params = np.append(new_wfn_params, new_energy)

        # Initialize perturbed Hamiltonian with the current value of lambda using the static method of fanpt_container.
        ham = fanpt_container.linear_comb_ham(ham1, ham0, final_l, 1 - final_l)

        # Initialize wfn with the perturbed Hamiltonian.
        pyci_wfn = AP1roG(ham, nocc, nproj=nproj, norm_det=[(0, 1.0)], wfn=None)

        # Solve fanci problem with fanpt_params as initial guess.
        # Take the params given by PyCI and use them as initial params in the FanPT calculation for the next lambda.
        results = pyci_wfn.optimize(fanpt_params)
        params = results.x

    assert np.allclose(params[-1], expected)

@pytest.mark.parametrize("filename, nocc, expected", [("he_ccpvqz", 2, -1.4828563704249667), ("be_ccpvdz", 4, -12.956853413604742), ("li2_ccpvdz", 4, -16.710930203005727), ("lih_sto6g", 4, -6.790870514399225), ("h2_631gdp", 2, -0.874693982384949)])
def test_fanpt_e_free(filename, nocc, expected):
    nsteps = 10
    order = 2

    # Define ham0 and ham1
    ham1 = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
    ham0 = pyci.hamiltonian(ham1.ecore, ham1.one_mo, reduce_to_fock(ham1.two_mo))

    # contruct empty fci wave function class instance from # of basis functions and occupation
    wfn0 = pyci.fullci_wfn(ham0.nbasis, nocc, nocc)
    wfn0.add_hartreefock_det()

    # initialize sparse matrix operator (hamiltonian into wave function)
    op = pyci.sparse_op(ham0, wfn0)

    # solve for the lowest eigenvalue and eigenvector
    e_hf, e_vecs0 = op.solve(n=1, tol=1.0e-9)

    # Get params as the solution of the fanci wfn with ham0 (last element will be the energy of the "ideal" system).
    nproj = int(comb(ham1.nbasis, nocc))
    pyci_wfn = AP1roG(ham1, nocc, nproj=nproj, norm_det=[(0, 1.0)], wfn=None)

    params_guess = np.zeros(pyci_wfn.nparam, dtype=pyci.c_double)
    params_guess[-1] = e_hf[0]
    params_guess[:-1] = np.eye(ham1.nbasis, nocc).reshape(-1)[len(params_guess)]

    results = pyci_wfn.optimize(params_guess, use_jac=True)

    # Set the initial variables:
    params = results.x
    ham0 = ham0
    ham1 = ham1
    ref_sd = 0
    ham_ci_op = None
    f_pot_ci_op = None
    ovlp_s = None
    d_ovlp_s = None

    # steps = int (number of steps that will be taken along the path)
    steps = nsteps

    # final_order = int (order up to which we'll solve the fanpt equations)
    final_order = order

    # inorm = bool (whether intermediate normalization is applied or not)
    inorm = True

    for l in np.linspace(0.0, 1.0, steps, endpoint=False):
        fanpt_container = FANPTContainerEFree(
            fanci_wfn=pyci_wfn,
            params=params,
            ham0=ham0,
            ham1=ham1,
            l=l,
            ref_sd=ref_sd,
            inorm=inorm,
            ham_ci_op=ham_ci_op,
            f_pot_ci_op=f_pot_ci_op,
            ovlp_s=ovlp_s,
            d_ovlp_s=d_ovlp_s,
        )

        final_l = l + 1 / steps
        fanpt_updater = FANPTUpdater(
            fanpt_container=fanpt_container,
            final_order=final_order,
            final_l=final_l,
            solver=None,
            resum=False,
        )
        new_wfn_params = fanpt_updater.new_wfn_params
        new_energy = fanpt_updater.new_energy

        # These params serve as initial guess to solve the FanCI equations for the given lambda.
        fanpt_params = np.append(new_wfn_params, new_energy)

        # Initialize perturbed Hamiltonian with the current value of lambda using the static method of fanpt_container.
        ham = fanpt_container.linear_comb_ham(ham1, ham0, final_l, 1 - final_l)

        # Initialize wfn with the perturbed Hamiltonian.
        pyci_wfn = AP1roG(ham, nocc, nproj=nproj, norm_det=[(0, 1.0)], wfn=None)

        # Solve fanci problem with fanpt_params as initial guess.
        # Take the params given by PyCI and use them as initial params in the FanPT calculation for the next lambda.
        results = pyci_wfn.optimize(fanpt_params)
        params = results.x

    assert np.allclose(params[-1], expected)

