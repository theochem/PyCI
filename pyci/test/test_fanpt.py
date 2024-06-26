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

from pyci.fanci.fanpt_wrapper import reduce_to_fock, solve_fanpt
from pyci.fanpt import FANPTUpdater, FANPTContainerEParam, FANPTContainerEFree

@pytest.mark.parametrize("filename, nocc, expected", [("he_ccpvqz",  1, -2.8868091056425156),
                                                      ("be_ccpvdz",  2, -14.600556820761211),
                                                      ("li2_ccpvdz", 3, -16.862861409549044),
                                                      ("lih_sto6g",  2, -8.963531095653355),
                                                      ("h2_631gdp",  1, -1.869682842154122),
                                                      ("h2o_ccpvdz", 5, -77.96987451399201)])
def test_fanpt_e_free(filename, nocc, expected):
    nsteps = 10
    order = 1

    # Define ham0 and ham1
    ham1 = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
    ham0 = pyci.hamiltonian(ham1.ecore, ham1.one_mo, reduce_to_fock(ham1.two_mo))

    # contruct empty fci wave function class instance from # of basis functions and occupation
    wfn0 = pyci.fullci_wfn(ham0.nbasis, nocc, nocc)
    wfn0.add_hartreefock_det()

    # initialize sparse matrix operator (hamiltonian into wave function)
    op = pyci.sparse_op(ham0, wfn0)

    # solve for the lowest eigenvalue and eigenvector
    e_hf, e_vecs0 = op.solve(n=1, tol=1.0e-8)

    # Get params as the solution of the fanci wfn with ham0 (last element will be the energy of the "ideal" system).
    nproj = int(comb(ham1.nbasis, nocc))
    pyci_wfn = AP1roG(ham1, nocc, nproj=nproj)

    params = np.zeros(pyci_wfn.nparam, dtype=pyci.c_double)
    params[-1] = e_hf[0]

    fill = 'excitation'
    fanpt_results = solve_fanpt(pyci_wfn, ham0, pyci_wfn.ham, params,
                                fill=fill, energy_active=False, resum=False, ref_sd=0,
                                final_order=order, lambda_i=0.0, lambda_f=1.0, steps=nsteps,
                                solver_kwargs={'mode':'lstsq', 'use_jac':True, 'xtol':1.0e-8,
                                'ftol':1.0e-8, 'gtol':1.0e-5, 'max_nfev':pyci_wfn.nparam, 'verbose':2})

    assert np.allclose(fanpt_results.x[-1], expected)

@pytest.mark.parametrize("filename, nocc, expected", [("he_ccpvqz",  1, -2.8868091056425156),
                                                      ("be_ccpvdz",  2, -14.600556842700215),
                                                      ("li2_ccpvdz", 3, -16.86286984124269),
                                                      ("lih_sto6g",  2, -8.96353109432708),
                                                      ("h2_631gdp",  1, -1.869682842154122),
                                                      ("h2o_ccpvdz", 5, -77.96987516399848)])
def test_fanpt_e_param(filename, nocc, expected):
    nsteps = 10
    order = 1

    # Define ham0 and ham1
    ham1 = pyci.hamiltonian(datafile("{0:s}.fcidump".format(filename)))
    ham0 = pyci.hamiltonian(ham1.ecore, ham1.one_mo, reduce_to_fock(ham1.two_mo))

    # contruct empty fci wave function class instance from # of basis functions and occupation
    wfn0 = pyci.fullci_wfn(ham0.nbasis, nocc, nocc)
    wfn0.add_hartreefock_det()

    # initialize sparse matrix operator (hamiltonian into wave function)
    op = pyci.sparse_op(ham0, wfn0)

    # solve for the lowest eigenvalue and eigenvector
    e_hf, e_vecs0 = op.solve(n=1, tol=1.0e-8)

    # Get params as the solution of the fanci wfn with ham0 (last element will be the energy of the "ideal" system).
    nproj = int(comb(ham1.nbasis, nocc))
    pyci_wfn = AP1roG(ham1, nocc, nproj=nproj)

    params = np.zeros(pyci_wfn.nparam, dtype=pyci.c_double)
    params[-1] = e_hf[0]

    fill = 'excitation'
    fanpt_results = solve_fanpt(pyci_wfn, ham0, pyci_wfn.ham, params,
                                fill=fill, energy_active=True, resum=False, ref_sd=0,
                                final_order=order, lambda_i=0.0, lambda_f=1.0, steps=nsteps,
                                solver_kwargs={'mode':'lstsq', 'use_jac':True, 'xtol':1.0e-8,
                                'ftol':1.0e-8, 'gtol':1.0e-5, 'max_nfev':pyci_wfn.nparam, 'verbose':2})

    assert np.allclose(fanpt_results.x[-1], expected)
