""" Test FANPTContainerEParam"""

import numpy as np

import pytest

import pyci

from fanci import APIG
from fanci.fanpt import FANPTUpdater, FANPTContainerEParam
from fanci.test import find_datafile


def run_fanpt(
    nocc,
    nproj,
    steps,
    fanci_wfn,
    params,
    ham0,
    ham1,
    final_order,
    ref_sd,
    inorm,
    ham_ci_op,
    f_pot_ci_op,
    ovlp_s,
    d_ovlp_s,
):
    for l in np.linspace(0.0, 1.0, steps, endpoint=False):
        fanpt_container = FANPTContainerEParam(
            fanci_wfn=fanci_wfn,
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

        # These params serve as initial guess to solve the fanci equations for the given lambda.
        fanpt_params = np.append(new_wfn_params, new_energy)

        # Initialize perturbed Hamiltonian with the current value of lambda using the static method of fanpt_container.
        ham = fanpt_container.linear_comb_ham(ham1, ham0, final_l, 1 - final_l)

        # Initialize fanci wfn with the perturbed Hamiltonian.
        fanci_wfn = APIG(ham, nocc, nproj=nproj, norm_det=[(0, 1.0)], wfn=None)

        # Solve fanci problem with fanpt_params as initial guess.
        # Take the params given by fanci and use them as initial params in the fanpt calculation for the next lambda.
        results = fanci_wfn.optimize(fanpt_params)
        params = results.x

    return params[-1]


@pytest.mark.parametrize(
    "filename, nocc, nproj, nsteps, order, hf, expected",
    [
        ("lih_hf_sto6g", 2, 15, 1, 2, -8.94728, -8.96353110963018),
        ("h2_hf_631gdp", 1, 10, 5, 2, -1.84444, -1.86968286083031),
        ("lih_hf_sto6g", 2, 15, 5, 5, -8.94728, -8.96353110963018),
    ],
)
def test_fanpt_e_param(filename, nocc, nproj, nsteps, order, hf, expected):
    """
    """
    # Define ham0 and ham1
    ham1 = pyci.hamiltonian(find_datafile("{0:s}.fcidump".format(filename)))
    two_int = np.zeros_like(ham1.two_mo, dtype=ham1.two_mo.dtype)
    fock = ham1.one_mo.copy()
    fock += np.einsum("piqi->pq", ham1.two_mo)
    fock -= np.einsum("piiq->pq", ham1.two_mo)
    ham0 = pyci.hamiltonian(ham1.ecore, fock, two_int)

    # Get params as the solution of the fanci wfn with ham0 (last element will be the energy of the "ideal" system).
    fanci_wfn = APIG(ham0, nocc, nproj=nproj, norm_det=[(0, 1.0)], wfn=None)
    params_guess = np.zeros(fanci_wfn.nparam, dtype=pyci.c_double)
    params_guess[-1] = hf
    params_guess[:-1].reshape(ham0.nbasis, nocc)[:, :] = np.eye(ham0.nbasis, nocc)

    results = fanci_wfn.optimize(params_guess, use_jac=True)

    # Set the initial variables:
    params = results.x
    ham0 = ham0
    ham1 = ham1
    l = 0
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

    fan_energy = run_fanpt(
        nocc,
        nproj,
        steps,
        fanci_wfn,
        params,
        ham0,
        ham1,
        final_order,
        ref_sd,
        inorm,
        ham_ci_op,
        f_pot_ci_op,
        ovlp_s,
        d_ovlp_s,
    )
    assert np.allclose(fan_energy, expected)
