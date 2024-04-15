""" FANPT wrapper"""

import numpy as np
import pyci

from .fanci import FanCI
from ..fanpt import FANPTUpdater, FANPTContainerEParam, FANPTContainerEFree

def solve_fanpt(
    fanci_wfn,
    ham0,
    ham1,
    guess_params,
    fill,
    energy_active=True,
    resum=False,
    ref_sd=0,
    final_order=1,
    lambda_i=0.0,
    lambda_f=1.0,
    steps=1,
    kwargs=None,
    solver_kwargs=None,
):
    """[summary]

    Args:
        fanci_wfn : FanCI class
            FanCI wavefunction.
        guess_params : np.ndarray
            Initial guess for wave function parameters.
        ham0 : pyci.hamiltonian
            PyCI Hamiltonian of the ideal system.
        ham1 : pyci.hamiltonian
            PyCI Hamiltonian of the real system.
        energy_active : bool, optional
            Whether the energy is an active parameter. It determines which FANPT
            method is used. If set to true, FANPTContainerEParam is used.
            Defaults to True.
        resum : bool, optional
            Indicates if we will solve the FANPT equations by re-summing the series.
            Defaults to False.
        ref_sd : int, optional
            Index of the Slater determinant used to impose intermediate normalization.
            <n[ref_sd]|Psi(l)> = 1. Defaults to 0.
        final_order : int, optional
            Final order of the FANPT calculation. Defaults to 1.
        lambda_i : float, optional
            Initial lambda value for the solution of the FANPT equations. Defaults to 0.0.
        lambda_f : float, optional
            Lambda value up to which the FANPT calculation will be performed. Defaults to 1.0.
        steps (int, optional): int, optional
            Solve FANPT in n stepts between lambda_i and lambda_f. Defaults to 1.
        kwargs (dict, optional):
            Additional keyword arguments for FanPTContainer class. Defaults to {}.

    Raises:
        TypeError: [description]

    Returns:
        params: np.ndarray
        Solution of the FANPT calculation.
    """
    if not isinstance(fanci_wfn, FanCI):
        raise TypeError("fanci_wfn must be a FanCI wavefunction")
    if kwargs is None:
        kwargs = {}
    if solver_kwargs is None:
        solver_kwargs = {}

    # Check for normalization constraint in FANCI wfn
    # Assumes intermediate normalization relative to ref_sd only
    if f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}" in fanci_wfn.constraints:
        inorm = True
        norm_det = [(ref_sd, 1.0)]
    else:
        inorm = False
        norm_det = None

    # Select FANPT method
    if energy_active:
        fanptcontainer = FANPTContainerEParam

    if resum:
        if energy_active:
            raise ValueError(
                "The energy parameter must be inactive with the resumation option."
            )
        nequation = fanci_wfn.nequation
        nactive = fanci_wfn.nactive
        steps = 1
        if not inorm and (nequation == nactive):
            norm_det = [(ref_sd, 1.0)]
        elif inorm and (nequation - 1) == nactive:
            fanci_wfn.remove_constraint(f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}")
            inorm = False
        else:
            raise ValueError(
                "The necesary condition of a determined system of equations is not met."
            )

    # Get initial guess for parameters at initial lambda value.
    results = fanci_wfn.optimize(guess_params, **solver_kwargs)
    params = results.x

    # Solve FANPT equations
    for l in np.linspace(lambda_i, lambda_f, steps, endpoint=False):
        fanpt_container = fanptcontainer(
            fanci_wfn=fanci_wfn,
            params=params,
            ham0=ham0,
            ham1=ham1,
            l=l,
            inorm=inorm,
            ref_sd=ref_sd,
            **kwargs,
        )

        final_l = l + (lambda_f - lambda_i) / steps
        print(f'Solving FanPT problem at lambda={final_l}')

        fanpt_updater = FANPTUpdater(
            fanpt_container=fanpt_container,
            final_order=final_order,
            final_l=final_l,
            solver=None,
            resum=resum,
        )
        new_wfn_params = fanpt_updater.new_wfn_params
        new_energy = fanpt_updater.new_energy

        # These params serve as initial guess to solve the fanci equations for the given lambda.
        fanpt_params = np.append(new_wfn_params, new_energy)
        print('Frobenius Norm of parameters: {}'.format(np.linalg.norm(fanpt_params - params)))
        print('Energy change: {}'.format(np.linalg.norm(fanpt_params[-1] - params[-1])))

        # Initialize perturbed Hamiltonian with the current value of lambda using the static method of fanpt_container.
        ham = fanpt_container.linear_comb_ham(ham1, ham0, final_l, 1 - final_l)

        # Initialize fanci wfn with the perturbed Hamiltonian.
        fanci_wfn = update_fanci_wfn(ham, fanci_wfn, norm_det, fill)

        # Solve the fanci problem with fanpt_params as initial guess.
        # Take the params given by fanci and use them as initial params in the
        # fanpt calculation for the next lambda.
        results = fanci_wfn.optimize(fanpt_params, **solver_kwargs)
        params = results.x

        if not energy_active:
            fanci_wfn.freeze_parameter([-1])

    # Output for debugging purposes
    results["energy"] = fanpt_params[-1]
    results["residuals"] = results.fun
    #output = {
    #    "x": params,
    #    "variables": {
    #        "steps": steps,
    #        "inorm": inorm,
    #        "norm_det": norm_det,
    #        "final_l": final_l,
    #    },
    #}
    return results


def update_fanci_wfn(ham, fanciwfn, norm_det, fill):
    fanci_class = fanciwfn.__class__

    if isinstance(fanciwfn.wfn, pyci.fullci_wfn):
        nocc = (fanciwfn.wfn.nocc_up, fanciwfn.wfn.nocc_dn)
    else:
        nocc = fanciwfn.wfn.nocc_up

    # Activate energy parameter
    fanciwfn.unfreeze_parameter([-1])

    # FIXME: for FanCI class
    #return fanci_class(
    #    ham, fanciwfn.wfn, fanciwfn.nproj, fanciwfn.nparam, norm_det=norm_det, fill=fill
    #)

    # for fanpy class
    # ASSUMING GeneratedFanCI class
    return fanci_class(
        ham, fanciwfn._fanpy_wfn, fanciwfn._wfn.nocc_up + fanciwfn._wfn.nocc_dn,
        nproj=fanciwfn.nproj, wfn=fanciwfn.wfn, fill=fill, seniority=fanciwfn.seniority,
        step_print=fanciwfn.step_print, step_save=fanciwfn.step_save, tmpfile=fanciwfn.tmpfile,
        objective_type=fanciwfn.objective_type, norm_det=norm_det,
        param_selection=fanciwfn.indices_component_params,
        constraints=fanciwfn._constraints
    )


def reduce_to_fock(two_int, lambda_val=0):
    """Reduce given two electron integrals to that of the correspoding Fock operator.

    Parameters
    ----------
    two_int : np.ndarray(K, K, K, K)
        Two electron integrals of restricted orbitals.

    """
    fock_two_int = two_int * lambda_val
    nspatial = two_int.shape[0]
    indices = np.arange(nspatial)
    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]]
    fock_two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]]
    #fock_two_int[indices[None, :, None], indices[:, None, None], indices[None, None, :], indices[None, :, None]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, None, :], indices[None, :, None]]
    #fock_two_int[indices[None, :, None], indices[:, None, None], indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], indices[None, :, None], indices[None, :, None], indices[None, None, :]]

    #occ_indices = np.arange(nelec // 2)
    #fock_two_int[indices[:, None, None], occ_indices[None, :, None], indices[None, None, :], occ_indices[None, :, None]] =  two_int[indices[:, None, None], occ_indices[None, :, None], indices[None, None, :], occ_indices[None, :, None]]
    #fock_two_int[indices[:, None, None], occ_indices[None, :, None], occ_indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], occ_indices[None, :, None], occ_indices[None, :, None], indices[None, None, :]]
    #fock_two_int[occ_indices[None, :, None], indices[:, None, None], indices[None, None, :], occ_indices[None, :, None]] =  two_int[indices[:, None, None], occ_indices[None, :, None], indices[None, None, :], occ_indices[None, :, None]]
    #fock_two_int[occ_indices[None, :, None], indices[:, None, None], occ_indices[None, :, None], indices[None, None, :]] =  two_int[indices[:, None, None], occ_indices[None, :, None], occ_indices[None, :, None], indices[None, None, :]]

    return fock_two_int
