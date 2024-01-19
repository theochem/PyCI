r"""FANPT updater."""

from functools import partial
from math import factorial

import numpy as np

import pyci

from .base_fanpt_container import FANPTContainer
from .fanpt_constant_terms import FANPTConstantTerms


class FANPTUpdater:
    r"""Solve the FANPT equations up to a given order, updates the wavefunction parameters and
    energy.

    Attributes
    ----------
    fanpt_container : FANPTContainer
        Object containing the FANPT matrices and vectors.
    final_order : int
        Final order of the FANPT calculation.
    final_l : float
        Lambda value up to which the FANPT calculation will be performed.
    solver : callable
        Solver that will be used to solve the FANPT equations.
    resum : bool
        Indicates if we will solve the FANPT equations by re-summing the series.
    inverse : np.ndarray
        Inverse of the coefficient matrix.
    A_matrix : np.ndarray
        A matrix in the resummation equations.
    b_vector : np.ndarray
        b vector in the resummation equations.
    responses : np.ndarray
        Responses up to the specified order.
    new_wfn_params : np.ndarray
        New wavefunction parameters.
    new_energy : float
        New energy.
    new_ham_op : pyci.sparse_op
        PyCI sparse operator of the perturbed Hamiltonian at the new value of lambda.
    new_ovlp_s : np.ndarray
        Overlaps of the wavefunction in the "S" projection space with the new parameters.
    fanpt_e : float
        Energy calculated as the sum of the FANPT responses of the variable E.

    Methods
    -------
    __init__(self, fanpt_container, final_order=1, final_l=1, solver=None)
        Initialize the updater.
    assign_fanpt_container(self, fanpt_container)
        Assign the FANPT container.
    assign_final_l(self, final_l)
        Assign the final lambda.
    inverse_coeff_matrix(self)
        Return the inverse of the matrix of derivatives of the FANPT equations with respect to
        the active parameters.
    resum_matrix(self)
        Return the matrix that appears in the resummation expressions.
    resum_vector(self)
        Return the vector that appears in the resummation expressions.
    fanpt_resum_correction(self):
        Return the resummation of all the responses.
    assign_final_order(self, final_order)
        Assign the final order.
    assign_solver(self, solver)
        Assign the solver.
    get_responses(self)
        Find the responses up to the final order.
    params_updater(self)
        Update the wavefunction parameters with the new responses up to the given value of
        final_lambda.
    energy_ham_ovlp_updater(self)
        Update the energy, Hamiltonian sparse operator, and wavefunction overlaps.
    fanpt_e_response(self)
        Return the energy calculated as the sum of the FANPT responses of the E variable.
    """

    def __init__(self, fanpt_container, final_order=1, final_l=1.0, solver=None, resum=False):
        r"""Initialize the updater.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            Object containing the FANPT matrices and vectors.
        final_order : int
            Final order of the FANPT calculation.
        final_l : float
            Lambda value up to which the FANPT calculation will be performed.
        solver : callable
            Solver that will be used to solve the FANPT equations.
        resum : bool
            Indicates if we will solve the FANPT equations by re-summing the series.
        """
        self.assign_fanpt_container(fanpt_container=fanpt_container)
        self.assign_final_l(final_l=final_l)
        self.assign_resum(resum)
        if self.resum:
            self.inverse_coeff_matrix()
            self.resum_matrix()
            self.resum_vector()
            self.fanpt_resum_correction()
        else:
            self.assign_final_order(final_order=final_order)
            self.assign_solver(solver=solver)
            self.get_responses()
        self.params_updater()
        self.energy_ham_ovlp_updater()
        if self.fanpt_container.active_energy:
            self.fanpt_e_response()

    def assign_fanpt_container(self, fanpt_container):
        r"""Assign the FANPT container.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            FANPTContainer object that contains all the matrices and vectors required to perform
            the FANPT calculation.

        Raises
        ------
        TypeError
            If fanpt_container is not a child of FANPTContainer.
        """
        if not isinstance(fanpt_container, FANPTContainer):
            raise TypeError("fanpt_container must be a child of FANPTContainer")
        self.fanpt_container = fanpt_container

    def assign_resum(self, resum):
        r"""Assign the value of resum.

        Returns
        ------
        resum : bool
            Indicates if we will solve the FANPT equations by re-summing the series.

        Note
        ----
        The resummation can only be performed when the energy is inactive, and for a determined
        system of equations, that is, if we have the same number of variables as FANPT equations,
        hence the condition: nequation == nactive.
        """
        if self.fanpt_container.active_energy:
            self.resum = False
        elif resum:
            if self.fanpt_container.nequation == self.fanpt_container.nactive:
                self.resum = resum
            else:
                self.resum = False
        else:
            self.resum = False

    def inverse_coeff_matrix(self):
        r"""Return the inverse of the matrix of derivatives of the FANPT equations with respect to
        the active parameters.

        Returns
        -------
        inverse : np.ndarray
            Inverse of the matrix of derivatives of the FANPT equations with respect to the
            active parameters.
        """
        self.inverse = np.linalg.inv(self.fanpt_container.c_matrix)

    def resum_matrix(self):
        r"""Return the matrix that appears in the resummation expressions.

        Returns
        -------
        A_matrix : np.ndarray
            Matrix that appears in the resummation expressions.

        A = [(dG/dp_k)^-1] * [d^2G/dldp_k]
        """
        self.A_matrix = np.dot(self.inverse, self.fanpt_container.d2_g_lambda_wfnparams)

    def resum_vector(self):
        r"""Return the vector that appears in the resummation expressions.

        Returns
        -------
        b_vector : np.ndarray
            Vector that appears in the resummation expressions.

        b = [(dG/dp_k)^-1] * dG/dl
        """
        self.b_vector = np.dot(self.inverse, self.fanpt_container.d_g_lambda)

    def fanpt_resum_correction(self):
        r"""Return the resummation of all the responses.

        Returns
        -------
        resum_result_correction : np.ndarray
            Sum of all the responses up to infinite order.

        -l * [(l * A + I)^-1] * b
        I : identity matrix
        """
        self.resum_correction = -(
            self.final_l
            * np.dot(
                np.linalg.inv(
                    self.final_l * self.A_matrix + np.identity(self.fanpt_container.nequation)
                ),
                self.b_vector,
            )
        )

    def assign_final_order(self, final_order):
        r"""Assign the final order.

        Parameters
        ----------
        final_order : int
            Order of the current FANPT iteration.

        Raises
        ------
        TypeError
            If final_order is not an int.
        ValueError
            If final_order is negative.
        """
        if not isinstance(final_order, int):
            raise TypeError("final_order must be an integer.")
        if final_order < 0:
            raise ValueError("final_order must be non-negative.")
        self.final_order = final_order

    def assign_final_l(self, final_l):
        r"""Assign the final lambda.

        Parameters
        ----------
        final_l : float
            Lambda value up to which the FANPT calculation will be performed.

        Raises
        ------
        TypeError
            If final_l is not a float.
        ValueError
            If final_l is not between self.fanpt_container.l_param and 1.
        """
        if not isinstance(final_l, float):
            raise TypeError("final_l must be given as a float.")
        if not self.fanpt_container.l < final_l <= 1.0:
            raise ValueError(
                "final_l must be greater than {} and lower or equal than 1.".format(
                    self.fanpt_container.l
                )
            )
        self.final_l = final_l

    def assign_solver(self, solver):
        r"""Assign solver."""
        if solver is None:
            self.solver = partial(np.linalg.lstsq, rcond=-1)

    def get_responses(self):
        r"""Find the responses up to the final order.

        Returns
        -------
        resp_matrix : np.ndarray
            Numpy array with shape (final_order, nactive).
            resp_matrix[k] contains the responses of order k+1.
            If the energy is active, the final element of each row is the response of the energy.
        """
        resp_matrix = np.zeros((self.final_order, self.fanpt_container.nactive))
        for o in range(1, self.final_order + 1):
            c_terms = FANPTConstantTerms(
                fanpt_container=self.fanpt_container,
                order=o,
                previous_responses=resp_matrix[: o - 1, :],
            )
            constant_terms = c_terms.constant_terms
            resp_matrix[o - 1] = self.solver(self.fanpt_container.c_matrix, constant_terms)[0]
        self.responses = resp_matrix

    def params_updater(self):
        r"""Update the wavefunction parameters with the new responses up to the given value of
        final_lambda.

        Returns
        -------
        new_wfn : BaseWavefunction
            Updated wavefunction with the FANPT responses.
            Does not modify the value of self.fanpt_container.wfn (it uses a deepcopy).

        Notes
        -----
        It only updates the active wavefunction parameters.
        """
        wfn_params = self.fanpt_container.wfn_params
        wfn_params = wfn_params.flatten()
        if self.resum:
            corrections = self.resum_correction
        else:
            l0 = self.fanpt_container.l
            dl = np.array(
                [
                    (self.final_l - l0) ** order / factorial(order)
                    for order in range(1, self.final_order + 1)
                ]
            )
            if self.fanpt_container.active_energy:
                wfn_responses = self.responses[:, :-1]
            else:
                wfn_responses = self.responses
            corrections = np.sum(wfn_responses * dl.reshape(self.final_order, 1), axis=0)
        active_wfn_indices = np.where(self.fanpt_container.fanci_wfn.mask[:-1])[0]
        for c, active_index in zip(corrections, active_wfn_indices):
            wfn_params[active_index] += c
        self.new_wfn_params = wfn_params

    def energy_ham_ovlp_updater(self):
        r""""Update the energy, Hamiltonian sparse operator, and wavefunctoin overlaps.

        Generates
        ---------
        new_energy : float
            E = <ref_sd|ham(final_l)|psi(final_l)>
        new_ham_op : pyci.sparse_op
            PyCI sparse operator of the perturbed Hamiltonian at the new value of lambda.
        new_ovlp_s : np.ndarray
            Overlaps of the wavefunction in the "S" projection space with the new parameters.

        Notes
        -----
        This E satisfies the 2n + 1 rule.
        """
        new_ham = FANPTContainer.linear_comb_ham(
            self.fanpt_container.ham1, self.fanpt_container.ham0, self.final_l, 1 - self.final_l
        )
        new_ham_op = pyci.sparse_op(
            new_ham, self.fanpt_container.fanci_wfn.wfn, self.fanpt_container.nproj
        )
        new_ovlp_s = self.fanpt_container.fanci_wfn.compute_overlap(self.new_wfn_params, "S")
        f = np.empty(self.fanpt_container.nproj, dtype=pyci.c_double)
        new_ham_op(new_ovlp_s, out=f)
        if self.fanpt_container.inorm:
            energy = f[self.fanpt_container.ref_sd]
        else:
            energy = f[self.fanpt_container.ref_sd] / new_ovlp_s[self.fanpt_container.ref_sd]
        self.new_energy = energy
        self.new_ham_op = new_ham_op
        self.new_ovlp_s = new_ovlp_s

    def fanpt_e_response(self):
        r"""Return the energy calculated as the sum of the FANPT responses of the E variable.

        Generates
        ---------
        fanpt_e : float
            Energy calculated as the sum of the FANPT responses of the variable E.
            E(final_l) = E(l) + sum_k{1/k! * (final_l - l)^k * d^kE/dl^k}

        Notes
        -----
        This E does not satisfy the 2n + 1 rule.
        """
        e0 = self.fanpt_container.energy
        l0 = self.fanpt_container.l
        dl = np.array(
            [
                (self.final_l - l0) ** order / factorial(order)
                for order in range(1, self.final_order + 1)
            ]
        )
        e_responses = self.responses[:, -1]
        self.fanpt_e = e0 + np.sum(e_responses * dl)
