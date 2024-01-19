r"""Base class that contains the elements required to perform a FANPT calculation."""

from abc import ABCMeta, abstractmethod

import pyci


class FANPTContainer(metaclass=ABCMeta):
    r"""Container for the matrices and vectors required ot perform a FANPT calculation.

    We assume that the equations to be solved have the following structure:

    A block of nproj equations obtained by projecting the Schrödinger equation in a space
    of nproj Slater determinants:

    G_1 = <1|ham(l)|psi(l)> - E * <1|psi(l)> = 0
    G_2 = <2|ham(l)|psi(l)> - E * <2|psi(l)> = 0
    ....
    G_nproj = <nproj|ham(l)|psi(l)> - E * <nproj|psi(l)> = 0

    We also have constraint equations (at least one, used to impose intermediate normalization).
    It is assumed that the constraint equations only depend on the wavefunction parameters, p_k,
    and are independent of the energy, E, and lambda, l. This implies that while all the vector
    have nequation elements and all the matrices have nequation rows, except for the coefficient
    matrix (dG/dp_k), only the first nproj elements of the vectors and the first nproj rows of
    the matrices are non-zero.

    Attributes
    ----------
    fanci_wfn : FanCI instance
        FanCI wavefunction.
    params : np.ndarray
        Wavefunction parameters and energy at for the given lambda value.
    ham0 : pyci.hamiltonian
        PyCI Hamiltonian of the ideal system.
    ham1 : pyci.hamiltonian
        PyCI Hamiltonian of the real system.
    l : float
        Lambda value.
    ref_sd : int
        Index of the Slater determinant used to impose intermediate normalization.
        <n[ref_sd]|Psi(l)> = 1.
    inorm : bool
        Indicates whether we will work with intermediate normalization or not.
    ham : pyci.hamiltonian
        PyCI Hamiltonian for the given value of lambda.
        ham = l * ham1 + (1 - l) * ham0
    f_pot : pyci.hamiltonian
        PyCI Hamiltonian corresponding to the fluctuation potential.
        f_pot = ham1 - ham0
    wfn_params : np.ndarray
        Wavefunction parameters.
    energy : float
        Energy for the current value of lambda.
    active_energy : bool
        Indicates if the energy will be varied in the calculations.
        It is False either when the energy is frozen in a E-param calculation
        or in any E-free calculation.
    ham_ci_op : pyci.sparse_op
        PyCI sparse operator corresponding to the perturbed Hamiltonian.
    f_pot_ci_op : pyci.sparse_op
        PyCI sparse operator corresponding to the fluctuation potential.
    ovlp_s : np.ndarray
        Overlaps of the wavefunction with the determinants in the "S" space.
    d_ovlp_s : np.ndarray
        Derivatives of the overlaps of the wavefunction with the determinants in the "S" space
        with respect to the active wavefunction parameters.
    d_g_lambda : np.ndarray
        Derivative of the FANPT equations with respect to the lambda parameter.
        numpy array with shape (self.nequations,).
    d2_g_lambda_wfnparams : np.ndarray
        Derivative of the FANPT equations with respect to lambda and the wavefunction
        parameters.
        numpy array with shape (self.nequations, len(self.wfn_params_active)).
    c_matrix : np.ndarray
            Coefficient matrix of the FANPT system of equations.
            numpy array with shape (self.nequations, len(self.nactive)).

    Properties
    ----------
    nactive : int
        Number of active parameters.
    nequation : int
        Number of equations.
    nproj : int
        Number of determinants in the projection ("P") space.

    Methods
    -------
    __init__(self, fanci_wfn, params, ham0, ham1, l=0, ref_sd=0)
        Initialize the FANPT container.
    linear_comb_ham(ham1, ham0, a1, a0)
        Return a linear combination of two PyCI Hamiltonians.
    der_g_lambda(self)
        Derivative of the FANPT equations with respect to the lambda parameter.
    der2_g_lambda_wfnparams(self)
        Derivative of the FANPT equations with respect to lambda and the wavefunction parameters.
    gen_coeff_matrix(self)
        Generate the coefficient matrix of the linear FANPT system of equations.
    """

    def __init__(
        self,
        fanci_wfn,
        params,
        ham0,
        ham1,
        l=0,
        ref_sd=0,
        inorm=False,
        ham_ci_op=None,
        f_pot_ci_op=None,
        ovlp_s=None,
        d_ovlp_s=None,
    ):
        r"""Initialize the FANPT container.

        Parameters
        ----------
        fanci_wfn : FanCI instance
            FanCI wavefunction.
        params : np.ndarray
            Wavefunction parameters and energy at for the given lambda value.
        ham0 : pyci.hamiltonian
            PyCI Hamiltonian of the ideal system.
        ham1 : pyci.hamiltonian
            PyCI Hamiltonian of the real system.
        l : float
            Lambda value.
        ref_sd : int
            Index of the Slater determinant used to impose intermediate normalization.
            <n[ref_sd]|Psi(l)> = 1.
        ham_ci_op : {pyci.sparse_op, None}
            PyCI sparse operator of the perturbed Hamiltonian.
        f_pot_ci_op : {pyci.sparse_op, None}
            PyCI sparse operator of the fluctuation potential.
        ovlp_s : {np.ndarray, None}
            Overlaps in the "S" projection space.
        d_ovlp_s : {np.ndarray, None}
            Derivatives of the overlaps in the "S" projection space.
        """
        # Separate parameters for better readability.
        self.params = params
        self.wfn_params = params[:-1]
        self.energy = params[-1]
        self.active_energy = fanci_wfn.mask[-1]
        self.inorm = inorm

        # Assign ideal and real Hamiltonians.
        self.ham1 = ham1
        self.ham0 = ham0

        # Lambda parameter.
        self.l = l

        # Assign FanCI wfn
        self.fanci_wfn = fanci_wfn

        # Build the perturbed Hamiltonian.
        self.ham = FANPTContainer.linear_comb_ham(self.ham1, self.ham0, self.l, 1 - self.l)

        # Construct the perturbed Hamiltonian and fluctuation potential sparse operators.
        if ham_ci_op:
            self.ham_ci_op = ham_ci_op
        else:
            self.ham_ci_op = pyci.sparse_op(self.ham, self.fanci_wfn.wfn, self.fanci_wfn.nproj)
        if f_pot_ci_op:
            self.f_pot_ci_op = f_pot_ci_op
        else:
            self.f_pot = FANPTContainer.linear_comb_ham(self.ham1, self.ham0, 1.0, -1.0)
            self.f_pot_ci_op = pyci.sparse_op(self.f_pot, self.fanci_wfn.wfn, self.fanci_wfn.nproj)

        if ovlp_s:
            self.ovlp_s = ovlp_s
        else:
            self.ovlp_s = self.fanci_wfn.compute_overlap(self.wfn_params, "S")
        if d_ovlp_s:
            self.d_ovlp_s = d_ovlp_s
        else:
            self.d_ovlp_s = self.fanci_wfn.compute_overlap_deriv(self.wfn_params, "S")

        # Update Hamilonian in the fanci_wfn.
        self.fanci_wfn._ham = self.ham

        # Assign ref_sd.
        if self.inorm:
            if f"<\\psi_{{{ref_sd}}}|\\Psi> - v_{{{ref_sd}}}" in self.fanci_wfn.constraints:
                self.ref_sd = ref_sd
            else:
                raise KeyError(
                    "The normalization of the Slater determinant is not constrained"
                    "in the FanCI wavefunction."
                )
        else:
            self.ref_sd = ref_sd

        # Generate the required vectors and matrices.
        self.der_g_lambda()
        self.der2_g_lambda_wfnparams()
        self.gen_coeff_matrix()

    @staticmethod
    def linear_comb_ham(ham1, ham0, a1, a0):
        r"""Return a linear combination of two PyCI Hamiltonians.

        Parameters
        ----------
        ham1 : pyci.hamiltonian
            PyCI Hamiltonian of the real system.
        ham0 : pyci.hamiltonian
            PyCI Hamiltonian of the ideal system.
        a1 : float
            Coefficient of the real Hamiltonian.
        a0 : float
            Coefficient of the ideal Hamiltonian.

        Returns
        -------
        pyci.hamiltonian
        """
        ecore = a1 * ham1.ecore + a0 * ham0.ecore
        one_mo = a1 * ham1.one_mo + a0 * ham0.one_mo
        two_mo = a1 * ham1.two_mo + a0 * ham0.two_mo
        return pyci.hamiltonian(ecore, one_mo, two_mo)

    @property
    def nactive(self):
        r"""Return the number of active parameters.

        Returns
        -------
        self.fanci_wfn.nactive : int
            Number of active parameters.
        """
        return self.fanci_wfn.nactive

    @property
    def nequation(self):
        r"""Return the number of equations.

        Returns
        -------
        self.fanci_wfn.nequation : int
            Number of equations (includes the number of constraints).
        """
        return self.fanci_wfn.nequation

    @property
    def nproj(self):
        r"""Return the number of determinants in the projection "P" space.

        Returns
        -------
        self.fanci_wfn.nproj
            Number of determinants in the "P" space.
        """
        return self.fanci_wfn.nproj

    @abstractmethod
    def der_g_lambda(self):
        r"""Derivative of the FANPT equations with respect to the lambda parameter.

        dG/dl

        Generates
        ---------
        d_g_lambda : np.ndarray
            Derivative of the FANPT equations with respect to the lambda parameter.
            numpy array with shape (self.nequations,).
        """
        self.d_g_lambda = None

    @abstractmethod
    def der2_g_lambda_wfnparams(self):
        r"""Derivative of the FANPT equations with respect to lambda and the wavefunction parameters.

        d^2G/dldp_k

        Generates
        ---------
        d2_g_lambda_wfnparams : np.ndarray
            Derivative of the FANPT equations with respect to lambda and the wavefunction
            parameters.
            numpy array with shape (self.nequations, len(self.wfn_params_active)).
        """
        self.d2_g_lambda_wfnparams = None

    @abstractmethod
    def gen_coeff_matrix(self):
        r"""Generate the coefficient matrix of the linear FANPT system of equations.

        dG/dp_k

        Generates
        ---------
        c_matrix : np.ndarray
            Coefficient matrix of the FANPT system of equations.
            numpy array with shape (self.nequations, len(self.nactive)).
        """
        self.c_matrix = None
