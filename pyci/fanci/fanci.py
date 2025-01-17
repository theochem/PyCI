r"""
FanCI base class module.

"""

from abc import ABCMeta, abstractmethod

from collections import OrderedDict

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from scipy.optimize import OptimizeResult, least_squares, root

from scipy.optimize import minimize

import pyci


__all__ = [
    "FanCI",
]


class FanCI(metaclass=ABCMeta):
    r"""
    FanCI problem class.

    """

    @property
    def nequation(self) -> int:
        r"""
        Number of nonlinear equations.

        """
        return self._nequation

    @property
    def nproj(self) -> int:
        r"""
        Number of determinants in projection ("P") space.

        """
        return self._nproj

    @property
    def nparam(self) -> int:
        r"""
        Number of parameters for this FanCI problem.

        """
        return self._nparam
    @property
    def constraints(self) -> Tuple[str]:
        r"""
        List of constraints.

        """
        return tuple(self._constraints.keys())

    @property
    def ham(self) -> pyci.hamiltonian:
        r"""
        PyCI Hamiltonian.

        """
        return self._ham

    @property
    def wfn(self) -> pyci.wavefunction:
        r"""
        PyCI wave function.

        """
        return self._wfn

    @property
    def ci_op(self) -> pyci.sparse_op:
        r"""
        PyCI sparse CI matrix operator.

        """
        return self._ci_op

    @property
    def pspace(self) -> np.ndarray:
        r"""
        Array of determinant occupations in projection ("P") space.

        """
        return self._pspace

    @property
    def sspace(self) -> np.ndarray:
        r"""
        Array of determinant occupations in auxiliiary ("S") space.

        """
        return self._sspace

    @property
    def nbasis(self) -> int:
        r"""
        Number of molecular orbital basis functions.

        """
        return self._wfn.nbasis

    @property
    def nocc_up(self) -> int:
        r"""
        Number of spin-up occupied orbitals.

        """
        return self._wfn.nocc_up

    @property
    def nocc_dn(self) -> int:
        r"""
        Number of spin-down occupied orbitals.

        """
        return self._wfn.nocc_dn

    @property
    def nvir_up(self) -> int:
        r"""
        Number of spin-up virtual orbitals.

        """
        return self._wfn.nvir_up

    @property
    def nvir_dn(self) -> int:
        r"""
        Number of spin-down virtual orbitals.

        """
        return self._wfn.nvir_dn

    def __init__(
        self,
        ham: pyci.hamiltonian,
        wfn: pyci.wavefunction,
        nproj: int,
        nparam: int,
        norm_param: Sequence[Tuple[int, float]] = None,
        norm_det: Sequence[Tuple[int, float]] = None,
        constraints: Dict[str, Tuple[Callable, Callable]] = None,
        fill: str = "excitation",
    ) -> None:
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        wfn : pyci.wavefunction
            PyCI wave function.
        nproj : int
            Number of determinants in projection ("P") space.
        nparam : int
            Number of parameters for this FanCI problem.
        norm_param : Sequence[Tuple[int, float]], optional
            Indices of parameters whose values to constrain, and the value to which to constrain
            them.
        norm_det : Sequence[Tuple[int, float]], optional
            Indices of determinant whose overlaps to constrain, and the value to which to constrain
            them.
        constraints : Dict[str, Tuple[Callable, Callable]], optional
            Pairs of functions (f, dfdx) corresponding to additional constraints.
        fill : ('excitation' | 'seniority' | None)
            Whether to fill the projection ("P") space by excitation level, by seniority, or not
            at all (in which case ``wfn`` must already be filled).

        """
        # Generate constraints dict
        if constraints is None:
            constraints = OrderedDict()
        elif isinstance(constraints, dict):
            constraints = OrderedDict(constraints)
        else:
            raise TypeError(f"Invalid `constraints` type `{type(constraints)}`; must be dictionary")

        # Add norm_det and norm_param constraints
        norm_param = list() if norm_param is None else norm_param
        norm_det = list() if norm_det is None else norm_det
        for index, value in norm_param:
            name = f"p_{{{index}}} - v_{{{index}}}"
            constraints[name] = self.make_param_constraint(index, value)
        for index, value in norm_det:
            name = f"<\\psi_{{{index}}}|\\Psi> - v_{{{index}}}"
            constraints[name] = self.make_det_constraint(index, value)

        # Number of nonlinear equations and active parameters
        nequation = nproj + len(constraints)

        # Generate determinant spaces
        wfn = fill_wavefunction(wfn, nproj, fill)

        # Compute CI matrix operator with nproj rows and len(wfn) columns
        if type(wfn).__name__ == "nonsingletci_wfn":
            ci_op = pyci.sparse_op(ham, wfn, nrow=nproj, ncol=len(wfn), symmetric=False, wfntype="nonsingletci")
        else:
            ci_op = pyci.sparse_op(ham, wfn, nrow=nproj, ncol=len(wfn), symmetric=False)

        # Compute arrays of occupations
        sspace = wfn.to_occ_array()

        # if type(wfn).__name__ == "nonsingletci_wfn":
        #     pspace = sspace[np.r_[0:13, 21:29, 37:45, 53:60, 66:72, 75:77, 81, 99:102, 104:108, 109:117, 126:129, 131:135, 
        #               136:144, 171:183, 189:195, 201:206, 209, 221:224, 225:231, 2316:239, 240:246, 261:267, 
        #               273:278, 282:286, 288, 300:302, 303:309, 314:316, 317:323, 337:344, 348:352, 356:359, 
        #               365:369, 371:375, 381:384, 388:391, 393:395, 400:403, 405:408, 413:416, 418:420, 422, 
        #               424:426, 427, 430]]
        #     # nproj = len(pspace)
        # else:
        pspace = sspace[:nproj]
        print("len(sspace), len(pspace): ", len(sspace), len(pspace))
        # dec_sspace = wfn.to_det_array()
        # print("Printing sspace")
        # for dec, bin in zip(dec_sspace, sspace):
        #     print("\n", dec, bin)

        # Assign attributes to instance
        self._nequation = nequation
        self._nproj = nproj
        self._nparam = nparam
        self._constraints = constraints
        self._ham = ham
        self._wfn = wfn
        self._ci_op = ci_op
        self._pspace = pspace
        self._sspace = sspace

        # Set read-only flag on public array attributes
        self._pspace.setflags(write=False)
        self._sspace.setflags(write=False)

    def optimize(
        self,
        x0: np.ndarray,
        mode: str = "lstsq",
        use_jac: bool = False,
        sigma: float = 0.1,
        **kwargs: Any,
    ) -> OptimizeResult:
        r"""
        Optimize the wave function parameters.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for wave function parameters.
        mode : ('lstsq' | 'root'), default='lstsq'
            Solver mode.
        use_jac : bool, default=False
            Whether to use the Jacobian function or a finite-difference approximation.
        kwargs : Any, optional
            Additional keyword arguments to pass to optimizer.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result of optimization.

        """
        # Convert x0 to proper dtype array
        x0 = np.asarray(x0, dtype=pyci.c_double)
        # Check input x0 length
        if x0.size != self.nparam:
            raise ValueError("length of `x0` does not match `param`")

        # Use bare functions
        f = self.compute_objective
        j = self.compute_jacobian

        # Set up initial arguments to optimizer
        opt_kwargs = kwargs.copy()
        if use_jac:
            opt_kwargs["jac"] = j

        # Parse mode parameter; choose optimizer and fix arguments
        if mode == "lstsq":
            opt_args = f, x0
            optimizer = least_squares
        elif mode == "root":
            opt_args = f, x0
            optimizer = root
        # FIXME: For BFGS to work, objective function must return a scalar value
        elif mode == "bfgs":
            opt_args = f, x0
            optimizer = minimize
        else:
            raise ValueError("invalid mode parameter")

        # Run optimizer
        return optimizer(*opt_args, **opt_kwargs)

    def optimize_stochastic(
        self,
        nsamp: int,
        x0: np.ndarray,
        mode: str = "lstsq",
        use_jac: bool = False,
        fill: str = "excitation",
        **kwargs: Any,
    ) -> List[Tuple[np.ndarray]]:
        r"""
        Run a stochastic optimization of a FanCI wave function.

        Parameters
        ----------
        nsamp: int
            Number of samples to compute.
        x0 : np.ndarray
            Initial guess for wave function parameters.
        mode : ('lstsq' | 'root'), default='lstsq'
            Solver mode.
        use_jac : bool, default=False
            Whether to use the Jacobian function or a finite-difference approximation.
        fill : ('excitation' | 'seniority' | None)
            Whether to fill the projection ("P") space by excitation level, by seniority,
            or not at all (in which case ``wfn`` must already be filled).
        kwargs : Any, optional
            Additional keyword arguments to pass to optimizer.

        Returns
        -------
        result : List[Tuple[np.ndarray]]
            List of (occs, coeffs, params) vectors for each solution.

        """
        # Get wave function information
        ham = self._ham
        nproj = self._nproj
        nparam = self._nparam
        nbasis = self._wfn.nbasis
        nocc_up = self._wfn.nocc_up
        nocc_dn = self._wfn.nocc_dn
        constraints = self._constraints
        ci_cls = self._wfn.__class__

        # Start at sample 1
        isamp = 1
        result = []
        # Iterate until nsamp samples are reached
        while True:
            # Optimize this FanCI wave function and get the result
            opt = self.optimize(x0, mode=mode, use_jac=use_jac, **kwargs)
            x0 = opt.x
            coeffs = self.compute_overlap(x0[:-1], "S")

            # Add the result to our list
            result.append((np.copy(self.sspace), coeffs, x0))

            # Check if we're done manually each time; this avoids an extra
            # CI matrix preparation with an equivalent "for" loop
            if isamp >= nsamp:
                return result

            # Try to get the garbage collector to remove the old CI matrix
            del self._ci_op
            self._ci_op = None

            # Generate new determinant indices from "S" space
            new_indices = np.random.choice(
                self.sspace.shape[0],
                size=nproj,
                replace=False,
                # Probability vector p_i = abs(c_i)
                p=np.abs(coeffs),
            )

            # Make new FanCI wave function in-place
            FanCI.__init__(
                self,
                ham,
                # Generate new determinants from "S" space
                ci_cls(nbasis, nocc_up, nocc_dn, self.sspace[new_indices]),
                nproj,
                nparam,
                constraints=constraints,
                fill=fill,
            )

            # Go to next iteration
            isamp += 1

    def add_constraint(self, name: str, f: Callable, dfdx: Callable = None) -> None:
        r"""
        Add a constraint to the system.

        ``dfdx`` must be specified to compute the Jacobian of the system.

        Parameters
        ----------
        name : str
            Label for constraint.
        f : Callable
            Constraint function.
        dfdx : Callable, optional
            Gradient of constraint function.

        """
        self._constraints[name] = f, dfdx
        # Update nequation
        self._nequation = self._nproj + len(self._constraints)

    def remove_constraint(self, name: str) -> None:
        r"""
        Remove a constraint from the system.

        Parameters
        ----------
        name : str
            Label for constraint.

        """
        del self._constraints[name]
        # Update nequation
        self._nequation = self._nproj + len(self._constraints)

    def compute_objective(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI objective function.

            f : x[k] -> y[n]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        obj : np.ndarray
            Objective vector.

        """
        # Allocate objective vector
        f = np.empty(self._nequation, dtype=pyci.c_double)
        f_proj = f[: self._nproj]
        f_cons = f[self._nproj :]

        # Assign Energy = x[-1]
        energy = x[-1]

        # Compute overlaps of determinants in sspace:
        #
        #   c_m
        #
        ovlp = self.compute_overlap(x[:-1], "S")

        # Compute objective function:
        #
        #   f_n = <n|H|\Psi> - E <n|\Psi>
        #
        #       = <m|H|n> c_m - E \delta_{mn} c_m
        #
        # Note: we update ovlp in-place here
        self._ci_op(ovlp, out=f_proj)
        ovlp_proj = ovlp[: self._nproj]
        ovlp_proj *= energy
        f_proj -= ovlp_proj

        # Compute constraint functions
        for i, constraint in enumerate(self._constraints.values()):
            f_cons[i] = constraint[0](x)

        # Return objective
        return f

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the Jacobian of the FanCI objective function.

            j : x[k] -> y[n, k]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        jac : np.ndarray
            Jacobian matrix.

        """
        # Allocate Jacobian matrix (in transpose memory order)
        jac = np.empty((self._nequation, self._nparam), order="F", dtype=pyci.c_double)
        jac_proj = jac[: self._nproj]
        jac_cons = jac[self._nproj :]

        # Assign Energy = x[-1]
        energy = x[-1]

        # Compute Jacobian:
        #
        #   J_{nk} = d(<n|H|\Psi>)/d(p_k) - E d(<n|\Psi>)/d(p_k) - dE/d(p_k) <n|\Psi>
        #
        # Compute overlap derivatives in sspace:
        #
        #   d(c_m)/d(p_k)
        #
        d_ovlp = self.compute_overlap_deriv(x[:-1], "S")

        # Compute final Jacobian column
        #
        #   dE/d(p_k) <n|\Psi> = dE/d(p_k) \delta_{nk} c_n
        #
        ovlp = self.compute_overlap(x[:-1], "P")
        ovlp *= -1
        jac_proj[:, -1] = ovlp
        #
        # Remove final column from jac_proj
        #
        jac_proj = jac_proj[:, :-1]

        # Iterate over remaining columns of Jacobian and d_ovlp
        for jac_col, d_ovlp_col in zip(jac_proj.transpose(), d_ovlp.transpose()):
            #
            # Compute each column of the Jacobian:
            #
            #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
            #
            #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
            #
            # Note: we update d_ovlp in-place here
            self._ci_op(d_ovlp_col, out=jac_col)
            d_ovlp_proj = d_ovlp_col[: self._nproj]
            d_ovlp_proj *= energy
            jac_col -= d_ovlp_proj

        # Compute Jacobian of constraint functions
        for i, constraint in enumerate(self._constraints.values()):
            jac_cons[i] = constraint[1](x)

        # Return Jacobian
        return jac

    def make_param_constraint(self, i: int, val: float) -> Tuple[Callable, Callable]:
        r"""
        Generate parameter constraint functions.

        Parameters
        ----------
        i : int
            Index of parameter whose value to constrain.
        val : float
            Value to which to constrain parameter.

        Returns
        -------
        f: Callable
            Constraint function.
        dfdx : Callable
            Gradient of constraint function.

        """

        def f(x: np.ndarray) -> float:
            r""" "
            Constraint function p_{i} - v_{i}.

            """
            return x[i] - val

        def dfdx(x: np.ndarray) -> np.ndarray:
            r"""
            Constraint gradient \delta_{ki}.

            """
            y = np.zeros(self.nparam, dtype=pyci.c_double)
            y[i] = 1
            return y

        return f, dfdx

    def make_det_constraint(self, i: int, val: float) -> Tuple[Callable, Callable]:
        r"""
        Generate determinant overlap constraint functions.

        Parameters
        ----------
        i : int
            Index of determinant whose overlap to constrain.
        val : float
            Value to which to constrain overlap.

        Returns
        -------
        f: Callable
            Constraint function.
        dfdx : Callable
            Gradient of constraint function.

        """

        def f(x: np.ndarray) -> float:
            r""" "
            Constraint function <\psi_{i}|\Psi> - v_{i}.

            """
            return self.compute_overlap(x[:-1], self._sspace[np.newaxis, i])[0] - val

        def dfdx(x: np.ndarray) -> np.ndarray:
            r""" "
            Constraint gradient d(<\psi_{i}|\Psi>)/d(p_{k}).

            """
            y = np.zeros(self._nparam, dtype=pyci.c_double)
            d_ovlp = self.compute_overlap_deriv(x[:-1], self._sspace[np.newaxis, i])[0]
            y[: -1] = d_ovlp
            return y

        return f, dfdx

    @abstractmethod
    def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap vector.

        """
        raise NotImplementedError("this method must be overwritten in a sub-class")

    @abstractmethod
    def compute_overlap_deriv(
        self, x: np.ndarray, occs_array: Union[np.ndarray, str]
    ) -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative matrix.

        """
        raise NotImplementedError("this method must be overwritten in a sub-class")


def fill_wavefunction(wfn: pyci.wavefunction, nproj: int, fill: str) -> None:
    r"""
    Fill the PyCI wave function object for the FanCI problem.

    Helper function for ``FanCI.__init__``.

    Parameters
    ----------
    wfn : pyci.wavefunction
        PyCI wave function.
    nproj : int
        Number of determinants in projection ("P") space.
    fill : ('excitation' | 'seniority' | None)
        Whether to fill the projection ("P") space by excitation level, by seniority, or not
        at all (in which case ``wfn`` must already be filled).

    """
    # Handle wfn parameter; decide values for generating determinants from its type
    if isinstance(wfn, pyci.doci_wfn):
        e_max = min(wfn.nocc_up, wfn.nvir_up)
        s_min = 0
        s_max = 0
        connections = (1,)
    elif isinstance(wfn, (pyci.fullci_wfn, pyci.genci_wfn)):
        e_max = min(wfn.nocc, wfn.nvir)
        s_min = wfn.nocc_up - wfn.nocc_dn
        s_max = min(wfn.nocc_up, wfn.nvir_up)
        connections = (1, 2)
    else:
        raise TypeError(f"invalid `wfn` type `{type(wfn)}`; must be `pyci.wavefunction`")

    # Use new wavefunction; don't modify original object
    wfn = wfn.__class__(wfn)

    if fill == "excitation":
        # Fill wfn with P space determinants in excitation order until len(wfn) >= nproj;
        # only add Hartree-Fock det. (zero order excitation) if wfn is empty
        # for nexc in range(bool(len(wfn)), e_max + 1):
        for nexc in range(e_max + 1):
            if len(wfn) >= nproj:
                break
            pyci.add_excitations(wfn, nexc)

    elif fill == "seniority":
        # Fill with determinants in increasing-seniority order
        if isinstance(wfn, pyci.doci_wfn) and len(wfn) < nproj:
            wfn.add_all_dets()
        else:
            # Valid seniorities increase by two from s_min
            for nsen in range(s_min, s_max + 1, 2):
                if len(wfn) >= nproj:
                    break
                pyci.add_seniorities(wfn, nsen)
    elif fill is not None:
        raise ValueError(f"Invalid `fill` value: '{fill}'")

    if len(wfn) < nproj:
        raise ValueError(f"unable to generate `nproj={nproj}` determinants")

    # Truncate wave function if we generated > nproj determinants
    if len(wfn) > nproj:
        wfn = wfn.__class__(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn, wfn.to_det_array(nproj))

    # Fill wfn with S space determinants
    for det in wfn.to_det_array(nproj):
        pyci.add_excitations(wfn, *connections, ref=det)

    return wfn
