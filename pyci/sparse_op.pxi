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


cdef class sparse_op:
    r"""
    Sparse matrix operator class.

    Attributes
    ----------
    shape : (int, int)
        Shape of matrix operator.
    ecore : float
        Constant/"zero-electron" integral.

    """
    cdef SparseOp _obj
    cdef tuple _shape
    cdef double _ecore

    @property
    def shape(self):
        r"""
        The shape of the sparse matrix operator.

        """
        return self._shape

    @property
    def ecore(self):
        r"""
        Constant/"zero-electron" integral.

        """
        return self._ecore

    def __init__(self, hamiltonian ham not None, wavefunction wfn not None, int_t nrow=-1):
        r"""
        Initialize a sparse matrix operator instance.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        wfn : (doci_wfn | fullci_wfn | genci_wfn)
            Wave function object.
        nrow : int, optional
            Number of rows (<= number of determinants in wavefunction). Default is square matrix.

        """
        # Check parameters
        if len(wfn) == 0:
            raise ValueError('wfn must contain at least one determinant')
        # DOCI wave function
        elif isinstance(wfn, doci_wfn):
            if ham._h is None:
                raise AttributeError('seniority-zero integrals were not computed')
            # Restricted DOCI operator
            elif isinstance(ham, restricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                self._obj.init_doci(
                        (<doci_wfn>wfn)._obj,
                        <double *>(&ham._h[0]),
                        <double *>(&ham._v[0, 0]),
                        <double *>(&ham._w[0, 0]),
                        nrow,
                        )
            # Unrestricted DOCI operator
            elif isinstance(ham, unrestricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            # Generalized DOCI operator
            elif isinstance(ham, generalized_ham):
                if wfn.nbasis * 2 != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            else:
                raise ValueError('invalid ham type')
        # FullCI wave function
        elif isinstance(wfn, fullci_wfn):
            if ham._one_mo is None:
                raise AttributeError('full integral arrays were not saved')
            # Restricted FullCI operator
            elif isinstance(ham, restricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                self._obj.init_fullci(
                        (<fullci_wfn>wfn)._obj,
                        <double *>(&ham._one_mo[0, 0]),
                        <double *>(&ham._two_mo[0, 0, 0, 0]),
                        nrow,
                        )
            # Unrestricted FullCI operator
            elif isinstance(ham, unrestricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            # Generalized FullCI operator
            elif isinstance(ham, generalized_ham):
                if wfn.nbasis * 2 != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            else:
                raise ValueError('invalid ham type')
        # Generalized CI wave function
        elif isinstance(wfn, genci_wfn):
            if not isinstance(ham, generalized_ham):
                raise ValueError('invalid ham type')
            elif ham._one_mo is None:
                raise AttributeError('full integral arrays were not saved')
            elif wfn.nbasis != ham.nbasis:
                raise ValueError('dimension of ham, wfn do not match')
            self._obj.init_genci((<genci_wfn>wfn)._obj, <double *>(&ham._one_mo[0, 0]),
                    <double *>(&ham._two_mo[0, 0, 0, 0]), nrow)
        else:
            raise TypeError('wfn type must be one of \'doci_wfn\', \'fullci_wfn\', \'genci_wfn\'')
        self._shape = self._obj.nrow, self._obj.ncol
        self._ecore = ham.ecore

    def __call__(self, double[::1] x not None, double[::1] out=None):
        r"""
        Compute the result of the sparse operator :math:`A` applied to vector :math:`x`.

        Parameters
        ----------
        x : np.ndarray(c_double(n))
            Operand vector.
        out : np.ndarray(c_double(n)), optional
            Output parameter, as in NumPy (e.g., numpy.dot).

        Returns
        -------
        y : np.ndarray(c_double(n))
           Result vector.

        """
        cdef np.ndarray y
        if x.size != self._obj.ncol:
            raise ValueError('Dimensions of operator and \'x\' do not match')
        # set y and out variables
        if out is None:
            y = np.empty(self._obj.nrow, dtype=c_double)
            out = y
        else:
            if out.size != self._obj.nrow:
                raise ValueError('Dimensions of operator and \'out\' do not match')
            y = np.asarray(out)
        # return result of operation
        self._obj.perform_op(&x[0], &out[0])
        return y

    def to_csr_matrix(self):
        r"""
        Convert the sparse matrix operator to CSR matrix data in NumPy arrays.

        Returns
        -------
        csr_mat : tuple(np.ndarray, np.ndarray, np.ndarray)
            CSR matrix data. Can be passed to `scipy.sparse.csr_matrix`.

        """
        cdef double *data_ptr = &self._obj.data[0]
        cdef int_t *indices_ptr = &self._obj.indices[0]
        cdef int_t *indptr_ptr = &self._obj.indptr[0]
        cdef np.ndarray data = np.copy(<double[:self._obj.data.size()]>data_ptr)
        cdef np.ndarray indices = np.copy(<int_t[:self._obj.indices.size()]>indices_ptr)
        cdef np.ndarray indptr = np.copy(<int_t[:self._obj.indptr.size()]>indptr_ptr)
        return data, indices, indptr

    def solve(self, int_t n=1, int_t ncv=-1, double[::1] c0=None, int_t maxiter=-1, double tol=1.0e-6):
        r"""
        Solve the CI problem for the energy/energies and coefficient vector(s).

        Parameters
        ----------
        n : int, default=1
            Number of lowest-energy solutions for which to solve.
        ncv : int, default=max(n + 1, min(20, nrow))
            Number of Lanczos vectors to use for eigensolver.
            More is generally faster and more reliably convergent.
        c0 : np.ndarray(c_double(nrow)), optional
            Initial guess for lowest-energy coefficient vector.
            If not provided, the default is [1, 0, 0, ..., 0, 0].
        maxiter : int, default=1000*n
            Maximum number of iterations for eigensolver to run.
        tol : float, default=1.0e-6
            Convergence tolerance for eigensolver.

        Returns
        -------
        evals : np.ndarray(c_double(n))
            Energies.
        evecs : np.ndarray(c_double(n, nrow))
            Coefficient vectors.

        """
        if self._shape[0] != self._shape[1]:
            raise ValueError('cannot solve for a rectangular operator')
        cdef int_t ndet = self._shape[0]
        # set default number of lanczos vectors n < ncv <= len(c0)
        if ncv == -1:
            ncv = max(n + 1, min(20, ndet))
        # default initial guess c = [1, 0, ..., 0]
        if c0 is None:
            c0 = np.zeros(ndet, dtype=c_double)
            c0[0] = 1.0
        elif ndet != c0.shape[0]:
            raise ValueError('dimension of sparse_op, c0 do not match')
        # default maxiter = 1000 * n
        if maxiter == -1:
            maxiter = 1000 * n
        # solve eigenproblem
        cdef np.ndarray evals_array = np.empty(n, dtype=c_double)
        cdef np.ndarray evecs_array = np.empty((n, ndet), dtype=c_double)
        cdef double[::1] evals = evals_array
        cdef double[:, ::1] evecs = evecs_array
        self._obj.solve(<double *>(&c0[0]), n, ncv, maxiter, tol,
                <double *>(&evals[0]), <double *>(&evecs[0, 0]))
        evals_array += self._ecore
        return evals_array, evecs_array


# vim: set ft=pyrex:
