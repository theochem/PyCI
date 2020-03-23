# cython : language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

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

r"""
PyCI Cython header.

"""

from libc.stdint cimport int64_t, uint64_t

from pyci.doci cimport DOCIWfn
from pyci.fullci cimport FullCIWfn


__all__ = [
    'SparseOp',
    #'doci_solve_sparse_',
    #'doci_solve_direct_',
    ]


ctypedef int64_t int_t


ctypedef uint64_t uint_t


cdef extern from 'pyci/solve.h' namespace 'pyci':

    cdef cppclass SparseOp:
        int_t nrow, ncol
        SparseOp()
        SparseOp(const DOCIWfn &, const double *, const double *, const double *, const int_t) except +
        void init(const DOCIWfn &, const double *, const double *, const double *, const int_t) except +
        double * data_ptr() except +
        int_t * indices_ptr() except +
        int_t * indptr_ptr() except +
        void perform_op(const double *, double *)
        void solve(const double *, const int_t, const int_t, const int_t, const double, double *, double *) except +
    
#    void doci_solve_sparse_ 'solve_sparse' (const DOCIWfn &, const double *, const double *,
#        const double *, const double *, const int_t, const int_t, const int_t,
#        const double, double *, double *) except +
#
#    void doci_solve_direct_ 'solve_direct' (const DOCIWfn &, const double *, const double *,
#        const double *, const double *, const int_t, const int_t, const int_t,
#        const double, double *, double *) except +
