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

from libcpp.vector cimport vector


__all__ = [
    'int_t',
    'uint_t',
    'binomial',
    'binomial_nocheck',
    'fill_det',
    'fill_occs',
    'fill_virs',
    'next_colex',
    'unrank_indices',
    'nword_det',
    'excite_det',
    'setbit_det',
    'clearbit_det',
    'popcnt_det',
    'ctz_det',
    'phase_single_det',
    'phase_double_det',
    'rank_det',
    'OneSpinWfn',
    'TwoSpinWfn',
    'SparseOp',
    ]


ctypedef int64_t int_t


ctypedef uint64_t uint_t


cdef extern from 'pyci/pyci.h' namespace 'pyci':

    int_t binomial(int_t, int_t) except +

    int_t binomial_nocheck(int_t, int_t)

    void fill_det(const int_t, const int_t *, uint_t *)

    void fill_occs(const int_t, const uint_t *, int_t *)

    void fill_virs(const int_t, int_t, const uint_t *, int_t *)

    void next_colex(int_t *)

    void unrank_indices(int_t, const int_t, int_t, int_t *)

    int_t nword_det(const int_t)

    void excite_det(const int_t, const int_t, uint_t *)

    void setbit_det(const int_t, const uint_t *)

    void clearbit_det(const int_t, const uint_t *)

    int_t popcnt_det(const int_t, const uint_t *)

    int_t ctz_det(const int_t, const uint_t *)

    int_t phase_single_det(const int_t, const int_t, const int_t, const uint_t *)

    int_t phase_double_det(const int_t, const int_t, const int_t, const int_t, const int_t, const uint_t *)

    int_t rank_det(const int_t, const int_t, const uint_t *)

    cdef cppclass OneSpinWfn:

        int_t nword, nbasis, nocc, nvir, ndet
        vector[uint_t] dets

        OneSpinWfn()

        OneSpinWfn(const int_t, const int_t) except +

        OneSpinWfn(const OneSpinWfn &) except +

        OneSpinWfn(const TwoSpinWfn &) except +

        OneSpinWfn(const char *) except +

        OneSpinWfn(const int_t, const int_t, const int_t, const uint_t *) except +

        OneSpinWfn(const int_t, const int_t, const int_t, const int_t *) except +

        void init(const int_t, const int_t) except +

        void from_onespinwfn(const OneSpinWfn &) except +

        void from_twospinwfn(const TwoSpinWfn &) except +

        void from_file(const char *) except +

        void from_det_array(const int_t, const int_t, const int_t, const uint_t *) except +

        void from_occs_array(const int_t, const int_t, const int_t, const int_t *) except +

        void to_file(const char *) except +

        void to_occs_array(const int_t, const int_t, int_t *)

        int_t index_det(const uint_t *)

        int_t index_det_from_rank(const int_t)

        int_t copy_det(const int_t, uint_t *)

        const uint_t * det_ptr(const int_t)

        int_t add_det(const uint_t *) except +

        int_t add_det_with_rank(const uint_t *, const int_t) except +

        int_t add_det_from_occs(const int_t *) except +

        int_t add_all_dets() except +

        int_t add_excited_dets(const uint_t *, const int_t) except +

        void reserve(const int_t) except +

        void squeeze()

        double compute_overlap(const double *, const OneSpinWfn &, const double *) except +

        void compute_rdms_doci(const double *, double *, double *) except +

        void compute_rdms_genci(const double *, double *, double *) except +

        double compute_enpt2_doci(const double *, const double *, const double *, const double, const double) except +

        double compute_enpt2_genci(const double *, const double *, const double *, const double, const double) except +

        int_t run_hci_doci(const double *, const double *, const double) except +

        int_t run_hci_genci(const double *, const double *, const double *, const double) except +

    cdef cppclass TwoSpinWfn:

        int_t nword, nword2, nbasis, nocc_up, nocc_dn, nvir_up, nvir_dn
        int_t ndet, maxdet_up, maxdet_dn
        vector[uint_t] dets

        TwoSpinWfn()

        TwoSpinWfn(const int_t, const int_t, const int_t) except +

        TwoSpinWfn(const OneSpinWfn &) except +

        TwoSpinWfn(const TwoSpinWfn &) except +

        TwoSpinWfn(const char *) except +

        TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *) except +

        TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const int_t *) except +

        void init(const int_t, const int_t, const int_t) except +

        void from_onespinwfn(const OneSpinWfn &) except +

        void from_twospinwfn(const TwoSpinWfn &) except +

        void from_file(const char *) except +

        void from_det_array(const int_t, const int_t, const int_t, const int_t, const uint_t *) except +

        void from_occs_array(const int_t, const int_t, const int_t, const int_t, const int_t *) except +

        void to_file(const char *) except +

        void to_occs_array(const int_t, const int_t, int_t *) except +

        int_t index_det(const uint_t *)

        int_t index_det_from_rank(const int_t rank)

        void copy_det(const int_t, uint_t *)

        const uint_t * det_ptr(const int_t)

        int_t add_det(const uint_t *) except +

        int_t add_det_with_rank(const uint_t *, const int_t) except +

        int_t add_det_from_occs(const int_t *) except +

        void add_all_dets() except +

        void add_excited_dets(const uint_t *, const int_t, const int_t) except +

        void reserve(const int_t) except +

        void squeeze()

        double compute_overlap(const double *, const TwoSpinWfn &, const double *) except +

        double compute_enpt2_fullci(const double *, const double *, const double *, const double, const double) except +

        void compute_rdms_fullci(const double *, double *, double *) except +

        int_t run_hci_fullci(const double *, const double *, const double *, const double) except +

    cdef cppclass SparseOp:

        int_t nrow, ncol
        vector[double] data
        vector[int_t] indices
        vector[int_t] indptr

        SparseOp()

        void perform_op(const double *, double *)

        void solve(const double *, const int_t, const int_t, const int_t, const double, double *, double *) except +

        void init_doci(const OneSpinWfn &, const double *, const double *, const double *, const int_t) except +

        void init_genci(const OneSpinWfn &, const double *, const double *, const int_t) except +

        void init_fullci(const TwoSpinWfn &, const double *, const double *, const int_t) except +
