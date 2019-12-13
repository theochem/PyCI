# cython : language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

# This file is part of DOCI.
#
# DOCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# DOCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with DOCI. If not, see <http://www.gnu.org/licenses/>.

r"""
DOCI C extension Cython header.

"""

from libc.stdint cimport int64_t, uint64_t


__all__ = [
    'int_t',
    'uint_t',
    'dociham',
    'dociwfn',
    'DOCIWfn',
    'doci_rdms',
    'doci_energy',
    'doci_hci',
    'solve_sparse',
    'solve_direct',
    'binomial',
    'fill_det',
    'fill_occs',
    'fill_virs',
    'excite_det',
    'setbit_det',
    'clearbit_det',
    'popcnt_det',
    'ctz_det',
    'hash_det',
    ]


ctypedef int64_t int_t

ctypedef uint64_t uint_t


cdef extern from "doci.h" namespace "doci":

    cdef cppclass DOCIWfn:
        int_t nword, nbasis, nocc, nvir, ndet
        DOCIWfn()
        DOCIWfn(const int_t, const int_t) except +
        DOCIWfn(const char *) except +
        void init(const int_t, const int_t) except +
        void from_file(const char *) except +
        void to_file(const char *) except +
        int_t index_det(const uint_t *)
        int_t copy_det(const int_t, uint_t *)
        int_t add_det(const uint_t *) except +
        int_t add_det_from_occs(const int_t *) except +
        int_t add_all_dets() nogil except +
        int_t add_excited_dets(const uint_t *, const int_t) except +
        void reserve(const int_t) except +
        void squeeze()

    void doci_rdms(const DOCIWfn &, const double *, double *, double *) except +

    double doci_energy(const DOCIWfn &, const double *, const double *, const double *, const double *) nogil except +

    int_t doci_hci(DOCIWfn &, const double *, const double *, const double) nogil except +

    void solve_sparse(const DOCIWfn &, const double *, const double *, const double *, const double *,
        const int_t, const int_t, const int_t, const double, double *, double *) nogil except +

    void solve_direct(const DOCIWfn &, const double *, const double *, const double *, const double *,
        const int_t, const int_t, const int_t, const double, double *, double *) nogil except +

    int_t binomial(int_t, int_t) except +

    void fill_det(const int_t, const int_t *, uint_t *)

    void fill_occs(const int_t, const uint_t *, int_t *)

    void fill_virs(const int_t, const int_t, const uint_t *, int_t *)

    void excite_det(const int_t, const int_t, uint_t *)

    void setbit_det(const int_t, const uint_t *)

    void clearbit_det(const int_t, const uint_t *)

    int_t popcnt_det(const int_t, const uint_t *)

    int_t ctz_det(const int_t, const uint_t *)

    int_t hash_det(const int_t, const int_t, const uint_t *)


cdef class dociham:
    cdef int_t _nbasis
    cdef double _ecore
    cdef double[::1] _h
    cdef double[:, ::1] _v
    cdef double[:, ::1] _w
    cdef double[:, ::1] _one_mo
    cdef double[:, :, :, ::1] _two_mo


cdef class dociwfn:
    cdef DOCIWfn _obj
