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
    ]


ctypedef int64_t int_t


ctypedef uint64_t uint_t


cdef extern from 'pyci/common.h' namespace 'pyci':

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
