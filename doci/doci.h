/* This file is part of DOCI.
 *
 * DOCI is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * DOCI is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DOCI. If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <cstdint>
#include <vector>

#include <parallel_hashmap/phmap.h>


namespace doci {


typedef std::int64_t int_t;

typedef std::uint64_t uint_t;

template<class KeyType, class ValueType, std::size_t N=4, class Mutex=phmap::NullMutex>
using hashmap = phmap::parallel_flat_hash_map<
    KeyType,
    ValueType,
    phmap::container_internal::hash_default_hash<KeyType>,
    phmap::container_internal::hash_default_eq<KeyType>,
    phmap::container_internal::Allocator<phmap::container_internal::Pair<const KeyType, ValueType>>,
    N,
    Mutex>;


struct DOCIWfn {
public:
    int_t nword, nbasis, nocc, nvir, ndet;
    std::vector<uint_t> dets;
    hashmap<int_t, int_t> dict;
public:
    DOCIWfn();
    DOCIWfn(const int_t, const int_t);
    DOCIWfn(const char *);
    ~DOCIWfn();
    void init(const int_t, const int_t);
    void from_file(const char *);
    void to_file(const char *) const;
    int_t index_det(const uint_t *) const;
    void copy_det(const int_t, uint_t *) const;
    int_t add_det(const uint_t *);
    int_t add_det_from_occs(const int_t *);
    void add_all_dets();
    void add_excited_dets(const uint_t *, const int_t);
    void reserve(const int_t);
    void squeeze();
};


void doci_rdms(const DOCIWfn &, const double *, double *, double *);


double doci_energy(const DOCIWfn &, const double *, const double *, const double *, const double *);


int_t doci_hci(DOCIWfn &, const double *, const double *, const double);


void solve_sparse(const DOCIWfn &, const double *, const double *, const double *, const double *,
    const int_t, const int_t, const int_t, const double, double *, double *);


void solve_direct(const DOCIWfn &, const double *, const double *, const double *, const double *,
    const int_t, const int_t, const int_t, const double, double *, double *);


int_t binomial(int_t, int_t);


int_t nword_det(const int_t);


void fill_det(const int_t, const int_t *, uint_t *);


void fill_occs(const int_t, const uint_t *, int_t *);


void fill_virs(const int_t, const int_t, const uint_t *, int_t *);


void excite_det(const int_t, const int_t, uint_t *);


void setbit_det(const int_t, uint_t *);


void clearbit_det(const int_t, uint_t *);


int_t popcnt_det(const int_t, const uint_t *);


int_t ctz_det(const int_t, const uint_t *);


int_t hash_det(const int_t, const int_t, const uint_t *);


} // namespace doci
