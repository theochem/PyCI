/* This file is part of PyCI.
 *
 * PyCI is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * PyCI is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PyCI. If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <climits>
#include <cstdint>

#include <parallel_hashmap/phmap_fwd_decl.h>

#define PYCI_INT_SIZE (std::int64_t)(sizeof(std::int64_t) * CHAR_BIT)
#define PYCI_UINT_SIZE (std::int64_t)(sizeof(std::uint64_t) * CHAR_BIT)
#define PYCI_INT_MAX (std::int64_t)INT64_MAX
#define PYCI_UINT_MAX (std::uint64_t)UINT64_MAX
#define PYCI_UINT_ZERO (std::uint64_t)0U
#define PYCI_UINT_ONE (std::uint64_t)1U
#if UINT64_MAX <= ULONG_MAX
#define PYCI_POPCNT(X) __builtin_popcountl(X)
#define PYCI_CTZ(X) __builtin_ctzl(X)
#elif UINT64_MAX <= ULLONG_MAX
#define PYCI_POPCNT(X) __builtin_popcountll(X)
#define PYCI_CTZ(X) __builtin_ctzll(X)
#else
#error Not compiling for a compatible 64-bit system.
#endif


namespace pyci {


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


int_t binomial(int_t, int_t);


int_t binomial_nocheck(int_t, int_t);


void fill_det(const int_t, const int_t *, uint_t *);


void fill_occs(const int_t, const uint_t *, int_t *);


void fill_virs(const int_t, int_t, const uint_t *, int_t *);


void next_colex(int_t *);


void unrank_indices(int_t, const int_t, int_t, int_t *);


int_t nword_det(const int_t);


void excite_det(const int_t, const int_t, uint_t *);


void setbit_det(const int_t, uint_t *);


void clearbit_det(const int_t, uint_t *);


int_t phase_single_det(const int_t, const int_t, const int_t, const uint_t *);


int_t phase_double_det(const int_t, const int_t, const int_t, const int_t, const int_t, const uint_t *);


int_t popcnt_det(const int_t, const uint_t *);


int_t ctz_det(const int_t, const uint_t *);


int_t rank_det(const int_t, const int_t, const uint_t *);


} // namespace pyci
