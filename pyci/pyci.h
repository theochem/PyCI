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

#include <vector>

#include <parallel_hashmap/phmap.h>

#define PYCI_NWORD_MAX 16

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


struct OneSpinWfn
{
    typedef hashmap<int_t, int_t> hashmap_type;

    int_t nword, nbasis, nocc, nvir, ndet;
    std::vector<uint_t> dets;
    hashmap_type dict;

    OneSpinWfn();

    OneSpinWfn(const int_t, const int_t);

    OneSpinWfn(const OneSpinWfn &);

    OneSpinWfn(const char *);

    OneSpinWfn(const int_t, const int_t, const int_t, const uint_t *);

    OneSpinWfn(const int_t, const int_t, const int_t, const int_t *);

    ~OneSpinWfn();

    void init(const int_t, const int_t);

    void from_onespinwfn(const OneSpinWfn &);

    void from_file(const char *);

    void from_det_array(const int_t, const int_t, const int_t, const uint_t *);

    void from_occs_array(const int_t, const int_t, const int_t, const int_t *);

    void to_file(const char *) const;

    void to_occs_array(const int_t, const int_t, int_t *) const;

    int_t index_det(const uint_t *) const;

    int_t index_det_from_rank(const int_t) const;

    void copy_det(const int_t, uint_t *) const;

    const uint_t * det_ptr(const int_t) const;

    int_t add_det(const uint_t *);

    int_t add_det_with_rank(const uint_t *, const int_t);

    int_t add_det_from_occs(const int_t *);

    void add_all_dets();

    void add_excited_dets(const uint_t *, const int_t);

    void reserve(const int_t);

    void squeeze();

    void compute_rdms_doci(const double *, double *, double *) const;

    void compute_rdms_genci(const double *, double *, double *) const;

    double compute_overlap(const double *, const OneSpinWfn &, const double *) const;

    int_t run_hci_doci(const double *, const double *, const double);

    int_t run_hci_genci(const double *, const double *, const double *, const double);
};


struct TwoSpinWfn
{
    typedef hashmap<int_t, int_t> hashmap_type;

    int_t nword, nword2, nbasis, nocc_up, nocc_dn, nvir_up, nvir_dn;
    int_t ndet, maxdet_up, maxdet_dn;
    std::vector<uint_t> dets;
    hashmap_type dict;

    TwoSpinWfn();

    TwoSpinWfn(const int_t, const int_t, const int_t);

    TwoSpinWfn(const TwoSpinWfn &);

    TwoSpinWfn(const char *);

    TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);

    ~TwoSpinWfn();

    void init(const int_t, const int_t, const int_t);

    void from_twospinwfn(const TwoSpinWfn &);

    void from_file(const char *);

    void from_det_array(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    void from_occs_array(const int_t, const int_t, const int_t, const int_t, const int_t *);

    void to_file(const char *) const;

    void to_occs_array(const int_t, const int_t, int_t *) const;

    int_t index_det(const uint_t *) const;

    int_t index_det_from_rank(const int_t) const;

    void copy_det(const int_t, uint_t *) const;

    const uint_t * det_ptr(const int_t) const;

    int_t add_det(const uint_t *);

    int_t add_det_with_rank(const uint_t *, const int_t);

    int_t add_det_from_occs(const int_t *);

    void add_all_dets();

    void add_excited_dets(const uint_t *, const int_t, const int_t);

    void reserve(const int_t);

    void squeeze();

    double compute_overlap(const double *, const TwoSpinWfn &, const double *) const;

    void compute_rdms_fullci(const double *, double *, double *) const;

    double compute_enpt2_fullci(const double *, const double *, const double *, const double, const double) const;

    int_t run_hci_fullci(const double *, const double *, const double *, const double);
};


struct SparseOp
{
    int_t nrow, ncol;
    std::vector<double> data;
    std::vector<int_t> indices;
    std::vector<int_t> indptr;

    SparseOp();

    inline int_t rows() const { return nrow; }

    inline int_t cols() const { return ncol; }

    void perform_op(const double *, double *) const;

    void solve(const double *, const int_t, const int_t, const int_t, const double, double *, double *) const;

    void init_doci(const OneSpinWfn &, const double *, const double *, const double *, const int_t);

    void init_fullci(const TwoSpinWfn &, const double *, const double *, const int_t);

    void init_genci(const OneSpinWfn &, const double *, const double *, const int_t);
};


} // namespace pyci
