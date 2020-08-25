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

#include <parallel_hashmap/phmap.h>

#include <climits>
#include <cstdint>
#include <string>
#include <vector>

/* Define integer types, popcnt and ctz functions. */
#define PYCI_INT_SIZE static_cast<std::int64_t>(sizeof(std::int64_t) * CHAR_BIT)
#define PYCI_UINT_SIZE static_cast<std::int64_t>(sizeof(std::uint64_t) * CHAR_BIT)
#define PYCI_INT_MAX static_cast<std::int64_t>(INT64_MAX)
#define PYCI_UINT_MAX static_cast<std::uint64_t>(UINT64_MAX)
#define PYCI_UINT_ZERO static_cast<std::uint64_t>(0U)
#define PYCI_UINT_ONE static_cast<std::uint64_t>(1U)
#if UINT64_MAX <= ULONG_MAX
#define PYCI_POPCNT(X) __builtin_popcountl(X)
#define PYCI_CTZ(X) __builtin_ctzl(X)
#elif UINT64_MAX <= ULLONG_MAX
#define PYCI_POPCNT(X) __builtin_popcountll(X)
#define PYCI_CTZ(X) __builtin_ctzll(X)
#else
#error Integer type definitions in pyci.h are incompatible with your compiler.
#endif

/* Seed for SpookyHash. */
#ifndef PYCI_SPOOKYHASH_SEED
#define PYCI_SPOOKYHASH_SEED static_cast<std::uint64_t>(0xdeadbeefdeadbeefUL)
#endif

namespace pyci {

/* Universal signed integer type. */
typedef std::int64_t int_t;

/* Universal unsigned integer type. */
typedef std::uint64_t uint_t;

/* Hash map template type. */
template<class KeyType, class ValueType>
using hashmap = phmap::flat_hash_map<KeyType, ValueType>;

/* Forward-declare classes. */
struct Ham;
struct Wfn;
struct OneSpinWfn;
struct TwoSpinWfn;
struct DOCIWfn;
struct FullCIWfn;
struct GenCIWfn;
struct SparseOp;

/* PyCI routines. */

int_t binomial(int_t, int_t);

int_t binomial_cutoff(int_t, int_t);

void fill_hartreefock_det(int_t, uint_t *);

void fill_det(const int_t, const int_t *, uint_t *);

void fill_occs(const int_t, const uint_t *, int_t *);

void fill_virs(const int_t, int_t, const uint_t *, int_t *);

void next_colex(int_t *);

int_t rank_colex(const int_t, const int_t, const uint_t *);

void unrank_colex(int_t, const int_t, int_t, int_t *);

int_t phase_single_det(const int_t, const int_t, const int_t, const uint_t *);

int_t phase_double_det(const int_t, const int_t, const int_t, const int_t, const int_t,
                       const uint_t *);

int_t popcnt_det(const int_t, const uint_t *);

int_t ctz_det(const int_t, const uint_t *);

inline int_t nword_det(const int_t n) {
    return n / PYCI_UINT_SIZE + ((n % PYCI_UINT_SIZE) ? 1 : 0);
}

inline void excite_det(const int_t i, const int_t a, uint_t *det) {
    det[i / PYCI_UINT_SIZE] &= ~(PYCI_UINT_ONE << (i % PYCI_UINT_SIZE));
    det[a / PYCI_UINT_SIZE] |= PYCI_UINT_ONE << (a % PYCI_UINT_SIZE);
}

inline void setbit_det(const int_t i, uint_t *det) {
    det[i / PYCI_UINT_SIZE] |= PYCI_UINT_ONE << (i % PYCI_UINT_SIZE);
}

inline void clearbit_det(const int_t i, uint_t *det) {
    det[i / PYCI_UINT_SIZE] &= ~(PYCI_UINT_ONE << (i % PYCI_UINT_SIZE));
}

int_t add_hci(const Ham &, DOCIWfn &, const double *, const double);

int_t add_hci(const Ham &, FullCIWfn &, const double *, const double);

int_t add_hci(const Ham &, GenCIWfn &, const double *, const double);

double compute_overlap(const OneSpinWfn &, const OneSpinWfn &, const double *, const double *);

double compute_overlap(const TwoSpinWfn &, const TwoSpinWfn &, const double *, const double *);

void compute_rdms(const DOCIWfn &, const double *, double *, double *);

void compute_rdms(const FullCIWfn &, const double *, double *, double *);

void compute_rdms(const GenCIWfn &, const double *, double *, double *);

double compute_enpt2(const Ham &, const DOCIWfn &, const double *, const double, const double);

double compute_enpt2(const Ham &, const FullCIWfn &, const double *, const double, const double);

double compute_enpt2(const Ham &, const GenCIWfn &, const double *, const double, const double);

/*
Section: Hamiltonian classes

Notes: We will subclass
*/

struct Ham {
public:
    int_t nbasis;
    double ecore, *one_mo, *two_mo, *h, *v, *w;

    inline Ham(void) {
    }

    inline Ham(const Ham &ham)
        : nbasis(ham.nbasis), ecore(ham.ecore), one_mo(ham.one_mo), two_mo(ham.two_mo), h(ham.h),
          v(ham.v), w(ham.w) {
    }

    inline Ham(Ham &&ham) noexcept
        : nbasis(std::exchange(ham.nbasis, 0)), ecore(std::exchange(ham.ecore, 0.0)),
          one_mo(std::exchange(ham.one_mo, nullptr)), two_mo(std::exchange(ham.two_mo, nullptr)),
          h(std::exchange(ham.h, nullptr)), v(std::exchange(ham.v, nullptr)),
          w(std::exchange(ham.w, nullptr)) {
    }
};

/* Wave function classes. */

struct Wfn {
public:
    int_t nbasis, nocc, nocc_up, nocc_dn, nvir, nvir_up, nvir_dn;
    int_t ndet, nword, nword2, maxrank_up, maxrank_dn;

protected:
    std::vector<uint_t> dets;
    hashmap<uint_t, int_t> dict;

public:
    Wfn(const Wfn &);

    Wfn(Wfn &&) noexcept;

    inline Wfn(const int_t nb, const int_t nu, const int_t nd) {
        init(nb, nu, nd);
    }

    void squeeze(void);

protected:
    inline Wfn(void) {
    }

    void init(const int_t, const int_t, const int_t);
};

struct OneSpinWfn : public Wfn {
public:
    using Wfn::maxrank_dn;
    using Wfn::maxrank_up;
    using Wfn::nbasis;
    using Wfn::ndet;
    using Wfn::nocc;
    using Wfn::nocc_dn;
    using Wfn::nocc_up;
    using Wfn::nvir;
    using Wfn::nvir_dn;
    using Wfn::nvir_up;
    using Wfn::nword;
    using Wfn::nword2;

protected:
    using Wfn::dets;
    using Wfn::dict;

public:
    OneSpinWfn(const OneSpinWfn &);

    OneSpinWfn(OneSpinWfn &&) noexcept;

    OneSpinWfn(const std::string &);

    OneSpinWfn(const int_t, const int_t, const int_t);

    OneSpinWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    OneSpinWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);

    inline const uint_t *det_ptr(const int_t i) const {
        return &dets[i * nword];
    }

    void to_file(const std::string &) const;

    void to_det_array(const int_t, const int_t, uint_t *) const;

    void to_occ_array(const int_t, const int_t, int_t *) const;

    int_t index_det(const uint_t *) const;

    int_t index_det_from_rank(const uint_t) const;

    void copy_det(const int_t, uint_t *) const;

    uint_t rank_det(const uint_t *) const;

    int_t add_det(const uint_t *);

    int_t add_det_with_rank(const uint_t *, const uint_t);

    int_t add_det_from_occs(const int_t *);

    void add_hartreefock_det(void);

    void add_all_dets(void);

    void add_excited_dets(const uint_t *, const int_t);

    void add_dets_from_wfn(const OneSpinWfn &);

    void reserve(const int_t);
};

struct TwoSpinWfn : public Wfn {
public:
    using Wfn::maxrank_dn;
    using Wfn::maxrank_up;
    using Wfn::nbasis;
    using Wfn::ndet;
    using Wfn::nocc;
    using Wfn::nocc_dn;
    using Wfn::nocc_up;
    using Wfn::nvir;
    using Wfn::nvir_dn;
    using Wfn::nvir_up;
    using Wfn::nword;
    using Wfn::nword2;

protected:
    using Wfn::dets;
    using Wfn::dict;

public:
    TwoSpinWfn(const TwoSpinWfn &);

    TwoSpinWfn(TwoSpinWfn &&) noexcept;

    TwoSpinWfn(const std::string &);

    TwoSpinWfn(const int_t, const int_t, const int_t);

    TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);

    inline const uint_t *det_ptr(const int_t i) const {
        return &dets[i * nword2];
    }

    void to_file(const std::string &) const;

    void to_det_array(const int_t, const int_t, uint_t *) const;

    void to_occ_array(const int_t, const int_t, int_t *) const;

    int_t index_det(const uint_t *) const;

    int_t index_det_from_rank(const uint_t) const;

    void copy_det(const int_t, uint_t *) const;

    uint_t rank_det(const uint_t *) const;

    int_t add_det(const uint_t *);

    int_t add_det_with_rank(const uint_t *, const uint_t);

    int_t add_det_from_occs(const int_t *);

    void add_hartreefock_det(void);

    void add_all_dets(void);

    void add_excited_dets(const uint_t *, const int_t, const int_t);

    void add_dets_from_wfn(const TwoSpinWfn &);

    void reserve(const int_t);
};

struct DOCIWfn final : public OneSpinWfn {
public:
    using Wfn::maxrank_dn;
    using Wfn::maxrank_up;
    using Wfn::nbasis;
    using Wfn::ndet;
    using Wfn::nocc;
    using Wfn::nocc_dn;
    using Wfn::nocc_up;
    using Wfn::nvir;
    using Wfn::nvir_dn;
    using Wfn::nvir_up;
    using Wfn::nword;
    using Wfn::nword2;

protected:
    using Wfn::dets;
    using Wfn::dict;

public:
    DOCIWfn(const DOCIWfn &);

    DOCIWfn(DOCIWfn &&) noexcept;

    DOCIWfn(const std::string &);

    DOCIWfn(const int_t, const int_t, const int_t);

    DOCIWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    DOCIWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);
};

struct FullCIWfn final : public TwoSpinWfn {
public:
    using Wfn::maxrank_dn;
    using Wfn::maxrank_up;
    using Wfn::nbasis;
    using Wfn::ndet;
    using Wfn::nocc;
    using Wfn::nocc_dn;
    using Wfn::nocc_up;
    using Wfn::nvir;
    using Wfn::nvir_dn;
    using Wfn::nvir_up;
    using Wfn::nword;
    using Wfn::nword2;

protected:
    using Wfn::dets;
    using Wfn::dict;

public:
    FullCIWfn(const FullCIWfn &);

    FullCIWfn(FullCIWfn &&) noexcept;

    FullCIWfn(const DOCIWfn &);

    FullCIWfn(const std::string &);

    FullCIWfn(const int_t, const int_t, const int_t);

    FullCIWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    FullCIWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);
};

struct GenCIWfn final : public OneSpinWfn {
public:
    using Wfn::maxrank_dn;
    using Wfn::maxrank_up;
    using Wfn::nbasis;
    using Wfn::ndet;
    using Wfn::nocc;
    using Wfn::nocc_dn;
    using Wfn::nocc_up;
    using Wfn::nvir;
    using Wfn::nvir_dn;
    using Wfn::nvir_up;
    using Wfn::nword;
    using Wfn::nword2;

protected:
    using Wfn::dets;
    using Wfn::dict;

public:
    GenCIWfn(const GenCIWfn &);

    GenCIWfn(GenCIWfn &&) noexcept;

    GenCIWfn(const DOCIWfn &);

    GenCIWfn(const FullCIWfn &);

    GenCIWfn(const std::string &);

    GenCIWfn(const int_t, const int_t, const int_t);

    GenCIWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

    GenCIWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);
};

/* Sparse matrix operator class. */

struct SparseOp final {
public:
    int_t nrow, ncol, size;
    double ecore;

protected:
    std::vector<double> data;
    std::vector<int_t> indices;
    std::vector<int_t> indptr;

public:
    SparseOp(const SparseOp &);

    SparseOp(SparseOp &&) noexcept;

    SparseOp(const int_t, const int_t);

    SparseOp(const Ham &, const DOCIWfn &, const int_t, const int_t);

    SparseOp(const Ham &, const FullCIWfn &, const int_t, const int_t);

    SparseOp(const Ham &, const GenCIWfn &, const int_t, const int_t);

    inline const double *data_ptr(const int_t index) const {
        return &data[index];
    }

    inline const int_t *indices_ptr(const int_t index) const {
        return &indices[index];
    }

    inline const int_t *indptr_ptr(const int_t index) const {
        return &indptr[index];
    }

    double get_element(const int_t, const int_t) const;

    void perform_op(const double *, double *) const;

    void perform_op_cepa0(const double *, double *, const int_t) const;

    void perform_op_transpose_cepa0(const double *, double *, const int_t) const;

    void rhs_cepa0(double *, const int_t) const;

protected:
    template<class WfnType>
    void init(const Ham &, const WfnType &, const int_t, const int_t);

    void init_thread_sort_row(const int_t);

    void init_thread_condense(SparseOp &, const int_t);

    void init_thread_add_row(const Ham &, const DOCIWfn &, const int_t, uint_t *, int_t *, int_t *);

    void init_thread_add_row(const Ham &, const FullCIWfn &, const int_t, uint_t *, int_t *,
                             int_t *);

    void init_thread_add_row(const Ham &, const GenCIWfn &, const int_t, uint_t *, int_t *,
                             int_t *);
};

} // namespace pyci
