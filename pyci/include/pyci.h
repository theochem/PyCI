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
#include <vector>

/* Uncomment this to use exact (colexicographical order) hashing.
 * This will not work for systems where binomial(nbasis, nocc) > 2 ** 63. */
/* #define PYCI_EXACT_HASH */

/* Define integer types, popcnt and ctz functions. */
#define PYCI_INT_SIZE (std::int64_t)(sizeof(std::int64_t) * CHAR_BIT)
#define PYCI_UINT_SIZE (std::int64_t)(sizeof(std::uint64_t) * CHAR_BIT)
#define PYCI_INT_MAX (std::int64_t) INT64_MAX
#define PYCI_UINT_MAX (std::uint64_t) UINT64_MAX
#define PYCI_UINT_ZERO (std::uint64_t)0U
#define PYCI_UINT_ONE (std::uint64_t)1U
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
#define PYCI_SPOOKYHASH_SEED (uint_t)0xdeadbeefdeadbeefU
#endif

namespace pyci {

/* Universal signed integer type. */
typedef std::int64_t int_t;

/* Universal unsigned integer type. */
typedef std::uint64_t uint_t;

/* Hash map template type. */
template <class KeyType, class ValueType> using hashmap = phmap::flat_hash_map<KeyType, ValueType>;

/* Forward-declare classes. */

struct OneSpinWfn;

struct TwoSpinWfn;

struct SparseOp;

/* Common functions. */

bool binomial_raises(int_t, int_t);

int_t binomial(int_t, int_t);

void fill_det(const int_t, const int_t *, uint_t *);

void fill_occs(const int_t, const uint_t *, int_t *);

void fill_virs(const int_t, int_t, const uint_t *, int_t *);

void next_colex(int_t *);

int_t rank_colex(const int_t, const int_t, const uint_t *);

void unrank_colex(int_t, const int_t, int_t, int_t *);

int_t nword_det(const int_t);

void excite_det(const int_t, const int_t, uint_t *);

void setbit_det(const int_t, uint_t *);

void clearbit_det(const int_t, uint_t *);

int_t phase_single_det(const int_t, const int_t, const int_t, const uint_t *);

int_t phase_double_det(const int_t, const int_t, const int_t, const int_t, const int_t,
                       const uint_t *);

int_t popcnt_det(const int_t, const uint_t *);

int_t ctz_det(const int_t, const uint_t *);

/* Wave function class with determinants made up of one bitstring. */
struct OneSpinWfn {
public:
  int_t nword, nbasis, nocc, nvir, ndet;
  std::vector<uint_t> dets;

private:
  hashmap<uint_t, int_t> dict;

public:
  OneSpinWfn(void);

  OneSpinWfn(OneSpinWfn &&) noexcept;

  OneSpinWfn(const int_t, const int_t);

  OneSpinWfn(const OneSpinWfn &);

  OneSpinWfn(const TwoSpinWfn &);

  OneSpinWfn(const char *);

  OneSpinWfn(const int_t, const int_t, const int_t, const uint_t *);

  OneSpinWfn(const int_t, const int_t, const int_t, const int_t *);

  void init(const int_t, const int_t);

  void from_onespinwfn(const OneSpinWfn &);

  void from_twospinwfn(const TwoSpinWfn &);

  void from_file(const char *);

  void from_det_array(const int_t, const int_t, const int_t, const uint_t *);

  void from_occs_array(const int_t, const int_t, const int_t, const int_t *);

  void to_file(const char *) const;

  void to_occs_array(const int_t, const int_t, int_t *) const;

  int_t index_det(const uint_t *) const;

  int_t index_det_from_rank(const uint_t) const;

  void copy_det(const int_t, uint_t *) const;

  const uint_t *det_ptr(const int_t) const;

  uint_t rank_det(const uint_t *) const;

  int_t add_det(const uint_t *);

  int_t add_det_with_rank(const uint_t *, const uint_t);

  int_t add_det_from_occs(const int_t *);

  void add_hartreefock_det(void);

  void add_all_dets(void);

  void add_excited_dets(const uint_t *, const int_t);

  void add_dets_from_wfn(const OneSpinWfn &);

  void reserve(const int_t);

  void squeeze(void);

  void clear(void);

  void compute_rdms_doci(const double *, double *, double *) const;

  void compute_rdms_genci(const double *, double *, double *) const;

  double compute_overlap(const double *, const OneSpinWfn &, const double *) const;

  double compute_enpt2_doci(const double *, const double *, const double *, const double,
                            const double) const;

  double compute_enpt2_genci(const double *, const double *, const double *, const double,
                             const double) const;

  int_t run_hci_doci(const double *, const double *, const double);

  int_t run_hci_genci(const double *, const double *, const double *, const double);
};

/* Wave function class with determinants made up of two bitstrings. */
struct TwoSpinWfn {
public:
  int_t nword, nword2, nbasis, nocc_up, nocc_dn, nvir_up, nvir_dn;
  int_t ndet, maxdet_up, maxdet_dn;
  std::vector<uint_t> dets;

private:
  hashmap<uint_t, int_t> dict;

public:
  TwoSpinWfn(void);

  TwoSpinWfn(TwoSpinWfn &&) noexcept;

  TwoSpinWfn(const int_t, const int_t, const int_t);

  TwoSpinWfn(const OneSpinWfn &);

  TwoSpinWfn(const TwoSpinWfn &);

  TwoSpinWfn(const char *);

  TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);

  TwoSpinWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);

  void init(const int_t, const int_t, const int_t);

  void from_onespinwfn(const OneSpinWfn &);

  void from_twospinwfn(const TwoSpinWfn &);

  void from_file(const char *);

  void from_det_array(const int_t, const int_t, const int_t, const int_t, const uint_t *);

  void from_occs_array(const int_t, const int_t, const int_t, const int_t, const int_t *);

  void to_file(const char *) const;

  void to_occs_array(const int_t, const int_t, int_t *) const;

  int_t index_det(const uint_t *) const;

  int_t index_det_from_rank(const uint_t) const;

  void copy_det(const int_t, uint_t *) const;

  const uint_t *det_ptr(const int_t) const;

  uint_t rank_det(const uint_t *) const;

  int_t add_det(const uint_t *);

  int_t add_det_with_rank(const uint_t *, const uint_t);

  int_t add_det_from_occs(const int_t *);

  void add_hartreefock_det(void);

  void add_all_dets(void);

  void add_excited_dets(const uint_t *, const int_t, const int_t);

  void add_dets_from_wfn(const TwoSpinWfn &);

  void reserve(const int_t);

  void squeeze(void);

  void clear(void);

  double compute_overlap(const double *, const TwoSpinWfn &, const double *) const;

  void compute_rdms_fullci(const double *, double *, double *, double *, double *, double *) const;

  double compute_enpt2_fullci(const double *, const double *, const double *, const double,
                              const double) const;

  int_t run_hci_fullci(const double *, const double *, const double *, const double);
};

/* Sparse matrix operator with eigensolver. */
struct SparseOp {
public:
  int_t nrow, ncol, size;
  double ecore;
  std::vector<double> data;
  std::vector<int_t> indices;
  std::vector<int_t> indptr;

  SparseOp(void);

  SparseOp(SparseOp &&) noexcept;

  inline int_t rows(void) const {
    return nrow;
  }

  inline int_t cols(void) const {
    return ncol;
  }

  const double *data_ptr(const int_t) const;

  const int_t *indices_ptr(const int_t) const;

  const int_t *indptr_ptr(const int_t) const;

  double get_element(const int_t, const int_t) const;

  void perform_op(const double *, double *) const;

  void perform_op_cepa0(const double *, double *, const int_t) const;

  void rhs_cepa0(double *, const int_t) const;

  void solve(const double *, const int_t, const int_t, const int_t, const double, double *,
             double *) const;

  void init_doci(const OneSpinWfn &, const double, const double *, const double *, const double *,
                 const int_t, const int_t);

  void init_fullci(const TwoSpinWfn &, const double, const double *, const double *, const int_t,
                   const int_t);

  void init_genci(const OneSpinWfn &, const double, const double *, const double *, const int_t,
                  const int_t);
};

} // namespace pyci
