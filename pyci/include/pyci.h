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

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <parallel_hashmap/phmap.h>

#ifndef PYCI_VERSION
#define PYCI_VERSION 0.0.0
#endif
#define LITERAL(S) #S
#define STRINGIZE(S) LITERAL(S)

/* Define integer types, popcnt and ctz functions. */
#define PYCI_INT_SIZE static_cast<std::int64_t>(std::numeric_limits<std::int64_t>::digits)
#define PYCI_UINT_SIZE static_cast<std::int64_t>(std::numeric_limits<std::uint64_t>::digits)
#define PYCI_INT_MAX std::numeric_limits<std::int64_t>::max()
#define PYCI_UINT_MAX std::numeric_limits<std::uint64_t>::max()
#define PYCI_UINT_ZERO static_cast<std::uint64_t>(0U)
#define PYCI_UINT_ONE static_cast<std::uint64_t>(1U)
#define PYCI_POPCNT(X) __builtin_popcountl(X)
#define PYCI_CTZ(X) __builtin_ctzl(X)

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
using HashMap = phmap::flat_hash_map<KeyType, ValueType>;

/* Pybind11 NumPy array type. */
template<typename Scalar>
using Array = pybind11::array_t<Scalar, pybind11::array::c_style | pybind11::array::forcecast>;

/* Forward-declare classes. */
struct Ham;
struct Wfn;
struct OneSpinWfn;
struct TwoSpinWfn;
struct DOCIWfn;
struct FullCIWfn;
struct GenCIWfn;
struct SparseOp;

/* Number of threads global variable. */
int_t g_number_threads{4};

/* PyCI routines. */

int_t get_num_threads(void);

void set_num_threads(const int_t);

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

int_t nword_det(const int_t);

void excite_det(const int_t, const int_t, uint_t *);

void setbit_det(const int_t, uint_t *);

void clearbit_det(const int_t, uint_t *);

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

/* Hamiltonian class. */

struct Ham final {
public:
    int_t nbasis;
    double ecore, *one_mo, *two_mo, *h, *v, *w;
    pybind11::object one_mo_array, two_mo_array, h_array, v_array, w_array;

    Ham(void);

    Ham(const Ham &);

    Ham(Ham &&) noexcept;

    Ham(const std::string &);

    Ham(const double, const Array<double>, const Array<double>);

    void to_file(const std::string &, const int_t, const int_t, const double) const;

private:
    void init_ham(const pybind11::tuple &);
};

/* Wave function classes. */

struct Wfn {
public:
    int_t nbasis, nocc, nocc_up, nocc_dn, nvir, nvir_up, nvir_dn;
    int_t ndet, nword, nword2, maxrank_up, maxrank_dn;

protected:
    std::vector<uint_t> dets;
    HashMap<uint_t, int_t> dict;

public:
    Wfn(const Wfn &);

    Wfn(Wfn &&) noexcept;

    Wfn(const int_t, const int_t, const int_t);

    int_t length(void) const;

    void squeeze(void);

protected:
    Wfn(void);

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

    OneSpinWfn(const int_t, const int_t, const int_t, const Array<uint_t>);

    OneSpinWfn(const int_t, const int_t, const int_t, const Array<int_t>);

    const uint_t *det_ptr(const int_t) const;

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

    Array<uint_t> py_getitem(const int_t) const;

    Array<uint_t> py_to_det_array(int_t, int_t) const;

    Array<int_t> py_to_occ_array(int_t, int_t) const;

    int_t py_index_det(const Array<uint_t>) const;

    uint_t py_rank_det(const Array<uint_t>) const;

    int_t py_add_det(const Array<uint_t>);

    int_t py_add_occs(const Array<int_t>);

    int_t py_add_excited_dets(const int_t, const pybind11::object);
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

    TwoSpinWfn(const int_t, const int_t, const int_t, const Array<uint_t>);

    TwoSpinWfn(const int_t, const int_t, const int_t, const Array<int_t>);

    const uint_t *det_ptr(const int_t) const;

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

    Array<uint_t> py_getitem(const int_t) const;

    Array<uint_t> py_to_det_array(int_t, int_t) const;

    Array<int_t> py_to_occ_array(int_t, int_t) const;

    int_t py_index_det(const Array<uint_t>) const;

    uint_t py_rank_det(const Array<uint_t>) const;

    int_t py_add_det(const Array<uint_t>);

    int_t py_add_occs(const Array<int_t>);

    int_t py_add_excited_dets(const int_t, const pybind11::object);
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

    DOCIWfn(const int_t, const int_t, const int_t, const Array<uint_t>);

    DOCIWfn(const int_t, const int_t, const int_t, const Array<int_t>);
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

    FullCIWfn(const int_t, const int_t, const int_t, const Array<uint_t>);

    FullCIWfn(const int_t, const int_t, const int_t, const Array<int_t>);
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

    GenCIWfn(const int_t, const int_t, const int_t, const Array<uint_t>);

    GenCIWfn(const int_t, const int_t, const int_t, const Array<int_t>);
};

/* Sparse matrix operator class. */

struct SparseOp final {
public:
    int_t nrow, ncol, size;
    double ecore;
    pybind11::object shape;
    std::vector<double> data;
    std::vector<int_t> indices;
    std::vector<int_t> indptr;

    SparseOp(const SparseOp &);

    SparseOp(SparseOp &&) noexcept;

    SparseOp(const int_t, const int_t);

    SparseOp(const Ham &, const DOCIWfn &, const int_t, const int_t);

    SparseOp(const Ham &, const FullCIWfn &, const int_t, const int_t);

    SparseOp(const Ham &, const GenCIWfn &, const int_t, const int_t);

    const double *data_ptr(const int_t) const;

    const int_t *indices_ptr(const int_t) const;

    const int_t *indptr_ptr(const int_t) const;

    double get_element(const int_t, const int_t) const;

    void perform_op(const double *, double *) const;

    void perform_op_cepa0(const double *, double *, const int_t) const;

    void perform_op_transpose_cepa0(const double *, double *, const int_t) const;

    void rhs_cepa0(double *, const int_t) const;

    Array<double> py_matvec(const Array<double>) const;

    Array<double> py_matvec_out(const Array<double>, Array<double>) const;

    Array<double> py_matvec_cepa0(const Array<double>, const int_t) const;

    Array<double> py_rmatvec_cepa0(const Array<double>, const int_t) const;

    Array<double> py_rhs_cepa0(const int_t) const;

    template<class WfnType>
    void init(const Ham &, const WfnType &, const int_t, const int_t);

    void init_thread_add_row(const Ham &, const DOCIWfn &, const int_t, uint_t *, int_t *, int_t *);

    void init_thread_add_row(const Ham &, const FullCIWfn &, const int_t, uint_t *, int_t *,
                             int_t *);

    void init_thread_add_row(const Ham &, const GenCIWfn &, const int_t, uint_t *, int_t *,
                             int_t *);

    void init_thread_sort_row(const int_t);

    void init_thread_condense(SparseOp &, const int_t);
};

/* Free Python interface functions. */

int_t py_popcnt(const Array<uint_t>);

int_t py_ctz(const Array<uint_t>);

int_t py_dociwfn_add_hci(const Ham &, DOCIWfn &, const Array<double>, const double);

int_t py_fullciwfn_add_hci(const Ham &, FullCIWfn &, const Array<double>, const double);

int_t py_genciwfn_add_hci(const Ham &, GenCIWfn &, const Array<double>, const double);

double py_dociwfn_compute_overlap(const DOCIWfn &, const DOCIWfn &, const Array<double>,
                                  const Array<double>);

double py_fullciwfn_compute_overlap(const FullCIWfn &, const FullCIWfn &, const Array<double>,
                                    const Array<double>);

double py_genciwfn_compute_overlap(const GenCIWfn &, const GenCIWfn &, const Array<double>,
                                   const Array<double>);

pybind11::tuple py_dociwfn_compute_rdms(const DOCIWfn &, const Array<double>);

pybind11::tuple py_fullciwfn_compute_rdms(const FullCIWfn &, const Array<double>);

pybind11::tuple py_genciwfn_compute_rdms(const GenCIWfn &, const Array<double>);

double py_dociwfn_compute_enpt2(const Ham &, const DOCIWfn &, const Array<double>, const double,
                                const double);

double py_fullciwfn_compute_enpt2(const Ham &, const FullCIWfn &, const Array<double>, const double,
                                  const double);

double py_genciwfn_compute_enpt2(const Ham &, const GenCIWfn &, const Array<double>, const double,
                                 const double);

} // namespace pyci
