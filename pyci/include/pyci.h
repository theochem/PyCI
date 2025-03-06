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

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <fstream>
#include <future>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE long
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <parallel_hashmap/phmap.h>

#ifdef _USE_RAPIDHASH
#include <rapidhash.h>
#else
#include <SpookyV2.h>
#endif

#include <sort_with_arg.h>

/* Macros to produce strings from literal macro parameters. */

#define LITERAL(S) #S
#define STRINGIZE(S) LITERAL(S)

/* PyCI version. */

#define PYCI_VERSION STRINGIZE(_PYCI_VERSION)
#define GIT_BRANCH STRINGIZE(_GIT_BRANCH)
#define BUILD_TIME STRINGIZE(_BUILD_TIME)
#define COMPILER_VERSION STRINGIZE(_COMPILER_VERSION)

/* Dynamic resize factor for SparseOp vectors. */

#ifndef PYCI_SPARSEOP_RESIZE_FACTOR
#define PYCI_SPARSEOP_RESIZE_FACTOR 1.5
#endif

/* Minimum number of individual jobs per thread. */

#ifndef PYCI_CHUNKSIZE_MIN
#define PYCI_CHUNKSIZE_MIN 1024
#endif

namespace pyci {

/* Integer types, popcnt and ctz functions. */

typedef unsigned long ulong;

template<typename T>
inline constexpr int Size(void) {
    return std::numeric_limits<T>::digits;
}

template<typename T>
inline constexpr T Max(void) {
    return std::numeric_limits<T>::max();
}

template<typename T>
inline int Pop(const T);

template<>
inline int Pop(const unsigned t) {
    return __builtin_popcount(t);
}

template<>
inline int Pop(const unsigned long t) {
    return __builtin_popcountl(t);
}

template<typename T>
inline int Ctz(const T);

template<>
inline int Ctz(const unsigned t) {
    return __builtin_ctz(t);
}

template<>
inline int Ctz(const unsigned long t) {
    return __builtin_ctzl(t);
}

template<>
inline int Ctz(const unsigned long long t) {
    return __builtin_ctzll(t);
}

/* Hash function. */

#ifdef _USE_RAPIDHASH
typedef std::uint64_t Hash;

template<typename T, typename U>
Hash spookyhash(T length, const U *data) {
    return rapidhash(reinterpret_cast<const void *>(data), length * sizeof(U));
}
#else
typedef std::pair<ulong, ulong> Hash;

template<typename T, typename U>
Hash spookyhash(T length, const U *data) {
    Hash h(0x23a23cf5033c3c81UL, 0xb3816f6a2c68e530UL);
    SpookyHash::Hash128(reinterpret_cast<const void *>(data), length * sizeof(U), &h.first,
                        &h.second);
    return h;
}
#endif

/* Vector template types. */

template<typename T>
using Vector = std::vector<T>;

template<typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

/* Eigen dense matrix template types. */

#define PYCI_MAT_DYNAMIC Eigen::Dynamic, Eigen::Dynamic

template<typename T>
using DenseMatrix = Eigen::Map<Eigen::Matrix<T, PYCI_MAT_DYNAMIC, Eigen::RowMajor>>;

template<typename T>
using CDenseMatrix = Eigen::Map<const Eigen::Matrix<T, PYCI_MAT_DYNAMIC, Eigen::RowMajor>>;

template<typename T>
using DenseVector = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template<typename T>
using CDenseVector = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

/* Hash map template type. */

template<class KeyType, class ValueType>
using HashMap = phmap::flat_hash_map<KeyType, ValueType>;

/* Pybind11 NumPy array types. */

template<typename Scalar>
using Array = pybind11::array_t<Scalar, pybind11::array::c_style | pybind11::array::forcecast>;

template<typename Scalar>
using ColMajorArray =
    pybind11::array_t<Scalar, pybind11::array::f_style | pybind11::array::forcecast>;

/* Forward-declare classes. */

struct SQuantOp;
struct Wfn;
struct OneSpinWfn;
struct TwoSpinWfn;
struct DOCIWfn;
struct FullCIWfn;
struct GenCIWfn;
struct SparseOp;

/* Number of threads global variable. */

extern long g_number_threads;

/* PyCI routines. */

long get_num_threads(void);

long end_chunk_idx(const long, const long, const long);

void set_num_threads(const long);

long binomial(long, long);

void fill_hartreefock_det(long, ulong *);

void fill_det(const long, const long *, ulong *);

void fill_occs(const long, const ulong *, long *);

void fill_virs(const long, long, const ulong *, long *);

void next_colex(long *);

long rank_colex(const long, const long, const ulong *);

void unrank_colex(long, const long, long, long *);

long phase_single_det(const long, const long, const long, const ulong *);

long phase_double_det(const long, const long, const long, const long, const long, const ulong *);

long popcnt_det(const long, const ulong *);

long ctz_det(const long, const ulong *);

long nword_det(const long);

void excite_det(const long, const long, ulong *);

void setbit_det(const long, ulong *);

void clearbit_det(const long, ulong *);

void compute_rdms(const DOCIWfn &, const double *, double *, double *);

void compute_rdms_1234(const DOCIWfn &, const double *, double *, double *, double *, double *);

void compute_rdms(const FullCIWfn &, const double *, double *, double *);

void compute_rdms(const GenCIWfn &, const double *, double *, double *);

void compute_transition_rdms(const DOCIWfn &, const DOCIWfn &, const double *, const double *,
                             double *, double *);

void compute_transition_rdms(const FullCIWfn &, const FullCIWfn &, const double *, const double *,
                             double *, double *);

void compute_transition_rdms(const GenCIWfn &, const GenCIWfn &, const double *, const double *,
                             double *, double *);

template<class WfnType>
double compute_overlap(const WfnType &, const WfnType &, const double *, const double *);

template<class WfnType>
long add_hci(const SQuantOp &, WfnType &, const double *, const double, const long = -1);

template<class WfnType>
double compute_enpt2(const SQuantOp &, const WfnType &, const double *, const double, const double,
                     const long = -1);

/* Free Python interface functions. */

long py_popcnt(const Array<ulong>);

long py_ctz(const Array<ulong>);

pybind11::tuple py_compute_rdms_doci(const DOCIWfn &, const Array<double>);

pybind11::tuple py_compute_rdms_1234_doci(const DOCIWfn &, const Array<double>);

pybind11::tuple py_compute_rdms_fullci(const FullCIWfn &, const Array<double>);

pybind11::tuple py_compute_rdms_genci(const GenCIWfn &, const Array<double>);

pybind11::tuple py_compute_transition_rdms_doci(const DOCIWfn &, const DOCIWfn &,
                                                const Array<double>, const Array<double>);

pybind11::tuple py_compute_transition_rdms_fullci(const FullCIWfn &, const FullCIWfn &,
                                                  const Array<double>, const Array<double>);

pybind11::tuple py_compute_transition_rdms_genci(const GenCIWfn &, const GenCIWfn &,
                                                 const Array<double>, const Array<double>);

template<class WfnType>
double py_compute_overlap(const WfnType &, const WfnType &, const Array<double>,
                          const Array<double>);

template<class WfnType>
long py_add_hci(const SQuantOp &, WfnType &, const Array<double>, const double, const long = -1);

template<class WfnType>
double py_compute_enpt2(const SQuantOp &, const WfnType &, const Array<double>, const double,
                        const double, const long = -1);

/* Second quantized operator class. */

struct SQuantOp final {
public:
    long nbasis;
    double ecore, *one_mo, *two_mo, *h, *v, *w;
    Array<double> one_mo_array, two_mo_array, h_array, v_array, w_array;

    SQuantOp(void);

    SQuantOp(const SQuantOp &);

    SQuantOp(SQuantOp &&) noexcept;

    SQuantOp(const std::string &);

    SQuantOp(const double, const Array<double>, const Array<double>);

    void to_file(const std::string &, const long, const long, const double) const;
};

/* Wave function classes. */

struct Wfn {
public:
    long nbasis, nocc, nocc_up, nocc_dn, nvir, nvir_up, nvir_dn;
    long ndet, nword, nword2, maxrank_up, maxrank_dn;

protected:
    AlignedVector<ulong> dets;
    HashMap<Hash, long> dict;

public:
    Wfn(const Wfn &);

    Wfn(Wfn &&) noexcept;

    Wfn(const long, const long, const long);

    long length(void) const;

    void squeeze(void);

protected:
    Wfn(void);

    void init(const long, const long, const long);
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

    OneSpinWfn(const long, const long, const long);

    OneSpinWfn(const long, const long, const long, const long, const ulong *);

    OneSpinWfn(const long, const long, const long, const long, const long *);

    OneSpinWfn(const long, const long, const long, const Array<ulong>);

    OneSpinWfn(const long, const long, const long, const Array<long>);

    const ulong *det_ptr(const long) const;

    void to_file(const std::string &) const;

    void to_det_array(const long, const long, ulong *) const;

    void to_occ_array(const long, const long, long *) const;

    long index_det(const ulong *) const;

    long index_det_from_rank(const Hash) const;

    void copy_det(const long, ulong *) const;

    Hash rank_det(const ulong *) const;

    long add_det(const ulong *);

    long add_det_with_rank(const ulong *, const Hash);

    long add_det_from_occs(const long *);

    void add_hartreefock_det(void);

    void add_all_dets(long = -1);

    void add_excited_dets(const ulong *, const long);

    void add_dets_from_wfn(const OneSpinWfn &);

    void reserve(const long);

    Array<ulong> py_getitem(const long) const;

    Array<ulong> py_to_det_array(long, long) const;

    Array<long> py_to_occ_array(long, long) const;

    long py_index_det(const Array<ulong>) const;

    Hash py_rank_det(const Array<ulong>) const;

    long py_add_det(const Array<ulong>);

    long py_add_occs(const Array<long>);

    long py_add_excited_dets(const long, const pybind11::object);
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

    TwoSpinWfn(const long, const long, const long);

    TwoSpinWfn(const long, const long, const long, const long, const ulong *);

    TwoSpinWfn(const long, const long, const long, const long, const long *);

    TwoSpinWfn(const long, const long, const long, const Array<ulong>);

    TwoSpinWfn(const long, const long, const long, const Array<long>);

    const ulong *det_ptr(const long) const;

    void to_file(const std::string &) const;

    void to_det_array(const long, const long, ulong *) const;

    void to_occ_array(const long, const long, long *) const;

    long index_det(const ulong *) const;

    long index_det_from_rank(const Hash) const;

    void copy_det(const long, ulong *) const;

    Hash rank_det(const ulong *) const;

    long add_det(const ulong *);

    long add_det_with_rank(const ulong *, const Hash);

    long add_det_from_occs(const long *);

    void add_hartreefock_det(void);

    void add_all_dets(long = -1);

    void add_excited_dets(const ulong *, const long, const long);

    void add_dets_from_wfn(const TwoSpinWfn &);

    void reserve(const long);

    Array<ulong> py_getitem(const long) const;

    Array<ulong> py_to_det_array(long, long) const;

    Array<long> py_to_occ_array(long, long) const;

    long py_index_det(const Array<ulong>) const;

    Hash py_rank_det(const Array<ulong>) const;

    long py_add_det(const Array<ulong>);

    long py_add_occs(const Array<long>);

    long py_add_excited_dets(const long, const pybind11::object);
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

    DOCIWfn(const long, const long, const long);

    DOCIWfn(const long, const long, const long, const long, const ulong *);

    DOCIWfn(const long, const long, const long, const long, const long *);

    DOCIWfn(const long, const long, const long, const Array<ulong>);

    DOCIWfn(const long, const long, const long, const Array<long>);
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

    FullCIWfn(const long, const long, const long);

    FullCIWfn(const long, const long, const long, const long, const ulong *);

    FullCIWfn(const long, const long, const long, const long, const long *);

    FullCIWfn(const long, const long, const long, const Array<ulong>);

    FullCIWfn(const long, const long, const long, const Array<long>);
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

    GenCIWfn(const long, const long, const long);

    GenCIWfn(const long, const long, const long, const long, const ulong *);

    GenCIWfn(const long, const long, const long, const long, const long *);

    GenCIWfn(const long, const long, const long, const Array<ulong>);

    GenCIWfn(const long, const long, const long, const Array<long>);
};

/* Sparse matrix operator class. */

struct SparseOp final {
public:
    long nrow, ncol, size;
    double ecore;
    bool symmetric;
    pybind11::tuple shape;

private:
    AlignedVector<double> data;
    AlignedVector<long> indices, indptr;

public:
    SparseOp(const SparseOp &);

    SparseOp(SparseOp &&) noexcept;

    SparseOp(const long, const long, const bool);

    SparseOp(const SQuantOp &, const DOCIWfn &, const long, const long, const bool);

    SparseOp(const SQuantOp &, const FullCIWfn &, const long, const long, const bool);

    SparseOp(const SQuantOp &, const GenCIWfn &, const long, const long, const bool);

    pybind11::object dtype(void) const;

    const double *data_ptr(const long) const;

    const long *indices_ptr(const long) const;

    const long *indptr_ptr(const long) const;

    double get_element(const long, const long) const;

    void perform_op(const double *, double *) const;

    void perform_op_symm(const double *, double *) const;

    void solve_ci(const long, const double *, const long, const long, const double, double *,
                  double *) const;

    template<class WfnType>
    void update(const SQuantOp &, const WfnType &, const long, const long, const long);

    void reserve(const long);

    void squeeze(void);

    Array<double> py_matvec(const Array<double>) const;

    Array<double> py_matvec_out(const Array<double>, Array<double>) const;

    pybind11::tuple py_solve_ci(const long, pybind11::object, const long, const long,
                                const double) const;

    template<class WfnType>
    void py_update(const SQuantOp &, const WfnType &);

    Array<double> py_data() const;

    Array<long> py_indices() const;

    Array<long> py_indptr() const;

private:
    void sort_row(const long);

    void add_row(const SQuantOp &, const DOCIWfn &, const long, ulong *, long *, long *);

    void add_row(const SQuantOp &, const FullCIWfn &, const long, ulong *, long *, long *);

    void add_row(const SQuantOp &, const GenCIWfn &, const long, ulong *, long *, long *);
};

/* FanCI objective classes. */

template<class Wfn>
class Objective {
public:
    std::size_t nproj;
    std::size_t nconn;
    std::size_t nparam;
    std::size_t n_detcons;
    std::size_t n_paramcons;
    std::vector<double> ovlp;
    std::vector<double> d_ovlp;
    std::vector<long> idx_detcons;
    std::vector<long> idx_paramcons;
    std::vector<double> val_detcons;
    std::vector<double> val_paramcons;

public:
    Objective(const SparseOp &, const Wfn &, const std::size_t = 0UL, const long * = nullptr,
              const double * = nullptr, const std::size_t = 0UL, const long * = nullptr,
              const double * = nullptr);

    Objective(const SparseOp &, const Wfn &, const pybind11::object, const pybind11::object,
              const pybind11::object, const pybind11::object);

    Objective(const Objective &);

    Objective(Objective &&) noexcept;

    void init(const std::size_t, const long *, const double *, const std::size_t, const long *,
              const double *);

    void objective(const SparseOp &, const double *, double *);

    void jacobian(const SparseOp &, const double *, double *);

    Array<double> py_objective(const SparseOp &, const Array<double> &);

    ColMajorArray<double> py_jacobian(const SparseOp &, const Array<double> &);

    Array<double> py_overlap(const Array<double> &);

    ColMajorArray<double> py_d_overlap(const Array<double> &);

    virtual void overlap(const std::size_t, const double *, double *) = 0;

    virtual void d_overlap(const std::size_t, const double *, double *) = 0;
};

class AP1roGObjective : public Objective<DOCIWfn> {
public:
    using Objective<DOCIWfn>::nproj;
    using Objective<DOCIWfn>::nconn;
    using Objective<DOCIWfn>::nparam;
    using Objective<DOCIWfn>::ovlp;
    using Objective<DOCIWfn>::d_ovlp;

    std::size_t nrow;
    std::size_t ncol;
    std::vector<std::size_t> nexc_list;
    std::vector<std::size_t> hole_list;
    std::vector<std::size_t> part_list;

public:
    AP1roGObjective(const SparseOp &, const DOCIWfn &, const std::size_t = 0UL,
                    const long * = nullptr, const double * = nullptr, const std::size_t = 0UL,
                    const long * = nullptr, const double * = nullptr);

    AP1roGObjective(const SparseOp &, const DOCIWfn &, const pybind11::object,
                    const pybind11::object, const pybind11::object, const pybind11::object);

    AP1roGObjective(const AP1roGObjective &);

    AP1roGObjective(AP1roGObjective &&) noexcept;

    void init_overlap(const DOCIWfn &);

    virtual void overlap(const size_t, const double *x, double *y);

    virtual void d_overlap(const size_t, const double *x, double *y);
};

class APIGObjective : public Objective<DOCIWfn> {
public:
    using Objective<DOCIWfn>::nproj;
    using Objective<DOCIWfn>::nconn;
    using Objective<DOCIWfn>::nparam;
    using Objective<DOCIWfn>::ovlp;
    using Objective<DOCIWfn>::d_ovlp;

    std::size_t nrow;
    std::size_t ncol;
    std::vector<std::size_t> part_list;

public:
    APIGObjective(const SparseOp &, const DOCIWfn &, const std::size_t = 0UL,
                  const long * = nullptr, const double * = nullptr, const std::size_t = 0UL,
                  const long * = nullptr, const double * = nullptr);

    APIGObjective(const SparseOp &, const DOCIWfn &, const pybind11::object, const pybind11::object,
                  const pybind11::object, const pybind11::object);

    APIGObjective(const APIGObjective &);

    APIGObjective(APIGObjective &&) noexcept;

    void init_overlap(const DOCIWfn &);

    virtual void overlap(const size_t, const double *x, double *y);

    virtual void d_overlap(const size_t, const double *x, double *y);
};

} // namespace pyci
