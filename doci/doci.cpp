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

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include <parallel_hashmap/phmap.h>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::int64_t
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

#include "doci.h"


#define DOCI_INT_SIZE (std::int64_t)(sizeof(std::int64_t) * CHAR_BIT)
#define DOCI_UINT_SIZE (std::int64_t)(sizeof(std::uint64_t) * CHAR_BIT)
#define DOCI_INT_MAX (std::int64_t)INT64_MAX
#define DOCI_UINT_MAX (std::uint64_t)UINT64_MAX
#define DOCI_UINT_ONE (std::uint64_t)1U

#if UINT64_MAX <= ULONG_MAX
#define DOCI_POPCNT(X) __builtin_popcountl(X)
#define DOCI_CTZ(X) __builtin_ctzl(X)
#else
#define DOCI_POPCNT(X) __builtin_popcountll(X)
#define DOCI_CTZ(X) __builtin_ctzll(X)
#endif


namespace doci {


DOCIWfn::DOCIWfn() {
    return;
};


DOCIWfn::DOCIWfn(const int_t nbasis_, const int_t nocc_) {
    init(nbasis_, nocc_);
}


DOCIWfn::DOCIWfn(const char *filename) {
    from_file(filename);
}


DOCIWfn::~DOCIWfn() {
    return;
}


void DOCIWfn::init(const int_t nbasis_, const int_t nocc_) {
    if (binomial(nbasis_, nocc_) >= DOCI_INT_MAX / nbasis_)
        throw std::runtime_error("nbasis, nocc too large for hash type");
    nword = nword_det(nbasis_);
    nbasis = nbasis_;
    nocc = nocc_;
    nvir = nbasis_ - nocc_;
    ndet = 0;
    dets.resize(0);
    dict.clear();
}


void DOCIWfn::from_file(const char *filename) {
    bool success = false;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file.read((char *)&ndet, sizeof(int_t))   &&
        file.read((char *)&nbasis, sizeof(int_t)) &&
        file.read((char *)&nocc, sizeof(int_t))) {
        nword = nword_det(nbasis);
        nvir = nbasis - nocc;
        dets.resize(0);
        dict.clear();
        dets.resize(nword * ndet);
        dict.reserve(ndet);
        if (file.read((char *)&dets[0], sizeof(uint_t) * nword * ndet)) success = true;
    }
    file.close();
    if (success)
        for (int_t i = 0; i < ndet; ++i)
            dict[hash_det(nbasis, nocc, &dets[nword * i])] = i;
    else throw std::runtime_error("Error in file");
}


void DOCIWfn::to_file(const char *filename) const {
    bool success = false;
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (file.write((char *)&ndet, sizeof(int_t))   &&
        file.write((char *)&nbasis, sizeof(int_t)) &&
        file.write((char *)&nocc, sizeof(int_t))   &&
        file.write((char *)&dets[0], sizeof(uint_t) * nword * ndet)) success = true;
    file.close();
    if (!success) throw std::runtime_error("Error writing file");
}


int_t DOCIWfn::index_det(const uint_t *det) const {
    hashmap<int_t, int_t>::const_iterator search = dict.find(hash_det(nbasis, nocc, det));
    return (search == dict.end()) ? -1 : search->second;
}


void DOCIWfn::copy_det(const int_t i, uint_t *det) const {
    std::memcpy(det, &dets[i * nword], sizeof(uint_t) * nword);
}


int_t DOCIWfn::add_det(const uint_t *det) {
    if (dict.insert(std::make_pair(hash_det(nbasis, nocc, det), ndet)).second) {
        dets.resize(dets.size() + nword);
        std::memcpy(&dets[nword * ndet], det, sizeof(uint_t) * nword);
        return ndet++;
    }
    return -1;
}


int_t DOCIWfn::add_det_from_occs(const int_t *occs) {
    std::vector<uint_t> det(nword);
    fill_det(nocc, occs, &det[0]);
    return add_det(&det[0]);
}


namespace {

int_t binomial_nocheck(int_t n, int_t k) {
    if (k == 0) return 1;
    else if (k == 1) return n;
    else if (k >= n) return (k == n);
    if (k > n / 2) k = n - k;
    int_t binom = 1;
    for (int_t d = 1; d <= k; ++d)
        binom = binom * n-- / d;
    return binom;
}

}


namespace {

void next_colex(int_t *indices) {
    int_t i = 0;
    while (indices[i + 1] - indices[i] == 1) {
        indices[i] = i;
        ++i;
    }
    ++(indices[i]);
}

}


namespace {

void unhash_indices(int_t nbasis, const int_t nocc, int_t hash, int_t* occs) {
    int_t i, j, k, binom;
    for (i = 0; i < nocc; ++i) {
        j = nocc - i;
        binom = binomial_nocheck(nbasis, j);
        if (binom <= hash) {
            for (k = 0; k < j; ++k)
                occs[k] = k;
            break;
        }
        while (binom > hash)
            binom = binomial_nocheck(--nbasis, j);
        occs[j - 1] = nbasis;
        hash -= binom;
    }
}

}


void DOCIWfn::add_all_dets() {
    ndet = binomial_nocheck(nbasis, nocc);
    dets.resize(0);
    dict.clear();
    dets.resize(ndet * nword);
    dict.reserve(ndet);
    for (int_t idet = 0; idet < ndet; ++idet)
        dict[idet] = idet;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t chunk = omp_get_thread_num();
        int_t start = chunk * chunksize;
        int_t end = (start + chunksize < ndet) ? start + chunksize : ndet;
        std::vector<int_t> occs(nocc + 1);
        unhash_indices(nbasis, nocc, start, &occs[0]);
        occs[nocc] = nbasis + 1;
        for (int_t idet = start; idet < end; ++idet) {
            fill_det(nocc, &occs[0], &dets[idet * nword]);
            next_colex(&occs[0]);
        }
    }
}


void DOCIWfn::add_excited_dets(const uint_t *rdet, const int_t e) {
    int_t i, j, k, no = binomial_nocheck(nocc, e), nv = binomial_nocheck(nvir, e);
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    std::vector<int_t> occinds(e + 1);
    std::vector<int_t> virinds(e + 1);
    fill_occs(nword, rdet, &occs[0]);
    fill_virs(nword, nbasis, rdet, &virs[0]);
    for (k = 0; k < e; ++k)
        virinds[k] = k;
    virinds[e] = nvir + 1;
    for (i = 0; i < nv; ++i) {
        for (k = 0; k < e; ++k)
            occinds[k] = k;
        occinds[e] = nocc + 1;
        for (j = 0; j < no; ++j) {
            std::memcpy(&det[0], rdet, sizeof(uint_t) * nword);
            for (k = 0; k < e; ++k)
                excite_det(occs[occinds[k]], virs[virinds[k]], &det[0]);
            add_det(&det[0]);
            next_colex(&occinds[0]);
        }
        next_colex(&virinds[0]);
    }
}


void DOCIWfn::reserve(const int_t n) {
    dets.reserve(n * nword);
    dict.reserve(n);
}


void DOCIWfn::squeeze() {
    dets.shrink_to_fit();
}


void doci_rdms(const DOCIWfn &wfn, const double *coeffs, double *d0, double *d2) {
    int_t idet, jdet, i, j, k, l;
    double val1, val2;
    // iterate over determinants
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    for (idet = 0; idet < wfn.ndet; ++idet) {
        wfn.copy_det(idet, &det[0]);
        fill_occs(wfn.nword, &det[0], &occs[0]);
        fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
        // diagonal elements
        val1 = coeffs[idet] * coeffs[idet];
        for (i = 0; i < wfn.nocc; ++i) {
            k = occs[i];
            d0[k * (wfn.nbasis + 1)] += val1;
            for (j = i + 1; j < wfn.nocc; ++j) {
                l = occs[j];
                d2[wfn.nbasis * k + l] += val1;
                d2[wfn.nbasis * l + k] += val1;
            }
            // pair excitation elements
            for (j = 0; j < wfn.nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                jdet = wfn.index_det(&det[0]);
                wfn.copy_det(idet, &det[0]);
                // check if excited determinant is in wfn
                if (jdet > idet) {
                    val2 = coeffs[idet] * coeffs[jdet];
                    d0[wfn.nbasis * k + l] += val2;
                    d0[wfn.nbasis * l + k] += val2;
                }
            }
        }
    }
}


double doci_energy(const DOCIWfn &wfn, const double *h, const double *v, const double *w, const double *coeffs) {
    int_t nthread = omp_get_max_threads();
    int_t chunksize = wfn.ndet / nthread + ((wfn.ndet % nthread) ? 1 : 0);
    double val = 0.0;
    #pragma omp parallel reduction(+:val)
    {
        int_t idet, jdet, i, j, k, l;
        int_t chunk = omp_get_thread_num();
        int_t start = chunk * chunksize;
        int_t end = (start + chunksize < wfn.ndet) ? start + chunksize : wfn.ndet;
        double val1, val2, val3;
        // iterate over determinants
        std::vector<uint_t> det(wfn.nword);
        std::vector<int_t> occs(wfn.nocc);
        std::vector<int_t> virs(wfn.nvir);
        for (idet = start; idet < end; ++idet) {
            wfn.copy_det(idet, &det[0]);
            fill_occs(wfn.nword, &det[0], &occs[0]);
            fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
            val1 = 0.0;
            val2 = 0.0;
            val3 = 0.0;
            // diagonal elements
            for (i = 0; i < wfn.nocc; ++i) {
                k = occs[i];
                val1 += v[k * (wfn.nbasis + 1)];
                val2 += h[k];
                for (j = i + 1; j < wfn.nocc; ++j)
                    val2 += w[k * wfn.nbasis + occs[j]];
                // pair excitation elements
                for (j = 0; j < wfn.nvir; ++j) {
                    l = virs[j];
                    excite_det(k, l, &det[0]);
                    jdet = wfn.index_det(&det[0]);
                    wfn.copy_det(idet, &det[0]);
                    // check if excited determinant is in wfn
                    if (jdet != -1) val3 += v[k * wfn.nbasis + l] * coeffs[jdet];
                }
            }
            val += ((val1 + val2 * 2) * coeffs[idet] + val3) * coeffs[idet];
        }
    }
    return val;
}


int_t doci_hci(DOCIWfn &wfn, const double *v, const double *coeffs, const double eps) {
    int_t ndet = wfn.ndet;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t idet, i, j, k, l;
        int_t chunk = omp_get_thread_num();
        int_t start = chunk * chunksize;
        int_t end = (start + chunksize < ndet) ? start + chunksize : ndet;
        // iterate over determinants
        std::vector<uint_t> det(wfn.nword);
        std::vector<int_t> occs(wfn.nocc);
        std::vector<int_t> virs(wfn.nvir);
        for (idet = start; idet < end; ++idet) {
            wfn.copy_det(idet, &det[0]);
            fill_occs(wfn.nword, &det[0], &occs[0]);
            fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
            // pair excitation elements
            for (i = 0; i < wfn.nocc; ++i) {
                k = occs[i];
                for (j = 0; j < wfn.nvir; ++j) {
                    l = virs[j];
                    excite_det(k, l, &det[0]);
                    // add determinant if |H*c| > eps
                    if (std::abs(v[k * wfn.nbasis + l] * coeffs[idet]) > eps)
                        #pragma omp critical
                        wfn.add_det(&det[0]);
                    wfn.copy_det(idet, &det[0]);
                }
            }
        }
    }
    return wfn.ndet - ndet;
}


namespace {

struct SparseOp {
    int_t nrow;
    std::vector<double> data;
    std::vector<int_t> indices;
    std::vector<int_t> indptr;
    inline SparseOp (const DOCIWfn &wfn_) : nrow(wfn_.ndet) {};
    inline int_t rows() { return nrow; }
    inline int_t cols() { return nrow; }
    void perform_op(const double *, double *);
};

}


namespace {

struct DirectOp {
    int_t nrow;
    const double *v;
    const DOCIWfn &wfn;
    std::vector<double> data;
    inline DirectOp (const DOCIWfn &wfn_, const double *v_) : nrow(wfn_.ndet), v(v_), wfn(wfn_) {};
    inline int_t rows() { return nrow; }
    inline int_t cols() { return nrow; }
    void perform_op(const double *, double *);
};

}


namespace {

void prepare_matvec_sparse(const DOCIWfn &wfn, SparseOp& op, const double *h, const double *v, const double *w) {
    int_t idet, jdet, i, j, k, l;
    double val1, val2;
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    // prepare sparse matrix
    op.data.resize(0);
    op.indices.resize(0);
    op.indptr.resize(0);
    op.data.reserve(wfn.ndet + 1);
    op.indices.reserve(wfn.ndet + 1);
    op.indptr.reserve(wfn.ndet + 1);
    op.indptr.push_back(0);
    // iterate over determinants
    for (idet = 0; idet < wfn.ndet; ++idet) {
        wfn.copy_det(idet, &det[0]);
        fill_occs(wfn.nword, &det[0], &occs[0]);
        fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
        val1 = 0.0;
        val2 = 0.0;
        // diagonal elements
        for (i = 0; i < wfn.nocc; ++i) {
            k = occs[i];
            val1 += v[k * (wfn.nbasis + 1)];
            val2 += h[k];
            for (j = i + 1; j < wfn.nocc; ++j)
                val2 += w[k * wfn.nbasis + occs[j]];
            // pair excitation elements
            for (j = 0; j < wfn.nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                jdet = wfn.index_det(&det[0]);
                wfn.copy_det(idet, &det[0]);
                // check if excited determinant is in wfn
                if (jdet != -1) {
                    // add pair-excited element to sparse matrix
                    op.data.push_back(v[k * wfn.nbasis + l]);
                    op.indices.push_back(jdet);
                }
            }
        }
        // add diagonal element to sparse matrix
        op.data.push_back(val1 + val2 * 2);
        op.indices.push_back(idet);
        // add pointer to next row's indices
        op.indptr.push_back(op.indices.size());
    }
    op.data.shrink_to_fit();
    op.indices.shrink_to_fit();
    op.indptr.shrink_to_fit();
}

}


namespace {

void prepare_matvec_direct(const DOCIWfn &wfn, DirectOp &op, const double *h, const double *v, const double *w) {
    op.data.resize(wfn.ndet);
    int_t nthread = omp_get_max_threads();
    int_t chunksize = wfn.ndet / nthread + ((wfn.ndet % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t idet, i, j, k;
        int_t chunk = omp_get_thread_num();
        int_t start = chunk * chunksize;
        int_t end = (start + chunksize < wfn.ndet) ? start + chunksize : wfn.ndet;
        double val1, val2;
        // iterate over determinants
        std::vector<int_t> occs(wfn.nocc);
        for (idet = start; idet < end; ++idet) {
            fill_occs(wfn.nword, &wfn.dets[idet * wfn.nword], &occs[0]);
            val1 = 0.0;
            val2 = 0.0;
            // diagonal elements
            for (i = 0; i < wfn.nocc; ++i) {
                k = occs[i];
                val1 += v[k * (wfn.nbasis + 1)];
                val2 += h[k];
                for (j = i + 1; j < wfn.nocc; ++j)
                    val2 += w[k * wfn.nbasis + occs[j]];
            }
            op.data[idet] = val1 + val2 * 2;
        }
    }
}

}


namespace {

void SparseOp::perform_op(const double *x, double *y) {
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t i, j;
        int_t chunk = omp_get_thread_num();
        int_t istart = chunk * chunksize;
        int_t iend = (istart + chunksize < nrow) ? istart + chunksize : nrow;
        int_t jstart, jend = indptr[istart];
        double val;
        for (i = istart; i < iend; ++i) {
            jstart = jend;
            jend = indptr[i + 1];
            val = 0.0;
            for (j = jstart; j < jend; ++j)
                val += data[j] * x[indices[j]];
            y[i] = val;
        }
    }
}

}


namespace {

void DirectOp::perform_op(const double *x, double *y) {
    int_t nthread = omp_get_max_threads();
    int_t chunksize = wfn.ndet / nthread + ((wfn.ndet % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t idet, jdet, i, j, k, l;
        int_t chunk = omp_get_thread_num();
        int_t start = chunk * chunksize;
        int_t end = (start + chunksize < wfn.ndet) ? start + chunksize : wfn.ndet;
        double val;
        // iterate over determinants
        std::vector<uint_t> det(wfn.nword);
        std::vector<int_t> occs(wfn.nocc);
        std::vector<int_t> virs(wfn.nvir);
        for (idet = start; idet < end; ++idet) {
            wfn.copy_det(idet, &det[0]);
            fill_occs(wfn.nword, &det[0], &occs[0]);
            fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
            // diagonal elements
            val = data[idet] * x[idet];
            for (i = 0; i < wfn.nocc; ++i) {
                k = occs[i];
                // pair excitation elements
                for (j = 0; j < wfn.nvir; ++j) {
                    l = virs[j];
                    excite_det(k, l, &det[0]);
                    jdet = wfn.index_det(&det[0]);
                    wfn.copy_det(idet, &det[0]);
                    // check if excited determinant is in wfn
                    if (jdet != -1) val += v[k * wfn.nbasis + l] * x[jdet];
                }
            }
            y[idet] = val;
        }
    }
}

}


void solve_sparse(const DOCIWfn &wfn, const double *h, const double *v, const double *w, const double *coeffs,
    const int_t n, const int_t ncv, const int_t maxit, const double tol, double *evals, double *evecs) {
    SparseOp op(wfn);
    prepare_matvec_sparse(wfn, op, h, v, w);
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, SparseOp> eigs(&op, n, ncv);
    eigs.init(coeffs);
    eigs.compute(maxit, tol, Spectra::SMALLEST_ALGE);
    if (eigs.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("Did not converge");
    Eigen::Map<Eigen::VectorXd> eigenvalues(evals, n);
    Eigen::Map<Eigen::MatrixXd> eigenvectors(evecs, wfn.ndet, n);
    eigenvalues = eigs.eigenvalues();
    eigenvectors = eigs.eigenvectors();
}


void solve_direct(const DOCIWfn &wfn, const double *h, const double *v, const double *w, const double *coeffs,
    const int_t n, const int_t ncv, const int_t maxit, const double tol, double *evals, double *evecs) {
    DirectOp op(wfn, v);
    prepare_matvec_direct(wfn, op, h, v, w);
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, DirectOp> eigs(&op, n, ncv);
    eigs.init(coeffs);
    eigs.compute(maxit, tol, Spectra::SMALLEST_ALGE);
    if (eigs.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("Did not converge");
    Eigen::Map<Eigen::VectorXd> eigenvalues(evals, n);
    Eigen::Map<Eigen::MatrixXd> eigenvectors(evecs, wfn.ndet, n);
    eigenvalues = eigs.eigenvalues();
    eigenvectors = eigs.eigenvectors();
}


int_t binomial(int_t n, int_t k) {
    if (k == 0) return 1;
    else if (k == 1) return n;
    else if (k >= n) return (k == n);
    if (k > n / 2) k = n - k;
    int_t binom = 1;
    for (int_t d = 1; d <= k; ++d) {
        if (binom >= DOCI_INT_MAX / n)
            throw std::runtime_error("Binomial computation overflowed");
        binom = binom * n-- / d;
    }
    return binom;
}


int_t nword_det(const int_t n) {
    return n / DOCI_UINT_SIZE + ((n % DOCI_UINT_SIZE) ? 1 : 0);
}


void fill_det(const int_t nocc, const int_t *occs, uint_t *det) {
    int_t j;
    for (int_t i = 0; i < nocc; ++i) {
        j = occs[i];
        det[j / DOCI_UINT_SIZE] |= DOCI_UINT_ONE << (j % DOCI_UINT_SIZE);
    }
}


void fill_occs(const int_t nword, const uint_t *det, int_t *occs) {
    int_t p, j = 0, offset = 0;
    uint_t word;
    for (int_t i = 0; i < nword; ++i) {
        word = det[i];
        while (word) {
            p = DOCI_CTZ(word);
            occs[j++] = p + offset;
            word &= ~(DOCI_UINT_ONE << p);
        }
        offset += DOCI_UINT_SIZE;
    }
}


void fill_virs(const int_t nword, const int_t nbasis, const uint_t *det, int_t *virs) {
    int_t p, n = nbasis, j = 0, offset = 0;
    uint_t word, mask;
    for (int_t i = 0; i < nword; ++i) {
        mask = (n < DOCI_UINT_SIZE) ? ((DOCI_UINT_ONE << n) - 1) : DOCI_UINT_MAX;
        word = det[i] ^ mask;
        while (word) {
            p = DOCI_CTZ(word);
            virs[j++] = p + offset;
            word &= ~(DOCI_UINT_ONE << p);
        }
        offset += DOCI_UINT_SIZE;
        n -= DOCI_UINT_SIZE;
    }
}


void excite_det(const int_t i, const int_t a, uint_t *det) {
    det[i / DOCI_UINT_SIZE] &= ~(DOCI_UINT_ONE << (i % DOCI_UINT_SIZE));
    det[a / DOCI_UINT_SIZE] |= DOCI_UINT_ONE << (a % DOCI_UINT_SIZE);
}


void setbit_det(const int_t i, uint_t *det) {
    det[i / DOCI_UINT_SIZE] |= DOCI_UINT_ONE << (i % DOCI_UINT_SIZE);
}


void clearbit_det(const int_t i, uint_t *det) {
    det[i / DOCI_UINT_SIZE] &= ~(DOCI_UINT_ONE << (i % DOCI_UINT_SIZE));
}


int_t popcnt_det(const int_t nword, const uint_t *det) {
    int_t popcnt = 0;
    for (int_t i = 0; i < nword; ++i)
        popcnt += DOCI_POPCNT(det[i]);
    return popcnt;
}


int_t ctz_det(const int_t nword, const uint_t *det) {
    uint_t word;
    for (int_t i = 0; i < nword; ++i) {
        word = det[i];
        if (word) return DOCI_CTZ(word) + i * DOCI_UINT_SIZE;
    }
    return 0;
}


int_t hash_det(const int_t nbasis, const int_t nocc, const uint_t *det) {
    int_t k = 0, binom = 1, hash = 0;
    for (int_t i = 0; i < nbasis; ++i) {
        if (k == nocc) break;
        else if (det[i / DOCI_UINT_SIZE] & (DOCI_UINT_ONE << (i % DOCI_UINT_SIZE))) {
            ++k;
            binom = (k == i) ? 1 : binom * i / k;
            hash += binom;
        }
        else binom = (k >= i) ? 1 : binom * i / (i - k);
    }
    return hash;
}


} // namespace doci
