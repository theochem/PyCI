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

#include <cstring>

#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include <pyci.h>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE pyci::int_t
#include <Eigen/Core>

#include <Spectra/SymEigsSolver.h>


namespace pyci {


namespace { // anonymous


void init_doci_run_thread(SparseOp &op, const OneSpinWfn &wfn, const double *h, const double *v, const double *w,
    const int_t istart, const int_t iend) {
    // prepare sparse matrix
    if (istart >= iend)
        return;
    op.data.reserve(wfn.ndet + 1);
    op.indices.reserve(wfn.ndet + 1);
    op.indptr.reserve(iend - istart + 1);
    op.indptr.push_back(0);
    // set nrow and ncol
    op.nrow = iend - istart;
    // prepare working vectors
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    // loop over determinants
    int_t i, j, k, l, jdet;
    double val1, val2;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        wfn.copy_det(idet, &det[0]);
        fill_occs(wfn.nword, &det[0], &occs[0]);
        fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
        val1 = 0.0;
        val2 = 0.0;
        // loop over occupied indices
        for (i = 0; i < wfn.nocc; ++i) {
            k = occs[i];
            // compute part of diagonal matrix element
            val1 += v[k * (wfn.nbasis + 1)];
            val2 += h[k];
            for (j = i + 1; j < wfn.nocc; ++j)
                val2 += w[k * wfn.nbasis + occs[j]];
            // loop over virtual indices
            for (j = 0; j < wfn.nvir; ++j) {
                // compute single/"pair"-excited elements
                l = virs[j];
                excite_det(k, l, &det[0]);
                jdet = wfn.index_det(&det[0]);
                // check if excited determinant is in wfn
                if (jdet != -1) {
                    // add single/"pair"-excited matrix element
                    op.data.push_back(v[k * wfn.nbasis + l]);
                    op.indices.push_back(jdet);
                }
                excite_det(l, k, &det[0]);
            }
        }
        // add diagonal element to matrix
        op.data.push_back(val1 + val2 * 2);
        op.indices.push_back(idet);
        // add pointer to next row's indices
        op.indptr.push_back(op.indices.size());
    }
    // finalize vectors
    op.data.shrink_to_fit();
    op.indices.shrink_to_fit();
    op.indptr.shrink_to_fit();
}


void init_fullci_run_thread(SparseOp &op, const TwoSpinWfn &wfn, const double *one_mo, const double *two_mo,
    const int_t istart, const int_t iend) {
    // prepare sparse matrix
    if (istart >= iend)
        return;
    op.data.reserve(wfn.ndet + 1);
    op.indices.reserve(wfn.ndet + 1);
    op.indptr.reserve(iend - istart + 1);
    op.indptr.push_back(0);
    // set nrow and ncol
    op.nrow = iend - istart;
    // working vectors
    std::vector<uint_t> det(wfn.nword2);
    std::vector<int_t> occs_up(wfn.nocc_up);
    std::vector<int_t> occs_dn(wfn.nocc_dn);
    std::vector<int_t> virs_up(wfn.nvir_up);
    std::vector<int_t> virs_dn(wfn.nvir_dn);
    const uint_t *rdet_up, *rdet_dn;
    uint_t *det_up = &det[0], *det_dn = &det[wfn.nword];
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, jdet, ioffset, koffset, sign_up;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        rdet_up = &wfn.dets[idet * wfn.nword2];
        rdet_dn = rdet_up + wfn.nword;
        std::memcpy(det_up, rdet_up, sizeof(uint_t) * wfn.nword2);
        fill_occs(wfn.nword, rdet_up, &occs_up[0]);
        fill_occs(wfn.nword, rdet_dn, &occs_dn[0]);
        fill_virs(wfn.nword, wfn.nbasis, rdet_up, &virs_up[0]);
        fill_virs(wfn.nword, wfn.nbasis, rdet_dn, &virs_dn[0]);
        val2 = 0.0;
        // loop over spin-up occupied indices
        for (i = 0; i < wfn.nocc_up; ++i) {
            ii = occs_up[i];
            ioffset = n3 * ii;
            // compute part of diagonal matrix element
            val2 += one_mo[(n1 + 1) * ii];
            for (k = i + 1; k < wfn.nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                val2 += two_mo[koffset + n1 * ii + kk] - two_mo[koffset + n1 * kk + ii];
            }
            for (k = 0; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                val2 += two_mo[ioffset + n2 * kk + n1 * ii + kk];
            }
            // loop over spin-up virtual indices
            for (j = 0; j < wfn.nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
                jdet = wfn.index_det(det_up);
                // check if 1-0 excited determinant is in wfn
                if (jdet != -1) {
                    // compute 1-0 matrix element
                    val1 = one_mo[n1 * ii + jj];
                    for (k = 0; k < wfn.nocc_up; ++k) {
                        kk = occs_up[k];
                        koffset = ioffset + n2 * kk;
                        val1 += two_mo[koffset + n1 * jj + kk] - two_mo[koffset + n1 * kk + jj];
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        kk = occs_dn[k];
                        val1 += two_mo[ioffset + n2 * kk + n1 * jj + kk];
                    }
                    // add 1-0 matrix element
                    op.data.push_back(sign_up * val1);
                    op.indices.push_back(jdet);
                }
                // loop over spin-down occupied indices
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = 0; l < wfn.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 1-1 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = wfn.index_det(det_up);
                        // check if 1-1 excited determinant is in wfn
                        if (jdet != -1) {
                            // add 1-1 matrix element
                            op.data.push_back(
                                sign_up * phase_single_det(wfn.nword, kk, ll, rdet_dn)
                              * two_mo[koffset + n1 * jj + ll]
                            );
                            op.indices.push_back(jdet);
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                // loop over spin-up occupied indices
                for (k = i + 1; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < wfn.nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        jdet = wfn.index_det(det_up);
                        // check if 2-0 excited determinant is in wfn
                        if (jdet != -1) {
                            // add 2-0 matrix element
                            op.data.push_back(
                                phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up)
                              * (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj])
                            );
                            op.indices.push_back(jdet);
                        }
                        excite_det(ll, kk, det_up);
                    }
                }
                excite_det(jj, ii, det_up);
            }
        }
        // loop over spin-down occupied indices
        for (i = 0; i < wfn.nocc_dn; ++i) {
            ii = occs_dn[i];
            ioffset = n3 * ii;
            // compute part of diagonal matrix element
            val2 += one_mo[(n1 + 1) * ii];
            for (k = i + 1; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                val2 += two_mo[koffset + n1 * ii + kk] - two_mo[koffset + n1 * kk + ii];
            }
            // loop over spin-down virtual indices
            for (j = 0; j < wfn.nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                jdet = wfn.index_det(det_up);
                // check if 0-1 excited determinant is in wfn
                if (jdet != -1) {
                    // compute 0-1 matrix element
                    val1 = one_mo[n1 * ii + jj];
                    for (k = 0; k < wfn.nocc_up; ++k) {
                        kk = occs_up[k];
                        val1 += two_mo[ioffset + n2 * kk + n1 * jj + kk];
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        kk = occs_dn[k];
                        koffset = ioffset + n2 * kk;
                        val1 += two_mo[koffset + n1 * jj + kk] - two_mo[koffset + n1 * kk + jj];
                    }
                    // add 0-1 matrix element
                    op.data.push_back(phase_single_det(wfn.nword, ii, jj, rdet_dn) * val1);
                    op.indices.push_back(jdet);
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < wfn.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = wfn.index_det(det_up);
                        // check if excited determinant is in wfn
                        if (jdet != -1) {
                            // add 0-2 matrix element
                            op.data.push_back(
                                phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn)
                              * (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj])
                            );
                            op.indices.push_back(jdet);
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
        // add diagonal element to matrix
        op.data.push_back(val2);
        op.indices.push_back(idet);
        // add pointer to next row's indices
        op.indptr.push_back(op.indices.size());
    }
    // finalize vectors
    op.data.shrink_to_fit();
    op.indices.shrink_to_fit();
    op.indptr.shrink_to_fit();
}


void init_genci_run_thread(SparseOp &op, const OneSpinWfn &wfn, const double *one_mo, const double *two_mo,
    const int_t istart, const int_t iend) {
    // prepare sparse matrix
    if (istart >= iend)
        return;
    op.data.reserve(wfn.ndet + 1);
    op.indices.reserve(wfn.ndet + 1);
    op.indptr.reserve(iend - istart + 1);
    op.indptr.push_back(0);
    // set nrow and ncol
    op.nrow = iend - istart;
    // working vectors
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    const uint_t *rdet;
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, jdet, ioffset, koffset;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        rdet = &wfn.dets[idet * wfn.nword];
        std::memcpy(&det[0], rdet, sizeof(uint_t) * wfn.nword);
        fill_occs(wfn.nword, rdet, &occs[0]);
        fill_virs(wfn.nword, wfn.nbasis, rdet, &virs[0]);
        val2 = 0.0;
        // loop over occupied indices
        for (i = 0; i < wfn.nocc; ++i) {
            ii = occs[i];
            ioffset = n3 * ii;
            // compute part of diagonal matrix element
            val2 += one_mo[(n1 + 1) * ii];
            for (k = i + 1; k < wfn.nocc; ++k) {
                kk = occs[k];
                koffset = ioffset + n2 * kk;
                val2 += two_mo[koffset + n1 * ii + kk] - two_mo[koffset + n1 * kk + ii];
            }
            // loop over virtual indices
            for (j = 0; j < wfn.nvir; ++j) {
                jj = virs[j];
                // single excitation elements
                excite_det(ii, jj, &det[0]);
                jdet = wfn.index_det(&det[0]);
                // check if singly-excited determinant is in wfn
                if (jdet != -1) {
                    // compute single excitation matrix element
                    val1 = one_mo[n1 * ii + jj];
                    for (k = 0; k < wfn.nocc; ++k) {
                        kk = occs[k];
                        koffset = ioffset + n2 * kk;
                        val1 += two_mo[koffset + n1 * jj + kk] - two_mo[koffset + n1 * kk + jj];
                    }
                    // add single excitation matrix element
                    op.data.push_back(phase_single_det(wfn.nword, ii, jj, rdet) * val1);
                    op.indices.push_back(jdet);
                }
                // loop over occupied indices
                for (k = i + 1; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    // loop over virtual indices
                    for (l = j + 1; l < wfn.nvir; ++l) {
                        ll = virs[l];
                        // double excitation elements
                        excite_det(kk, ll, &det[0]);
                        jdet = wfn.index_det(&det[0]);
                        // check if double excited determinant is in wfn
                        if (jdet != -1) {
                            // add double matrix element
                            op.data.push_back(
                                phase_double_det(wfn.nword, ii, kk, jj, ll, rdet)
                              * (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj])
                            );
                            op.indices.push_back(jdet);
                        }
                        excite_det(ll, kk, &det[0]);
                    }
                }
                excite_det(jj, ii, &det[0]);
            }
        }
        // add diagonal element to matrix
        op.data.push_back(val2);
        op.indices.push_back(idet);
        // add pointer to next row's indices
        op.indptr.push_back(op.indices.size());
    }
    // finalize vectors
    op.data.shrink_to_fit();
    op.indices.shrink_to_fit();
    op.indptr.shrink_to_fit();
}


void init_condense_thread(SparseOp &op, SparseOp &thread_op, const int_t ithread) {
    if (ithread == 0) {
        std::swap(op.data, thread_op.data);
        std::swap(op.indices, thread_op.indices);
        std::swap(op.indptr, thread_op.indptr);
        op.indptr.pop_back();
        return;
    } else if (thread_op.nrow == 0)
        return;
    int_t indptr_val = op.indices.size();
    // copy over data array
    int_t istart = op.data.size();
    int_t iend = thread_op.data.size();
    op.data.resize(istart + iend);
    std::memcpy(&op.data[istart], &thread_op.data[0], sizeof(double) * iend);
    thread_op.data.resize(0);
    thread_op.data.shrink_to_fit();
    // copy over indices array
    istart = op.indices.size();
    iend = thread_op.indices.size();
    op.indices.resize(istart + iend);
    std::memcpy(&op.indices[istart], &thread_op.indices[0], sizeof(int_t) * iend);
    thread_op.indices.resize(0);
    thread_op.indices.shrink_to_fit();
    // copy over indptr array
    iend = thread_op.indptr.size() - 1;
    for (int_t i = 0; i < iend; ++i)
        op.indptr.push_back(thread_op.indptr[i] + indptr_val);
    thread_op.indptr.resize(0);
    thread_op.indptr.shrink_to_fit();
}


} // namespace // anonymous


SparseOp::SparseOp(void) {
    return;
}


SparseOp::SparseOp(SparseOp &&op) noexcept
    : nrow(std::exchange(op.nrow, 0)), ncol(std::exchange(op.ncol, 0)),
      size(std::exchange(op.size, 0)), ecore(std::exchange(op.ecore, 0)), 
      data(std::move(op.data)), indices(std::move(op.indices)), indptr(std::move(op.indptr)) {
}


const double * SparseOp::data_ptr(const int_t index) const {
    return &data[index];
}


const int_t * SparseOp::indices_ptr(const int_t index) const {
    return &indices[index];
}


const int_t * SparseOp::indptr_ptr(const int_t index) const {
    return &indptr[index];
}


void SparseOp::perform_op(const double *x, double *y) const {
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t i, j;
        int_t istart = omp_get_thread_num() * chunksize;
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


void SparseOp::solve(const double *coeffs, const int_t n, const int_t ncv, const int_t maxit,
    const double tol, double *evals, double *evecs) const {
    if (nrow == 1) {
        *evals = data[0];
        *evecs = 1.0;
        return;
    }
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, const SparseOp> eigs(this, n, ncv);
    eigs.init(coeffs);
    eigs.compute(maxit, tol, Spectra::SMALLEST_ALGE);
    if (eigs.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("did not converge");
    Eigen::Map<Eigen::VectorXd> eigenvalues(evals, n);
    Eigen::Map<Eigen::MatrixXd> eigenvectors(evecs, ncol, n);
    eigenvalues = eigs.eigenvalues();
    for (int_t i = 0; i < n; ++i)
        evals[i] += ecore;
    eigenvectors = eigs.eigenvectors();
}


void SparseOp::init_doci(const OneSpinWfn &wfn, const double ecore_, const double *h,
        const double *v, const double *w, const int_t nrow_) {
    // set attributes
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
    ecore = ecore_;
    // prepare vectors
    data.resize(0);
    indices.resize(0);
    indptr.resize(0);
    // do computations in chunks by making smaller SparseOps in parallel
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
    std::vector<SparseOp> ops(nthread);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < nrow) ? istart + chunksize : nrow;
        init_doci_run_thread(ops[ithread], wfn, h, v, w, istart, iend);
        // construct larger SparseOp (this instance) from chunks
#pragma omp for ordered schedule(static,1)
        for (int_t t = 0; t < nthread; ++t)
#pragma omp ordered
            init_condense_thread(*this, ops[t], t);
    }
    // finalize vectors
    size = indices.size();
    indptr.push_back(size);
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


void SparseOp::init_fullci(const TwoSpinWfn &wfn, const double ecore_, const double *one_mo,
        const double *two_mo, const int_t nrow_) {
    // set attributes
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
    ecore = ecore_;
    // prepare vectors
    data.resize(0);
    indices.resize(0);
    indptr.resize(0);
    // do computations in chunks by making smaller SparseOps in parallel
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
    std::vector<SparseOp> ops(nthread);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < nrow) ? istart + chunksize : nrow;
        init_fullci_run_thread(ops[ithread], wfn, one_mo, two_mo, istart, iend);
        // construct larger SparseOp (this instance) from chunks
#pragma omp for ordered schedule(static,1)
        for (int_t t = 0; t < nthread; ++t)
#pragma omp ordered
            init_condense_thread(*this, ops[t], t);
    }
    // finalize vectors
    size = indices.size();
    indptr.push_back(size);
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


void SparseOp::init_genci(const OneSpinWfn &wfn, const double ecore_, const double *one_mo,
        const double *two_mo, const int_t nrow_) {
    // set attributes
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
    ecore = ecore_;
    // prepare vectors
    data.resize(0);
    indices.resize(0);
    indptr.resize(0);
    // do computations in chunks by making smaller SparseOps in parallel
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
    std::vector<SparseOp> ops(nthread);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < nrow) ? istart + chunksize : nrow;
        init_genci_run_thread(ops[ithread], wfn, one_mo, two_mo, istart, iend);
        // construct larger SparseOp (this instance) from chunks
#pragma omp for ordered schedule(static,1)
        for (int_t t = 0; t < nthread; ++t)
#pragma omp ordered
            init_condense_thread(*this, ops[t], t);
    }
    // finalize vectors
    size = indices.size();
    indptr.push_back(size);
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


} // namespace pyci
