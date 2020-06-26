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

#include <cassert>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include <pyci/doci.h>
#include <pyci/fullci.h>
#include <pyci/solve.h>

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE pyci::int_t
#include <Eigen/Core>

#include <Spectra/SymEigsSolver.h>


namespace pyci {


SparseOp::SparseOp() : nrow(0), ncol(0) {
    return;
}


SparseOp::SparseOp(const DOCIWfn &wfn, const double *h, const double *v, const double *w, const int_t nrow_) {
    init(wfn, h, v, w, nrow_);
}


SparseOp::SparseOp(const FullCIWfn &wfn, const double *one_mo, const double *two_mo, const int_t nrow_) {
    init(wfn, one_mo, two_mo, nrow_);
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
    const double tol, double *evals, double *evecs) {
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, SparseOp> eigs(this, n, ncv);
    eigs.init(coeffs);
    eigs.compute(maxit, tol, Spectra::SMALLEST_ALGE);
    if (eigs.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("Did not converge");
    Eigen::Map<Eigen::VectorXd> eigenvalues(evals, n);
    Eigen::Map<Eigen::MatrixXd> eigenvectors(evecs, ncol, n);
    eigenvalues = eigs.eigenvalues();
    eigenvectors = eigs.eigenvectors();
}


void SparseOp::condense_threads(std::vector<SparseOp> &ops) {
    // make one large SparseOp
    int_t i, istart, iend, val_ptr;
    for (std::vector<SparseOp>::iterator it = ops.begin(); it != ops.end(); ++it) {
        val_ptr = indices.size();
        if (!(it->nrow)) continue;
        // copy over data array
        istart = data.size();
        iend = it->data.size();
        data.resize(istart + iend);
        std::memcpy(&data[istart], &(it->data[0]), sizeof(double) * iend);
        it->data.resize(0);
        // copy over indices array
        istart = indices.size();
        iend = it->indices.size();
        indices.resize(istart + iend);
        std::memcpy(&indices[istart], &(it->indices[0]), sizeof(int_t) * iend);
        it->indices.resize(0);
        // copy over indptr array
        iend = it->indptr.size() - 1;
        for (i = 0; i < iend; ++i)
            indptr.push_back(it->indptr[i] + val_ptr);
        it->indptr.resize(0);
    }
    indptr.push_back(indices.size());
    // finalize vectors
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


void SparseOp::init(const DOCIWfn &wfn, const double *h, const double *v, const double *w, const int_t nrow_) {
    // set attributes
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
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
        ops[ithread].init_thread(wfn, h, v, w, istart, iend);
    }
    condense_threads(ops);
}


void SparseOp::init(const FullCIWfn &wfn, const double *one_mo, const double *two_mo, const int_t nrow_) {
    // set attributes
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
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
        ops[ithread].init_thread(wfn, one_mo, two_mo, istart, iend);
    }
    condense_threads(ops);
}


void SparseOp::init_thread(const DOCIWfn &wfn, const double *h, const double *v, const double *w,
    const int_t istart, const int_t iend) {
    // prepare sparse matrix
    indptr.push_back(0);
    if (istart >= iend) return;
    data.reserve(wfn.ndet + 1);
    indices.reserve(wfn.ndet + 1);
    indptr.reserve(iend - istart + 1);
    // set nrow
    nrow = istart - iend;
    // prepare working vectors
    std::vector<uint_t> vdet(wfn.nword);
    std::vector<int_t> voccs(wfn.nocc);
    std::vector<int_t> vvirs(wfn.nvir);
    uint_t *det = &vdet[0];
    int_t *occs = &voccs[0];
    int_t *virs = &vvirs[0];
    // loop over determinants
    int_t i, j, k, l, jdet;
    double val1, val2;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        std::memcpy(det, &wfn.dets[idet * wfn.nword], sizeof(uint_t) * wfn.nword);
        fill_occs(wfn.nword, det, occs);
        fill_virs(wfn.nword, wfn.nbasis, det, virs);
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
                // compute single/"pair" excitation elements
                l = virs[j];
                excite_det(k, l, det);
                jdet = wfn.index_det(det);
                // check if excited determinant is in wfn
                if (jdet != -1) {
                    // add single/"pair" excitation matrix element
                    data.push_back(v[k * wfn.nbasis + l]);
                    indices.push_back(jdet);
                }
                excite_det(l, k, det);
            }
        }
        // add diagonal element to matrix
        data.push_back(val1 + val2 * 2);
        indices.push_back(idet);
        // add pointer to next row's indices
        indptr.push_back(indices.size());
    }
    // finalize vectors
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


void SparseOp::init_thread(const FullCIWfn &wfn, const double *one_mo, const double *two_mo,
    const int_t istart, const int_t iend) {
    // prepare sparse matrix
    indptr.push_back(0);
    if (istart >= iend) return;
    data.reserve(wfn.ndet + 1);
    indices.reserve(wfn.ndet + 1);
    indptr.reserve(iend - istart + 1);
    // set nrow
    nrow = istart - iend;
    // working vectors
    std::vector<uint_t> vrdet(wfn.nword2), vdet(wfn.nword2);
    std::vector<int_t> voccs_up(wfn.nocc_up), voccs_dn(wfn.nocc_dn);
    std::vector<int_t> vvirs_up(wfn.nvir_up), vvirs_dn(wfn.nvir_dn);
    uint_t *rdet_up = &vrdet[0];
    uint_t *rdet_dn = rdet_up + wfn.nword;
    uint_t *det_up = &vdet[0];
    uint_t *det_dn = det_up + wfn.nword;
    int_t *occs_up = &voccs_up[0];
    int_t *occs_dn = &voccs_dn[0];
    int_t *virs_up = &vvirs_up[0];
    int_t *virs_dn = &vvirs_dn[0];
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, jdet, ioffset, koffset;
    int_t rank_up_ref, rank_dn_ref, rank_up, sign_up;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        wfn.copy_det(idet, rdet_up);
        std::memcpy(det_up, rdet_up, sizeof(uint_t) * wfn.nword2);
        fill_occs(wfn.nword, rdet_up, occs_up);
        fill_occs(wfn.nword, rdet_dn, occs_dn);
        fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
        fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
        rank_up_ref = rank_det(n1, wfn.nocc_up, rdet_up) * wfn.maxdet_dn;
        rank_dn_ref = rank_det(n1, wfn.nocc_dn, rdet_dn);
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
                rank_up = rank_det(n1, wfn.nocc_up, det_up) * wfn.maxdet_dn;
                jdet = wfn.index_det_from_rank(rank_up + rank_dn_ref);
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
                    data.push_back(sign_up * val1);
                    indices.push_back(jdet);
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
                        jdet = wfn.index_det_from_rank(rank_up + rank_det(n1, wfn.nocc_dn, det_dn));
                        // check if 1-1 excited determinant is in wfn
                        if (jdet != -1) {
                            // add 1-1 matrix element
                            data.push_back(
                                sign_up
                              * phase_single_det(wfn.nword, kk, ll, rdet_dn)
                              * two_mo[koffset + n1 * jj + ll]
                            );
                            indices.push_back(jdet);
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
                        jdet = wfn.index_det_from_rank(rank_det(n1, wfn.nocc_up, det_up) * wfn.maxdet_dn
                                                     + rank_dn_ref);
                        // check if 2-0 excited determinant is in wfn
                        if (jdet != -1) {
                            // add 2-0 matrix element
                            data.push_back(
                                phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up)
                              * (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj])
                            );
                            indices.push_back(jdet);
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
                jdet = wfn.index_det_from_rank(rank_up_ref + rank_det(n1, wfn.nocc_dn, det_dn));
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
                    data.push_back(phase_single_det(wfn.nword, ii, jj, rdet_dn) * val1);
                    indices.push_back(jdet);
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
                        jdet = wfn.index_det_from_rank(rank_up_ref + rank_det(n1, wfn.nocc_dn, det_dn));
                        // check if excited determinant is in wfn
                        if (jdet != -1) {
                            // add 0-2 matrix element
                            data.push_back(
                                phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn)
                              * (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj])
                            );
                            indices.push_back(jdet);
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
        // add diagonal element to matrix
        data.push_back(val2);
        indices.push_back(idet);
        // add pointer to next row's indices
        indptr.push_back(indices.size());
    }
    // finalize vectors
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


} // namespace pyci
