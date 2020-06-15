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


SparseOp::SparseOp() {
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


void SparseOp::solve(const double *coeffs, const int_t n, const int_t ncv, const int_t maxit, const double tol,
    double *evals, double *evecs) {
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


void SparseOp::init(const DOCIWfn &wfn, const double *h, const double *v, const double *w, const int_t nrow_) {
    int_t idet, jdet, i, j, k, l;
    double val1, val2;
    // set nrow <= ncol (value <1 defaults to nrow = ncol = wfn.ndet)
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
    // prepare sparse matrix
    data.resize(0);
    indices.resize(0);
    indptr.resize(0);
    data.reserve(ncol + 1);
    indices.reserve(ncol + 1);
    indptr.reserve(nrow + 1);
    indptr.push_back(0);
    // working vectors
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    // compute elements
    for (idet = 0; idet < nrow; ++idet) {
        // fill working vectors
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
                    data.push_back(v[k * wfn.nbasis + l]);
                    indices.push_back(jdet);
                }
            }
        }
        // add diagonal element to sparse matrix
        data.push_back(val1 + val2 * 2);
        indices.push_back(idet);
        // add pointer to next row's indices
        indptr.push_back(indices.size());
    }
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


void SparseOp::init(const FullCIWfn &wfn, const double *one_mo, const double *two_mo, const int_t nrow_) {
    int_t idet, jdet, i, j, k, l;
    // set nrow <= ncol (value <1 defaults to nrow = ncol = wfn.ndet)
    nrow = (nrow_ > 0) ? nrow_ : wfn.ndet;
    ncol = wfn.ndet;
    // prepare sparse matrix
    data.resize(0);
    indices.resize(0);
    indptr.resize(0);
    data.reserve(ncol + 1);
    indices.reserve(ncol + 1);
    indptr.reserve(nrow + 1);
    indptr.push_back(0);
    // compute elements
    for (idet = 0; idet < nrow; ++idet) {
        //
        // 0-0 excitation elements
        //
        for (i = 0; i < wfn.nocc_up; ++i) {
            for (j = 0; j < wfn.nvir_up; ++j) {
                //
                // 1-0 excitation elements
                //
                for (k = i + 1; k < wfn.nocc_up; ++k) {
                    for (l = j + 1; l < wfn.nvir_up; ++l) {
                        //
                        // 2-0 excitation elements
                        //
                    }
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    for (l = 0; j < wfn.nvir_dn; ++j) {
                        //
                        // 1-1 excitation elements
                        //
                    }
                }
            }
        }
        for (i = 0; i < wfn.nocc_dn; ++i) {
            for (j = 0; j < wfn.nvir_dn; ++j) {
                //
                // 0-1 excitation elements
                //
                for (k = i + 1; k < wfn.nocc_dn; ++k) {
                    for (l = j + 1; l < wfn.nvir_dn; ++l) {
                        //
                        // 0-2 excitation elements
                        //
                    }
                }
            }
        }
        // add pointer to next row's indices
        indptr.push_back(indices.size());
    }
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
}


} // namespace pyci
