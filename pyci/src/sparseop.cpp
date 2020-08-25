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

#include <algorithm>

#include <omp.h>

#include <pyci.h>
#include <sort_helper.h>

namespace pyci {

namespace {

void perform_op_thread(const double *, const int_t *, const int_t *, const double *, double *,
                       const int_t, const int_t);

void perform_op_cepa0_thread(const double *, const int_t *, const int_t *, const double *, double *,
                             const int_t, const int_t, const int_t, const double);

} // namespace

SparseOp::SparseOp(const SparseOp &op)
    : nrow(op.nrow), ncol(op.ncol), size(op.size), ecore(op.ecore), data(op.data),
      indices(op.indices), indptr(op.indptr) {
}

SparseOp::SparseOp(SparseOp &&op) noexcept
    : nrow(std::exchange(op.nrow, 0)), ncol(std::exchange(op.ncol, 0)),
      size(std::exchange(op.size, 0)), ecore(std::exchange(op.ecore, 0.0)),
      data(std::move(op.data)), indices(std::move(op.indices)), indptr(std::move(op.indptr)) {
}

SparseOp::SparseOp(const int_t rows, const int_t cols)
    : nrow(rows), ncol(cols), size(0), ecore(0.0) {
    indptr.push_back(0);
}

SparseOp::SparseOp(const Ham &ham, const DOCIWfn &wfn, const int_t rows, const int_t cols)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore) {
    init<DOCIWfn>(ham, wfn, rows, cols);
}

SparseOp::SparseOp(const Ham &ham, const FullCIWfn &wfn, const int_t rows, const int_t cols)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore) {
    init<FullCIWfn>(ham, wfn, rows, cols);
}

SparseOp::SparseOp(const Ham &ham, const GenCIWfn &wfn, const int_t rows, const int_t cols)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore) {
    init<GenCIWfn>(ham, wfn, rows, cols);
}

const double *SparseOp::data_ptr(const int_t index) const {
    return &data[index];
}

const int_t *SparseOp::indices_ptr(const int_t index) const {
    return &indices[index];
}

const int_t *SparseOp::indptr_ptr(const int_t index) const {
    return &indptr[index];
}

double SparseOp::get_element(const int_t i, const int_t j) const {
    const int_t *start = &indices[indptr[i]];
    const int_t *end = &indices[indptr[i + 1]];
    const int_t *e = std::find(start, end, j);
    return (e == end) ? 0.0 : data[indptr[i] + e - start];
}

void SparseOp::perform_op(const double *x, double *y) const {
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < nrow) ? start + chunksize : nrow;
        perform_op_thread(&data[0], &indptr[0], &indices[0], x, y, start, end);
    }
}

void SparseOp::perform_op_cepa0(const double *x, double *y, const int_t refind) const {
    int_t nthread = omp_get_max_threads();
    int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < nrow) ? start + chunksize : nrow;
        perform_op_cepa0_thread(&data[0], &indptr[0], &indices[0], x, y, start, end, refind,
                                get_element(refind, refind));
    }
}

void SparseOp::perform_op_transpose_cepa0(const double *x, double *y, const int_t refind) const {
    int_t i;
    for (i = 0; i < ncol; ++i)
        y[i] = 0;
    int_t j, jstart, jend = indptr[0];
    double h_refind = get_element(refind, refind);
    for (i = 0; i < nrow; ++i) {
        jstart = jend;
        jend = indptr[i + 1];
        for (j = jstart; j < jend; ++j) {
            if (indices[j] == i)
                y[indices[j]] += (h_refind - data[j]) * x[i];
            else
                y[indices[j]] -= data[j] * x[i];
        }
    }
    y[refind] = x[refind];
}

void SparseOp::rhs_cepa0(double *b, const int_t refind) const {
    int_t iend = indptr[refind + 1];
    for (int_t i = indptr[refind]; i != iend; ++i) {
        if (indices[i] >= nrow)
            break;
        b[indices[i]] = data[i];
    }
}

Array<double> SparseOp::py_matvec(Array<double> x) const {
    Array<double> y(nrow);
    perform_op(reinterpret_cast<const double *>(x.request().ptr),
               reinterpret_cast<double *>(y.request().ptr));
    return y;
}

Array<double> SparseOp::py_matvec_cepa0(Array<double> x, const int_t refind) const {
    Array<double> y(nrow);
    perform_op_cepa0(reinterpret_cast<const double *>(x.request().ptr),
                     reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

Array<double> SparseOp::py_rmatvec_cepa0(Array<double> x, const int_t refind) const {
    Array<double> y(ncol);
    perform_op_transpose_cepa0(reinterpret_cast<const double *>(x.request().ptr),
                               reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

Array<double> SparseOp::py_rhs_cepa0(const int_t refind) const {
    Array<double> y(nrow);
    rhs_cepa0(reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

pybind11::tuple SparseOp::py_get_shape(void) const {
    return pybind11::make_tuple(nrow, ncol);
}

template<class WfnType>
void SparseOp::init(const Ham &ham, const WfnType &wfn, const int_t rows, const int_t cols) {
    indptr.push_back(0);
#pragma omp parallel
    {
        int_t nthread = omp_get_max_threads();
        int_t chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
        int_t ithread = omp_get_thread_num();
        int_t start = ithread * chunksize;
        int_t end = (start + chunksize < nrow) ? start + chunksize : nrow;
        SparseOp op(end - start, ncol);
        int_t row = 0;
        std::vector<uint_t> det(wfn.nword2);
        std::vector<int_t> occs(wfn.nocc);
        std::vector<int_t> virs(wfn.nvir);
        for (int_t i = start; i < end; ++i) {
            op.init_thread_add_row(ham, wfn, i, &det[0], &occs[0], &virs[0]);
            op.init_thread_sort_row(row++);
        }
        op.size = op.indices.size();
#pragma omp for ordered schedule(static, 1)
        for (int_t i = 0; i < nthread; ++i)
#pragma omp ordered
            op.init_thread_condense(*this, i);
    }
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
    size = indices.size();
    ecore = ham.ecore;
}

void SparseOp::init_thread_sort_row(const int_t idet) {
    typedef std::sort_helper::value_iterator_t<double, int_t> iter;
    int_t start = indptr[idet], end = indptr[idet + 1];
    std::sort(iter(&data[start], &indices[start]), iter(&data[end], &indices[end]));
}

void SparseOp::init_thread_condense(SparseOp &op, const int_t ithread) {
    int_t i, start, end;
    if (!nrow)
        return;
    // copy over data array
    start = op.data.size();
    op.data.resize(start + size);
    std::memcpy(&op.data[start], &data[0], sizeof(double) * size);
    std::vector<double>().swap(data);
    // copy over indices array
    start = op.indices.size();
    op.indices.resize(start + size);
    std::memcpy(&op.indices[start], &indices[0], sizeof(int_t) * size);
    std::vector<int_t>().swap(indices);
    // copy over indptr array
    start = op.indptr.back();
    end = indptr.size();
    for (i = 1; i < end; ++i)
        op.indptr.push_back(indptr[i] + start);
    std::vector<int_t>().swap(indptr);
}

void SparseOp::init_thread_add_row(const Ham &ham, const DOCIWfn &wfn, const int_t idet,
                                   uint_t *det, int_t *occs, int_t *virs) {
    int_t i, j, k, l, jdet;
    double val1 = 0.0, val2 = 0.0;
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
    // loop over occupied indices
    for (i = 0; i < wfn.nocc_up; ++i) {
        k = occs[i];
        // compute part of diagonal matrix element
        val1 += ham.v[k * (wfn.nbasis + 1)];
        val2 += ham.h[k];
        for (j = i + 1; j < wfn.nocc_up; ++j)
            val2 += ham.w[k * wfn.nbasis + occs[j]];
        // loop over virtual indices
        for (j = 0; j < wfn.nvir_up; ++j) {
            // compute single/"pair"-excited elements
            l = virs[j];
            excite_det(k, l, det);
            jdet = wfn.index_det(det);
            // check if excited determinant is in wfn
            if ((jdet != -1) && (jdet < ncol)) {
                // add single/"pair"-excited matrix element
                data.push_back(ham.v[k * wfn.nbasis + l]);
                indices.push_back(jdet);
            }
            excite_det(l, k, det);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        data.push_back(val1 + val2 * 2);
        indices.push_back(idet);
    }
    // add pointer to next row's indices
    indptr.push_back(indices.size());
}

void SparseOp::init_thread_add_row(const Ham &ham, const FullCIWfn &wfn, const int_t idet,
                                   uint_t *det_up, int_t *occs_up, int_t *virs_up) {
    int_t i, j, k, l, ii, jj, kk, ll, jdet, ioffset, koffset, sign_up;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2 = 0.0;
    const uint_t *rdet_up = wfn.det_ptr(idet);
    const uint_t *rdet_dn = rdet_up + wfn.nword;
    uint_t *det_dn = det_up + wfn.nword;
    int_t *occs_dn = occs_up + wfn.nocc_up;
    int_t *virs_dn = virs_up + wfn.nvir_up;
    std::memcpy(det_up, rdet_up, sizeof(uint_t) * wfn.nword2);
    fill_occs(wfn.nword, rdet_up, occs_up);
    fill_occs(wfn.nword, rdet_dn, occs_dn);
    fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
    fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
    // loop over spin-up occupied indices
    for (i = 0; i < wfn.nocc_up; ++i) {
        ii = occs_up[i];
        ioffset = n3 * ii;
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < wfn.nocc_up; ++k) {
            kk = occs_up[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        for (k = 0; k < wfn.nocc_dn; ++k) {
            kk = occs_dn[k];
            val2 += ham.two_mo[ioffset + n2 * kk + n1 * ii + kk];
        }
        // loop over spin-up virtual indices
        for (j = 0; j < wfn.nvir_up; ++j) {
            jj = virs_up[j];
            // 1-0 excitation elements
            excite_det(ii, jj, det_up);
            sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
            jdet = wfn.index_det(det_up);
            // check if 1-0 excited determinant is in wfn
            if ((jdet != -1) && (jdet < ncol)) {
                // compute 1-0 matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
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
                    jdet = wfn.index_det(det_up);
                    // check if 1-1 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < ncol)) {
                        // add 1-1 matrix element
                        data.push_back(sign_up * phase_single_det(wfn.nword, kk, ll, rdet_dn) *
                                       ham.two_mo[koffset + n1 * jj + ll]);
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
                    jdet = wfn.index_det(det_up);
                    // check if 2-0 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < ncol)) {
                        // add 2-0 matrix element
                        data.push_back(phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                                       (ham.two_mo[koffset + n1 * jj + ll] -
                                        ham.two_mo[koffset + n1 * ll + jj]));
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
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < wfn.nocc_dn; ++k) {
            kk = occs_dn[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        // loop over spin-down virtual indices
        for (j = 0; j < wfn.nvir_dn; ++j) {
            jj = virs_dn[j];
            // 0-1 excitation elements
            excite_det(ii, jj, det_dn);
            jdet = wfn.index_det(det_up);
            // check if 0-1 excited determinant is in wfn
            if ((jdet != -1) && (jdet < ncol)) {
                // compute 0-1 matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
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
                    jdet = wfn.index_det(det_up);
                    // check if excited determinant is in wfn
                    if ((jdet != -1) && (jdet < ncol)) {
                        // add 0-2 matrix element
                        data.push_back(phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn) *
                                       (ham.two_mo[koffset + n1 * jj + ll] -
                                        ham.two_mo[koffset + n1 * ll + jj]));
                        indices.push_back(jdet);
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            excite_det(jj, ii, det_dn);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        data.push_back(val2);
        indices.push_back(idet);
    }
    // add pointer to next row's indices
    indptr.push_back(indices.size());
}

void SparseOp::init_thread_add_row(const Ham &ham, const GenCIWfn &wfn, const int_t idet,
                                   uint_t *det, int_t *occs, int_t *virs) {
    int_t i, j, k, l, ii, jj, kk, ll, jdet, ioffset, koffset;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2 = 0.0;
    const uint_t *rdet = wfn.det_ptr(idet);
    // fill working vectors
    std::memcpy(det, rdet, sizeof(uint_t) * wfn.nword);
    fill_occs(wfn.nword, rdet, occs);
    fill_virs(wfn.nword, wfn.nbasis, rdet, virs);
    // loop over occupied indices
    for (i = 0; i < wfn.nocc; ++i) {
        ii = occs[i];
        ioffset = n3 * ii;
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < wfn.nocc; ++k) {
            kk = occs[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        // loop over virtual indices
        for (j = 0; j < wfn.nvir; ++j) {
            jj = virs[j];
            // single excitation elements
            excite_det(ii, jj, det);
            jdet = wfn.index_det(det);
            // check if singly-excited determinant is in wfn
            if ((jdet != -1) && (jdet < ncol)) {
                // compute single excitation matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                // add single excitation matrix element
                data.push_back(phase_single_det(wfn.nword, ii, jj, rdet) * val1);
                indices.push_back(jdet);
            }
            // loop over occupied indices
            for (k = i + 1; k < wfn.nocc; ++k) {
                kk = occs[k];
                koffset = ioffset + n2 * kk;
                // loop over virtual indices
                for (l = j + 1; l < wfn.nvir; ++l) {
                    ll = virs[l];
                    // double excitation elements
                    excite_det(kk, ll, det);
                    jdet = wfn.index_det(det);
                    // check if double excited determinant is in wfn
                    if ((jdet != -1) && (jdet < ncol)) {
                        // add double matrix element
                        data.push_back(phase_double_det(wfn.nword, ii, kk, jj, ll, rdet) *
                                       (ham.two_mo[koffset + n1 * jj + ll] -
                                        ham.two_mo[koffset + n1 * ll + jj]));
                        indices.push_back(jdet);
                    }
                    excite_det(ll, kk, det);
                }
            }
            excite_det(jj, ii, det);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        data.push_back(val2);
        indices.push_back(idet);
    }
    // add pointer to next row's indices
    indptr.push_back(indices.size());
}

namespace {

void perform_op_thread(const double *data, const int_t *indptr, const int_t *indices,
                       const double *x, double *y, const int_t start, const int_t end) {
    int_t j, jstart, jend = indptr[start];
    double val;
    for (int_t i = start; i < end; ++i) {
        jstart = jend;
        jend = indptr[i + 1];
        val = 0.0;
        for (j = jstart; j < jend; ++j)
            val += data[j] * x[indices[j]];
        y[i] = val;
    }
}

void perform_op_cepa0_thread(const double *data, const int_t *indptr, const int_t *indices,
                             const double *x, double *y, const int_t start, const int_t end,
                             const int_t refind, const double h_refind) {
    int_t j, jstart, jend = indptr[start];
    double val;
    for (int_t i = start; i < end; ++i) {
        jstart = jend;
        jend = indptr[i + 1];
        val = 0.0;
        if (i == refind) {
            for (j = jstart; j < jend; ++j) {
                if (indices[j] == refind)
                    val -= x[indices[j]];
                else
                    val += data[j] * x[indices[j]];
            }
        } else {
            for (j = jstart; j < jend; ++j) {
                if (indices[j] == i)
                    val += (data[j] - h_refind) * x[indices[j]];
                else if (indices[j] != refind)
                    val += data[j] * x[indices[j]];
            }
        }
        y[i] = -val;
    }
}

} // namespace

} // namespace pyci
