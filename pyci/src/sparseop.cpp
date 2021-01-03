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

#include <pyci.h>

namespace pyci {

namespace {

template<class T>
inline void append(AlignedVector<T> &v, const T &t) {
    if (v.size() + 1 >= v.capacity())
        v.reserve(std::lround(PYCI_SPARSEOP_RESIZE_FACTOR * v.size() + 0.5));
    v.push_back(t);
}

} // namespace

SparseOp::SparseOp(const SparseOp &op)
    : nrow(op.nrow), ncol(op.ncol), size(op.size), ecore(op.ecore), symmetric(op.symmetric),
      shape(op.shape), data(op.data), indices(op.indices), indptr(op.indptr) {
}

SparseOp::SparseOp(SparseOp &&op) noexcept
    : nrow(std::exchange(op.nrow, 0)), ncol(std::exchange(op.ncol, 0)),
      size(std::exchange(op.size, 0)), ecore(std::exchange(op.ecore, 0.0)),
      symmetric(std::exchange(op.symmetric, 0)), shape(std::move(op.shape)),
      data(std::move(op.data)), indices(std::move(op.indices)), indptr(std::move(op.indptr)) {
}

SparseOp::SparseOp(const long rows, const long cols, const bool symm)
    : nrow(rows), ncol(cols), size(0), ecore(0.0), symmetric(symm) {
    shape = pybind11::make_tuple(pybind11::cast(nrow), pybind11::cast(ncol));
    append<long>(indptr, 0);
}

SparseOp::SparseOp(const Ham &ham, const DOCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    shape = pybind11::make_tuple(pybind11::cast(nrow), pybind11::cast(ncol));
    append<long>(indptr, 0);
    init<DOCIWfn>(ham, wfn, rows, cols);
}

SparseOp::SparseOp(const Ham &ham, const FullCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    shape = pybind11::make_tuple(pybind11::cast(nrow), pybind11::cast(ncol));
    append<long>(indptr, 0);
    init<FullCIWfn>(ham, wfn, rows, cols);
}

SparseOp::SparseOp(const Ham &ham, const GenCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    shape = pybind11::make_tuple(pybind11::cast(nrow), pybind11::cast(ncol));
    append<long>(indptr, 0);
    init<GenCIWfn>(ham, wfn, rows, cols);
}

long SparseOp::rows(void) const {
    return nrow;
}

long SparseOp::cols(void) const {
    return ncol;
}

pybind11::object SparseOp::dtype(void) const {
    return pybind11::dtype::of<double>();
}

const double *SparseOp::data_ptr(const long index) const {
    return &data[index];
}

const long *SparseOp::indices_ptr(const long index) const {
    return &indices[index];
}

const long *SparseOp::indptr_ptr(const long index) const {
    return &indptr[index];
}

double SparseOp::get_element(const long i, const long j) const {
    const long *start = &indices[indptr[i]];
    const long *end = &indices[indptr[i + 1]];
    const long *e = std::lower_bound(start, end, j);
    return (*e == j) ? data[indptr[i] + e - start] : 0.0;
}

void SparseOp::perform_op(const double *x, double *y) const {
    if (symmetric)
        return perform_op_symm(x, y);
    CSparseMatrix<double> mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], nullptr);
    CDenseVector<double> xvec(x, ncol);
    DenseVector<double> yvec(y, nrow);
    yvec = mat * xvec;
}

void SparseOp::perform_op_transpose(const double *x, double *y) const {
    if (symmetric)
        return perform_op_symm(x, y);
    CSparseMatrix<double> mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], nullptr);
    CDenseVector<double> xvec(x, nrow);
    DenseVector<double> yvec(y, ncol);
    yvec = mat.transpose() * xvec;
}

void SparseOp::perform_op_symm(const double *x, double *y) const {
    CSparseMatrix<double> mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], nullptr);
    CDenseVector<double> xvec(x, ncol);
    DenseVector<double> yvec(y, nrow);
    yvec = mat.selfadjointView<Eigen::Upper>() * xvec;
}

void SparseOp::solve_ci(const long n, const double *coeffs, const long ncv, const long maxiter,
                        const double tol, double *evals, double *evecs) const {
    if (n > nrow)
        throw std::runtime_error("cannot find >n eigenpairs for sparse operator with n rows");
    else if (nrow == 1) {
        *evals = get_element(0, 0) + ecore;
        *evecs = 1.0;
        return;
    }
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, const SparseOp> eigs(
        this, n, (ncv != -1) ? ncv : std::min(nrow, std::max(n * 2 + 1, 20L)));
    AlignedVector<double> c0;
    if (coeffs == nullptr) {
        c0.resize(nrow);
        c0[0] = 1.0;
        coeffs = &c0[0];
    }
    eigs.init(coeffs);
    eigs.compute((maxiter != -1) ? maxiter : n * nrow * 10, tol, Spectra::SMALLEST_ALGE);
    if (eigs.info() != Spectra::SUCCESSFUL)
        throw std::runtime_error("did not converge");
    DenseVector<double> eigenvalues(evals, n);
    DenseMatrix<double> eigenvectors(evecs, nrow, n);
    eigenvalues = eigs.eigenvalues();
    for (long i = 0; i < n; ++i)
        evals[i] += ecore;
    eigenvectors = eigs.eigenvectors();
}

Array<double> SparseOp::py_matvec(const Array<double> x) const {
    Array<double> y(nrow);
    perform_op(reinterpret_cast<const double *>(x.request().ptr),
               reinterpret_cast<double *>(y.request().ptr));
    return y;
}

Array<double> SparseOp::py_matvec_out(const Array<double> x, Array<double> y) const {
    perform_op(reinterpret_cast<const double *>(x.request().ptr),
               reinterpret_cast<double *>(y.request().ptr));
    return y;
}

Array<double> SparseOp::py_rmatvec(const Array<double> x) const {
    Array<double> y(nrow);
    perform_op_transpose(reinterpret_cast<const double *>(x.request().ptr),
                         reinterpret_cast<double *>(y.request().ptr));
    return y;
}

Array<double> SparseOp::py_rmatvec_out(const Array<double> x, Array<double> y) const {
    perform_op_transpose(reinterpret_cast<const double *>(x.request().ptr),
                         reinterpret_cast<double *>(y.request().ptr));
    return y;
}

Array<double> SparseOp::py_matmat(const Array<double> x) const {
    Array<double> y({nrow, x.request().size / ncol});
    return py_matmat_out(x, y);
}

Array<double> SparseOp::py_matmat_out(const Array<double> x, Array<double> y) const {
    pybind11::buffer_info xbuf = x.request();
    pybind11::buffer_info ybuf = y.request();
    const double *xptr = reinterpret_cast<const double *>(xbuf.ptr);
    double *yptr = reinterpret_cast<double *>(ybuf.ptr);
    CSparseMatrix<double> mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], nullptr);
    CDenseMatrix<double> xmat(xptr, ncol, xbuf.size / ncol);
    DenseMatrix<double> ymat(yptr, nrow, ybuf.size / nrow);
    if (symmetric)
        ymat = mat.selfadjointView<Eigen::Upper>() * xmat;
    else
        ymat = mat * xmat;
    return y;
}

Array<double> SparseOp::py_rmatmat(const Array<double> x) const {
    Array<double> y({ncol, x.request().size / nrow});
    return py_rmatmat_out(x, y);
}

Array<double> SparseOp::py_rmatmat_out(const Array<double> x, Array<double> y) const {
    pybind11::buffer_info xbuf = x.request();
    pybind11::buffer_info ybuf = y.request();
    const double *xptr = reinterpret_cast<const double *>(xbuf.ptr);
    double *yptr = reinterpret_cast<double *>(ybuf.ptr);
    CSparseMatrix<double> mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], nullptr);
    CDenseMatrix<double> xmat(xptr, nrow, xbuf.size / nrow);
    DenseMatrix<double> ymat(yptr, ncol, ybuf.size / ncol);
    if (symmetric)
        ymat = mat.selfadjointView<Eigen::Upper>() * xmat;
    else
        ymat = mat.transpose() * xmat;
    return y;
}

pybind11::tuple SparseOp::py_solve_ci(const long n, pybind11::object coeffs, const long ncv,
                                      const long maxiter, const double tol) const {
    Array<double> eigvals(n);
    Array<double> eigvecs({n, nrow});
    const double *cptr =
        coeffs.is(pybind11::none())
            ? nullptr
            : reinterpret_cast<const double *>(coeffs.cast<Array<double>>().request().ptr);
    double *evals = reinterpret_cast<double *>(eigvals.request().ptr);
    double *evecs = reinterpret_cast<double *>(eigvecs.request().ptr);
    solve_ci(n, cptr, ncv, maxiter, tol, evals, evecs);
    return pybind11::make_tuple(eigvals, eigvecs);
}

template<class WfnType>
void SparseOp::init_thread(SparseOp &op, const Ham &ham, const WfnType &wfn, const long start,
                           const long end) {
    AlignedVector<ulong> det(wfn.nword2);
    AlignedVector<long> occs(wfn.nocc);
    AlignedVector<long> virs(wfn.nvir);
    long row = 0;
    for (long i = start; i < end; ++i) {
        op.init_thread_add_row(ham, wfn, i, &det[0], &occs[0], &virs[0]);
        op.init_thread_sort_row(row++);
    }
    op.data.shrink_to_fit();
    op.indices.shrink_to_fit();
    op.indptr.shrink_to_fit();
    op.size = op.indices.size();
}

template<class WfnType>
void SparseOp::init(const Ham &ham, const WfnType &wfn, const long rows, const long cols) {
    long nthread = get_num_threads();
    long chunksize = nrow / nthread + static_cast<bool>(nrow % nthread);
    long start, end = 0;
    while (nthread > 1 && chunksize < PYCI_CHUNKSIZE_MIN) {
        nthread /= 2;
        chunksize = nrow / nthread + static_cast<bool>(nrow % nthread);
    }
    Vector<SparseOp> v_ops;
    Vector<std::thread> v_threads;
    v_ops.reserve(nthread);
    v_threads.reserve(nthread);
    for (long i = 0; i < nthread; ++i) {
        start = end;
        end = std::min(start + chunksize, nrow);
        v_ops.emplace_back(end - start, ncol, symmetric);
        v_threads.emplace_back(&SparseOp::init_thread<WfnType>, std::ref(v_ops.back()),
                               std::ref(ham), std::ref(wfn), start, end);
    }
    long ithread = 0;
    for (auto &thread : v_threads) {
        thread.join();
        v_ops[ithread].init_thread_condense(*this, ithread);
        ++ithread;
    }
    data.shrink_to_fit();
    indices.shrink_to_fit();
    indptr.shrink_to_fit();
    size = indices.size();
}

void SparseOp::init_thread_sort_row(const long idet) {
    typedef std::sort_with_arg::value_iterator_t<double, long> iter;
    long start = indptr[idet], end = indptr[idet + 1];
    std::sort(iter(&data[start], &indices[start]), iter(&data[end], &indices[end]));
}

void SparseOp::init_thread_condense(SparseOp &op, const long ithread) {
    long i, j, start, end, offset;
    if (!ithread) {
        op.data.swap(data);
        op.indptr.swap(indptr);
        op.indices.swap(indices);
    } else if (nrow) {
        // copy over data array
        start = op.data.size();
        op.data.resize(start + size);
        std::memcpy(&op.data[start], &data[0], sizeof(double) * size);
        AlignedVector<double>().swap(data);
        // copy over indices array
        start = op.indices.size();
        op.indices.resize(start + size);
        std::memcpy(&op.indices[start], &indices[0], sizeof(long) * size);
        AlignedVector<long>().swap(indices);
        // copy over indptr array
        start = op.indptr.size();
        end = start + indptr.size() - 1;
        offset = op.indptr.back();
        op.indptr.resize(end);
        j = 0;
        for (i = start; i < end; ++i)
            op.indptr[i] = indptr[++j] + offset;
        AlignedVector<long>().swap(indptr);
    }
}

void SparseOp::init_thread_add_row(const Ham &ham, const DOCIWfn &wfn, const long idet, ulong *det,
                                   long *occs, long *virs) {
    long i, j, k, l, jdet, jmin = symmetric ? idet - 1 : -1;
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
            if ((jdet > jmin) && (jdet < ncol)) {
                // add single/"pair"-excited matrix element
                append<double>(data, ham.v[k * wfn.nbasis + l]);
                append<long>(indices, jdet);
            }
            excite_det(l, k, det);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val1 + val2 * 2);
        append<long>(indices, idet);
    }
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}

void SparseOp::init_thread_add_row(const Ham &ham, const FullCIWfn &wfn, const long idet,
                                   ulong *det_up, long *occs_up, long *virs_up) {
    long i, j, k, l, ii, jj, kk, ll, jdet, jmin = symmetric ? idet - 1 : -1;
    long ioffset, koffset, sign_up;
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val1, val2 = 0.0;
    const ulong *rdet_up = wfn.det_ptr(idet);
    const ulong *rdet_dn = rdet_up + wfn.nword;
    ulong *det_dn = det_up + wfn.nword;
    long *occs_dn = occs_up + wfn.nocc_up;
    long *virs_dn = virs_up + wfn.nvir_up;
    std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword2);
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
            if ((jdet > jmin) && (jdet < ncol)) {
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
                append<double>(data, sign_up * val1);
                append<long>(indices, jdet);
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
                    if ((jdet > jmin) && (jdet < ncol)) {
                        // add 1-1 matrix element
                        append<double>(data, sign_up *
                                                 phase_single_det(wfn.nword, kk, ll, rdet_dn) *
                                                 ham.two_mo[koffset + n1 * jj + ll]);
                        append<long>(indices, jdet);
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
                    if ((jdet > jmin) && (jdet < ncol)) {
                        // add 2-0 matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
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
            if ((jdet > jmin) && (jdet < ncol)) {
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
                append<double>(data, phase_single_det(wfn.nword, ii, jj, rdet_dn) * val1);
                append<long>(indices, jdet);
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
                    if ((jdet > jmin) && (jdet < ncol)) {
                        // add 0-2 matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            excite_det(jj, ii, det_dn);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val2);
        append<long>(indices, idet);
    }
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}

void SparseOp::init_thread_add_row(const Ham &ham, const GenCIWfn &wfn, const long idet, ulong *det,
                                   long *occs, long *virs) {
    long i, j, k, l, ii, jj, kk, ll, jdet, jmin = symmetric ? idet - 1 : -1, ioffset, koffset;
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val1, val2 = 0.0;
    const ulong *rdet = wfn.det_ptr(idet);
    // fill working vectors
    std::memcpy(det, rdet, sizeof(ulong) * wfn.nword);
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
            if ((jdet > jmin) && (jdet < ncol)) {
                // compute single excitation matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                // add single excitation matrix element
                append<double>(data, phase_single_det(wfn.nword, ii, jj, rdet) * val1);
                append<long>(indices, jdet);
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
                    if ((jdet > jmin) && (jdet < ncol)) {
                        // add double matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det);
                }
            }
            excite_det(jj, ii, det);
        }
    }
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val2);
        append<long>(indices, idet);
    }
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}

} // namespace pyci
