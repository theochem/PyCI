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
#include <iostream>

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

SparseOp::SparseOp(const SQuantOp &ham, const DOCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    update<DOCIWfn>(ham, wfn, nrow, ncol, 0);
}

SparseOp::SparseOp(const SQuantOp &ham, const FullCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    update<FullCIWfn>(ham, wfn, nrow, ncol, 0);
}

SparseOp::SparseOp(const SQuantOp &ham, const GenCIWfn &wfn, const long rows, const long cols,
                   const bool symm)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    std::cout << "Inside SparseOp constructor" << std::endl;
    update<GenCIWfn>(ham, wfn, nrow, ncol, 0);
}

SparseOp::SparseOp(const SQuantOp &ham, const NonSingletCI &wfn, const long rows, const long cols,
                   const bool symm, const std::string wfntype)
    : nrow((rows > -1) ? rows : wfn.ndet), ncol((cols > -1) ? cols : wfn.ndet), size(0),
      ecore(ham.ecore), symmetric(symm) {
    append<long>(indptr, 0);
    std::cout << "Inside SparseOp NonSingletCI constructor" << std::endl;
    std::cout << "nrow: " << nrow << ", ncol: " << ncol << std::endl;
    update<NonSingletCI>(ham, wfn, nrow, ncol, 0);
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
    typedef Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, long>> SparseMatrix;
    SparseMatrix mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], 0);
    Eigen::Map<const Eigen::VectorXd> xvec(x, ncol);
    Eigen::Map<Eigen::VectorXd> yvec(y, nrow);
    yvec = mat * xvec;
}

void SparseOp::perform_op_symm(const double *x, double *y) const {
    typedef Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, long>> SparseMatrix;
    SparseMatrix mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], 0);
    Eigen::Map<const Eigen::VectorXd> xvec(x, ncol);
    Eigen::Map<Eigen::VectorXd> yvec(y, nrow);
    yvec = mat.selfadjointView<Eigen::Lower>() * xvec;
}

void SparseOp::solve_ci(const long n, const double *coeffs, const long ncv, const long maxiter,
                        const double tol, double *evals, double *evecs) const {
    if ((nrow > 1 && n >= nrow) || (nrow == 1 && n > 1)) {
        throw std::invalid_argument("cannot find >=n eigenpairs for sparse operator with n rows");
    } else if (!symmetric) {
        throw pybind11::type_error("Can only solve sparse symmetric matrix operators");
    } else if (nrow == 1) {
        *evals = get_element(0, 0) + ecore;
        *evecs = 1.0;
        return;
    }
    typedef Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, long>> SparseMatrix;
    SparseMatrix mat(nrow, ncol, size, &indptr[0], &indices[0], &data[0], 0);
    Spectra::SparseSymMatProd<double, Eigen::Lower, Eigen::RowMajor, long> op(mat);
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double, Eigen::Lower, Eigen::RowMajor, long>>
        eigs(op, n, (ncv != -1) ? ncv : std::min(nrow, std::max(n * 2 + 1, 20L)));
    if (coeffs == nullptr)
        eigs.init();
    else
        eigs.init(coeffs);
    eigs.compute(Spectra::SortRule::SmallestAlge, (maxiter != -1) ? maxiter : n * nrow * 10, tol);
    if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("did not converge");
    DenseVector<double> eigenvalues(evals, n);
    DenseMatrix<double> eigenvectors(evecs, n, nrow);
    eigenvalues = eigs.eigenvalues();
    for (long i = 0; i < n; ++i)
        evals[i] += ecore;
    // This is needed so that the eigenvectors are in the proper order
    // when passed back to Python as NumPy arrays
    eigenvectors.transpose() = eigs.eigenvectors();

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
void SparseOp::py_update(const SQuantOp &ham, const WfnType &wfn) {
    update<WfnType>(ham, wfn, wfn.ndet, wfn.ndet, nrow);
}

template void SparseOp::py_update(const SQuantOp &, const DOCIWfn &);

template void SparseOp::py_update(const SQuantOp &, const FullCIWfn &);

template void SparseOp::py_update(const SQuantOp &, const GenCIWfn &);

template void SparseOp::py_update(const SQuantOp &ham, const NonSingletCI &wfn);
// {    update<NonSingletCI>(ham, wfn, wfn.ndet, wfn.ndet, nrow);}

template<class WfnType>
void SparseOp::update(const SQuantOp &ham, const WfnType &wfn, const long rows, const long cols,
                      const long startrow) {
    std::cout << "Inside SparseOp update" << std::endl;
    std::cout << "Type of WfnType: " << typeid(WfnType).name() << std::endl;

    AlignedVector<ulong> det(wfn.nword2);
    AlignedVector<long> occs(wfn.nocc);
    AlignedVector<long> virs(wfn.nvir);
    std::cout << "wfn.nvir: " << wfn.nvir << std::endl;
    shape = pybind11::make_tuple(pybind11::cast(rows), pybind11::cast(cols));
    nrow = rows;
    ncol = cols;
    std::cout << "nrow: " << nrow << std::endl;
    indptr.reserve(nrow + 1);
    for (long idet = startrow; idet < rows; ++idet) {
        add_row(ham, wfn, idet, &det[0], &occs[0], &virs[0]);
        sort_row(idet);
    }
    size = indices.size();
    std::cout << "size: " << size << std::endl;
}


void SparseOp::update(const SQuantOp &ham, const NonSingletCI &wfn, const long rows, const long cols,
                      const long startrow) {
    std::cout << "Inside NonsingletCI SparseOp update" << std::endl;
    AlignedVector<ulong> det(wfn.nword);
    AlignedVector<long> occs(wfn.nocc);
    AlignedVector<long> virs(wfn.nbasis - wfn.nocc);
    std::cout << "wfn.nvir: " << wfn.nvir << std::endl;
    shape = pybind11::make_tuple(pybind11::cast(rows), pybind11::cast(cols));
    nrow = rows;
    ncol = cols;
    std::cout << "nrow: " << nrow << std::endl;
    indptr.reserve(nrow + 1);
    for (long idet = startrow; idet < rows; ++idet) {
        add_row(ham, wfn, idet, &det[0], &occs[0], &virs[0]);
        sort_row(idet);
    }
    size = indices.size();
    std::cout << "size: " << size << std::endl;
}


void SparseOp::reserve(const long n) {
    indices.reserve(n);
    data.reserve(n);
}

void SparseOp::squeeze(void) {
    indptr.shrink_to_fit();
    indices.shrink_to_fit();
    data.shrink_to_fit();
}

void SparseOp::sort_row(const long idet) {
    typedef std::sort_with_arg::value_iterator_t<double, long> iter;
    long start = indptr[idet], end = indptr[idet + 1];
    std::sort(iter(&data[start], &indices[start]), iter(&data[end], &indices[end]));
}

void SparseOp::add_row(const SQuantOp &ham, const DOCIWfn &wfn, const long idet, ulong *det, long *occs,
                       long *virs) {
    /* long i, j, k, l, jdet, jmin = symmetric ? idet - 1 : -1; */
    long  jdet, jmin = symmetric ? idet : Max<long>();
    double val1 = 0.0, val2 = 0.0;
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
    // loop over occupied indices
    for (long i = 0, j, k, l ; i < wfn.nocc_up; ++i) {
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
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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

void SparseOp::add_row(const SQuantOp &ham, const FullCIWfn &wfn, const long idet, ulong *det_up,
                       long *occs_up, long *virs_up) {
    long i, j, k, l, ii, jj, kk, ll, jdet, jmin = symmetric ? idet : Max<long>();
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
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
                 ;   ll = virs_dn[l];
                    // 0-2 excitation elements
                    excite_det(kk, ll, det_dn);
                    jdet = wfn.index_det(det_up);
                    // check if excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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

void SparseOp::add_row(const SQuantOp &ham, const GenCIWfn &wfn, const long idet, ulong *det, long *occs,
                       long *virs) {
    long jdet, jmin = symmetric ? idet : Max<long>();
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
    for (long i = 0, j, k, l, ii, jj, kk, ll, ioffset, koffset; i < wfn.nocc; ++i) {
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
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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


void SparseOp::add_row(const SQuantOp &ham, const NonSingletCI &wfn, const long idet, ulong *det_up,
                       long *occs, long *virs) {
    std::cout << "---Inside add_row NonSingletCI--" << std::endl;
    std::cout << "idet: " << idet << std::endl;
    std::cout << "wfn.nbasis: " << wfn.nbasis << std::endl;
    std::cout << "wfn.nword: " << wfn.nword << std::endl;
    long i, j, k, l, ii, jj, kk, ll, jdet, jmin = symmetric ? idet : Max<long>();
    long ioffset, koffset, sign_up;
    long nbasis = wfn.nbasis / 2;
    long n1 = nbasis; // Check if nbasis or nbasis * 2!
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val1, val2 = 0.0;
    const ulong *rdet_up = wfn.det_ptr(idet);
    const ulong *rdet_dn = rdet_up + nbasis;
    std::cout << "rdet_up: " << *rdet_up << std::endl;
    ulong *det_dn = det_up + nbasis;
    std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword); // !Check nword or nword2 

    fill_occs(wfn.nword, rdet_up, occs);
    fill_virs(wfn.nword, nbasis, rdet_up, virs);
    
    long nocc_up = __builtin_popcount(*det_up & ((1 << nbasis / 2) - 1));
    long nocc_dn = wfn.nocc - nocc_up;  // std:popcount(det_up>> wfn.nbasis / 2);
    long nvir_up = nbasis - nocc_up;
    long nvir_dn = nbasis - nocc_dn;
    long nvir = nvir_up + nvir_dn;
    std::cout << "nocc_up: " << nocc_up << ", nocc_dn: " << nocc_dn << std::endl;
    std::cout << "nvir_up: " << nvir_up << ", nvir_dn: " << nvir_dn << std::endl;
    std::cout << "Occs: " ;
    for (i = 0; i < wfn.nocc; ++i) {
        std::cout << occs[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Virs: " ;
    for (i = 0; i < (nvir); ++i) {
        std::cout << virs[i] << " ";
    }
    std::cout << std::endl;
    long *virs_up = virs;
    long *virs_dn = nullptr;
    for (long i = 0; i < nvir; ++i) {
        if (virs[i] >= nbasis) {
            virs_dn = &virs[i];
            break;
        }
    }
    long *occs_up = occs;
    long *occs_dn = nullptr;
    for (long i = 0; i < wfn.nocc; ++i) {
        if (occs[i] >= nbasis) {
            occs_dn = &occs[i];
            break;
        }
    }
    // std::cout << "ham.one_mo.size(): " << ham.one_mo.size() << std::endl;
    // loop over spin-up occupied indices
    for (i = 0; i < nocc_up; ++i) {
        ii = occs_up[i];
        ioffset = n3 * ii;
        
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < nocc_up; ++k) {
            kk = occs_up[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
            std::cout << "ii, kk: " << ii << ", " << kk << std::endl;
        }
        for (k = 0; k < nocc_dn; ++k) {
            kk = occs_dn[k];
            val2 += ham.two_mo[ioffset + n2 * kk + n1 * ii + kk];
        }
        // loop over spin-up virtual indices
        for (j = 0; j < nvir_up; ++j) {
            jj = virs_up[j];
            // alpha -> alpha excitation elements
            excite_det(ii, jj, det_up);
            sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
            jdet = wfn.index_det(det_up);
            
            // check if 1-0 excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute 1-0 matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                for (k = 0; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                // add 1-0 matrix element
                append<double>(data, sign_up * val1);
                append<long>(indices, jdet);
                std::cout << "jdet: " << jdet << std::endl;
            }
            // loop over spin-down occupied indices
            for (k = 0; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                for (l = 0; l < nvir_up; ++l) {
                    ll = virs_up[l];
                    // beta -> alpha excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if 1-1 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 1-1 matrix element
                        append<double>(data, sign_up *
                                                 phase_single_det(wfn.nword, kk, ll, rdet_up) *
                                                 ham.two_mo[koffset + n1 * jj + ll]); 
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_up); 
                }
                // loop over spin-down virtual indices
                for (l = 0; l < nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // 1-1 excitation elements
                    excite_det(kk, ll, det_dn);
                    jdet = wfn.index_det(det_up);
                    // check if 1-1 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
            for (k = i + 1; k < nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                for (l = j + 1; l < nvir_up; ++l) {
                    ll = virs_up[l];
                    // alpha -> alpha excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if the excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 2-0 matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_up);
                }
                // loop over spin-dn virtual indices
                for (l = j + 1; l < nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // alpha -> beta excitation elements
                    excite_det(kk, ll, det_dn);
                    jdet = wfn.index_det(det_up);
                    // check if the excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 2-0 matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            excite_det(jj, ii, det_up);
        }

        // loop over spin-dn virtual indices
        for (j = 0; j < nvir_dn; ++j) {
            jj = virs_dn[j];
            // alpha -> beta excitation elements
            excite_det(ii, jj, det_up);
            sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
            jdet = wfn.index_det(det_up);
            // check if the excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute the matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                for (k = 0; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                // add the matrix element
                append<double>(data, sign_up * val1);
                append<long>(indices, jdet);
            }
            // loop over spin-down occupied indices
            for (k = 0; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                for (l = 0; l < nvir_up; ++l) {
                    ll = virs_up[l];
                    // beta -> alpha excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if 1-1 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 1-1 matrix element
                        append<double>(data, sign_up *
                                                 phase_single_det(wfn.nword, kk, ll, rdet_up) *
                                                 ham.two_mo[koffset + n1 * jj + ll]); 
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_up); 
                }
                // loop over spin-down virtual indices
                for (l = 0; l < nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // 1-1 excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if 1-1 excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 1-1 matrix element
                        append<double>(data, sign_up *
                                                 phase_single_det(wfn.nword, kk, ll, rdet_up) *
                                                 ham.two_mo[koffset + n1 * jj + ll]); 
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_up); 
                }
            }
            // loop over spin-up occupied indices
            for (k = i + 1; k < nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                // for (l = j + 1; l < nvir_up; ++l) {
                //     ll = virs_up[l];
                //     // alpha -> alpha excitation elements
                //     excite_det(kk, ll, det_up);
                //     jdet = wfn.index_det(det_up);
                //     // check if the excited determinant is in wfn
                //     if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                //         // add 2-0 matrix element
                //         append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                //                                  (ham.two_mo[koffset + n1 * jj + ll] -
                //                                   ham.two_mo[koffset + n1 * ll + jj]));
                //         append<long>(indices, jdet);
                //     }
                //     excite_det(ll, kk, det_up);
                // }
                // loop over spin-dn virtual indices
                for (l = j + 1; l < nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // alpha -> beta excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if the excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
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
    for (i = 0; i < nocc_dn; ++i) {
        ii = occs_dn[i];
        ioffset = n3 * ii;
        // compute part of diagonal matrix element
        val2 += ham.one_mo[(n1 + 1) * ii];
        for (k = i + 1; k < nocc_dn; ++k) {
            kk = occs_dn[k];
            koffset = ioffset + n2 * kk;
            val2 += ham.two_mo[koffset + n1 * ii + kk] - ham.two_mo[koffset + n1 * kk + ii];
        }
        // loop over spin-up virtual indices
        for (j = 0; j < nvir_up; ++j) {
            jj = virs_up[j];
            // beta -> alpha excitation elements
            excite_det(ii, jj, det_up);
            jdet = wfn.index_det(det_up);
            // check if the excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute the matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < nocc_up; ++k) {
                    kk = occs_up[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                for (k = 0; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                // add the matrix element
                append<double>(data, phase_single_det(wfn.nword, ii, jj, rdet_up) * val1);
                append<long>(indices, jdet);
            }
            // loop over spin-down occupied indices
            for (k = i + 1; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                for (l = j + 1; l < nvir_up; ++l) {
                    ll = virs_up[l];
                    // beta -> alpha excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 0-2 matrix element
                        append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                                                 (ham.two_mo[koffset + n1 * jj + ll] -
                                                  ham.two_mo[koffset + n1 * ll + jj]));
                        append<long>(indices, jdet);
                    }
                    excite_det(ll, kk, det_up);
                }
                // loop over spin-down virtual indices
                for (l = j + 1; l < nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // beta -> beta excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 0-2 matrix element
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
        // loop over spin-down virtual indices
        for (j = 0; j < nvir_dn; ++j) {
            jj = virs_dn[j];
            // beta -> beta excitation elements
            excite_det(ii, jj, det_up);
            jdet = wfn.index_det(det_up);
            // check if 0-1 excited determinant is in wfn
            if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                // compute 0-1 matrix element
                val1 = ham.one_mo[n1 * ii + jj];
                for (k = 0; k < nocc_up; ++k) {
                    kk = occs_up[k];
                    val1 += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                for (k = 0; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    val1 += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
                }
                // add 0-1 matrix element
                append<double>(data, phase_single_det(wfn.nword, ii, jj, rdet_up) * val1);
                append<long>(indices, jdet);
            }
            // loop over spin-down occupied indices
            for (k = i + 1; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                // for (l = j + 1; l < nvir_up; ++l) {
                //     ll = virs_up[l];
                //     // beta -> alpha excitation elements
                //     excite_det(kk, ll, det_up);
                //     jdet = wfn.index_det(det_up);
                //     // check if excited determinant is in wfn
                //     if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                //         // add 0-2 matrix element
                //         append<double>(data, phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up) *
                //                                  (ham.two_mo[koffset + n1 * jj + ll] -
                //                                   ham.two_mo[koffset + n1 * ll + jj]));
                //         append<long>(indices, jdet);
                //     }
                //     excite_det(ll, kk, det_up);
                // }
                // loop over spin-down virtual indices
                for (l = j + 1; l < nvir_dn; ++l) {
                    ll = virs_dn[l];
                    // beta -> beta excitation elements
                    excite_det(kk, ll, det_up);
                    jdet = wfn.index_det(det_up);
                    // check if excited determinant is in wfn
                    if ((jdet != -1) && (jdet < jmin) && (jdet < ncol)) {
                        // add 0-2 matrix element
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
    // add diagonal element to matrix
    if (idet < ncol) {
        append<double>(data, val2);
        append<long>(indices, idet);
    }
    std::cout << "Insinde nonsinglet add_row indices.size(): " << indices.size() << std::endl;
    // add pointer to next row's indices
    append<long>(indptr, indices.size());
}



} // namespace pyci
