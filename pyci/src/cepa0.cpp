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

void perform_op_cepa0_thread(const double *data, const long *indptr, const long *indices,
                             const double *x, double *y, const long start, const long end,
                             const long refind, const double h_refind) {
    long j, jstart, jend = indptr[start];
    double val;
    for (long i = start; i < end; ++i) {
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

void SparseOp::perform_op_cepa0(const double *x, double *y, const long refind) const {
    long nthread = get_num_threads(), start, end;
    long chunksize = nrow / nthread + ((nrow % nthread) ? 1 : 0);
    double h_refind = get_element(refind, refind);
    Vector<std::thread> v_threads;
    v_threads.reserve(nthread);
    for (long i = 0; i < nthread; ++i) {
        start = i * chunksize;
        end = (start + chunksize < nrow) ? start + chunksize : nrow;
        v_threads.emplace_back(&perform_op_cepa0_thread, &data[0], &indptr[0], &indices[0], x, y,
                               start, end, refind, h_refind);
    }
    for (auto &thread : v_threads)
        thread.join();
}

void SparseOp::perform_op_transpose_cepa0(const double *x, double *y, const long refind) const {
    long i;
    for (i = 0; i < ncol; ++i)
        y[i] = 0;
    long j, jstart, jend = indptr[0];
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

void SparseOp::rhs_cepa0(double *b, const long refind) const {
    long iend = indptr[refind + 1];
    for (long i = indptr[refind]; i != iend; ++i) {
        if (indices[i] >= nrow)
            break;
        b[indices[i]] = data[i];
    }
}

Array<double> SparseOp::py_matvec_cepa0(const Array<double> x, const long refind) const {
    if (symmetric)
        throw std::runtime_error("cannot run CEPA0 with sparse_op(symmetric=True)");
    Array<double> y(nrow);
    perform_op_cepa0(reinterpret_cast<const double *>(x.request().ptr),
                     reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

Array<double> SparseOp::py_rmatvec_cepa0(Array<double> x, const long refind) const {
    if (symmetric)
        throw std::runtime_error("cannot run CEPA0 with sparse_op(symmetric=True)");
    Array<double> y(ncol);
    perform_op_transpose_cepa0(reinterpret_cast<const double *>(x.request().ptr),
                               reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

Array<double> SparseOp::py_rhs_cepa0(const long refind) const {
    if (symmetric)
        throw std::runtime_error("cannot run CEPA0 with sparse_op(symmetric=True)");
    Array<double> y(nrow);
    rhs_cepa0(reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

} // namespace pyci
