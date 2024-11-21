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

template<class Wfn>
Objective<Wfn>::Objective(const SparseOp &op_, const Wfn &wfn_,
                          const std::size_t n_detcons_,
                          const long *idx_detcons_,
                          const double *val_detcons_,
                          const std::size_t n_paramcons_,
                          const long *idx_paramcons_,
                          const double *val_paramcons_)
: nproj(op_.nrow), nconn(op_.ncol), nparam(0UL), n_detcons(n_detcons_), n_paramcons(n_paramcons_)
{
    (void)wfn_;

    init(n_detcons_, idx_detcons_, val_detcons_, n_paramcons_, idx_paramcons_, val_paramcons_);
}

template<class Wfn>
Objective<Wfn>::Objective(const SparseOp &op_, const Wfn &wfn_,
                          const pybind11::object idx_detcons_,
                          const pybind11::object val_detcons_,
                          const pybind11::object idx_paramcons_,
                          const pybind11::object val_paramcons_)
: nproj(op_.nrow), nconn(op_.ncol), nparam(0UL)
{
    (void)wfn_;

    long *idxptr_detcons, *idxptr_paramcons;
    double *valptr_detcons, *valptr_paramcons;

    if (idx_detcons_.is(pybind11::none()) && val_detcons_.is(pybind11::none())) {
        n_detcons = 0;
        idxptr_detcons = nullptr;
        valptr_detcons = nullptr;
    } else if (!(idx_detcons_.is(pybind11::none()) || val_detcons_.is(pybind11::none()))) {
        pybind11::buffer_info idx_detcons_info = idx_detcons_.cast<Array<long>>().request();
        pybind11::buffer_info val_detcons_info = val_detcons_.cast<Array<double>>().request();
        n_detcons = 1;
        for (pybind11::ssize_t dim : idx_detcons_info.shape)
            n_detcons *= dim;
        idxptr_detcons = reinterpret_cast<long *>(idx_detcons_info.ptr);
        valptr_detcons = reinterpret_cast<double *>(val_detcons_info.ptr);
    } else {
        throw std::exception();
    }

    if (idx_paramcons_.is(pybind11::none()) && val_paramcons_.is(pybind11::none())) {
        n_paramcons = 0;
        idxptr_paramcons = nullptr;
        valptr_paramcons = nullptr;
    } else if (!(idx_paramcons_.is(pybind11::none()) || val_paramcons_.is(pybind11::none()))) {
        pybind11::buffer_info idx_paramcons_info = idx_paramcons_.cast<Array<long>>().request();
        pybind11::buffer_info val_paramcons_info = val_paramcons_.cast<Array<double>>().request();
        n_paramcons = 1;
        for (pybind11::ssize_t dim : idx_paramcons_info.shape)
            n_paramcons *= dim;
        idxptr_paramcons = reinterpret_cast<long *>(idx_paramcons_info.ptr);
        valptr_paramcons = reinterpret_cast<double *>(val_paramcons_info.ptr);
    } else {
        throw std::exception();
    }
    init(n_detcons, idxptr_detcons, valptr_detcons, n_paramcons, idxptr_paramcons, valptr_paramcons);
}

template<class Wfn>
Objective<Wfn>::Objective(const Objective<Wfn> &obj)
: nproj(obj.nproj), nconn(obj.nconn), nparam(obj.nparam), n_detcons(obj.n_detcons),
  n_paramcons(obj.n_paramcons), ovlp(obj.ovlp), d_ovlp(obj.d_ovlp), idx_detcons(obj.idx_detcons),
  idx_paramcons(obj.idx_paramcons), val_detcons(obj.val_detcons), val_paramcons(obj.val_paramcons)
{
    return;
}

template<class Wfn>
Objective<Wfn>::Objective(Objective<Wfn> &&obj) noexcept
: nproj(std::exchange(obj.nproj, 0)), nconn(std::exchange(obj.nconn, 0)), nparam(std::exchange(obj.nparam, 0)), n_detcons(std::exchange(obj.n_detcons, 0)),
  n_paramcons(std::exchange(obj.n_paramcons, 0)), ovlp(std::move(obj.ovlp)), d_ovlp(std::move(obj.d_ovlp)), idx_detcons(std::move(obj.idx_detcons)),
  idx_paramcons(std::move(obj.idx_paramcons)), val_detcons(std::move(obj.val_detcons)), val_paramcons(std::move(obj.val_paramcons))
{
    return;
}

template<class Wfn>
void Objective<Wfn>::init(const std::size_t n_detcons_,
                          const long *idx_detcons_,
                          const double *val_detcons_,
                          const std::size_t n_paramcons_,
                          const long *idx_paramcons_,
                          const double *val_paramcons_)
{
    if (n_detcons_ != 0) {
        idx_detcons.resize(n_detcons_);
        val_detcons.resize(n_detcons_);
        for (size_t i = 0; i != n_detcons_; ++i) {
            idx_detcons[i] = idx_detcons_[i];
            val_detcons[i] = val_detcons_[i];
        }
    }

    if (n_paramcons_ != 0) {
        idx_paramcons.resize(n_paramcons_);
        val_paramcons.resize(n_paramcons_);
        for (std::size_t i = 0; i != n_paramcons_; ++i) {
            idx_paramcons[i] = idx_paramcons_[i];
            val_paramcons[i] = val_paramcons_[i];
        }
    }
}

template<class Wfn>
void Objective<Wfn>::objective(const SparseOp &op, const double *x, double *y)
{
    double e = x[nparam];

    /* Compute overlaps of determinants in connection space ("S space"):

           c_m
    */
    this->overlap(nconn, x, &ovlp[0]);

    /* Compute objective function:

           f_n = <m|H|n> c_m - E <n|\Psi>
    */
    op.perform_op(&ovlp[0], y);
    for (std::size_t i = 0; i != nproj; ++i) {
        y[i] -= e * ovlp[i];
    }
    y += nproj;

    /* Compute determinant constraints */
    for (std::size_t i = 0; i != n_detcons; ++i) {
        y[i] = ovlp[idx_detcons[i]] - val_detcons[i];
    }
    y += n_detcons;

    /* Compute parameter constraints. */
    for (std::size_t i = 0; i != n_paramcons; ++i) {
        y[i] = x[idx_paramcons[i]] - val_paramcons[i];
    }
}

template<class Wfn>
void Objective<Wfn>::jacobian(const SparseOp &op, const double *x, double *y)
{
    double e = x[nparam];

    /* Compute overlaps of determinants in projection space ("P space"):

           c_n
    */
    this->overlap(nproj, x, &ovlp[0]);

    /* Compute gradient of overlaps of determinants in connection space ("S space"):

           d(c_m)/d(p_k)
    */
    this->d_overlap(nconn, x, &d_ovlp[0]);

    /* Compute determinant constraint gradients */
    for (std::size_t j = 0, i; j != n_detcons; ++j) {
        for (i = 0; i != nparam; ++i) {
            y[(nproj + n_detcons + n_paramcons) * i + nproj + j] = d_ovlp[nconn * i + idx_detcons[j]];
        }
    }

    /* Compute parameter constraint gradients. */
    for (std::size_t i = 0; i != n_paramcons; ++i) {
        y[(nproj + n_detcons + n_paramcons) * idx_paramcons[i] + nproj + n_detcons + i] = 1;
    }

    /* Compute each column of the Jacobian:

           d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)

           E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
    */
    double *d_ovlp_col = &d_ovlp[0];
    for (std::size_t i = 0, j; i != nparam; ++i) {
        op.perform_op(d_ovlp_col, y);
        for (j = 0; j != nproj; ++j) {
            y[j] -= e * d_ovlp_col[j];
        }
        d_ovlp_col += nconn;
        y += nproj + n_detcons + n_paramcons;
    }

    /* Compute final Jacobian column:

           dE/d(p_k) <n|\Psi> = dE/d(p_k) \delta_{nk} c_n
    */
    std::size_t i = 0;
    for (; i != nproj; ++i) {
        y[i] = -ovlp[i];
    }
    for (; i != nproj + n_detcons + n_paramcons; ++i) {
        y[i] = 0.0;
    }
}

template<class Wfn>
Array<double> Objective<Wfn>::py_objective(const SparseOp &op, const Array<double> &x)
{
    Array<double> y(nproj + n_detcons + n_paramcons);
    objective(op, reinterpret_cast<const double *>(x.request().ptr),
              reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template<class Wfn>
ColMajorArray<double> Objective<Wfn>::py_jacobian(const SparseOp &op, const Array<double> &x)
{
    ColMajorArray<double> y({(long)(nproj + n_detcons + n_paramcons), (long)nparam + 1});
    jacobian(op, reinterpret_cast<const double *>(x.request().ptr),
             reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template<class Wfn>
Array<double> Objective<Wfn>::py_overlap(const Array<double> &x)
{
    Array<double> y(nconn);
    this->overlap(nconn, reinterpret_cast<const double *>(x.request().ptr),
              reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template<class Wfn>
ColMajorArray<double> Objective<Wfn>::py_d_overlap(const Array<double> &x)
{
    ColMajorArray<double> y({nconn, nparam});
    this->d_overlap(nconn, reinterpret_cast<const double *>(x.request().ptr),
              reinterpret_cast<double *>(y.request().ptr));
    return y;
}

template class Objective<DOCIWfn>;

template class Objective<FullCIWfn>;

template class Objective<GenCIWfn>;

template class Objective<NonSingletCI>;

} // namespace pyci
