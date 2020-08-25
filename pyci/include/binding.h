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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pyci.h>

#include <vector>

#ifndef PYCI_VERSION
#define PYCI_VERSION 0.0.0
#endif
#define LITERAL(S) #S
#define STRINGIZE(S) LITERAL(S)

namespace py = pybind11;

namespace pyci {

typedef typename py::array_t<int_t, py::array::c_style | py::array::forcecast> i_array_t;

typedef typename py::array_t<uint_t, py::array::c_style | py::array::forcecast> u_array_t;

typedef typename py::array_t<double, py::array::c_style | py::array::forcecast> d_array_t;

int_t wfn_length(const Wfn &wfn) {
    return wfn.ndet;
}

u_array_t onespinwfn_getitem(const OneSpinWfn &wfn, int_t index) {
    u_array_t array(wfn.nword);
    wfn.copy_det(index, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

u_array_t twospinwfn_getitem(const TwoSpinWfn &wfn, int_t index) {
    u_array_t array({static_cast<int_t>(2), wfn.nword});
    wfn.copy_det(index, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

u_array_t onespinwfn_to_det_array(const OneSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    u_array_t array({end - start, wfn.nword});
    wfn.to_det_array(start, end, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

u_array_t twospinwfn_to_det_array(const TwoSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    u_array_t array({end - start, static_cast<int_t>(2), wfn.nword});
    wfn.to_det_array(start, end, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

i_array_t onespinwfn_to_occ_array(const OneSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    i_array_t array({end - start, wfn.nocc_up});
    wfn.to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

i_array_t twospinwfn_to_occ_array(const TwoSpinWfn &wfn, int_t start, int_t end) {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = wfn.ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    i_array_t array({end - start, static_cast<int_t>(2), wfn.nocc_up});
    wfn.to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

int_t onespinwfn_index_det(const OneSpinWfn &wfn, const u_array_t det) {
    return wfn.index_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t twospinwfn_index_det(const TwoSpinWfn &wfn, const u_array_t det) {
    return wfn.index_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

uint_t onespinwfn_rank_det(const OneSpinWfn &wfn, const u_array_t det) {
    return wfn.rank_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

uint_t twospinwfn_rank_det(const TwoSpinWfn &wfn, const u_array_t det) {
    return wfn.rank_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t onespinwfn_add_det(OneSpinWfn &wfn, const u_array_t det) {
    return wfn.add_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t twospinwfn_add_det(TwoSpinWfn &wfn, const u_array_t det) {
    return wfn.add_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t onespinwfn_add_excited_dets(OneSpinWfn &wfn, const int_t exc, const py::object ref) {
    std::vector<uint_t> v_ref;
    uint_t *ptr;
    if (ref.is(py::none())) {
        v_ref.resize(wfn.nword);
        ptr = &v_ref[0];
        fill_hartreefock_det(wfn.nocc_up, ptr);
    } else
        ptr = reinterpret_cast<uint_t *>(ref.cast<u_array_t>().request().ptr);
    int_t ndet_old = wfn.ndet;
    wfn.add_excited_dets(ptr, exc);
    return wfn.ndet - ndet_old;
}

int_t twospinwfn_add_excited_dets(TwoSpinWfn &wfn, const int_t exc, const py::object ref) {
    std::vector<uint_t> v_ref;
    uint_t *ptr;
    if (ref.is(py::none())) {
        v_ref.resize(wfn.nword2);
        ptr = &v_ref[0];
        fill_hartreefock_det(wfn.nocc_up, ptr);
        fill_hartreefock_det(wfn.nocc_dn, ptr + wfn.nword);
    } else
        ptr = reinterpret_cast<uint_t *>(ref.cast<u_array_t>().request().ptr);
    int_t ndet_old = wfn.ndet;
    int_t maxup = (wfn.nocc_up < wfn.nvir_up) ? wfn.nocc_up : wfn.nvir_up;
    int_t maxdn = (wfn.nocc_dn < wfn.nvir_dn) ? wfn.nocc_dn : wfn.nvir_dn;
    int_t a = (exc < maxup) ? exc : maxup;
    int_t b = exc - a;
    while ((a >= 0) && (b <= maxdn))
        wfn.add_excited_dets(ptr, a--, b++);
    return wfn.ndet - ndet_old;
}

int_t onespinwfn_add_occs(OneSpinWfn &wfn, const i_array_t occs) {
    return wfn.add_det_from_occs(reinterpret_cast<const int_t *>(occs.request().ptr));
}

int_t twospinwfn_add_occs(TwoSpinWfn &wfn, const i_array_t occs) {
    return wfn.add_det_from_occs(reinterpret_cast<const int_t *>(occs.request().ptr));
}

d_array_t sparse_op_matvec(const SparseOp &op, const d_array_t x) {
    py::buffer_info buf = x.request();
    d_array_t y(op.nrow);
    op.perform_op(reinterpret_cast<const double *>(buf.ptr),
                  reinterpret_cast<double *>(y.request().ptr));
    return y;
}

d_array_t sparse_op_matvec_cepa0(const SparseOp &op, const d_array_t x, const int_t refind) {
    py::buffer_info buf = x.request();
    d_array_t y(op.nrow);
    op.perform_op_cepa0(reinterpret_cast<const double *>(buf.ptr),
                        reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

d_array_t sparse_op_rmatvec_cepa0(const SparseOp &op, const d_array_t x, const int_t refind) {
    py::buffer_info buf = x.request();
    d_array_t y(op.ncol);
    op.perform_op_transpose_cepa0(reinterpret_cast<const double *>(buf.ptr),
                                  reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

d_array_t sparse_op_rhs_cepa0(const SparseOp &op, const int_t refind) {
    d_array_t y(op.nrow);
    op.rhs_cepa0(reinterpret_cast<double *>(y.request().ptr), refind);
    return y;
}

int_t py_popcnt(const u_array_t det) {
    py::buffer_info buf = det.request();
    return popcnt_det(buf.shape[0], reinterpret_cast<const uint_t *>(buf.ptr));
}

int_t py_ctz(const u_array_t det) {
    py::buffer_info buf = det.request();
    return ctz_det(buf.shape[0], reinterpret_cast<const uint_t *>(buf.ptr));
}

int_t dociwfn_add_hci(const Ham &ham, DOCIWfn &wfn, const d_array_t coeffs, const double eps) {
    return add_hci(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps);
}

int_t fullciwfn_add_hci(const Ham &ham, FullCIWfn &wfn, const d_array_t coeffs, const double eps) {
    return add_hci(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps);
}

int_t genciwfn_add_hci(const Ham &ham, GenCIWfn &wfn, const d_array_t coeffs, const double eps) {
    return add_hci(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps);
}

double dociwfn_compute_overlap(const DOCIWfn &wfn1, const DOCIWfn &wfn2, const d_array_t coeffs1,
                               const d_array_t coeffs2) {
    return compute_overlap(wfn1, wfn2, reinterpret_cast<const double *>(coeffs1.request().ptr),
                           reinterpret_cast<const double *>(coeffs2.request().ptr));
}

double fullciwfn_compute_overlap(const FullCIWfn &wfn1, const FullCIWfn &wfn2,
                                 const d_array_t coeffs1, const d_array_t coeffs2) {
    return compute_overlap(wfn1, wfn2, reinterpret_cast<const double *>(coeffs1.request().ptr),
                           reinterpret_cast<const double *>(coeffs2.request().ptr));
}

double genciwfn_compute_overlap(const GenCIWfn &wfn1, const GenCIWfn &wfn2, const d_array_t coeffs1,
                                const d_array_t coeffs2) {
    return compute_overlap(wfn1, wfn2, reinterpret_cast<const double *>(coeffs1.request().ptr),
                           reinterpret_cast<const double *>(coeffs2.request().ptr));
}

py::tuple dociwfn_compute_rdms(const DOCIWfn &wfn, const d_array_t coeffs) {
    d_array_t d0({wfn.nbasis, wfn.nbasis});
    d_array_t d2({wfn.nbasis, wfn.nbasis});
    compute_rdms(wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                 reinterpret_cast<double *>(d0.request().ptr),
                 reinterpret_cast<double *>(d2.request().ptr));
    return py::make_tuple(d0, d2);
}

py::tuple fullciwfn_compute_rdms(const FullCIWfn &wfn, const d_array_t coeffs) {
    d_array_t rdm1({static_cast<int_t>(2), wfn.nbasis, wfn.nbasis});
    d_array_t rdm2({static_cast<int_t>(3), wfn.nbasis, wfn.nbasis, wfn.nbasis, wfn.nbasis});
    compute_rdms(wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                 reinterpret_cast<double *>(rdm1.request().ptr),
                 reinterpret_cast<double *>(rdm2.request().ptr));
    return py::make_tuple(rdm1, rdm2);
}

py::tuple genciwfn_compute_rdms(const GenCIWfn &wfn, const d_array_t coeffs) {
    d_array_t rdm1({wfn.nbasis, wfn.nbasis});
    d_array_t rdm2({wfn.nbasis, wfn.nbasis, wfn.nbasis, wfn.nbasis});
    compute_rdms(wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                 reinterpret_cast<double *>(rdm1.request().ptr),
                 reinterpret_cast<double *>(rdm2.request().ptr));
    return py::make_tuple(rdm1, rdm2);
}

double dociwfn_compute_enpt2(const Ham &ham, const DOCIWfn &wfn, const d_array_t coeffs,
                             const double energy, const double eps) {
    return compute_enpt2(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), energy,
                         eps);
}

double fullciwfn_compute_enpt2(const Ham &ham, const FullCIWfn &wfn, const d_array_t coeffs,
                               const double energy, const double eps) {
    return compute_enpt2(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), energy,
                         eps);
}

double genciwfn_compute_enpt2(const Ham &ham, const GenCIWfn &wfn, const d_array_t coeffs,
                              const double energy, const double eps) {
    return compute_enpt2(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), energy,
                         eps);
}

template<class ArrayType>
struct PyHamTemplate final : public Ham {
public:
    ArrayType one_mo_array, two_mo_array, h_array, v_array, w_array;

    PyHamTemplate(const PyHamTemplate &ham)
        : Ham(ham), one_mo_array(ham.one_mo_array), two_mo_array(ham.two_mo_array),
          h_array(ham.h_array), v_array(ham.v_array), w_array(ham.w_array) {
    }

    PyHamTemplate(PyHamTemplate &&ham) noexcept
        : Ham(ham), one_mo_array(std::move(ham.one_mo_array)),
          two_mo_array(std::move(ham.two_mo_array)), h_array(std::move(ham.h_array)),
          v_array(std::move(ham.v_array)), w_array(std::move(ham.w_array)) {
    }

    PyHamTemplate(const std::string &filename) : Ham() {
        py::tuple args = py::module::import("pyci.fcidump").attr("_load_ham")(filename);
        init(args);
    }

    PyHamTemplate(const double e, const ArrayType mo1, const ArrayType mo2) : Ham() {
        py::tuple args = py::module::import("pyci.fcidump").attr("_load_ham")(e, mo1, mo2);
        init(args);
    }

    void to_file(const std::string &filename, const int_t nelec, const int_t ms2,
                 const double tol) const {
        py::module::import("pyci.fcidump")
            .attr("write_fcidump")(filename, Ham::ecore, one_mo_array, two_mo_array, nelec, ms2,
                                   tol);
    }

private:
    void init(const py::tuple &args) {
        one_mo_array = args[1].cast<ArrayType>();
        two_mo_array = args[2].cast<ArrayType>();
        h_array = args[3].cast<ArrayType>();
        v_array = args[4].cast<ArrayType>();
        w_array = args[5].cast<ArrayType>();
        Ham::nbasis = one_mo_array.request().shape[0];
        Ham::ecore = args[0].cast<double>();
        Ham::one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
        Ham::two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
        Ham::h = reinterpret_cast<double *>(h_array.request().ptr);
        Ham::v = reinterpret_cast<double *>(v_array.request().ptr);
        Ham::w = reinterpret_cast<double *>(w_array.request().ptr);
    }
};

} // namespace pyci
