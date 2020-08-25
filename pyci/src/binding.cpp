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

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pyci.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef PYCI_VERSION
#define PYCI_VERSION 0.0.0
#endif
#define LITERAL(S) #S
#define STRINGIZE(S) LITERAL(S)

namespace py = pybind11;

using namespace pyci;

/*
Section: Pybind11 typedefs
*/

typedef typename py::array_t<int_t, py::array::c_style | py::array::forcecast> i_array_t;

typedef typename py::array_t<uint_t, py::array::c_style | py::array::forcecast> u_array_t;

typedef typename py::array_t<double, py::array::c_style | py::array::forcecast> d_array_t;

/*
Section: Python wave function interface functions
*/

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

/*
Section: Python sparse matrix operator interface functions
*/

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

/*
Section: Other Python interface functions
*/

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

/*
Section: Python C extension.
*/

PYBIND11_MODULE(pyci, m) {

    /*
    Section: Python interface for Hamiltonian class

    Note: classes with py::array_t as members must be declared within the PYBIND_MODULE() construct.
    */

    struct PyHam final : public Ham {
    public:
        d_array_t one_mo_array, two_mo_array, h_array, v_array, w_array;

        PyHam(const PyHam &ham)
            : Ham(ham), one_mo_array(ham.one_mo_array), two_mo_array(ham.two_mo_array),
              h_array(ham.h_array), v_array(ham.v_array), w_array(ham.w_array) {
        }

        PyHam(PyHam &&ham) noexcept
            : Ham(ham), one_mo_array(std::move(ham.one_mo_array)),
              two_mo_array(std::move(ham.two_mo_array)), h_array(std::move(ham.h_array)),
              v_array(std::move(ham.v_array)), w_array(std::move(ham.w_array)) {
        }

        PyHam(const std::string &filename) : Ham() {
            py::tuple args = py::module::import("pyci.fcidump").attr("_load_ham")(filename);
            init(args);
        }

        PyHam(const double e, const d_array_t mo1, const d_array_t mo2) : Ham() {
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
            one_mo_array = args[1].cast<d_array_t>();
            two_mo_array = args[2].cast<d_array_t>();
            h_array = args[3].cast<d_array_t>();
            v_array = args[4].cast<d_array_t>();
            w_array = args[5].cast<d_array_t>();
            Ham::nbasis = one_mo_array.request().shape[0];
            Ham::ecore = args[0].cast<double>();
            Ham::one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
            Ham::two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
            Ham::h = reinterpret_cast<double *>(h_array.request().ptr);
            Ham::v = reinterpret_cast<double *>(v_array.request().ptr);
            Ham::w = reinterpret_cast<double *>(w_array.request().ptr);
        }
    };

    /*
    Section: Initialization
    */

    m.doc() = "PyCI C extension module.";

    m.attr("__version__") = STRINGIZE(PYCI_VERSION);
    m.attr("c_int") = py::dtype::of<int_t>();
    m.attr("c_uint") = py::dtype::of<uint_t>();
    m.attr("c_double") = py::dtype::of<double>();

    if (std::getenv("OMP_NUM_THREADS") == nullptr)
        omp_set_num_threads(1);

    /*
    Section: Hamiltonian class
    */

    py::class_<Ham> base_ham(m, "__base_ham");
    base_ham.def_readonly("nbasis", &Ham::nbasis);
    base_ham.def_readonly("ecore", &Ham::ecore);

    py::class_<PyHam, Ham> hamiltonian(m, "hamiltonian");
    hamiltonian.doc() = "Hamiltonian class.";

    hamiltonian.def_readonly("one_mo", &PyHam::one_mo_array);
    hamiltonian.def_readonly("two_mo", &PyHam::two_mo_array);
    hamiltonian.def_readonly("h", &PyHam::h_array);
    hamiltonian.def_readonly("v", &PyHam::v_array);
    hamiltonian.def_readonly("w", &PyHam::w_array);

    hamiltonian.def(py::init<const std::string &>(), py::arg("filename"));

    hamiltonian.def(py::init<const double, const d_array_t, const d_array_t>(), py::arg("ecore"),
                    py::arg("one_mo"), py::arg("two_mo"));

    hamiltonian.def("to_file", &PyHam::to_file, py::arg("filename"), py::arg("nelec") = 0,
                    py::arg("ms2") = 0, py::arg("tol") = 1.0e-18);

    /*
    Section: Wavefunction class
    */

    py::class_<Wfn> wavefunction(m, "wavefunction");
    wavefunction.doc() = "Wave function class.";

    wavefunction.def_readonly("nbasis", &Wfn::nbasis);
    wavefunction.def_readonly("nocc", &Wfn::nocc);
    wavefunction.def_readonly("nocc_up", &Wfn::nocc_up);
    wavefunction.def_readonly("nocc_dn", &Wfn::nocc_dn);
    wavefunction.def_readonly("nvir", &Wfn::nvir);
    wavefunction.def_readonly("nvir_up", &Wfn::nvir_up);
    wavefunction.def_readonly("nvir_dn", &Wfn::nvir_dn);

    wavefunction.def("__len__", &wfn_length);

    wavefunction.def("squeeze", &Wfn::squeeze);

    /*
    Section: One-spin wavefunction class
    */

    py::class_<OneSpinWfn, Wfn> one_spin_wfn(m, "one_spin_wfn");
    one_spin_wfn.doc() = "Single-spin wave function class.";

    one_spin_wfn.def("__getitem__", &onespinwfn_getitem, py::arg("index"));

    one_spin_wfn.def("to_file", &OneSpinWfn::to_file, py::arg("filename"));

    one_spin_wfn.def("to_det_array", &onespinwfn_to_det_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    one_spin_wfn.def("to_occ_array", &onespinwfn_to_occ_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    one_spin_wfn.def("index_det", &onespinwfn_index_det, py::arg("det"));

    one_spin_wfn.def("index_det_from_rank", &OneSpinWfn::index_det_from_rank, py::arg("rank"));

    one_spin_wfn.def("rank_det", &onespinwfn_rank_det, py::arg("det"));

    one_spin_wfn.def("add_det", &onespinwfn_add_det, py::arg("det"));

    one_spin_wfn.def("add_occs", &onespinwfn_add_occs, py::arg("occs"));

    one_spin_wfn.def("add_hartreefock_det", &OneSpinWfn::add_hartreefock_det);

    one_spin_wfn.def("add_all_dets", &OneSpinWfn::add_all_dets);

    one_spin_wfn.def("add_excited_dets", &onespinwfn_add_excited_dets, py::arg("exc"),
                     py::arg("ref") = py::none());

    one_spin_wfn.def("add_dets_from_wfn", &OneSpinWfn::add_dets_from_wfn, py::arg("wfn"));

    one_spin_wfn.def("reserve", &OneSpinWfn::reserve, py::arg("n"));

    /*
    Section: Two-spin wavefunction class
    */

    py::class_<TwoSpinWfn, Wfn> two_spin_wfn(m, "two_spin_wfn");
    two_spin_wfn.doc() = "Two-spin wave function class.";

    two_spin_wfn.def("__getitem__", &twospinwfn_getitem, py::arg("index"));

    two_spin_wfn.def("to_file", &TwoSpinWfn::to_file, py::arg("filename"));

    two_spin_wfn.def("to_det_array", &twospinwfn_to_det_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    two_spin_wfn.def("to_occ_array", &twospinwfn_to_occ_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    two_spin_wfn.def("index_det", &twospinwfn_index_det, py::arg("det"));

    two_spin_wfn.def("index_det_from_rank", &TwoSpinWfn::index_det_from_rank, py::arg("rank"));

    two_spin_wfn.def("rank_det", &twospinwfn_rank_det, py::arg("det"));

    two_spin_wfn.def("add_det", &twospinwfn_add_det, py::arg("det"));

    two_spin_wfn.def("add_occs", &twospinwfn_add_occs, py::arg("occs"));

    two_spin_wfn.def("add_hartreefock_det", &TwoSpinWfn::add_hartreefock_det);

    two_spin_wfn.def("add_all_dets", &TwoSpinWfn::add_all_dets);

    two_spin_wfn.def("add_excited_dets", &twospinwfn_add_excited_dets, py::arg("exc"),
                     py::arg("ref") = py::none());

    two_spin_wfn.def("add_dets_from_wfn", &TwoSpinWfn::add_dets_from_wfn, py::arg("wfn"));

    two_spin_wfn.def("reserve", &TwoSpinWfn::reserve, py::arg("n"));

    /*
    Section: DOCI wave function class
    */

    py::class_<DOCIWfn, OneSpinWfn> doci_wfn(m, "doci_wfn");
    doci_wfn.doc() = "DOCI wave function class.";

    doci_wfn.def(py::init<const DOCIWfn &>(), py::arg("wfn"));

    doci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

    doci_wfn.def(py::init<const int_t, const int_t, const int_t>(), py::arg("nbasis"),
                 py::arg("nocc_up"), py::arg("nocc_dn"));

    doci_wfn.def(py::init([](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn,
                             const u_array_t array) {
                     py::buffer_info buf = array.request();
                     return DOCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0],
                                    reinterpret_cast<const uint_t *>(buf.ptr));
                 }),
                 py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    doci_wfn.def(py::init([](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn,
                             const i_array_t array) {
                     py::buffer_info buf = array.request();
                     return DOCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0],
                                    reinterpret_cast<const int_t *>(buf.ptr));
                 }),
                 py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    /*
    Section: FullCI wave function class
    */

    py::class_<FullCIWfn, TwoSpinWfn> fullci_wfn(m, "fullci_wfn");
    fullci_wfn.doc() = "FullCI wave function class.";

    fullci_wfn.def(py::init<const DOCIWfn &>(), py::arg("wfn"));
    fullci_wfn.def(py::init<const FullCIWfn &>(), py::arg("wfn"));

    fullci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

    fullci_wfn.def(py::init<const int_t, const int_t, const int_t>(), py::arg("nbasis"),
                   py::arg("nocc_up"), py::arg("nocc_dn"));

    fullci_wfn.def(py::init([](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn,
                               const u_array_t array) {
                       py::buffer_info buf = array.request();
                       return FullCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0],
                                        reinterpret_cast<const uint_t *>(buf.ptr));
                   }),
                   py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    fullci_wfn.def(py::init([](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn,
                               const i_array_t array) {
                       py::buffer_info buf = array.request();
                       return FullCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0],
                                        reinterpret_cast<const int_t *>(buf.ptr));
                   }),
                   py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    /*
    Section: GenCI wave function class
    */

    py::class_<GenCIWfn, OneSpinWfn> genci_wfn(m, "genci_wfn");
    genci_wfn.doc() = "Generalized CI wave function class.";

    genci_wfn.def(py::init<const DOCIWfn &>(), py::arg("wfn"));
    genci_wfn.def(py::init<const FullCIWfn &>(), py::arg("wfn"));
    genci_wfn.def(py::init<const GenCIWfn &>(), py::arg("wfn"));

    genci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

    genci_wfn.def(py::init<const int_t, const int_t, const int_t>(), py::arg("nbasis"),
                  py::arg("nocc_up"), py::arg("nocc_dn"));

    genci_wfn.def(py::init([](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn,
                              const u_array_t array) {
                      py::buffer_info buf = array.request();
                      return GenCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0],
                                      reinterpret_cast<const uint_t *>(buf.ptr));
                  }),
                  py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    genci_wfn.def(py::init([](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn,
                              const i_array_t array) {
                      py::buffer_info buf = array.request();
                      return GenCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0],
                                      reinterpret_cast<const int_t *>(buf.ptr));
                  }),
                  py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    /*
    Section: Sparse CI matrix operator class
    */

    py::class_<SparseOp> sparse_op(m, "sparse_op");
    sparse_op.doc() = "Sparse CI matrix operator class.";

    sparse_op.def_readonly("ecore", &SparseOp::ecore);
    sparse_op.def_readonly("size", &SparseOp::size);

    sparse_op.def_property_readonly(
        "shape", [](const SparseOp &op) { return py::make_tuple(op.nrow, op.ncol); });

    sparse_op.def(py::init<const Ham &, const DOCIWfn &, const int_t, const int_t>(),
                  py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1);

    sparse_op.def(py::init<const Ham &, const FullCIWfn &, const int_t, const int_t>(),
                  py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1);

    sparse_op.def(py::init<const Ham &, const GenCIWfn &, const int_t, const int_t>(),
                  py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1);

    sparse_op.def("__call__", &sparse_op_matvec, py::arg("x"));

    sparse_op.def("get_element", &SparseOp::get_element, py::arg("i"), py::arg("j"));

    sparse_op.def("matvec_cepa0", &sparse_op_matvec_cepa0, py::arg("x"), py::arg("refind") = 0);

    sparse_op.def("rmatvec_cepa0", &sparse_op_rmatvec_cepa0, py::arg("x"), py::arg("refind") = 0);

    sparse_op.def("rhs_cepa0", &sparse_op_rhs_cepa0, py::arg("refind") = 0);

    /*
    Section: Free functions
    */

    m.def("set_num_threads", &omp_set_num_threads, py::arg("nthread"));

    m.def("get_num_threads", &omp_get_max_threads);

    m.def("popcnt", &py_popcnt, py::arg("det"));
    m.def("ctz", &py_ctz, py::arg("det"));

    m.def(
        "add_hci",
        [](const PyHam &ham, DOCIWfn &wfn, const d_array_t coeffs, const double eps) {
            return add_hci(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps);
        },
        py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("eps") = 1.0e-5);

    m.def(
        "add_hci",
        [](const PyHam &ham, FullCIWfn &wfn, const d_array_t coeffs, const double eps) {
            return add_hci(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps);
        },
        py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("eps") = 1.0e-5);

    m.def(
        "add_hci",
        [](const PyHam &ham, GenCIWfn &wfn, const d_array_t coeffs, const double eps) {
            return add_hci(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps);
        },
        py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("eps") = 1.0e-5);

    m.def("compute_overlap", &dociwfn_compute_overlap, py::arg("wfn1"), py::arg("wfn2"),
          py::arg("coeffs1"), py::arg("coeffs2"));

    m.def("compute_overlap", &fullciwfn_compute_overlap, py::arg("wfn1"), py::arg("wfn2"),
          py::arg("coeffs1"), py::arg("coeffs2"));

    m.def("compute_overlap", &genciwfn_compute_overlap, py::arg("wfn1"), py::arg("wfn2"),
          py::arg("coeffs1"), py::arg("coeffs2"));

    m.def("compute_rdms", &dociwfn_compute_rdms, py::arg("wfn"), py::arg("coeff"));
    m.def("compute_rdms", &fullciwfn_compute_rdms, py::arg("wfn"), py::arg("coeff"));
    m.def("compute_rdms", &genciwfn_compute_rdms, py::arg("wfn"), py::arg("coeff"));

    m.def(
        "compute_enpt2",
        [](const PyHam &ham, DOCIWfn &wfn, const d_array_t coeffs, const double energy,
           const double eps) {
            return compute_enpt2(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                                 energy, eps);
        },
        py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("energy"),
        py::arg("eps") = 1.0e-5);

    m.def(
        "compute_enpt2",
        [](const PyHam &ham, FullCIWfn &wfn, const d_array_t coeffs, const double energy,
           const double eps) {
            return compute_enpt2(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                                 energy, eps);
        },
        py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("energy"),
        py::arg("eps") = 1.0e-5);

    m.def(
        "compute_enpt2",
        [](const PyHam &ham, GenCIWfn &wfn, const d_array_t coeffs, const double energy,
           const double eps) {
            return compute_enpt2(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                                 energy, eps);
        },
        py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("energy"),
        py::arg("eps") = 1.0e-5);

} // PYBIND_MODULE
