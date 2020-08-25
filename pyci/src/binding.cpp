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
#include <pyci.h>

#include <cstdlib>

namespace py = pybind11;

using namespace pyci;

/*
Section: Python C extension.
*/

PYBIND11_MODULE(pyci, m) {

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

    py::class_<Ham> hamiltonian(m, "hamiltonian");
    hamiltonian.doc() = "Hamiltonian class.";

    hamiltonian.def_readonly("nbasis", &Ham::nbasis);
    hamiltonian.def_readonly("ecore", &Ham::ecore);

    hamiltonian.def_readonly("one_mo", &Ham::one_mo_array);
    hamiltonian.def_readonly("two_mo", &Ham::two_mo_array);
    hamiltonian.def_readonly("h", &Ham::h_array);
    hamiltonian.def_readonly("v", &Ham::v_array);
    hamiltonian.def_readonly("w", &Ham::w_array);

    hamiltonian.def(py::init<const std::string &>(), py::arg("filename"));

    hamiltonian.def(py::init<const double, const Array<double>, const Array<double>>(),
                    py::arg("ecore"), py::arg("one_mo"), py::arg("two_mo"));

    hamiltonian.def("to_file", &Ham::to_file, py::arg("filename"), py::arg("nelec") = 0,
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

    wavefunction.def("__len__", &Wfn::length);

    wavefunction.def("squeeze", &Wfn::squeeze);

    /*
    Section: One-spin wavefunction class
    */

    py::class_<OneSpinWfn, Wfn> one_spin_wfn(m, "one_spin_wfn");
    one_spin_wfn.doc() = "Single-spin wave function class.";

    one_spin_wfn.def("__getitem__", &OneSpinWfn::py_getitem, py::arg("index"));

    one_spin_wfn.def("to_file", &OneSpinWfn::to_file, py::arg("filename"));

    one_spin_wfn.def("to_det_array", &OneSpinWfn::py_to_det_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    one_spin_wfn.def("to_occ_array", &OneSpinWfn::py_to_occ_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    one_spin_wfn.def("index_det", &OneSpinWfn::py_index_det, py::arg("det"));

    one_spin_wfn.def("index_det_from_rank", &OneSpinWfn::index_det_from_rank, py::arg("rank"));

    one_spin_wfn.def("rank_det", &OneSpinWfn::py_rank_det, py::arg("det"));

    one_spin_wfn.def("add_det", &OneSpinWfn::py_add_det, py::arg("det"));

    one_spin_wfn.def("add_occs", &OneSpinWfn::py_add_occs, py::arg("occs"));

    one_spin_wfn.def("add_hartreefock_det", &OneSpinWfn::add_hartreefock_det);

    one_spin_wfn.def("add_all_dets", &OneSpinWfn::add_all_dets);

    one_spin_wfn.def("add_excited_dets", &OneSpinWfn::py_add_excited_dets, py::arg("exc"),
                     py::arg("ref") = py::none());

    one_spin_wfn.def("add_dets_from_wfn", &OneSpinWfn::add_dets_from_wfn, py::arg("wfn"));

    one_spin_wfn.def("reserve", &OneSpinWfn::reserve, py::arg("n"));

    /*
    Section: Two-spin wavefunction class
    */

    py::class_<TwoSpinWfn, Wfn> two_spin_wfn(m, "two_spin_wfn");
    two_spin_wfn.doc() = "Two-spin wave function class.";

    two_spin_wfn.def("__getitem__", &TwoSpinWfn::py_getitem, py::arg("index"));

    two_spin_wfn.def("to_file", &TwoSpinWfn::to_file, py::arg("filename"));

    two_spin_wfn.def("to_det_array", &TwoSpinWfn::py_to_det_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    two_spin_wfn.def("to_occ_array", &TwoSpinWfn::py_to_occ_array, py::arg("low") = -1,
                     py::arg("high") = -1);

    two_spin_wfn.def("index_det", &TwoSpinWfn::py_index_det, py::arg("det"));

    two_spin_wfn.def("index_det_from_rank", &TwoSpinWfn::index_det_from_rank, py::arg("rank"));

    two_spin_wfn.def("rank_det", &TwoSpinWfn::py_rank_det, py::arg("det"));

    two_spin_wfn.def("add_det", &TwoSpinWfn::py_add_det, py::arg("det"));

    two_spin_wfn.def("add_occs", &TwoSpinWfn::py_add_occs, py::arg("occs"));

    two_spin_wfn.def("add_hartreefock_det", &TwoSpinWfn::add_hartreefock_det);

    two_spin_wfn.def("add_all_dets", &TwoSpinWfn::add_all_dets);

    two_spin_wfn.def("add_excited_dets", &TwoSpinWfn::py_add_excited_dets, py::arg("exc"),
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

    doci_wfn.def(py::init<const int_t, const int_t, const int_t, const Array<uint_t>>(),
                 py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    doci_wfn.def(py::init<const int_t, const int_t, const int_t, const Array<int_t>>(),
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

    fullci_wfn.def(py::init<const int_t, const int_t, const int_t, const Array<uint_t>>(),
                   py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    fullci_wfn.def(py::init<const int_t, const int_t, const int_t, const Array<int_t>>(),
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

    genci_wfn.def(py::init<const int_t, const int_t, const int_t, const Array<uint_t>>(),
                  py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

    genci_wfn.def(py::init<const int_t, const int_t, const int_t, const Array<int_t>>(),
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

    sparse_op.def("__call__", &SparseOp::py_matvec, py::arg("x"));

    sparse_op.def("get_element", &SparseOp::get_element, py::arg("i"), py::arg("j"));

    sparse_op.def("matvec_cepa0", &SparseOp::py_matvec_cepa0, py::arg("x"), py::arg("refind") = 0);

    sparse_op.def("rmatvec_cepa0", &SparseOp::py_rmatvec_cepa0, py::arg("x"),
                  py::arg("refind") = 0);

    sparse_op.def("rhs_cepa0", &SparseOp::py_rhs_cepa0, py::arg("refind") = 0);

    /*
    Section: Free functions
    */

    m.def("set_num_threads", &omp_set_num_threads, py::arg("nthread"));

    m.def("get_num_threads", &omp_get_max_threads);

    m.def("popcnt", &py_popcnt, py::arg("det"));
    m.def("ctz", &py_ctz, py::arg("det"));

    m.def("add_hci", &py_dociwfn_add_hci, py::arg("ham"), py::arg("wfn"), py::arg("coeffs"),
          py::arg("eps") = 1.0e-5);

    m.def("add_hci", &py_fullciwfn_add_hci, py::arg("ham"), py::arg("wfn"), py::arg("coeffs"),
          py::arg("eps") = 1.0e-5);

    m.def("add_hci", &py_genciwfn_add_hci, py::arg("ham"), py::arg("wfn"), py::arg("coeffs"),
          py::arg("eps") = 1.0e-5);

    m.def("compute_overlap", &py_dociwfn_compute_overlap, py::arg("wfn1"), py::arg("wfn2"),
          py::arg("coeffs1"), py::arg("coeffs2"));

    m.def("compute_overlap", &py_fullciwfn_compute_overlap, py::arg("wfn1"), py::arg("wfn2"),
          py::arg("coeffs1"), py::arg("coeffs2"));

    m.def("compute_overlap", &py_genciwfn_compute_overlap, py::arg("wfn1"), py::arg("wfn2"),
          py::arg("coeffs1"), py::arg("coeffs2"));

    m.def("compute_rdms", &py_dociwfn_compute_rdms, py::arg("wfn"), py::arg("coeff"));
    m.def("compute_rdms", &py_fullciwfn_compute_rdms, py::arg("wfn"), py::arg("coeff"));
    m.def("compute_rdms", &py_genciwfn_compute_rdms, py::arg("wfn"), py::arg("coeff"));

    m.def("compute_enpt2", &py_dociwfn_compute_enpt2, py::arg("ham"), py::arg("wfn"),
          py::arg("coeffs"), py::arg("energy"), py::arg("eps") = 1.0e-5);

    m.def("compute_enpt2", &py_fullciwfn_compute_enpt2, py::arg("ham"), py::arg("wfn"),
          py::arg("coeffs"), py::arg("energy"), py::arg("eps") = 1.0e-5);

    m.def("compute_enpt2", &py_genciwfn_compute_enpt2, py::arg("ham"), py::arg("wfn"),
          py::arg("coeffs"), py::arg("energy"), py::arg("eps") = 1.0e-5);

} // PYBIND_MODULE
