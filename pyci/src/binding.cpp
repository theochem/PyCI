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

#define BEGIN_MODULE(NAME, MODULE) PYBIND11_MODULE(NAME, MODULE) {
#define END_MODULE }

namespace py = pybind11;

using namespace pyci;

/*
Section: Python C extension.
*/

BEGIN_MODULE(pyci, m)

/*
Section: Initialization
*/

py::options options;
options.disable_function_signatures();

m.doc() = "PyCI C++ extension module"
          " v" PYCI_VERSION
          " (" GIT_BRANCH ", " BUILD_TIME ")"
          " [" COMPILER_VERSION "]";

m.attr("__version__") = PYCI_VERSION;

m.attr("c_long") = py::dtype::of<long>();

m.attr("c_ulong") = py::dtype::of<ulong>();

m.attr("c_double") = py::dtype::of<double>();

char *env_threads = std::getenv("PYCI_NUM_THREADS");
set_num_threads((env_threads == nullptr) ? g_number_threads : std::atol(env_threads));

/*
Section: Second quantized operator class
*/

py::class_<SQuantOp> secondquant_op(m, "secondquant_op");

secondquant_op.doc() = R"""(
Second-quantized operator class.

For arbitrary-seniority systems:

.. math::

    O = \sum_{pq}{t_{pq} a^\dagger_p a_q} + \sum_{pqrs}{g_{pqrs} a^\dagger_p a^\dagger_q a_s a_r}

For seniority-zero systems:

    .. math::

    O = \sum_{p}{h_p N_p} + \sum_{p \neq q}{v_{pq} P^\dagger_p P_q} + \sum_{pq}{w_{pq} N_p N_q}

where

.. math::

    h_{p} = \left<p|T|p\right> = t_{pp}

.. math::

    v_{pq} = \left<pp|V|qq\right> = g_{ppqq}

.. math::

    w_{pq} = 2 \left<pq|V|pq\right> - \left<pq|V|qp\right> = 2 * g_{pqpq} - g_{pqqp}

)""";

secondquant_op.def_readonly("nbasis", &SQuantOp::nbasis, R"""(
Number of spatial orbital basis functions.

Returns
-------
nbasis : int
    Number of spatial orbital basis functions.

)""");

secondquant_op.def_readonly("ecore", &SQuantOp::ecore, R"""(
Constant (or \"zero-particle\") integral.

Returns
-------
ecore : float
    Constant (or \"zero-particle\") integral.
)""");

secondquant_op.def_readonly("one_mo", &SQuantOp::one_mo_array, R"""(
One-particle molecular integral array.

Returns
-------
one_mo : numpy.ndarray
    One-particle molecular integral array.

)""");

secondquant_op.def_readonly("two_mo", &SQuantOp::two_mo_array, R"""(
Two-particle molecular integral array.

Returns
-------
two_mo : numpy.ndarray
    Two-particle molecular integral array.

)""");

secondquant_op.def_readonly("h", &SQuantOp::h_array, R"""(
Seniority-zero one-particle molecular integral array.

Returns
-------
h : numpy.ndarray
    Seniority-zero one-particle molecular integral array.

)""");

secondquant_op.def_readonly("v", &SQuantOp::v_array, R"""(
Seniority-zero two-particle molecular integral array.

Returns
-------
v : numpy.ndarray
    Seniority-zero two-particle molecular integral array.

)""");

secondquant_op.def_readonly("w", &SQuantOp::w_array, R"""(
Seniority-two two-particle molecular integral array.

Returns
-------
w : numpy.ndarray
    Seniority-two two-particle molecular integral array.

)""");

secondquant_op.def(py::init<const std::string &>(), R"""(
Initialize a second-quantized operator instance.

If doing a generalized CI problem, the dimension of the operator should be equal to the total
number of spin-orbitals. Otherwise, if doing a DOCI or FullCI problem, it should be equal to the
number of spatial orbitals. This applies whether one is loading the operator from an FCIDUMP
file or from NumPy arrays.

Parameters
----------
filename : TextIO
    Name of FCIDUMP file to load.

or

Parameters
----------
ecore : float
    Constant (or "zero-particle") integral.
one_mo : np.ndarray
    One-particle molecular integral array.
two_mo : np.ndarray
    Two-particle molecular integral array.

)""",
                py::arg("filename"));

secondquant_op.def(py::init<const double, const Array<double>, const Array<double>>(),
                py::arg("ecore"), py::arg("one_mo"), py::arg("two_mo"));

secondquant_op.def("to_file", &SQuantOp::to_file, R"""(
Write this operator to an FCIDUMP file.

Parameters
----------
filename : TextIO
    Name of FCIDUMP file to write.
nelec : int, default=0
    Number of electrons to write.
ms2 : int, default=0
    Spin number to write.
tol : float, default=0.0
    Only write integrals with magnitude larger than this value.

)""",
                py::arg("filename"), py::arg("nelec") = 0, py::arg("ms2") = 0,
                py::arg("tol") = 0.0);

/*
Section: Wavefunction class
*/

py::class_<Wfn> wavefunction(m, "wavefunction");

wavefunction.doc() = R"""(
Wave function base class.
)""";

wavefunction.def_readonly("nbasis", &Wfn::nbasis, R"""(
Number of spatial orbital functions.

Returns
-------
nbasis : int
    Number of spatial orbital functions.

)""");

wavefunction.def_readonly("nocc", &Wfn::nocc, R"""(
Number of occupied spin-orbitals.

Returns
-------
nocc : int
    Number of occupied spin-orbitals.

)""");

wavefunction.def_readonly("nocc_up", &Wfn::nocc_up, R"""(
Number of occupied spin-up orbitals.

Returns
-------
nocc_up : int
    Number of occupied spin-up orbitals.

)""");

wavefunction.def_readonly("nocc_dn", &Wfn::nocc_dn, R"""(
Number of occupied spin-down orbitals.

Returns
-------
nocc_dn : int
    Number of occupied spin-down orbitals.

)""");

wavefunction.def_readonly("nvir", &Wfn::nvir, R"""(
Number of virtual spin-orbitals.

Returns
-------
nvir : int
    Number of virtual spin-orbitals.

)""");

wavefunction.def_readonly("nvir_up", &Wfn::nvir_up, R"""(
Number of virtual spin-up orbitals.

Returns
-------
nvir_up : int
    Number of virtual spin-up orbitals.

)""");

wavefunction.def_readonly("nvir_dn", &Wfn::nvir_dn, R"""(
Number of virtual spin-down orbitals.

Returns
-------
nvir_dn : int
    Number of virtual spin-down orbitals.

)""");

wavefunction.def("__len__", &Wfn::length, R"""(
Return the number of determinants in the wave function.

Returns
-------
ndet : int
    Number of determinants in the wave function.

)""");

wavefunction.def("squeeze", &Wfn::squeeze, "Free any unused memory allocated to this object.");

/*
Section: One-spin wavefunction class
*/

py::class_<OneSpinWfn, Wfn> one_spin_wfn(m, "one_spin_wfn");

one_spin_wfn.doc() = R"""(
One-spin wave function base class.
)""";

one_spin_wfn.def("__getitem__", &OneSpinWfn::py_getitem, R"""(
Return a determinant from the wave function.

Parameters
----------
index : int
    Index of determinant.

Returns
-------
array : numpy.ndarray
    Determinant.

)""",
                 py::arg("index"));

one_spin_wfn.def("to_file", &OneSpinWfn::to_file, R"""(
Write the wave function to a binary file.

Parameters
----------
filename : TextIO
    Name of the file to write.

)""",
                 py::arg("filename"));

one_spin_wfn.def("to_det_array", &OneSpinWfn::py_to_det_array, R"""(
Return a section of the wave function as a numpy.ndarray of determinants.

Arguments ``low`` and ``high`` behave like the first two arguments to ``range``. By default, if
neither are specified, the whole determinant is returned.

Parameters
----------
low : int, default=-1
    Beginning of section.
end : int, default=-1
    End of section.

Returns
-------
array : numpy.ndarray
    Array of determinants.

)""",
                 py::arg("low") = -1, py::arg("high") = -1);

one_spin_wfn.def("to_occ_array", &OneSpinWfn::py_to_occ_array, R"""(
Return a section of the wave function as a numpy.ndarray of occupation vectors.

Arguments ``low`` and ``high`` behave like the first two arguments to ``range()``. By default, if
neither are specified, the whole determinant is returned.

Parameters
----------
low : int, default=-1
    Beginning of section.
end : int, default=-1
    End of section.

Returns
-------
array : numpy.ndarray
    Array of occupation vectors.

)""",
                 py::arg("low") = -1, py::arg("high") = -1);

one_spin_wfn.def("index_det", &OneSpinWfn::py_index_det, R"""(
Return the index of determinant ``det`` in the wave function.

If the determinant is not in the wave function, this function returns -1.

Parameters
----------
det : numpy.ndarray
    Determinant.

Returns
-------
index : int
    Index of determinant or -1.

)""",
                 py::arg("det"));

one_spin_wfn.def("index_det_from_rank", &OneSpinWfn::index_det_from_rank, R"""(
Return the index of determinant with rank ``rank`` in the wave function.

If the determinant is not in the wave function, this function returns -1.

Parameters
----------
rank : int
    Rank of determinant.

Returns
-------
index : int
    Index of determinant or -1.

)""",
                 py::arg("rank"));

one_spin_wfn.def("rank_det", &OneSpinWfn::py_rank_det, R"""(
Return the rank of determinant ``det``.

Parameters
----------
det : numpy.ndarray
    Determinant.

Returns
-------
rank : int
    Rank of determinant.

)""",
                 py::arg("det"));

one_spin_wfn.def("add_det", &OneSpinWfn::py_add_det, R"""(
Add determinant ``det`` to the wave function.

Parameters
----------
det : numpy.ndarray
    Determinant.

Returns
-------
index : int
    Index of added determinant in wave function, or -1 if the determinant was not added.

)""",
                 py::arg("det"));

one_spin_wfn.def("add_occs", &OneSpinWfn::py_add_occs, R"""(
Add occupation vector ``occs`` to the wave function.

Parameters
----------
occs : numpy.ndarray
    Occupation vector.

Returns
-------
index : int
    Index of added determinant in wave function, or -1 if the determinant was not added.

)""",
                 py::arg("occs"));

one_spin_wfn.def("add_hartreefock_det", &OneSpinWfn::add_hartreefock_det,
                 "Add the Hartree-Fock determinant to the wave function.");

one_spin_wfn.def("add_all_dets", &OneSpinWfn::add_all_dets, R"""(
Add all determinants to the wave function.

Parameters
----------
nthread : int
    Number of threads to use.

)""",
                 py::arg("nthread") = -1);

one_spin_wfn.def("add_excited_dets", &OneSpinWfn::py_add_excited_dets, R"""(
Add excited determinants to the wave function.

Parameters
----------
exc : int
    Excitation order.
ref : numpy.ndarray, default=None
    Reference determinant. Default is the Hartree-Fock determinant.

)""",
                 py::arg("exc"), py::arg("ref") = py::none());

one_spin_wfn.def("add_dets_from_wfn", &OneSpinWfn::add_dets_from_wfn, R"""(
Add the determinants from another wave function.

Parameters
----------
wfn : pyci.one_spin_wfn
    Wave function.

)""",
                 py::arg("wfn"));

one_spin_wfn.def("reserve", &OneSpinWfn::reserve, R"""(
Reserve space in memory for ``n`` determinants in the wave function object.

Parameters
----------
n : int
    Number of determinants for which to reserve space.

)""",
                 py::arg("n"));

/*
Section: Two-spin wavefunction class
*/

py::class_<TwoSpinWfn, Wfn> two_spin_wfn(m, "two_spin_wfn");

two_spin_wfn.doc() = R"""(
Two-spin wave function base class.
)""";

two_spin_wfn.def("__getitem__", &TwoSpinWfn::py_getitem, R"""(
Return a determinant from the wave function.

Parameters
----------
index : int
    Index of determinant.

Returns
-------
array : numpy.ndarray
    Determinant.

)""",
                 py::arg("index"));

two_spin_wfn.def("to_file", &TwoSpinWfn::to_file, R"""(
Write the wave function to a binary file.

Parameters
----------
filename : TextIO
    Name of the file to write.

)""",
                 py::arg("filename"));

two_spin_wfn.def("to_det_array", &TwoSpinWfn::py_to_det_array, R"""(
Return a section of the wave function as a numpy.ndarray of determinants.

Arguments ``low`` and ``high`` behave like the first two arguments to ``range``. By default, if
neither are specified, the whole determinant is returned.

Parameters
----------
low : int, default=-1
    Beginning of section.
end : int, default=-1
    End of section.

Returns
-------
array : numpy.ndarray
    Array of determinants.

)""",
                 py::arg("low") = -1, py::arg("high") = -1);

two_spin_wfn.def("to_occ_array", &TwoSpinWfn::py_to_occ_array, R"""(
Return a section of the wave function as a numpy.ndarray of occupation vectors.

Arguments ``low`` and ``high`` behave like the first two arguments to ``range()``. By default, if
neither are specified, the whole determinant is returned.

Parameters
----------
low : int, default=-1
    Beginning of section.
end : int, default=-1
    End of section.

Returns
-------
array : numpy.ndarray
    Array of occupation vectors.

)""",
                 py::arg("low") = -1, py::arg("high") = -1);

two_spin_wfn.def("index_det", &TwoSpinWfn::py_index_det, R"""(
Return the index of determinant ``det`` in the wave function.

If the determinant is not in the wave function, this function returns -1.

Parameters
----------
det : numpy.ndarray
    Determinant.

Returns
-------
index : int
    Index of determinant or -1.

)""",
                 py::arg("det"));

two_spin_wfn.def("index_det_from_rank", &TwoSpinWfn::index_det_from_rank, R"""(
Return the index of determinant with rank ``rank`` in the wave function.

If the determinant is not in the wave function, this function returns -1.

Parameters
----------
rank : int
    Rank of determinant.

Returns
-------
index : int
    Index of determinant or -1.

)""",
                 py::arg("rank"));

two_spin_wfn.def("rank_det", &TwoSpinWfn::py_rank_det, R"""(
Return the rank of determinant ``det``.

Parameters
----------
det : numpy.ndarray
    Determinant.

Returns
-------
rank : int
    Rank of determinant.

)""",
                 py::arg("det"));

two_spin_wfn.def("add_det", &TwoSpinWfn::py_add_det, R"""(
Add determinant ``det`` to the wave function.

Parameters
----------
det : numpy.ndarray
    Determinant.

Returns
-------
index : int
    Index of added determinant in wave function, or -1 if the determinant was not added.

)""",
                 py::arg("det"));

two_spin_wfn.def("add_occs", &TwoSpinWfn::py_add_occs, R"""(
Add occupation vector ``occs`` to the wave function.

Parameters
----------
occs : numpy.ndarray
    Occupation vector.

Returns
-------
index : int
    Index of added determinant in wave function, or -1 if the determinant was not added.

)""",
                 py::arg("occs"));

two_spin_wfn.def("add_hartreefock_det", &TwoSpinWfn::add_hartreefock_det,
                 "Add the Hartree-Fock determinant to the wave function.");

two_spin_wfn.def("add_all_dets", &TwoSpinWfn::add_all_dets, R"""(
Add all determinants to the wave function.

Parameters
----------
nthread : int
    Number of threads to use.

)""",
                 py::arg("nthread") = -1);

two_spin_wfn.def("add_excited_dets", &TwoSpinWfn::py_add_excited_dets, R"""(
Add excited determinants to the wave function.

Parameters
----------
exc : int
    Excitation order.
ref : numpy.ndarray, default=None
    Reference determinant. Default is the Hartree-Fock determinant.

)""",
                 py::arg("exc"), py::arg("ref") = py::none());

two_spin_wfn.def("add_dets_from_wfn", &TwoSpinWfn::add_dets_from_wfn, R"""(
Add the determinants from another wave function.

Parameters
----------
wfn : pyci.two_spin_wfn
    Wave function.

)""",
                 py::arg("wfn"));

two_spin_wfn.def("reserve", &TwoSpinWfn::reserve, R"""(
Reserve space in memory for ``n`` determinants in the wave function object.

Parameters
----------
n : int
    Number of determinants for which to reserve space.

)""",
                 py::arg("n"));

/*
Section: DOCI wave function class
*/

py::class_<DOCIWfn, OneSpinWfn> doci_wfn(m, "doci_wfn");

doci_wfn.doc() = R"""(
DOCI wave function base class.
)""";

doci_wfn.def(py::init<const DOCIWfn &>(), R"""(
Initialize a DOCI wave function.

Parameters
----------
wfn : pyci.doci_wfn
    Wave function from which to copy data.

or

Parameters
----------
filename : TextIO
    Filename of binary file from which to load wave function.

or

Parameters
----------
nbasis : int
    Number of spatial orbital functions.
nocc_up : int
    Number of occupied spin-up orbitals.
nocc_dn : int
    Number of occupied spin-down orbitals.

)""",
             py::arg("wfn"));

doci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

doci_wfn.def(py::init<const long, const long, const long>(), py::arg("nbasis"), py::arg("nocc_up"),
             py::arg("nocc_dn"));

doci_wfn.def(py::init<const long, const long, const long, const Array<ulong>>(), py::arg("nbasis"),
             py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

doci_wfn.def(py::init<const long, const long, const long, const Array<long>>(), py::arg("nbasis"),
             py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

/*
Section: FullCI wave function class
*/

py::class_<FullCIWfn, TwoSpinWfn> fullci_wfn(m, "fullci_wfn");

fullci_wfn.doc() = R"""(
FullCI wave function base class.
)""";

fullci_wfn.def(py::init<const DOCIWfn &>(), R"""(
Initialize a FullCI wave function.

Parameters
----------
wfn : (pyci.doci_wfn | pyci.fullci_wfn)
    Wave function from which to copy data.

or

Parameters
----------
filename : TextIO
    Filename of binary file from which to load wave function.

or

Parameters
----------
nbasis : int
    Number of spatial orbital functions.
nocc_up : int
    Number of occupied spin-up orbitals.
nocc_dn : int
    Number of occupied spin-down orbitals.

)""",
               py::arg("wfn"));

fullci_wfn.def(py::init<const FullCIWfn &>(), py::arg("wfn"));

fullci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

fullci_wfn.def(py::init<const long, const long, const long>(), py::arg("nbasis"),
               py::arg("nocc_up"), py::arg("nocc_dn"));

fullci_wfn.def(py::init<const long, const long, const long, const Array<ulong>>(),
               py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

fullci_wfn.def(py::init<const long, const long, const long, const Array<long>>(), py::arg("nbasis"),
               py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

/*
Section: GenCI wave function class
*/

py::class_<GenCIWfn, OneSpinWfn> genci_wfn(m, "genci_wfn");

genci_wfn.doc() = R"""(
Generalized CI wave function base class.
)""";

genci_wfn.def(py::init<const DOCIWfn &>(), R"""(
Initialize a Generalized CI wave function.

Parameters
----------
wfn : (pyci.doci_wfn | pyci.fullci_wfn | pyci.genci_wfn)
    Wave function from which to copy data.

or

Parameters
----------
filename : TextIO
    Filename of binary file from which to load wave function.

or

Parameters
----------
nbasis : int
    Number of spatial orbital functions.
nocc_up : int
    Number of occupied spin-up orbitals.
nocc_dn : int
    Number of occupied spin-down orbitals.

)""",
              py::arg("wfn"));

genci_wfn.def(py::init<const FullCIWfn &>(), py::arg("wfn"));

genci_wfn.def(py::init<const GenCIWfn &>(), py::arg("wfn"));

genci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

genci_wfn.def(py::init<const long, const long, const long>(), py::arg("nbasis"), py::arg("nocc_up"),
              py::arg("nocc_dn"));

genci_wfn.def(py::init<const long, const long, const long, const Array<ulong>>(), py::arg("nbasis"),
              py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

genci_wfn.def(py::init<const long, const long, const long, const Array<long>>(), py::arg("nbasis"),
              py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

/*
Section: NonSingletCI wave function class
*/

py::class_<NonSingletCI, GenCIWfn> nonsingletci_wfn(m, "nonsingletci_wfn");

nonsingletci_wfn.doc() = R"""(
Non-singlet CI wave function base class.
)""";

nonsingletci_wfn.def(py::init<const DOCIWfn &>(), R"""(
Initialize a Non-singlet CI wave function.

Parameters
----------
wfn : (pyci.doci_wfn | pyci.fullci_wfn | pyci.genci_wfn | pyci.nonsingletci_wfn)
    Wave function from which to copy data.

or

Parameters
----------
filename : TextIO
    Filename of binary file from which to load wave function.

or

Parameters
----------
nbasis : int
    Number of spatial orbital functions.
nocc_up : int
    Number of occupied spin-up orbitals.
nocc_dn : int
    Number of occupied spin-down orbitals.

)""",
              py::arg("wfn"));

nonsingletci_wfn.def(py::init<const FullCIWfn &>(), py::arg("wfn"));

nonsingletci_wfn.def(py::init<const NonSingletCI &>(), py::arg("wfn"));

nonsingletci_wfn.def(py::init<const std::string &>(), py::arg("filename"));

nonsingletci_wfn.def(py::init<const long, const long, const long>(), py::arg("nbasis"), py::arg("nocc_up"),
              py::arg("nocc_dn"));

nonsingletci_wfn.def(py::init<const long, const long, const long, const Array<ulong>>(), py::arg("nbasis"),
              py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

nonsingletci_wfn.def(py::init<const long, const long, const long, const Array<long>>(), py::arg("nbasis"),
              py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("array"));

nonsingletci_wfn.def("fill_hartreefock_det", &NonSingletCI::fill_hartreefock_det, R"""(
Fill the Hartree-Fock determinant in the given ptr. 

Parameters
----------
nb2 : int 
    Number of basis functions for total number of spin orbitals.
    Because of the way GenCI handles bit strings, it will be 
    nonsingletci_wfn.nbasis =  ham.nbasis * 2.
nocc : int
    Total number of occupied orbitals.
det : array
    Pointer to the array where the Hartree-Fock determinant will be stored. 
)""",
              py::arg("nb2"), py::arg("nocc"), py::arg("det"));

nonsingletci_wfn.def("add_excited_dets", &NonSingletCI::py_add_excited_dets, R"""(
Add excited determinants to the wave function.

Parameters
----------
exc : int
    Excitation order.
ref : numpy.ndarray, default=None
    Reference determinant. Default is the Hartree-Fock determinant.

)""",
              py::arg("exc"), py::arg("ref") = py::none());


/*
Section: Sparse CI matrix operator class
*/

py::class_<SparseOp> sparse_op(m, "sparse_op");

sparse_op.doc() = R"""(
Sparse matrix operator class.
)""";

sparse_op.def_readonly("ecore", &SparseOp::ecore, R"""(
Constant (or "zero-particle") integral.

Returns
-------
ecore : float
    Constant (or "zero-particle") integral.

)""");

sparse_op.def_readonly("symmetric", &SparseOp::symmetric, R"""(
Whether the sparse matrix operator is symmetric/Hermitian.

Returns
-------
symmetric : bool
    Whether the sparse matrix operator is symmetric/Hermitian.

)""");

sparse_op.def_readonly("size", &SparseOp::size, R"""(
Number of non-zero matrix elements.

Returns
-------
size : int
    Number of non-zero matrix elements.

)""");

sparse_op.def_readonly("shape", &SparseOp::shape, R"""(
Shape of the matrix.

Returns
-------
nrow : int
    Number of rows.
ncol : int
    Number of columns.

)""");

sparse_op.def_property_readonly("dtype", &SparseOp::dtype, R"""(
/*
Section: Sparse CI matrix operator class
*/

py::class_<SparseOp> sparse_op(m, "sparse_op");

sparse_op.doc() = R"""(
Sparse matrix operator class.
)""");

sparse_op.def_readonly("ecore", &SparseOp::ecore, R"""(
Constant (or "zero-particle") integral.

Returns
-------
ecore : float
    Constant (or "zero-particle") integral.

)""");

sparse_op.def_readonly("symmetric", &SparseOp::symmetric, R"""(
Whether the sparse matrix operator is symmetric/Hermitian.

Returns
-------
symmetric : bool
    Whether the sparse matrix operator is symmetric/Hermitian.

)""");

sparse_op.def_readonly("size", &SparseOp::size, R"""(
Number of non-zero matrix elements.

Returns
-------
size : int
    Number of non-zero matrix elements.

)""");

sparse_op.def_readonly("shape", &SparseOp::shape, R"""(
Shape of the matrix.

Returns
-------
nrow : int
    Number of rows.
ncol : int
    Number of columns.

)""");

sparse_op.def_property_readonly("dtype", &SparseOp::dtype, R"""(
Data type of matrix.

Returns
-------
dtype : numpy.dtype
    Data type of matrix.

)""");

sparse_op.def(py::init<const SQuantOp &, const DOCIWfn &, const long, const long, const bool>(), R"""(
Initialize a sparse matrix operator.

Parameters
----------
ham : pyci.secondquant_op
    Hamiltonian.
wfn : pyci.wavefunction
    Wave function.
nrow : int, default=len(wfn)
    Number of rows in matrix, using the first ``nrow`` determinants in ``wfn``.
ncol : int, default=len(wfn)
    Number of columns in matrix, using the first ``ncol`` determinants in ``wfn``.
symmetric : bool, default=False
    Whether to make the sparse matrix operator symmetric/Hermitian.

)""",
              py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1,
              py::arg("symmetric") = true);

sparse_op.def(py::init<const SQuantOp &, const FullCIWfn &, const long, const long, const bool>(),
              py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1,
              py::arg("symmetric") = true);

sparse_op.def(py::init<const SQuantOp &, const GenCIWfn &, const long, const long, const bool>(),
              py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1,
              py::arg("symmetric") = true);
sparse_op.def(py::init<const SQuantOp &, const NonSingletCI &, const long, const long, const bool, const std::string>(),
              py::arg("ham"), py::arg("wfn"), py::arg("nrow") = -1, py::arg("ncol") = -1,
              py::arg("symmetric") = true, py::arg("wfntype") = "nonsingletci");

sparse_op.def("update", &SparseOp::py_update<DOCIWfn>, R"""(
Update a sparse matrix operator for the HCI algorithm.

Parameters
----------
ham : pyci.secondquant_op
    Hamiltonian.
wfn : pyci.wavefunction
    Wave function.

Notes
-----
The user is responsible for using the same ``ham`` and ``wfn``.
This method doesn't work with rectangular operators. Those must
be re-initialized from the wave function object.

)""",
              py::arg("ham"), py::arg("wfn"));

sparse_op.def("update", &SparseOp::py_update<FullCIWfn>, py::arg("ham"), py::arg("wfn"));

sparse_op.def("update", &SparseOp::py_update<GenCIWfn>, py::arg("ham"), py::arg("wfn"));

sparse_op.def("update", &SparseOp::py_update<NonSingletCI>, py::arg("ham"), py::arg("wfn"));

sparse_op.def("__call__", &SparseOp::py_matvec, R"""(
Compute the matrix vector product of the sparse matrix operator with vector ``x``.

.. math::

    A \mathbf{x} = \mathbf{y}

Parameters
----------
x : numpy.ndarray
    Vector to which the operator will be applied.
out : numpy.ndarray, default=None
    Array in which to store the result. One will be created if this is not specified.

Returns
-------
y : numpy.ndarray
    Result vector.

)""",
              py::arg("x"));

sparse_op.def("__call__", &SparseOp::py_matvec_out, py::arg("x"), py::arg("out"));

sparse_op.def("matvec", &SparseOp::py_matvec, R"""(
Compute the matrix vector product of the sparse matrix operator with vector ``x``.

.. math::

    A \mathbf{x} = \mathbf{y}

Parameters
----------
x : numpy.ndarray
    Vector to which the operator will be applied.
out : numpy.ndarray, default=None
    Array in which to store the result. One will be created if this is not specified.

Returns
-------
y : numpy.ndarray
    Result vector.

)""",
              py::arg("x"));

sparse_op.def("matvec", &SparseOp::py_matvec_out, py::arg("x"), py::arg("out"));

sparse_op.def("get_element", &SparseOp::get_element, R"""(
Return the :math:`\left(i, j\right)`-th element of the sparse matrix operator.
)""",
              py::arg("i"), py::arg("j"));

sparse_op.def("solve", &SparseOp::py_solve_ci, R"""(
Solve a CI eigenproblem.

Parameters
----------
n : int, default=1
    Number of lowest eigenpairs to find.
c0 : np.ndarray, default=[1,0,...,0]
    Initial guess for lowest eigenvector.
ncv : int, default=min(nrow, max(2 * n + 1, 20))
    Number of Lanczos vectors to use.
maxiter : int, default=nrow * n * 10
    Maximum number of iterations to perform.
tol : float, default=1.0e-12
    Convergence tolerance.

Returns
-------
es : np.ndarray
    Energies.
cs : np.ndarray
    Coefficient vectors.

)""",
              py::arg("n") = 1, py::arg("c0") = py::none(), py::arg("ncv") = -1,
              py::arg("maxiter") = -1, py::arg("tol") = 1.0e-12);

sparse_op.def("reserve", &SparseOp::reserve, R"""(
Reserve space in memory for ``n`` nonzero elements in the sparse matrix operator.

Parameters
----------
n : int
    Number of elements for which to reserve space.

)""",
              py::arg("n"));

sparse_op.def("squeeze", &SparseOp::squeeze, "Free any unused memory allocated to this object.");

/*
Section: Free functions
*/

m.def("get_num_threads", &get_num_threads, R"""(
Return the default number of threads to use.

Returns
-------
nthread : int
    Number of threads.

)""");

m.def("set_num_threads", &set_num_threads, R"""(
Set the default number of threads to use.

Parameters
----------
nthread : int
    Number of threads.

)""",
      py::arg("n"));

m.def("popcnt", &py_popcnt, R"""(
Return the number of bits set to 1 in a determinant array.

Parameters
----------
det : numpy.ndarray
    Determinant array.

Returns
-------
pop : int
    Number of bits set to 1.

)""",
      py::arg("det"));

m.def("ctz", &py_ctz, R"""(
Return the number of trailing zeros in a determinant array.

Parameters
----------
det : numpy.ndarray
    Determinant array.

Returns
-------
ctz : int
    Number of trailing zeros.

)""",
      py::arg("det"));

m.def("add_hci", &py_add_hci<DOCIWfn>, R"""(
Add determinants to a wave function by running an iteration of Heat-Bath CI [HCI1]_.

.. [HCI1] Holmes, Adam A., Norm M. Tubman, and C. J. Umrigar. "Heat-bath configuration
          interaction: An efficient selected configuration interaction algorithm
          inspired by heat-bath sampling." *Journal of chemical theory and computation*
          12.8 (2016): 3674-3680.

Parameters
----------
ham : pyci.secondquant_op
    Hamiltonian.
wfn : pyci.wavefunction
    Wave function.
coeffs : numpy.ndarray
    Coefficient vector.
eps : float, default=1.0e-5
    :math:`\epsilon` value for Heat-Bath CI routine.
nthread : int
    Number of threads to use.

Returns
-------
ndet : int
    Number of determinants added.

)""",
      py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("eps") = 1.0e-5,
      py::arg("nthread") = -1);

m.def("add_hci", &py_add_hci<FullCIWfn>, py::arg("ham"), py::arg("wfn"), py::arg("coeffs"),
      py::arg("eps") = 1.0e-5, py::arg("nthread") = -1);

m.def("add_hci", &py_add_hci<GenCIWfn>, py::arg("ham"), py::arg("wfn"), py::arg("coeffs"),
      py::arg("eps") = 1.0e-5, py::arg("nthread") = -1);

m.def("compute_overlap", &py_compute_overlap<OneSpinWfn>, R"""(
Compute the overlap :math:`\left<\Psi_1|\Psi_2\right>` of two wave functions.

Parameters
----------
wfn1 : pyci.wavefunction
    First wave function.
wfn2 : pyci.wavefunction
    Second wave function.
coeffs1 : numpy.ndarray
    First coefficient vector.
coeffs2 : numpy.ndarray
    Second coefficient vector.

Returns
-------
olp : float
    Overlap.

)""",
      py::arg("wfn1"), py::arg("wfn2"), py::arg("coeffs1"), py::arg("coeffs2"));

m.def("compute_overlap", &py_compute_overlap<TwoSpinWfn>, py::arg("wfn1"), py::arg("wfn2"),
      py::arg("coeffs1"), py::arg("coeffs2"));

m.def("compute_rdms", &py_compute_rdms_doci, R"""(
Compute the one- and two- particle reduced density matrices (RDMs) of a wave function.

Parameters
----------
wfn : pyci.wavefunction
    Wave function.
coeffs : numpy.ndarray
    Coefficient vector.

Returns
-------
d1 : numpy.ndarray
    One-particle RDM matrix.
d2 : numpy.ndarray
    Two-particle RDM matrix.

Notes
-----
For DOCI wave functions, this method returns two nbasis-by-nbasis matrices, which include the unique
seniority-zero and seniority-two terms from the full 2-RDMs:

.. math::

    D_0 = \left<pp|qq\right>

.. math::

    D_2 = \left<pq|pq\right>

The diagonal elements of :math:`D_0` are equal to the 1-RDM elements :math:`\left<p|p\right>`.

For FullCI wave functions, the leading dimension of ``rdm1`` has length 2 and specifies the
spin-block 0) "up-up" or 1) "down-down", and the leading dimensions of ``rdm2`` has length 3 and
specifies the spin-block 0) "up-up-up-up", 1) "down-down-down-down', or 2) "up-down-up-down".

For Generalized CI wave functions, ``rdm1`` and ``rdm2`` are the full 1-RDM and 2-RDM, respectively.

)""",
      py::arg("wfn"), py::arg("coeffs"));

m.def("compute_rdms", &py_compute_rdms_fullci, py::arg("wfn"), py::arg("coeffs"));

m.def("compute_rdms", &py_compute_rdms_genci, py::arg("wfn"), py::arg("coeffs"));

m.def("compute_transition_rdms", &py_compute_transition_rdms_doci, R"""(
Compute the one- and two- particle transition reduced density matrices (RDMs) of two wave functions.

Parameters
----------
wfn1 : pyci.wavefunction
    Wave function.
wfn2 : pyci.wavefunction
    Wave function.
coeffs1 : numpy.ndarray
    Coefficient vector.
coeffs2 : numpy.ndarray
    Coefficient vector.

Returns
-------
d1 : numpy.ndarray
    One-particle TRDM matrix.
d2 : numpy.ndarray
    Two-particle TRDM matrix.

Notes
-----
For DOCI wave functions, this method returns two nbasis-by-nbasis matrices, which include the unique
seniority-zero and seniority-two terms from the full 2-TRDMs:

.. math::

    D_0 = \left<pp|qq\right>

.. math::

    D_2 = \left<pq|pq\right>

The diagonal elements of :math:`D_0` are equal to the 1-TRDM elements :math:`\left<p'|p\right>`.

For FullCI wave functions, the leading dimension of ``rdm1`` has length 2 and specifies the
spin-block 0) "up-up" or 1) "down-down", and the leading dimensions of ``rdm2`` has length 3 and
specifies the spin-block 0) "up-up-up-up", 1) "down-down-down-down', or 2) "up-down-up-down".

For Generalized CI wave functions, ``rdm1`` and ``rdm2`` are the full 1-TRDM and 2-TRDM, respectively.

)""",
      py::arg("wfn1"), py::arg("wfn2"), py::arg("coeffs1"), py::arg("coeffs2"));

m.def("compute_transition_rdms", &py_compute_transition_rdms_fullci,
      py::arg("wfn1"), py::arg("wfn2"), py::arg("coeffs1"), py::arg("coeffs2"));

m.def("compute_transition_rdms", &py_compute_transition_rdms_genci,
      py::arg("wfn1"), py::arg("wfn2"), py::arg("coeffs1"), py::arg("coeffs2"));

m.def("compute_enpt2", &py_compute_enpt2<DOCIWfn>, R"""(
Compute the second-order multi-reference Epstein-Nesbet (ENPT2) energy for a wave function.

Parameters
----------
ham : pyci.secondquant_op
    Hamiltonian.
wfn : pyci.wavefunction
    Wave function.
coeffs : numpy.ndarray
    Coefficient vector.
energy : float
    Variational CI energy for this wave function and Hamiltonian.
eps : float, default=1.0e-5
    :math:`\epsilon` value for ENPT2 routine.
nthread : int
    Number of threads to use.

Returns
-------
pt_energy : float
    ENPT2 energy.

)""",
      py::arg("ham"), py::arg("wfn"), py::arg("coeffs"), py::arg("energy"), py::arg("eps") = 1.0e-5,
      py::arg("nthread") = -1);

m.def("compute_enpt2", &py_compute_enpt2<FullCIWfn>, py::arg("ham"), py::arg("wfn"),
      py::arg("coeffs"), py::arg("energy"), py::arg("eps") = 1.0e-5, py::arg("nthread") = -1);

m.def("compute_enpt2", &py_compute_enpt2<GenCIWfn>, py::arg("ham"), py::arg("wfn"),
      py::arg("coeffs"), py::arg("energy"), py::arg("eps") = 1.0e-5, py::arg("nthread") = -1);

/*
Section: FanCI classes
*/

static constexpr char DOCSTRING_OVERLAP[] = R"""(
)""";

static constexpr char DOCSTRING_D_OVERLAP[] = R"""(
)""";

static constexpr char DOCSTRING_OBJECTIVE[] = R"""(
)""";

static constexpr char DOCSTRING_JACOBIAN[] = R"""(
)""";

py::class_<Objective<DOCIWfn>> doci_objective(m, "DOCIObjective");

doci_objective.doc() = R"""(
DOCI-FanCI objective class.
)""";

doci_objective.def("overlap", &Objective<DOCIWfn>::py_overlap, DOCSTRING_OVERLAP,
    py::arg("x"));

doci_objective.def("d_overlap", &Objective<DOCIWfn>::py_d_overlap, DOCSTRING_D_OVERLAP,
    py::arg("x"));

doci_objective.def("objective", &Objective<DOCIWfn>::py_objective, DOCSTRING_OBJECTIVE,
    py::arg("op"), py::arg("x"));

doci_objective.def("jacobian", &Objective<DOCIWfn>::py_jacobian, DOCSTRING_JACOBIAN,
    py::arg("op"), py::arg("x"));

py::class_<Objective<FullCIWfn>> fullci_objective(m, "FullCIObjective");

fullci_objective.doc() = R"""(
FullCI-FanCI objective class.
)""";

fullci_objective.def("overlap", &Objective<FullCIWfn>::py_overlap, DOCSTRING_OVERLAP,
    py::arg("x"));

fullci_objective.def("d_overlap", &Objective<FullCIWfn>::py_d_overlap, DOCSTRING_D_OVERLAP,
    py::arg("x"));

fullci_objective.def("objective", &Objective<FullCIWfn>::py_objective, DOCSTRING_OBJECTIVE,
    py::arg("op"), py::arg("x"));

fullci_objective.def("jacobian", &Objective<FullCIWfn>::py_jacobian, DOCSTRING_JACOBIAN,
    py::arg("op"), py::arg("x"));

py::class_<Objective<GenCIWfn>> genci_objective(m, "GenCIObjective");

genci_objective.doc() = R"""(
GenCI-FanCI objective class.
)""";

genci_objective.def("overlap", &Objective<GenCIWfn>::py_overlap, DOCSTRING_OVERLAP,
    py::arg("x"));

genci_objective.def("d_overlap", &Objective<GenCIWfn>::py_d_overlap, DOCSTRING_D_OVERLAP,
    py::arg("x"));

genci_objective.def("objective", &Objective<GenCIWfn>::py_objective, DOCSTRING_OBJECTIVE,
    py::arg("op"), py::arg("x"));

genci_objective.def("jacobian", &Objective<GenCIWfn>::py_jacobian, DOCSTRING_JACOBIAN,
    py::arg("op"), py::arg("x"));


py::class_<Objective<NonSingletCI>> nonsingletci_objective(m, "NonSingletCIObjective");

nonsingletci_objective.doc() = R"""(
NonSingletCI-FanCI objective class.
)""";

nonsingletci_objective.def("overlap", &Objective<NonSingletCI>::py_overlap, DOCSTRING_OVERLAP,
    py::arg("x"));

nonsingletci_objective.def("d_overlap", &Objective<NonSingletCI>::py_d_overlap, DOCSTRING_D_OVERLAP,
    py::arg("x"));

nonsingletci_objective.def("objective", &Objective<NonSingletCI>::py_objective, DOCSTRING_OBJECTIVE,
    py::arg("op"), py::arg("x"));

nonsingletci_objective.def("jacobian", &Objective<NonSingletCI>::py_jacobian, DOCSTRING_JACOBIAN,
    py::arg("op"), py::arg("x"));

py::class_<AP1roGObjective, Objective<DOCIWfn>> ap1rog_objective(m, "AP1roGObjective");

ap1rog_objective.doc() = R"""(
AP1roG objective class.
)""";

ap1rog_objective.def(py::init<const SparseOp &, const DOCIWfn &, const py::object, const py::object, const py::object, const py::object>(),
R"""(
Initialize the AP1roG objective instance.

Parameters
----------
op : pyci.sparse_op
    Sparse operator instance with ``nproj`` rows ("P" space) and ``nconn`` columns ("S" space).
wfn : pyci.doci_wfn
    DOCI wave function with ``nconn`` determinants.
    The first ``nproj`` determinants should correspond to the "S" space.
idx_det_cons : np.ndarray(Nd, dtype=pyci.c_long)
    The indices of the determinants on which constraints should be placed.
det_cons : np.ndarray(Nd, dtype=pyci.c_double)
    The values to which the constrained determinants' overlaps should be equal.
idx_param_cons : np.ndarray(Np, dtype=pyci.c_long)
    The indices of the parameters on which constraints should be placed.
param_cons : np.ndarray(Np, dtype=pyci.c_double)
    The values to which the constrained parameters should be equal.

)""",
    py::arg("op"), py::arg("wfn"), py::arg("idx_det_cons") = py::none(), py::arg("det_cons") = py::none(),
    py::arg("idx_param_cons") = py::none(), py::arg("param_cons") = py::none());

py::class_<APIGObjective, Objective<DOCIWfn>> apig_objective(m, "APIGObjective");

apig_objective.doc() = R"""(
APIG objective class.
)""";

apig_objective.def(py::init<const SparseOp &, const DOCIWfn &, const py::object, const py::object, const py::object, const py::object>(),
R"""(
Initialize the APIG objective instance.

Parameters
----------
op : pyci.sparse_op
    Sparse operator instance with ``nproj`` rows ("P" space) and ``nconn`` columns ("S" space).
wfn : pyci.doci_wfn
    DOCI wave function with ``nconn`` determinants.
    The first ``nproj`` determinants should correspond to the "S" space.
idx_det_cons : np.ndarray(Nd, dtype=pyci.c_long)
    The indices of the determinants on which constraints should be placed.
det_cons : np.ndarray(Nd, dtype=pyci.c_double)
    The values to which the constrained determinants' overlaps should be equal.
idx_param_cons : np.ndarray(Np, dtype=pyci.c_long)
    The indices of the parameters on which constraints should be placed.
param_cons : np.ndarray(Np, dtype=pyci.c_double)
    The values to which the constrained parameters should be equal.

)""",
    py::arg("op"), py::arg("wfn"), py::arg("idx_det_cons") = py::none(), py::arg("det_cons") = py::none(),
    py::arg("idx_param_cons") = py::none(), py::arg("param_cons") = py::none());

// Declare Python class based on AP1roGeneralizedSenoObjective
py::class_<AP1roGeneralizedSenoObjective, Objective<NonSingletCI>> ap1rogen_objective(m, "AP1roGeneralizedSenoObjective");

// Class documentation
ap1rogen_objective.doc() = R"""(
AP1roGSDGeneralized_sen-o objective class.
)""";

// Initializer (keep in mind the Wfn argument should be the right class)
ap1rogen_objective.def(py::init<const SparseOp &, const NonSingletCI &, const py::object, const py::object, const py::object, const py::object>(),
R"""(
Initialize the AP1roGSDGeneralized_sen-o objective instance.

Parameters
----------
op : pyci.sparse_op
    Sparse operator instance with ``nproj`` rows ("P" space) and ``nconn`` columns ("S" space).
wfn : pyci.doci_wfn
    NonSingletCI wave function with ``nconn`` determinants.
    The first ``nproj`` determinants should correspond to the "S" space.
idx_det_cons : np.ndarray(Nd, dtype=pyci.c_long)
    The indices of the determinants on which constraints should be placed.
det_cons : np.ndarray(Nd, dtype=pyci.c_double)
    The values to which the constrained determinants' overlaps should be equal.
idx_param_cons : np.ndarray(Np, dtype=pyci.c_long)
    The indices of the parameters on which constraints should be placed.
param_cons : np.ndarray(Np, dtype=pyci.c_double)
    The values to which the constrained parameters should be equal.

)""",
    py::arg("op"), py::arg("wfn"), py::arg("idx_det_cons") = py::none(), py::arg("det_cons") = py::none(),
    py::arg("idx_param_cons") = py::none(), py::arg("param_cons") = py::none());

END_MODULE
