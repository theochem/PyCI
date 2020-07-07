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
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <pyci.h>

#include "SpookyV2.cpp"
#include "common.cpp"
#include "onespin.cpp"
#include "twospin.cpp"
#include "rdm.cpp"
#include "hci.cpp"
#include "enpt2.cpp"
#include "solve.cpp"


namespace py = pybind11;


using namespace pyci;


/* Pybind11 typedefs. */


typedef typename py::array_t<int_t, py::array::c_style | py::array::forcecast> i_array_t;


typedef typename py::array_t<uint_t, py::array::c_style | py::array::forcecast> u_array_t;


typedef typename py::array_t<double, py::array::c_style | py::array::forcecast> d_array_t;


/* Python C extension. */


PYBIND11_MODULE(pyci, m) {


/* Hamiltonian C++ classes. */


struct Hamiltonian
{
    int_t nbasis;
    double ecore;
    d_array_t one_mo, two_mo, h, v, w;

    Hamiltonian(void) {
    }

    Hamiltonian(Hamiltonian &&ham) noexcept
        : nbasis(std::exchange(ham.nbasis, 0)), ecore(std::exchange(ham.ecore, 0)),
        one_mo(std::move(one_mo)), two_mo(std::move(two_mo)),
        h(std::move(h)), v(std::move(v)), w(std::move(w)) {
    }

    Hamiltonian(const Hamiltonian &ham)
        : nbasis(ham.nbasis), ecore(ham.ecore), one_mo(ham.one_mo), two_mo(ham.two_mo),
          h(ham.h), v(ham.v), w(ham.w) {
    }

    void init_rest(const double ecore_, const d_array_t one_mo_, const d_array_t two_mo_,
            const bool keep_mo, const bool doci) {
        py::buffer_info buf1 = one_mo_.request();
        py::buffer_info buf2 = two_mo_.request();
        if ((buf1.ndim != 2) || (buf2.ndim != 4) || (buf1.shape[0] != buf1.shape[1])
                || (buf1.shape[0] != buf2.shape[0]) || (buf1.shape[0] != buf2.shape[1])
                || (buf1.shape[0] != buf2.shape[2]) || (buf1.shape[0] != buf2.shape[3]))
            throw std::domain_error("one_mo/two_mo have mismatched dimensions");
        nbasis = buf1.shape[0];
        ecore = ecore_;
        if (keep_mo) {
            one_mo = one_mo_;
            two_mo = two_mo_;
        } else {
            one_mo = py::none();
            two_mo = py::none();
        }
        if (doci) {
            py::tuple senzero = py::module::import("pyci.utils").attr("make_senzero_integrals")(one_mo_, two_mo_);
            h = senzero[0].cast<d_array_t>();
            v = senzero[1].cast<d_array_t>();
            w = senzero[2].cast<d_array_t>();
        } else {
            h = py::none();
            v = py::none();
            w = py::none();
        }
    }

    void init_unrest(const double ecore_, const d_array_t one_mo_, const d_array_t two_mo_,
            const bool keep_mo, const bool doci) {
        py::buffer_info buf1 = one_mo_.request();
        py::buffer_info buf2 = two_mo_.request();
        if ((buf1.ndim != 3) || (buf1.shape[0] != 2) || (buf1.shape[1] != buf1.shape[2])
                || (buf2.ndim != 5) || (buf2.shape[0] != 3)
                || (buf1.shape[1] != buf2.shape[1]) || (buf1.shape[1] != buf2.shape[2])
                || (buf1.shape[1] != buf2.shape[3]) || (buf1.shape[1] != buf2.shape[4]))
            throw std::domain_error("one_mo/two_mo have mismatched dimensions");
        nbasis = buf1.shape[1];
        ecore = ecore_;
        if (keep_mo) {
            one_mo = one_mo_;
            two_mo = two_mo_;
        } else {
            one_mo = py::none();
            two_mo = py::none();
        }
        if (doci) {
            py::tuple senzero = py::module::import("pyci.utils").attr("make_senzero_integrals")(one_mo_, two_mo_);
            h = senzero[0].cast<d_array_t>();
            v = senzero[1].cast<d_array_t>();
            w = senzero[2].cast<d_array_t>();
        } else {
            h = py::none();
            v = py::none();
            w = py::none();
        }
    }

    void to_file(const std::string &filename, const int_t nelec, const int_t ms2, const double tol) {
        py::module::import("pyci.utils").attr("write_fcidump")(filename, ecore, one_mo, two_mo, nelec, ms2, tol);
    }
};


struct RestrictedHam : public Hamiltonian
{
    RestrictedHam(RestrictedHam &&ham) noexcept : Hamiltonian((Hamiltonian &&)ham) {
    }

    RestrictedHam(const RestrictedHam &ham) : Hamiltonian((const Hamiltonian &)ham) {
    }

    RestrictedHam(const double ecore_, const d_array_t one_mo_, const d_array_t two_mo_,
            const bool keep_mo, const bool doci) {
        Hamiltonian::init_rest(ecore_, one_mo_, two_mo_, keep_mo, doci);
    }

    RestrictedHam(const std::string &filename, const bool keep_mo, const bool doci) {
        py::tuple args = py::module::import("pyci.utils").attr("read_fcidump")(filename);
        Hamiltonian::init_rest(
            args[0].cast<double>(), args[1].cast<d_array_t>(), args[2].cast<d_array_t>(), keep_mo, doci
            );
    }
};


struct UnrestrictedHam : public Hamiltonian
{
    UnrestrictedHam(UnrestrictedHam &&ham) noexcept : Hamiltonian((Hamiltonian &&)ham) {
    }

    UnrestrictedHam(const UnrestrictedHam &ham) : Hamiltonian((const Hamiltonian &)ham) {
    }

    UnrestrictedHam(const double ecore_, const d_array_t one_mo_, const d_array_t two_mo_,
            const bool keep_mo, const bool doci) {
        Hamiltonian::init_unrest(ecore_, one_mo_, two_mo_, keep_mo, doci);
    }

    UnrestrictedHam(const std::string &filename, const bool keep_mo, const bool doci) {
        py::tuple args = py::module::import("pyci.utils").attr("read_fcidump")(filename);
        Hamiltonian::init_unrest(
            args[0].cast<double>(), args[1].cast<d_array_t>(), args[2].cast<d_array_t>(), keep_mo, doci
            );
    }
};


struct GeneralizedHam : public Hamiltonian
{
    GeneralizedHam(GeneralizedHam &&ham) noexcept : Hamiltonian((Hamiltonian &&)ham) {
    }

    GeneralizedHam(const GeneralizedHam &ham) : Hamiltonian((const Hamiltonian &)ham) {
    }

    GeneralizedHam(const double ecore_, const d_array_t one_mo_, const d_array_t two_mo_,
            const bool keep_mo, const bool doci) {
        Hamiltonian::init_rest(ecore_, one_mo_, two_mo_, keep_mo, doci);
    }

    GeneralizedHam(const std::string &filename, const bool keep_mo, const bool doci) {
        py::tuple args = py::module::import("pyci.utils").attr("read_fcidump")(filename);
        Hamiltonian::init_rest(
            args[0].cast<double>(), args[1].cast<d_array_t>(), args[2].cast<d_array_t>(), keep_mo, doci
            );
    }
};


/* Wave function C++ classes. */


struct Wavefunction
{
};


struct DOCIWfn : public OneSpinWfn
{
    DOCIWfn(DOCIWfn &&wfn) noexcept : OneSpinWfn((OneSpinWfn &&)wfn) {
    }

    DOCIWfn(const DOCIWfn &wfn) : OneSpinWfn((const OneSpinWfn &)wfn) {
    }

    DOCIWfn(const int_t nbasis, const int_t nocc) : OneSpinWfn(nbasis, nocc) {
    }

    DOCIWfn(const char *filename) : OneSpinWfn(filename) {
    }

    DOCIWfn(const int_t nbasis, const int_t nocc, const int_t ndet, const uint_t *det_array)
        : OneSpinWfn(nbasis, nocc, ndet, det_array) {
    }

    DOCIWfn(const int_t nbasis, const int_t nocc, const int_t ndet, const int_t *occs_array)
        : OneSpinWfn(nbasis, nocc, ndet, occs_array) {
    }
};


struct FullCIWfn : public TwoSpinWfn
{
    FullCIWfn(FullCIWfn &&wfn) noexcept : TwoSpinWfn((TwoSpinWfn &&)wfn) {
    }

    FullCIWfn(const FullCIWfn &wfn) : TwoSpinWfn((const TwoSpinWfn &)wfn) {
    }

    FullCIWfn(const int_t nbasis, const int_t nocc_up, const int_t nocc_dn)
        : TwoSpinWfn(nbasis, nocc_up, nocc_dn) {
    }

    FullCIWfn(const char *filename) : TwoSpinWfn(filename) {
    }

    FullCIWfn(const int_t nbasis, const int_t nocc_up, const int_t nocc_dn, const int_t ndet, const uint_t *det_array)
        : TwoSpinWfn(nbasis, nocc_up, nocc_dn, ndet, det_array) {
    }

    FullCIWfn(const int_t nbasis, const int_t nocc_up, const int_t nocc_dn, const int_t ndet, const int_t *occs_array)
        : TwoSpinWfn(nbasis, nocc_up, nocc_dn, ndet, occs_array) {
    }

    FullCIWfn(const DOCIWfn &wfn) : TwoSpinWfn((const OneSpinWfn &)wfn) {
    }
};


struct GenCIWfn : public OneSpinWfn
{
    GenCIWfn(GenCIWfn &&wfn) noexcept : OneSpinWfn((OneSpinWfn &&)wfn) {
    }

    GenCIWfn(const GenCIWfn &wfn) : OneSpinWfn((const OneSpinWfn &)wfn) {
    }

    GenCIWfn(const int_t nbasis, const int_t nocc) : OneSpinWfn(nbasis, nocc) {
    }

    GenCIWfn(const char *filename) : OneSpinWfn(filename) {
    }

    GenCIWfn(const int_t nbasis, const int_t nocc, const int_t ndet, const uint_t *det_array)
        : OneSpinWfn(nbasis, nocc, ndet, det_array) {
    }

    GenCIWfn(const int_t nbasis, const int_t nocc, const int_t ndet, const int_t *occs_array)
        : OneSpinWfn(nbasis, nocc, ndet, occs_array) {
    }

    GenCIWfn(const DOCIWfn &wfn) : OneSpinWfn(TwoSpinWfn((const OneSpinWfn)wfn)) {
    }

    GenCIWfn(const FullCIWfn &wfn) : OneSpinWfn((const TwoSpinWfn &)wfn) {
    }
};


/* Module documentation. */


py::options options;
options.disable_function_signatures();


m.doc() = "PyCI C extension module.";


/* Module attributes. */


#ifndef PYCI_VERSION
#define PYCI_VERSION 0.0.0
#endif
#define LITERAL(S) #S
#define STRINGIZE(S) LITERAL(S)
m.attr("__version__") = STRINGIZE(PYCI_VERSION);
#undef LITERAL
#undef STRINGIZE


m.attr("c_int") = py::dtype::of<int_t>();
m.attr("c_uint") = py::dtype::of<uint_t>();
m.attr("c_double") = py::dtype::of<double>();


/* Hamiltonian Python class. */


py::class_<Hamiltonian> hamiltonian(m, "hamiltonian");


hamiltonian.doc() = R"""(
Hamiltonian class.

.. math::

    H = \sum_{pq}{t_{pq} a^\dagger_p a_q} + \sum_{pqrs}{g_{pqrs} a^\dagger_p a^\dagger_q a_s a_r}

.. math::

    H = \sum_{p}{h_p N_p} + \sum_{p \neq q}{v_{pq} P^\dagger_p P_q} + \sum_{pq}{w_{pq} N_p N_q}

where

.. math::

    h_{p} = \left<p|T|p\right> = t_{pp}

.. math::

    v_{pq} = \left<pp|V|qq\right> = g_{ppqq}

.. math::

    w_{pq} = 2 \left<pq|V|pq\right> - \left<pq|V|qp\right> = 2 * g_{pqpq} - g_{pqqp}

)""";


hamiltonian.def_readonly("nbasis", &Hamiltonian::nbasis, "Number of orbital basis functions.");
hamiltonian.def_readonly("ecore", &Hamiltonian::ecore,   "Constant/\"zero-electron\" integral");
hamiltonian.def_readonly("one_mo", &Hamiltonian::one_mo, "Full one-electron integral array.");
hamiltonian.def_readonly("two_mo", &Hamiltonian::two_mo, "Full two-electron integral array.");
hamiltonian.def_readonly("h", &Hamiltonian::h, "Seniority-zero one-electron integral array.");
hamiltonian.def_readonly("v", &Hamiltonian::v, "Seniority-zero two-electron integral array.");
hamiltonian.def_readonly("w", &Hamiltonian::w, "Seniority-two two-electron integral array.");


hamiltonian.def("to_file", &Hamiltonian::to_file,
R"""(
Write a Hamiltonian to an FCIDUMP file.

Parameters
----------
filename : str
    Name of FCIDUMP file to write.
nelec : int, default=0
    Electron number to write to FCIDUMP file.
ms2 : int, default=0
    Spin number to write to FCIDUMP file.
tol : float, default=1.0e-18
    Write elements with magnitude larger than this value.

)""",
py::arg("filename"), py::arg("nelec") = 0, py::arg("ms2") = 0, py::arg("tol") = 1.0e-18);


/* Restricted Hamiltonian Python class. */


py::class_<RestrictedHam, Hamiltonian> restricted_ham(m, "restricted_ham");


restricted_ham.doc() = "Restricted Hamiltonian class.";


restricted_ham.def(py::init<const double, const d_array_t, const d_array_t, const bool, const bool>(),
R"""(
Initialize a restricted Hamiltonian.

Parameters
----------
ecore : float
    Constant/"zero-electron" integral.
one_mo : np.ndarray
    Full one-electron integral array.
two_mo : np.ndarray
    Full two-electron integral array.
keep_mo : bool, default=True
    Whether to keep the full one- and two- electron integral arrays.
doci : bool, default=True
    Whether to compute the seniority-zero and seniority-two integrals for DOCI.

or

Parameters
----------
filename : str
    Name of FCIDUMP file from which to read.
keep_mo : bool, default=True
    Whether to keep the full one- and two- electron integral arrays.
doci : bool, default=True
    Whether to compute the seniority-zero and seniority-two integrals for DOCI.

)""",
py::arg("ecore"), py::arg("one_mo"), py::arg("two_mo"), py::arg("keep_mo") = true, py::arg("doci") = true);


restricted_ham.def(py::init<const std::string &, const bool, const bool>(),
py::arg("filename"), py::arg("keep_mo") = true, py::arg("doci") = true);


/* Unrestricted Hamiltonian Python class. */


py::class_<UnrestrictedHam, Hamiltonian> unrestricted_ham(m, "unrestricted_ham");


unrestricted_ham.doc() = "Unrestricted Hamiltonian class.";


unrestricted_ham.def(py::init<const double, const d_array_t, const d_array_t, const bool, const bool>(),
R"""(
Initialize an unrestricted Hamiltonian.

Parameters
----------
ecore : float
    Constant/"zero-electron" integral.
one_mo : np.ndarray
    Full one-electron integral array.
two_mo : np.ndarray
    Full two-electron integral array.
keep_mo : bool, default=True
    Whether to keep the full one- and two- electron integral arrays.
doci : bool, default=True
    Whether to compute the seniority-zero and seniority-two integrals for DOCI.

or

Parameters
----------
filename : str
    Name of FCIDUMP file from which to read.
keep_mo : bool, default=True
    Whether to keep the full one- and two- electron integral arrays.
doci : bool, default=True
    Whether to compute the seniority-zero and seniority-two integrals for DOCI.

)""",
py::arg("ecore"), py::arg("one_mo"), py::arg("two_mo"), py::arg("keep_mo") = true, py::arg("doci") = true);


unrestricted_ham.def(py::init<const std::string &, const bool, const bool>(),
py::arg("filename"), py::arg("keep_mo") = true, py::arg("doci") = true);


/* Generalized Hamiltonian Python class. */


py::class_<GeneralizedHam, Hamiltonian> generalized_ham(m, "generalized_ham");


generalized_ham.doc() = "Generalized Hamiltonian class.";


generalized_ham.def(py::init<const double, const d_array_t, const d_array_t, const bool, const bool>(),
R"""(
Initialize a generalized Hamiltonian.

Parameters
----------
ecore : float
    Constant/"zero-electron" integral.
one_mo : np.ndarray
    Full one-electron integral array.
two_mo : np.ndarray
    Full two-electron integral array.
keep_mo : bool, default=True
    Whether to keep the full one- and two- electron integral arrays.
doci : bool, default=True
    Whether to compute the seniority-zero and seniority-two integrals for DOCI.

or

Parameters
----------
filename : str
    Name of FCIDUMP file from which to read.
keep_mo : bool, default=True
    Whether to keep the full one- and two- electron integral arrays.
doci : bool, default=True
    Whether to compute the seniority-zero and seniority-two integrals for DOCI.

)""",
py::arg("ecore"), py::arg("one_mo"), py::arg("two_mo"), py::arg("keep_mo") = true, py::arg("doci") = true);


generalized_ham.def(py::init<const std::string &, const bool, const bool>(),
py::arg("filename"), py::arg("keep_mo") = true, py::arg("doci") = true);


/* Wave function Python class. */


py::class_<Wavefunction> wavefunction(m, "wavefunction");


wavefunction.doc() = "Wave function class.";


/* One-spin wave function Python class. */


py::class_<OneSpinWfn> one_spin_wfn(m, "one_spin_wfn", "wavefunction");


one_spin_wfn.doc() = "One-spin wave function class.";


one_spin_wfn.def_readonly("nbasis", &OneSpinWfn::nbasis, "Number of orbital basis functions.");


one_spin_wfn.def("to_file", &OneSpinWfn::to_file,
R"""(
Write a one-spin wave function to a ONESPIN file.

Parameters
----------
filename : str
    Name of ONESPIN file to write.

)""", py::arg("filename"));


one_spin_wfn.def("add_hartreefock_det", &OneSpinWfn::add_hartreefock_det,
R"""(
Add the Hartree-Fock determinant to the wave function.

)""");


one_spin_wfn.def("add_all_dets", &OneSpinWfn::add_all_dets,
R"""(
Add all determinants to the wave function.

)""");


one_spin_wfn.def("reserve", &OneSpinWfn::reserve,
R"""(
Reserve space for :math:`n` determinants in the wave function.

Parameters
----------
n : int
    Number of determinants for which to reserve space.

)""", py::arg("n"));


one_spin_wfn.def("squeeze", &OneSpinWfn::squeeze,
R"""(
Release extra memory held by the wave function.

)""");


one_spin_wfn.def("clear", &OneSpinWfn::clear,
R"""(
Clear all determinants from the wave function.

)""");


one_spin_wfn.def("__len__", [](const OneSpinWfn &self) {
        return self.ndet;
    },
R"""(
Return the number of determinants in the wave function.

)""");


one_spin_wfn.def("__getitem__", [](const OneSpinWfn &self, const int_t index) {
        if ((index < 0) || (index >= self.ndet))
            throw std::out_of_range("index out of range");
        return u_array_t({self.nword}, {sizeof(uint_t)}, self.det_ptr(index));
    },
R"""(
Return the :math:`index`th determinant from the wave function.

Parameters
----------
index : int
    Index of determinant to be returned.

Returns
-------
det : np.ndarray
    Determinant.

)""", py::arg("index"));


one_spin_wfn.def("to_det_array", [](const OneSpinWfn &self, int_t start, int_t end) {
        if (start == -1) {
            start = 0;
            if (end == -1) end = self.ndet;
        } else if (end == -1) {
            end = start;
            start = 0;
        }
        if ((start < 0) || (end > self.ndet))
            throw std::out_of_range("start,end indices out of range");
        return u_array_t({end - start, self.nword}, self.det_ptr(start));
    },
R"""(
Convert the wave function to a NumPy array of determinants (bitstrings).

Parameters
----------
start : int, optional
    Starting determinant (works like Python's range() function).
end : int, optional
    Ending determinant (works like Python's range() function).

Returns
-------
det_array : np.ndarray
    Array of determinants.

)""", py::arg("start") = -1, py::arg("end") = -1);


one_spin_wfn.def("to_occ_array", [](const OneSpinWfn &self, int_t start, int_t end) {
        if (start == -1) {
            start = 0;
            if (end == -1) end = self.ndet;
        } else if (end == -1) {
            end = start;
            start = 0;
        }
        if ((start < 0) || (end > self.ndet))
            throw std::out_of_range("start,end indices out of range");
        i_array_t array({end - start, self.nocc});
        self.to_occs_array(start, end, (int_t *)array.request().ptr);
        return array;
    },
R"""(
Convert the wave function to a NumPy array of occupation vectors.

Parameters
----------
start : int, optional
    Starting determinant (works like Python's range() function).
end : int, optional
    Ending determinant (works like Python's range() function).

Returns
-------
occ_array : np.ndarray
    Array of occupation vectors.

)""", py::arg("start") = -1, py::arg("end") = -1);


one_spin_wfn.def("index_det", [](const OneSpinWfn &self, const u_array_t det) {
        py::buffer_info buf = det.request();
        if ((buf.ndim != 1) || (buf.shape[0] != self.nword))
            throw std::domain_error("det has mismatched dimensions");
        return self.index_det((uint_t *)buf.ptr);
    },
R"""(
Return the index of the specified determinant in the wave function (or -1 if it is not found).

Parameters
----------
det : np.ndarray
    Determinant.

Returns
-------
index : int
    Index of determinant or -1.

)""", py::arg("det"));


one_spin_wfn.def("add_det", [](OneSpinWfn &self, const u_array_t det) {
        py::buffer_info buf = det.request();
        if ((buf.ndim != 1) || (buf.shape[0] != self.nword))
        throw std::domain_error("det has mismatched dimensions");
        return self.add_det((uint_t *)buf.ptr);
    },
R"""(
Add a determinant to the wave function.

Parameters
----------
det : np.ndarray
    Determinant.

)""", py::arg("det"));


one_spin_wfn.def("add_occs", [](OneSpinWfn &self, const i_array_t occs) {
        py::buffer_info buf = occs.request();
        if ((buf.ndim != 1) || (buf.shape[0] != self.nocc))
            throw std::domain_error("occs has mismatched dimensions");
        return self.add_det_from_occs((int_t *)buf.ptr);
    },
R"""(
Add an occupation vector to the wave function.

Parameters
----------
occs : np.ndarray
    Occupation vector.

)""", py::arg("occs"));


one_spin_wfn.def("add_excited_dets", [](OneSpinWfn &self, const int_t exc, py::object ref) {
        int_t n, i;
        py::buffer_info buf;
        std::vector<uint_t> det;
        uint_t *ptr;
        if ((exc < 0) || (exc > (self.nvir > self.nocc ? self.nvir : self.nocc)))
            throw std::out_of_range("invalid excitation level");
        else if (py::cast<py::object>(ref).is(py::none())) {
            n = self.nocc; i = 0;
            det.resize(self.nword);
            while (n >= PYCI_UINT_SIZE) {
                det[i++] = PYCI_UINT_MAX;
                n -= PYCI_UINT_SIZE;
            }
            if (n) det[i] = (PYCI_UINT_ONE << n) - 1;
            ptr = &det[0];
        } else {
            buf = ref.cast<u_array_t>().request();
            if ((buf.ndim != 1) || (buf.shape[0] != self.nword))
                throw std::domain_error("ref has mismatched dimensions");
            ptr = (uint_t *)buf.ptr;
        }
        self.add_excited_dets(ptr, exc);
    },
R"""(
Add excited determinants from a reference determinant to the wave function.

Parameters
----------
exc : int
    Excitation level to add. Zero corresponds to no excitation.
ref : np.ndarray, optional
    Reference determinant. If not provided, the Hartree-Fock determinant is used.

)""", py::arg("exc"), py::arg("ref") = py::none());


/* Two-spin wave function Python class. */


py::class_<TwoSpinWfn> two_spin_wfn(m, "two_spin_wfn", "wavefunction");


two_spin_wfn.doc() = "Two-spin wave function class.";


two_spin_wfn.def_readonly("nbasis", &TwoSpinWfn::nbasis);


two_spin_wfn.def("to_file", &TwoSpinWfn::to_file,
R"""(
Write a two-spin wave function to a TWOSPIN file.

Parameters
----------
filename : str
    Name of TWOSPIN file to write.

)""", py::arg("filename"));


two_spin_wfn.def("add_hartreefock_det", &TwoSpinWfn::add_hartreefock_det,
R"""(
Add the Hartree-Fock determinant to the wave function.

)""");


two_spin_wfn.def("add_all_dets", &TwoSpinWfn::add_all_dets,
R"""(
Add all determinants to the wave function.

)""");


two_spin_wfn.def("reserve", &TwoSpinWfn::reserve,
R"""(
Reserve space for :math:`n` determinants in the wave function.

Parameters
----------
n : int
    Number of determinants for which to reserve space.

)""", py::arg("n"));


two_spin_wfn.def("squeeze", &TwoSpinWfn::squeeze,
R"""(
Release extra memory held by the wave function.

)""");


two_spin_wfn.def("clear", &TwoSpinWfn::clear,
R"""(
Clear all determinants from the wave function.

)""");


two_spin_wfn.def("__len__", [](const TwoSpinWfn &self) {
        return self.ndet;
    },
R"""(
Return the number of determinants in the wave function.

)""");


two_spin_wfn.def("__getitem__", [](const TwoSpinWfn &self, const int_t index) {
        if ((index < 0) || (index >= self.ndet))
            throw std::out_of_range("index out of range");
        return u_array_t(
            {2U, (unsigned)self.nword},
            self.det_ptr(index)
        );
    },
R"""(
Return the :math:`index`th determinant from the wave function.

Parameters
----------
index : int
    Index of determinant to be returned.

Returns
-------
det : np.ndarray
    Determinant.

)""", py::arg("index"));


two_spin_wfn.def("to_det_array", [](const TwoSpinWfn &self, int_t start, int_t end) {
        if (start == -1) {
            start = 0;
            if (end == -1) end = self.ndet;
        } else if (end == -1) {
            end = start;
            start = 0;
        }
        if ((start < 0) || (end > self.ndet))
            throw std::out_of_range("start,end indices out of range");
        return u_array_t(
            {(unsigned)(end - start), 2U, (unsigned)self.nword},
            self.det_ptr(start)
        );
    },
R"""(
Convert the wave function to a NumPy array of determinants (bitstrings).

Parameters
----------
start : int, optional
    Starting determinant (works like Python's range() function).
end : int, optional
    Ending determinant (works like Python's range() function).

Returns
-------
det_array : np.ndarray
    Array of determinants.

)""", py::arg("start") = -1, py::arg("end") = -1);


two_spin_wfn.def("to_occ_array", [](const TwoSpinWfn &self, int_t start, int_t end) {
        if (start == -1) {
            start = 0;
            if (end == -1) end = self.ndet;
        } else if (end == -1) {
            end = start;
            start = 0;
        }
        if ((start < 0) || (end > self.ndet))
            throw std::out_of_range("start,end indices out of range");
        i_array_t array({(unsigned)(end - start), 2U, (unsigned)self.nocc_up});
        self.to_occs_array(start, end, (int_t *)array.request().ptr);
        return array;
    },
R"""(
Convert the wave function to a NumPy array of occupation vectors.

Parameters
----------
start : int, optional
    Starting determinant (works like Python's range() function).
end : int, optional
    Ending determinant (works like Python's range() function).

Returns
-------
occ_array : np.ndarray
    Array of occupation vectors.

)""", py::arg("start") = -1, py::arg("end") = -1);


two_spin_wfn.def("index_det", [](const TwoSpinWfn &self, const u_array_t det) {
        py::buffer_info buf = det.request();
        if ((buf.ndim != 2) || (buf.shape[0] != 2) || (buf.shape[1] != self.nword))
            throw std::domain_error("det has mismatched dimensions");
        return self.index_det((uint_t *)buf.ptr);
    },
R"""(
Return the index of the specified determinant in the wave function (or -1 if it is not found).

Parameters
----------
det : np.ndarray
    Determinant.

Returns
-------
index : int
    Index of determinant or -1.

)""", py::arg("det"));


two_spin_wfn.def("add_det", [](TwoSpinWfn &self, const u_array_t det) {
        py::buffer_info buf = det.request();
        if ((buf.ndim != 2) || (buf.shape[0] != 2) || (buf.shape[1] != self.nword))
            throw std::domain_error("det has mismatched dimensions");
        return self.add_det((uint_t *)buf.ptr);
    },
R"""(
Add a determinant to the wave function.

Parameters
----------
det : np.ndarray
    Determinant.

)""", py::arg("det"));


two_spin_wfn.def("add_occs", [](TwoSpinWfn &self, const i_array_t occs) {
        py::buffer_info buf = occs.request();
        if ((buf.ndim != 2) || (buf.shape[0] != 2) || (buf.shape[1] != self.nocc_up))
            throw std::domain_error("occs has mismatched dimensions");
        return self.add_det_from_occs((int_t *)buf.ptr);
    },
R"""(
Add an occupation vector to the wave function.

Parameters
----------
occs : np.ndarray
    Occupation vector.

)""", py::arg("occs"));


two_spin_wfn.def("add_excited_dets", [](TwoSpinWfn &self, const int_t exc, py::object ref) {
        int_t n, i;
        int_t maxup = self.nocc_up < self.nvir_up ? self.nocc_up : self.nvir_up;
        int_t maxdn = self.nocc_dn < self.nvir_dn ? self.nocc_dn : self.nvir_dn;
        int_t maxexc = self.nocc_up + self.nocc_dn;
        if (self.nvir_up + self.nvir_dn < maxexc)
            maxexc = self.nvir_up + self.nvir_dn;
        py::buffer_info buf;
        std::vector<uint_t> det;
        uint_t *ptr;
        if ((exc < 0) || (exc > maxexc))
            throw std::out_of_range("invalid excitation level");
        else if (py::cast<py::object>(ref).is(py::none())) {
            n = self.nocc_up; i = 0;
            det.resize(self.nword2);
            while (n >= PYCI_UINT_SIZE) {
                det[i++] = PYCI_UINT_MAX;
                n -= PYCI_UINT_SIZE;
            }
            if (n) det[i] = (PYCI_UINT_ONE << n) - 1;
            n = self.nocc_dn; i = self.nword;
            det.resize(self.nword);
            while (n >= PYCI_UINT_SIZE) {
                det[i++] = PYCI_UINT_MAX;
                n -= PYCI_UINT_SIZE;
            }
            if (n) det[i] = (PYCI_UINT_ONE << n) - 1;
            ptr = &det[0];
        } else {
            buf = ref.cast<u_array_t>().request();
            if ((buf.ndim != 2) || (buf.shape[0] != 2) || (buf.shape[1] != self.nword))
                throw std::domain_error("ref has mismatched dimensions");
            ptr = (uint_t *)buf.ptr;
        }
        int_t a = (exc < maxup) ? exc : maxup;
        int_t b = exc - a;
        while ((a >= 0) && (b <= maxdn))
            self.add_excited_dets(ptr, a--, b++);
    },
R"""(
Add excited determinants from a reference determinant to the wave function.

Parameters
----------
exc : int
    Excitation level to add. Zero corresponds to no excitation.
ref : np.ndarray, optional
    Reference determinant. If not provided, the Hartree-Fock determinant is used.

)""", py::arg("exc"), py::arg("ref") = py::none());


/* DOCI wave function Python class. */


py::class_<DOCIWfn, OneSpinWfn> doci_wfn(m, "doci_wfn");


doci_wfn.doc() = "DOCI wave function class.";


doci_wfn.def_property_readonly("nocc", [](const OneSpinWfn &self) {
        return self.nocc * 2;
    },
"Number of occupied orbitals.");


doci_wfn.def_property_readonly("nocc_up", [](const OneSpinWfn &self) {
        return self.nocc;
    },
"Number of spin-up occupied orbitals.");


doci_wfn.def_property_readonly("nocc_dn", [](const OneSpinWfn &self) {
        return self.nocc;
    },
"Number of spin-down occupied orbitals.");


doci_wfn.def_property_readonly("nvir", [](const OneSpinWfn &self) {
        return self.nvir * 2;
    },
"Number of virtual orbitals.");


doci_wfn.def_property_readonly("nvir_up", [](const OneSpinWfn &self) {
        return self.nvir;
    },
"Number of spin-up virtual orbitals.");


doci_wfn.def_property_readonly("nvir_dn", [](const OneSpinWfn &self) {
        return self.nvir;
    },
"Number of spin-down virtual orbitals.");


doci_wfn.def(py::init<const char *>(),
R"""(
Initialize a DOCI wave function.

Parameters
----------
filename : str
    Name of ONESPIN file to read.

or

Parameters
----------
wfn : doci_wfn
    Wave function from which to initialize.

or

Parameters
----------
nbasis : int
    Number of orbital basis functions.
nocc : int
    Number of occupied spin-{up,down} orbitals.

or

Parameters
----------
nbasis : int
    Number of orbital basis functions.
nocc : int
    Number of occupied spin-{up,down} orbitals.
array : np.ndarray
    Array of determinants or occupation vectors.

)""",
py::arg("filename"));


doci_wfn.def(py::init<const int_t, const int_t>(),
py::arg("nbasis"), py::arg("nocc"));


doci_wfn.def(py::init<const DOCIWfn &>(),
py::arg("wfn"));


doci_wfn.def(py::init([](const int_t nbasis, const int_t nocc, const u_array_t det_array) {
        py::buffer_info buf = det_array.request();
        if ((buf.ndim != 2) || (buf.shape[1] != nword_det(nbasis)))
            throw std::domain_error("det_array has mismatched dimensions");
        return DOCIWfn(nbasis, nocc, buf.shape[0], (uint_t *)(buf.ptr));
    }),
py::arg("nbasis"), py::arg("nocc"), py::arg("dets"));


doci_wfn.def(py::init([](const int_t nbasis, const int_t nocc, const i_array_t occ_array) {
        py::buffer_info buf = occ_array.request();
        if ((buf.ndim != 2) || (buf.shape[1] != nocc))
            throw std::domain_error("occ_array has mismatched dimensions");
        return DOCIWfn(nbasis, nocc, buf.shape[0], (int_t *)(buf.ptr));
    }),
py::arg("nbasis"), py::arg("nocc"), py::arg("occs"));


doci_wfn.def("copy", [](const DOCIWfn &self) { return DOCIWfn(self); },
R"""(
Copy the wave function.

Returns
-------
wfn : doci_wfn
    Copy of the wave function.

)""");


doci_wfn.def("truncated", [](const DOCIWfn &self, const int_t n) {
        if ((n < 0) || (n > self.ndet))
            throw std::out_of_range("index out of range");
        return DOCIWfn(self.nbasis, self.nocc, n, self.det_ptr(0));
    },
R"""(
Return a truncated copy of the wave function.

Returns
-------
wfn : doci_wfn
    Truncated wave function.

)""");


/* FullCI wave function Python class. */


py::class_<FullCIWfn, TwoSpinWfn> fullci_wfn(m, "fullci_wfn");


fullci_wfn.doc() = "FullCI wave function class.";


fullci_wfn.def_property_readonly("nocc", [](const TwoSpinWfn &self) {
        return self.nocc_up + self.nocc_dn;
    },
"Number of occupied orbitals.");


fullci_wfn.def_property_readonly("nocc_up", [](const TwoSpinWfn &self) {
        return self.nocc_up;
    },
"Number of spin-up occupied orbitals.");


fullci_wfn.def_property_readonly("nocc_dn", [](const TwoSpinWfn &self) {
        return self.nocc_dn;
    },
"Number of spin-down occupied orbitals.");


fullci_wfn.def_property_readonly("nvir", [](const TwoSpinWfn &self) {
        return self.nvir_up + self.nvir_dn;
    },
"Number of virtual orbitals.");


fullci_wfn.def_property_readonly("nvir_up", [](const TwoSpinWfn &self) {
        return self.nvir_up;
    },
"Number of spin-up virtual orbitals.");


fullci_wfn.def_property_readonly("nvir_dn", [](const TwoSpinWfn &self) {
        return self.nvir_dn;
    },
"Number of spin-down virtual orbitals.");


fullci_wfn.def(py::init<const char *>(),
R"""(
Initialize a FullCI wave function.

Parameters
----------
filename : str
    Name of TWOSPIN file to read.

or

Parameters
----------
wfn : (fullci_wfn | doci_wfn)
    Wave function from which to initialize.

or

Parameters
----------
nbasis : int
    Number of orbital basis functions.
nocc_up : int
    Number of occupied spin-up orbitals.
nocc_dn : int
    Number of occupied spin-down orbitals.

or

Parameters
----------
nbasis : int
    Number of orbital basis functions.
nocc_up : int
    Number of occupied spin-up orbitals.
nocc_dn : int
    Number of occupied spin-down orbitals.
array : np.ndarray
    Array of determinants or occupation vectors.

)""",
py::arg("filename"));


fullci_wfn.def(py::init<const int_t, const int_t, const int_t>(),
py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"));


fullci_wfn.def(py::init<const DOCIWfn &>(),
py::arg("wfn"));


fullci_wfn.def(py::init<const FullCIWfn &>(),
py::arg("wfn"));


fullci_wfn.def(py::init(
    [](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn, const u_array_t det_array) {
        py::buffer_info buf = det_array.request();
        if ((buf.ndim != 3) || (buf.shape[1] != 2) || (buf.shape[2] != nword_det(nbasis)))
            throw std::domain_error("det_array has mismatched dimensions");
        return FullCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0], (uint_t *)(buf.ptr));
    }),
py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("dets"));


fullci_wfn.def(py::init(
    [](const int_t nbasis, const int_t nocc_up, const int_t nocc_dn, const i_array_t occ_array) {
        py::buffer_info buf = occ_array.request();
        if ((buf.ndim != 3) || (buf.shape[1] != 2) || (buf.shape[2] != nocc_up))
            throw std::domain_error("occ_array has mismatched dimensions");
        return FullCIWfn(nbasis, nocc_up, nocc_dn, buf.shape[0], (int_t *)(buf.ptr));
    }),
py::arg("nbasis"), py::arg("nocc_up"), py::arg("nocc_dn"), py::arg("occs"));


fullci_wfn.def("copy", [](const FullCIWfn &self) { return FullCIWfn(self); },
R"""(
Copy the wave function.

Returns
-------
wfn : fullci_wfn
    Copy of the wave function.

)""");


fullci_wfn.def("truncated", [](const FullCIWfn &self, const int_t n) {
        if ((n < 0) || (n > self.ndet))
            throw std::out_of_range("index out of range");
        return FullCIWfn(self.nbasis, self.nocc_up, self.nocc_dn, n, self.det_ptr(0));
    },
R"""(
Return a truncated copy of the wave function.

Returns
-------
wfn : fullci_wfn
    Truncated wave function.

)""");


/* GenCI wave function Python class. */


py::class_<GenCIWfn, OneSpinWfn> genci_wfn(m, "genci_wfn");


genci_wfn.doc() = "Generalized CI wave function class.";


genci_wfn.def_property_readonly("nocc", [](const OneSpinWfn &self) {
        return self.nocc;
    },
"Number of occupied orbitals.");


genci_wfn.def_property_readonly("nocc_up", [](const OneSpinWfn &self) {
        return self.nocc;
    },
"Number of spin-up occupied orbitals.");


genci_wfn.def_property_readonly("nocc_dn", [](const OneSpinWfn &self) {
        return 0;
    },
"Number of spin-down occupied orbitals.");


genci_wfn.def_property_readonly("nvir", [](const OneSpinWfn &self) {
        return self.nvir;
    },
"Number of virtual orbitals.");


genci_wfn.def_property_readonly("nvir_up", [](const OneSpinWfn &self) {
        return self.nvir;
    },
"Number of spin-up virtual orbitals.");


genci_wfn.def_property_readonly("nvir_dn", [](const OneSpinWfn &self) {
        return 0;
    },
"Number of spin-down virtual orbitals.");


genci_wfn.def(py::init<const char *>(),
R"""(
Initialize a generalized CI wave function.

Parameters
----------
filename : str
    Name of ONESPIN file to read.

or

Parameters
----------
wfn : (doci_wfn | fullci_wfn | genci_wfn)
    Wave function from which to initialize.

or

Parameters
----------
nbasis : int
    Number of orbital basis functions.
nocc : int
    Number of occupied orbitals.

or

Parameters
----------
nbasis : int
    Number of orbital basis functions.
nocc : int
    Number of occupied orbitals.
array : np.ndarray
    Array of determinants or occupation vectors.

)""",
py::arg("filename"));


genci_wfn.def(py::init<const int_t, const int_t>(),
py::arg("nbasis"), py::arg("nocc"));


genci_wfn.def(py::init<const DOCIWfn &>(),
py::arg("wfn"));


genci_wfn.def(py::init<const FullCIWfn &>(),
py::arg("wfn"));


genci_wfn.def(py::init<const GenCIWfn &>(),
py::arg("wfn"));


genci_wfn.def(py::init([](const int_t nbasis, const int_t nocc, const u_array_t det_array) {
        py::buffer_info buf = det_array.request();
        if ((buf.ndim != 2) || (buf.shape[1] != nword_det(nbasis)))
            throw std::domain_error("det_array has mismatched dimensions");
        return GenCIWfn(nbasis, nocc, buf.shape[0], (uint_t *)(buf.ptr));
    }),
py::arg("nbasis"), py::arg("nocc"), py::arg("dets"));


genci_wfn.def(py::init([](const int_t nbasis, const int_t nocc, const i_array_t occ_array) {
        py::buffer_info buf = occ_array.request();
        if ((buf.ndim != 2) || (buf.shape[1] != nocc))
            throw std::domain_error("occ_array has mismatched dimensions");
        return GenCIWfn(nbasis, nocc, buf.shape[0], (int_t *)(buf.ptr));
    }),
py::arg("nbasis"), py::arg("nocc"), py::arg("occs"));


genci_wfn.def("copy", [](const GenCIWfn &self) { return GenCIWfn(self); },
R"""(
Copy the wave function.

Returns
-------
wfn : genci_wfn
    Copy of the wave function.

)""");


genci_wfn.def("truncated", [](const GenCIWfn &self, const int_t n) {
        if ((n < 0) || (n > self.ndet))
            throw std::out_of_range("index out of range");
        return GenCIWfn(self.nbasis, self.nocc, n, self.det_ptr(0));
    },
R"""(
Return a truncated copy of the wave function.

Returns
-------
wfn : genci_wfn
    Truncated wave function.

)""");


/* Sparse CI matrix operator Python class. */


py::class_<SparseOp> sparse_op(m, "sparse_op");


sparse_op.doc() = "Sparse CI matrix operator class.";


sparse_op.def_property_readonly("shape", [](const SparseOp &self) {
        return py::make_tuple(self.nrow, self.ncol);
    },
"The shape of the sparse CI matrix.");


sparse_op.def(py::init([](const RestrictedHam &ham, const DOCIWfn &wfn, const int_t rows) {
        if (py::cast<py::object>(ham.h).is(py::none()))
            throw std::invalid_argument("ham does not have seniority-zero integrals");
        SparseOp obj;
        obj.init_doci(
            wfn, ham.ecore, (double *)ham.h.request().ptr, (double *)ham.v.request().ptr,
            (double *)ham.w.request().ptr, rows
            );
        return obj;
    }),
R"""(
Initialize a sparse CI matrix operator.

Parameters
----------
ham : (restricted_ham | unrestricted_ham | generalized_ham)
    Hamiltonian of the system.
wfn : (doci_wfn | fullci_wfn | genci_wfn)
    Wave function of the system.
rows : int, default=(number of columns)
    Number of rows (<= number of columns) of the matrix to construct.

)""",
py::arg("ham"), py::arg("wfn"), py::arg("rows") = -1);


sparse_op.def(py::init([](const RestrictedHam &ham, const FullCIWfn &wfn, const int_t rows) {
        if (py::cast<py::object>(ham.one_mo).is(py::none()))
            throw std::invalid_argument("ham does not have full integrals");
        SparseOp obj;
        obj.init_fullci(
            wfn, ham.ecore, (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, rows
            );
        return obj;
    }),
py::arg("ham"), py::arg("wfn"), py::arg("rows") = -1);


sparse_op.def(py::init([](const GeneralizedHam &ham, const GenCIWfn &wfn, const int_t rows) {
        if (py::cast<py::object>(ham.one_mo).is(py::none()))
            throw std::invalid_argument("ham does not have full integrals");
        SparseOp obj;
        obj.init_genci(
            wfn, ham.ecore, (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, rows
            );
        return obj;
    }),
py::arg("ham"), py::arg("wfn"), py::arg("rows") = -1);


sparse_op.def("__call__",
    [](const SparseOp &self, const d_array_t x) {
    py::buffer_info buf = x.request();
    if ((buf.ndim != 1) || (buf.shape[0] != self.ncol))
        throw std::domain_error("x has mismatched dimensions");
    d_array_t y(self.nrow);
    self.perform_op((const double *)buf.ptr, (double *)y.request().ptr);
    return y;
    },
R"""(
Compute the result of the sparse CI matrix :math:`A` applied to vector :math:`x`.

Parameters
----------
x : np.ndarray
    Operand vector.
out : np.ndarray, optional
    Output parameter, as in NumPy (e.g., numpy.dot).

Returns
-------
y : np.ndarray
   Result vector.

)""",
py::arg("x"));


sparse_op.def("__call__",
    [](const SparseOp &self, const d_array_t x, d_array_t out) {
    py::buffer_info bufx = x.request(), bufy = out.request();
    if ((bufx.ndim != 1) || (bufx.shape[0] != self.ncol) || (bufy.ndim != 1) || (bufy.shape[0] != self.nrow))
        throw std::domain_error("x,y have mismatched dimensions");
    self.perform_op((const double *)bufx.ptr, (double *)bufy.ptr);
    return out;
    },
py::arg("x"), py::arg("out"));


sparse_op.def("solve",
    [](const SparseOp &self, const int_t n, int_t ncv, py::object c0, int_t maxit, const double tol) {
        py::buffer_info buf;
        const double *c_ptr;
        std::vector<double> c;
        if (self.nrow != self.ncol)
            throw std::invalid_argument("cannot solve a rectangular op");
        if (ncv == -1) {
            ncv = self.nrow < 20 ? self.nrow : 20;
            ncv = ncv < n + 1 ? n + 1: ncv;
        }
        if (py::cast<py::object>(c0).is(py::none())) {
            c.resize(self.nrow);
            c[0] = 1.;
            c_ptr = &c[0];
        } else {
            buf = c0.cast<d_array_t>().request();
            if ((buf.ndim != 1) || (buf.shape[0] != self.nrow))
                throw std::domain_error("c0 has mismatched dimensions");
            c_ptr = (const double *)buf.ptr;
        }
        if (maxit == -1) {
            maxit = 1000 * n;
        }
        d_array_t evals(n);
        d_array_t evecs({(unsigned)n, (unsigned)self.nrow});
        self.solve(c_ptr, n, ncv, maxit, tol, (double *)(evals.request().ptr), (double *)(evecs.request().ptr));
        return py::make_tuple(evals, evecs);
    },
R"""(
Solve the CI problem for the energy/energies and coefficient vector(s).

Parameters
----------
n : int, default=1
    Number of lowest-energy solutions for which to solve.
ncv : int, default=max(n + 1, min(20, rows))
    Number of Lanczos vectors to use for eigensolver. More is generally faster and more reliably convergent.
c0 : np.ndarray, optional
    Initial guess for lowest-energy coefficient vector. If not provided, the default is [1, 0, 0, ..., 0, 0].
maxiter : int, default=1000*n
    Maximum number of iterations for eigensolver to run.
tol : float, default=1.0e-6
    Convergence tolerance for eigensolver.

Returns
-------
evals : np.ndarray
    Energies.
evecs : np.ndarray
    Coefficient vectors.

)""",
py::arg("n") = 1, py::arg("ncv") = -1, py::arg("c0") = py::none(), py::arg("maxit") = -1, py::arg("tol") = 1.e-6);


/* Python functions. */


m.def("popcnt", [](const u_array_t x) {
        py::buffer_info buf = x.request();
        unsigned i = 1U;
        for (auto x_it = buf.shape.begin(); x_it != buf.shape.end(); ++x_it)
            i *= *x_it;
        return popcnt_det(i, (const uint_t*)buf.ptr);
    },
R"""(
Compute the population count (number of 1s in the bitstring) of a determinant.

Parameters
----------
det : np.ndarray

Returns
-------
popcnt : int
    Population count.

)""",
py::arg("det"));


m.def("ctz", [](const u_array_t x) {
        py::buffer_info buf = x.request();
        unsigned i = 1U;
        for (auto x_it = buf.shape.begin(); x_it != buf.shape.end(); ++x_it)
            i *= *x_it;
        return ctz_det(i, (const uint_t*)buf.ptr);
    },
R"""(
Compute the trailing zero count of a determinant.

Parameters
----------
det : np.ndarray

Returns
-------
ctz : int
    Trailing zero count.

)""",
py::arg("det"));


m.def("compute_overlap", [](const DOCIWfn &wfn1, const d_array_t c1, const DOCIWfn &wfn2, const d_array_t c2) {
        py::buffer_info buf1 = c1.request();
        py::buffer_info buf2 = c2.request();
        if ((buf1.ndim != 1) || (buf2.ndim != 1) || (buf1.shape[0] != wfn1.ndet) || (buf2.shape[0] != wfn2.ndet))
            throw std::domain_error("c1,c2 have mismatched dimensions");
        return wfn1.compute_overlap((const double *)buf1.ptr, wfn2, (const double *)buf2.ptr);
    },
R"""(
Compute the overlap of two wave functions.

Parameters
----------
wfn1 : (doci_wfn | fullci_wfn | genci_wfn)
    The first wave function.
c1 : np.ndarray
    The first wave function's coefficient vector.
wfn2 : (doci_wfn | fullci_wfn | genci_wfn)
    The second wave function.
c2 : np.ndarray
    The second wave function's coefficient vector.

Returns
-------
olp : float
    Overlap.

Notes
-----
The wave functions must be of the same type.

)""",
py::arg("wfn1"), py::arg("c1"), py::arg("wfn2"), py::arg("c2"));


m.def("compute_overlap", [](const FullCIWfn &wfn1, const d_array_t c1, const FullCIWfn &wfn2, const d_array_t c2) {
        py::buffer_info buf1 = c1.request();
        py::buffer_info buf2 = c2.request();
        if ((buf1.ndim != 1) || (buf2.ndim != 1) || (buf1.shape[0] != wfn1.ndet) || (buf2.shape[0] != wfn2.ndet))
            throw std::domain_error("c1,c2 have mismatched dimensions");
        return wfn1.compute_overlap((const double *)buf1.ptr, wfn2, (const double *)buf2.ptr);
    },
py::arg("wfn1"), py::arg("c1"), py::arg("wfn2"), py::arg("c2"));


m.def("compute_overlap", [](const GenCIWfn &wfn1, const d_array_t c1, const GenCIWfn &wfn2, const d_array_t c2) {
        py::buffer_info buf1 = c1.request();
        py::buffer_info buf2 = c2.request();
        if ((buf1.ndim != 1) || (buf2.ndim != 1) || (buf1.shape[0] != wfn1.ndet) || (buf2.shape[0] != wfn2.ndet))
            throw std::domain_error("c1,c2 have mismatched dimensions");
        return wfn1.compute_overlap((const double *)buf1.ptr, wfn2, (const double *)buf2.ptr);
    },
py::arg("wfn1"), py::arg("c1"), py::arg("wfn2"), py::arg("c2"));


m.def("compute_rdms", [](const DOCIWfn &wfn, const d_array_t c) {
        py::buffer_info buf = c.request();
        if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
            throw std::domain_error("c has mismatched dimensions");
        d_array_t d0({(unsigned)wfn.nbasis, (unsigned)wfn.nbasis});
        d_array_t d2({(unsigned)wfn.nbasis, (unsigned)wfn.nbasis});
        wfn.compute_rdms_doci((const double *)buf.ptr, (double *)d0.request().ptr, (double *)d2.request().ptr);
        return py::make_tuple(d0, d2);
    },
R"""(
Compute the one- and two- electron reduced density matrices (RDMs) of a wave function.

.. math::

    d_{pq} = \left<p|q\right>

.. math::

    D_{pqrs} = \left<pq|rs\right>

Parameters
----------
wfn : (doci_wfn | fullci_wfn | genci_wfn)
    Wave function.
coeffs : np.ndarray(c_double(ndet))
    Coefficient vector.

Returns
-------
rdm1 : np.ndarray
    One-electron reduced density matrix.
rdm2 : np.ndarray(c_double(...))
    Two-electron reduced density matrix.

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
spin-block "up-up" or "down-down", and the leading dimensions of ``rdm2`` has length 2 and
specifies the spin-block "up-up-up-up", "down-down-down-down', or "up-down-up-down".

)""",
py::arg("wfn"), py::arg("c"));


m.def("compute_rdms", [](const FullCIWfn &wfn, const d_array_t c) {
        py::buffer_info buf = c.request();
        if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
            throw std::domain_error("c has mismatched dimensions");
        d_array_t rdm1({2U, (unsigned)wfn.nbasis, (unsigned)wfn.nbasis});
        d_array_t rdm2({3U, (unsigned)wfn.nbasis, (unsigned)wfn.nbasis, (unsigned)wfn.nbasis, (unsigned)wfn.nbasis});
        double *ptr_aa = (double *)rdm1.request().ptr;
        double *ptr_bb = ptr_aa + wfn.nbasis * wfn.nbasis;
        double *ptr_aaaa = (double *)rdm2.request().ptr;
        double *ptr_bbbb = ptr_aaaa + wfn.nbasis * wfn.nbasis * wfn.nbasis * wfn.nbasis;
        double *ptr_abab = ptr_bbbb + wfn.nbasis * wfn.nbasis * wfn.nbasis * wfn.nbasis;
        wfn.compute_rdms_fullci((const double *)buf.ptr, ptr_aa, ptr_bb, ptr_aaaa, ptr_bbbb, ptr_abab);
        return py::make_tuple(rdm1, rdm2);
    },
py::arg("wfn"), py::arg("c"));


m.def("compute_rdms", [](const GenCIWfn &wfn, const d_array_t c) {
        py::buffer_info buf = c.request();
        if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
            throw std::domain_error("c has mismatched dimensions");
        d_array_t rdm1({(unsigned)wfn.nbasis, (unsigned)wfn.nbasis});
        d_array_t rdm2({(unsigned)wfn.nbasis, (unsigned)wfn.nbasis, (unsigned)wfn.nbasis, (unsigned)wfn.nbasis});
        wfn.compute_rdms_doci((const double *)buf.ptr, (double *)rdm1.request().ptr, (double *)rdm2.request().ptr);
        return py::make_tuple(rdm1, rdm2);
    },
py::arg("wfn"), py::arg("c"));


m.def("run_hci", [](const RestrictedHam &ham, const DOCIWfn &wfn, const d_array_t c, const double eps) {
    py::buffer_info buf = c.request();
    if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
        throw std::domain_error("c has mismatched dimensions");
    else if (py::cast<py::object>(ham.v).is(py::none()))
        throw std::invalid_argument("ham does not have seniority-zero integrals");
    return ((OneSpinWfn &)wfn).run_hci_doci((double *)ham.v.request().ptr, (double *)buf.ptr, eps);
    },
R"""(
Run an iteration of heat-bath CI.

This routine adds all determinants connected to determinants currently in the wave function, if they
satisfy the following criteria;

For DOCI: :math:`|\left<f|H|d\right> c_d| > \epsilon` for :math:`f = P^\dagger_i P_a d`.

For Full/Generalized CI: :math:`|\left<f|H|d\right> c_d| > \epsilon`
for :math:`f = a^\dagger_i a_a d` or :math:`f = a^\dagger_i a^\dagger_j a_b a_a d`.

Parameters
----------
ham : (restricted_ham | unrestricted_ham | generalized_ham)
    Hamiltonian of the system.
wfn : (doci_wfn | fullci_wfn | genci_wfn)
    Wave function of the system.
coeffs : np.ndarray(c_double(ndet))
    Coefficient vector.
eps : float
    Threshold value for which determinants to include.

Returns
-------
n : int
    Number of determinants added to wave function.

)""",
py::arg("ham"), py::arg("wfn"), py::arg("c"), py::arg("eps"));


m.def("run_hci", [](const RestrictedHam &ham, const FullCIWfn &wfn, const d_array_t c, const double eps) {
    py::buffer_info buf = c.request();
    if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
        throw std::domain_error("c has mismatched dimensions");
    else if (py::cast<py::object>(ham.one_mo).is(py::none()))
        throw std::invalid_argument("ham does not have full integrals");
    return ((TwoSpinWfn &)wfn).run_hci_fullci(
        (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, (double *)buf.ptr, eps
        );
    },
py::arg("ham"), py::arg("wfn"), py::arg("c"), py::arg("eps"));


m.def("run_hci", [](const RestrictedHam &ham, const GenCIWfn &wfn, const d_array_t c, const double eps) {
    py::buffer_info buf = c.request();
    if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
        throw std::domain_error("c has mismatched dimensions");
    else if (py::cast<py::object>(ham.one_mo).is(py::none()))
        throw std::invalid_argument("ham does not have full integrals");
    return ((OneSpinWfn &)wfn).run_hci_genci(
        (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, (double *)buf.ptr, eps
        );
    },
py::arg("ham"), py::arg("wfn"), py::arg("c"), py::arg("eps"));


m.def("compute_enpt2",
    [](const RestrictedHam &ham, const DOCIWfn &wfn, const d_array_t c, const double energy, const double eps) {
        py::buffer_info buf = c.request();
        if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
            throw std::domain_error("c has mismatched dimensions");
        else if (py::cast<py::object>(ham.one_mo).is(py::none()))
            throw std::invalid_argument("ham does not have full integrals");
        return ((OneSpinWfn &)wfn).compute_enpt2_doci(
            (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, (double *)buf.ptr,
            energy - ham.ecore, eps
            ) + energy;
        },
R"""(
Compute the second-order Epstein-Nesbet perturbation theory (ENPT2) correction to the energy.

Parameters
----------
ham : (restricted_ham | unrestricted_ham | generalized_ham)
    Hamiltonian of the system.
wfn : (doci_wfn | fullci_wfn | genci_wfn)
    Wave function of the system.
coeffs : np.ndarray(c_double(ndet))
    Coefficient vector.
energy : float
    Variational energy of the system.
eps : float, default=1.0e-6
    Threshold value for which determinants to include.

Returns
-------
ecorr : float
    ENPT2-corrected energy of the system.

)""",
py::arg("ham"), py::arg("wfn"), py::arg("c"), py::arg("energy"), py::arg("eps") = 1.0e-6);


m.def("compute_enpt2",
    [](const RestrictedHam &ham, const FullCIWfn &wfn, const d_array_t c, const double energy, const double eps) {
        py::buffer_info buf = c.request();
        if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
            throw std::domain_error("c has mismatched dimensions");
        else if (py::cast<py::object>(ham.one_mo).is(py::none()))
            throw std::invalid_argument("ham does not have full integrals");
        return ((TwoSpinWfn &)wfn).compute_enpt2_fullci(
            (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, (double *)buf.ptr,
            energy - ham.ecore, eps
            ) + energy;
        },
py::arg("ham"), py::arg("wfn"), py::arg("c"), py::arg("energy"), py::arg("eps") = 1.0e-6);


m.def("compute_enpt2",
    [](const RestrictedHam &ham, const GenCIWfn &wfn, const d_array_t c, const double energy, const double eps) {
        py::buffer_info buf = c.request();
        if ((buf.ndim != 1) || (buf.shape[0] != wfn.ndet))
            throw std::domain_error("c has mismatched dimensions");
        else if (py::cast<py::object>(ham.one_mo).is(py::none()))
            throw std::invalid_argument("ham does not have full integrals");
        return ((OneSpinWfn &)wfn).compute_enpt2_genci(
            (double *)ham.one_mo.request().ptr, (double *)ham.two_mo.request().ptr, (double *)buf.ptr,
            energy - ham.ecore, eps
            ) + energy;
        },
py::arg("ham"), py::arg("wfn"), py::arg("c"), py::arg("energy"), py::arg("eps") = 1.0e-6);


} // PYBIND11_MODULE(pyci, m)
