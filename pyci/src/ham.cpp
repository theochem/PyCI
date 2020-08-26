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

Ham::Ham(void) {
}

Ham::Ham(const Ham &ham)
    : nbasis(ham.nbasis), ecore(ham.ecore), one_mo(ham.one_mo), two_mo(ham.two_mo), h(ham.h),
      v(ham.v), w(ham.w), one_mo_array(ham.one_mo_array), two_mo_array(ham.two_mo_array),
      h_array(ham.h_array), v_array(ham.v_array), w_array(ham.w_array) {
}

Ham::Ham(Ham &&ham) noexcept
    : nbasis(std::exchange(ham.nbasis, 0)), ecore(std::exchange(ham.ecore, 0.0)),
      one_mo(std::exchange(ham.one_mo, nullptr)), two_mo(std::exchange(ham.two_mo, nullptr)),
      h(std::exchange(ham.h, nullptr)), v(std::exchange(ham.v, nullptr)),
      w(std::exchange(ham.w, nullptr)), one_mo_array(std::move(ham.one_mo_array)),
      two_mo_array(std::move(ham.two_mo_array)), h_array(std::move(ham.h_array)),
      v_array(std::move(ham.v_array)), w_array(std::move(ham.w_array)) {
}

Ham::Ham(const std::string &filename) {
    pybind11::tuple args = pybind11::module::import("pyci.fcidump").attr("_load_ham")(filename);
    init_ham(args);
}

Ham::Ham(const double e, const Array<double> mo1, const Array<double> mo2) {
    pybind11::tuple args = pybind11::module::import("pyci.fcidump").attr("_load_ham")(e, mo1, mo2);
    init_ham(args);
}

void Ham::to_file(const std::string &filename, const long nelec, const long ms2,
                  const double tol) const {
    long n0 = sizeof(double);
    long n1 = n0 * nbasis;
    long n2 = n1 * nbasis;
    long n3 = n2 * nbasis;
    Array<double> one_mo_array({nbasis, nbasis}, {n1, n0}, one_mo);
    Array<double> two_mo_array({nbasis, nbasis, nbasis, nbasis}, {n3, n2, n1, n0}, two_mo);
    pybind11::module::import("pyci.fcidump")
        .attr("write_fcidump")(filename, ecore, one_mo_array, two_mo_array, nelec, ms2, tol);
}

void Ham::init_ham(const pybind11::tuple &args) {
    one_mo_array = args[1];
    two_mo_array = args[2];
    h_array = args[3];
    v_array = args[4];
    w_array = args[5];
    nbasis = one_mo_array.cast<Array<double>>().request().shape[0];
    ecore = args[0].cast<double>();
    one_mo = reinterpret_cast<double *>(one_mo_array.cast<Array<double>>().request().ptr);
    two_mo = reinterpret_cast<double *>(two_mo_array.cast<Array<double>>().request().ptr);
    h = reinterpret_cast<double *>(h_array.cast<Array<double>>().request().ptr);
    v = reinterpret_cast<double *>(v_array.cast<Array<double>>().request().ptr);
    w = reinterpret_cast<double *>(w_array.cast<Array<double>>().request().ptr);
}

} // namespace pyci
