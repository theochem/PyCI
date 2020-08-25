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

#include <pyci.h>

namespace pyci {

DOCIWfn::DOCIWfn(const DOCIWfn &wfn) : OneSpinWfn(wfn) {
}

DOCIWfn::DOCIWfn(DOCIWfn &&wfn) noexcept : OneSpinWfn(wfn) {
}

DOCIWfn::DOCIWfn(const std::string &filename) : OneSpinWfn(filename) {
    if (nocc_up != nocc_dn)
        throw std::runtime_error("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const int_t nb, const int_t nu, const int_t nd) : OneSpinWfn(nb, nu, nd) {
    if (nocc_up != nocc_dn)
        throw std::runtime_error("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n, const uint_t *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_up != nocc_dn)
        throw std::runtime_error("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n, const int_t *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_up != nocc_dn)
        throw std::runtime_error("nocc_up != nocc_dn");
}

} // namespace pyci
