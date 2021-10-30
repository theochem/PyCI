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

DOCIWfn::DOCIWfn(const DOCIWfn &wfn) : OneSpinWfn(wfn) {
}

DOCIWfn::DOCIWfn(DOCIWfn &&wfn) noexcept : OneSpinWfn(wfn) {
}

DOCIWfn::DOCIWfn(const std::string &filename) : OneSpinWfn(filename) {
    if (nocc_up != nocc_dn)
        throw std::invalid_argument("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const long nb, const long nu, const long nd) : OneSpinWfn(nb, nu, nd) {
    if (nocc_up != nocc_dn)
        throw std::invalid_argument("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_up != nocc_dn)
        throw std::invalid_argument("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_up != nocc_dn)
        throw std::invalid_argument("nocc_up != nocc_dn");
}

DOCIWfn::DOCIWfn(const long nb, const long nu, const long nd, const Array<ulong> array)
    : DOCIWfn(nb, nu, nd, array.request().shape[0],
              reinterpret_cast<const ulong *>(array.request().ptr)) {
}

DOCIWfn::DOCIWfn(const long nb, const long nu, const long nd, const Array<long> array)
    : DOCIWfn(nb, nu, nd, array.request().shape[0],
              reinterpret_cast<const long *>(array.request().ptr)) {
}

} // namespace pyci
