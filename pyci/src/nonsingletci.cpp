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

NonSingletCI::NonSingletCI(const NonSingletCI &wfn) : GenCIWfn(wfn) {
}

NonSingletCI::NonSingletCI(NonSingletCI &&wfn) noexcept : GenCIWfn(wfn) {
}

NonSingletCI::NonSingletCI(const DOCIWfn &wfn) : GenCIWfn(wfn){
}

NonSingletCI::NonSingletCI(const FullCIWfn &wfn) : GenCIWfn(wfn){
}

NonSingletCI::NonSingletCI(const std::string &filename) : GenCIWfn(filename) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd) : GenCIWfn(nb, nu, nd) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : GenCIWfn(nb, nu, nd, n, ptr) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : GenCIWfn(nb, nu, nd, n, ptr) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const Array<ulong> array)
    : NonSingletCI(nb, nu, nd, array.request().shape[0],
                   reinterpret_cast<const ulong *>(array.request().ptr)) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const Array<long> array)
    : NonSingletCI(nb, nu, nd, array.request().shape[0],
                   reinterpret_cast<const long *>(array.request().ptr)) {
}


} //namespace pyci



