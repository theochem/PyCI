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

FullCIWfn::FullCIWfn(const FullCIWfn &wfn) : TwoSpinWfn(wfn) {
}

FullCIWfn::FullCIWfn(FullCIWfn &&wfn) noexcept : TwoSpinWfn(wfn) {
}

FullCIWfn::FullCIWfn(const DOCIWfn &wfn) : TwoSpinWfn(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn) {
    ndet = wfn.ndet;
    dets.resize(wfn.ndet * nword2);
    dict.clear();
    for (long i = 0; i < wfn.ndet; ++i) {
        std::memcpy(&dets[i * wfn.nword2], wfn.det_ptr(i), sizeof(ulong) * wfn.nword);
        std::memcpy(&dets[i * wfn.nword2 + wfn.nword], wfn.det_ptr(i), sizeof(ulong) * wfn.nword);
        dict[rank_det(&dets[i * wfn.nword2])] = i;
    }
}

FullCIWfn::FullCIWfn(const std::string &filename) : TwoSpinWfn(filename) {
}

FullCIWfn::FullCIWfn(const long nb, const long nu, const long nd) : TwoSpinWfn(nb, nu, nd) {
}

FullCIWfn::FullCIWfn(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : TwoSpinWfn(nb, nu, nd, n, ptr) {
}

FullCIWfn::FullCIWfn(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : TwoSpinWfn(nb, nu, nd, n, ptr) {
}

FullCIWfn::FullCIWfn(const long nb, const long nu, const long nd, const Array<ulong> array)
    : FullCIWfn(nb, nu, nd, array.request().shape[0],
                reinterpret_cast<const ulong *>(array.request().ptr)) {
}

FullCIWfn::FullCIWfn(const long nb, const long nu, const long nd, const Array<long> array)
    : FullCIWfn(nb, nu, nd, array.request().shape[0],
                reinterpret_cast<const long *>(array.request().ptr)) {
}

} // namespace pyci
