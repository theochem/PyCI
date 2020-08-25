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

FullCIWfn::FullCIWfn(const FullCIWfn &wfn) : TwoSpinWfn(wfn) {
}

FullCIWfn::FullCIWfn(FullCIWfn &&wfn) noexcept : TwoSpinWfn(wfn) {
}

FullCIWfn::FullCIWfn(const DOCIWfn &wfn) : TwoSpinWfn(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn) {
    ndet = wfn.ndet;
    dets.resize(wfn.ndet * nword2);
    dict.clear();
    for (int_t i = 0; i < wfn.ndet; ++i) {
        std::memcpy(&dets[i * wfn.nword2], wfn.det_ptr(i), sizeof(uint_t) * wfn.nword);
        std::memcpy(&dets[i * wfn.nword2 + wfn.nword], wfn.det_ptr(i), sizeof(uint_t) * wfn.nword);
        dict[rank_det(&dets[i * wfn.nword2])] = i;
    }
}

FullCIWfn::FullCIWfn(const std::string &filename) : TwoSpinWfn(filename) {
}

FullCIWfn::FullCIWfn(const int_t nb, const int_t nu, const int_t nd) : TwoSpinWfn(nb, nu, nd) {
}

FullCIWfn::FullCIWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n,
                     const uint_t *ptr)
    : TwoSpinWfn(nb, nu, nd, n, ptr) {
}

FullCIWfn::FullCIWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n,
                     const int_t *ptr)
    : TwoSpinWfn(nb, nu, nd, n, ptr) {
}

} // namespace pyci
