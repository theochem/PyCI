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

GenCIWfn::GenCIWfn(const GenCIWfn &wfn) : OneSpinWfn(wfn) {
}

GenCIWfn::GenCIWfn(GenCIWfn &&wfn) noexcept : OneSpinWfn(wfn) {
}

GenCIWfn::GenCIWfn(const DOCIWfn &wfn) : GenCIWfn(FullCIWfn(wfn)) {
}

GenCIWfn::GenCIWfn(const FullCIWfn &wfn) : OneSpinWfn(wfn.nbasis * 2, wfn.nocc, 0) {
    ndet = wfn.ndet;
    dets.resize(wfn.ndet * wfn.nword2);
    std::vector<int_t> occs(wfn.nocc);
    int_t *occs_up = &occs[0], *occs_dn = &occs[wfn.nocc_up];
    int_t j, k = 0;
    for (int_t i = 0; i < wfn.ndet; ++i) {
        fill_occs(wfn.nword, wfn.det_ptr(i), occs_up);
        fill_occs(wfn.nword, wfn.det_ptr(i), occs_dn);
        for (j = 0; j < wfn.nocc_dn; ++j)
            occs_dn[j] += wfn.nbasis;
        fill_det(wfn.nocc, occs_up, &dets[k]);
        dict[rank_det(&dets[k])] = i;
        k += wfn.nword2;
    }
}

GenCIWfn::GenCIWfn(const std::string &filename) : OneSpinWfn(filename) {
    if (nocc_dn)
        throw std::runtime_error("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const int_t nb, const int_t nu, const int_t nd) : OneSpinWfn(nb, nu, nd) {
    if (nocc_dn)
        throw std::runtime_error("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n, const uint_t *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_dn)
        throw std::runtime_error("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n, const int_t *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_dn)
        throw std::runtime_error("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const int_t nb, const int_t nu, const int_t nd, const Array<uint_t> array)
    : GenCIWfn(nb, nu, nd, array.request().shape[0],
               reinterpret_cast<const uint_t *>(array.request().ptr)) {
}

GenCIWfn::GenCIWfn(const int_t nb, const int_t nu, const int_t nd, const Array<int_t> array)
    : GenCIWfn(nb, nu, nd, array.request().shape[0],
               reinterpret_cast<const int_t *>(array.request().ptr)) {
}

} // namespace pyci
