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
#include <iostream>
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
    AlignedVector<long> occs(wfn.nocc);
    long *occs_up = &occs[0], *occs_dn = &occs[wfn.nocc_up];
    long j, k = 0;
    for (long i = 0; i < wfn.ndet; ++i) {
        fill_occs(wfn.nword, wfn.det_ptr(i), occs_up);
        fill_occs(wfn.nword, wfn.det_ptr(i), occs_dn);
        for (j = 0; j < wfn.nocc_dn; ++j)
            occs_dn[j] += wfn.nbasis;
        fill_det(wfn.nocc, occs_up, &dets[k]);
        dict[rank_det(&dets[k])] = i;
        k += wfn.nword;       
    }
}

GenCIWfn::GenCIWfn(const std::string &filename) : OneSpinWfn(filename) {
    if (nocc_dn)
        throw std::invalid_argument("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const long nb, const long nu, const long nd) : OneSpinWfn(nb, nu, nd) {
    if (nocc_dn)
        throw std::invalid_argument("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_dn)
        throw std::invalid_argument("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : OneSpinWfn(nb, nu, nd, n, ptr) {
    if (nocc_dn)
        throw std::invalid_argument("nocc_dn != 0");
}

GenCIWfn::GenCIWfn(const long nb, const long nu, const long nd, const Array<ulong> array)
    : GenCIWfn(nb, nu, nd, array.request().shape[0],
               reinterpret_cast<const ulong *>(array.request().ptr)) {
}

GenCIWfn::GenCIWfn(const long nb, const long nu, const long nd, const Array<long> array)
    : GenCIWfn(nb, nu, nd, array.request().shape[0],
               reinterpret_cast<const long *>(array.request().ptr)) {
}

} // namespace pyci
