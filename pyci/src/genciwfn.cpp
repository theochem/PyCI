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

// GenCIWfn::GenCIWfn(const FullCIWfn &wfn) : OneSpinWfn(wfn.nbasis * 2, wfn.nocc, 0) {
GenCIWfn::GenCIWfn(const FullCIWfn &wfn) : OneSpinWfn(wfn.nbasis * 2, wfn.nocc, 0) {
    ndet = wfn.ndet;
    dets.resize(wfn.ndet * wfn.nword);
    AlignedVector<long> occs(wfn.nocc);
    std::cout << "Inside GenCIWfn constructor" << std::endl;
    std::cout << "wfn.nbasis: " << wfn.nbasis << ", wfn.nocc: " << wfn.nocc << std::endl;
    std::cout << "wfn.nocc: " << wfn.nocc << ", wfn.nocc_up, _dn: " << wfn.nocc_up << ", " << wfn.nocc_dn  << std::endl;
    std::cout << "wfn.ndet: " << wfn.ndet << std::endl;
    long *occs_up = &occs[0], *occs_dn = &occs[wfn.nocc_up];
    long j, k = 0;

    std::cout << "occs: " ;
    for (int i = 0; i < wfn.nocc; i++) {
        std::cout << occs[i] << " ";
    }
    std::cout << std::endl;
    
    for (long i = 0; i < wfn.ndet; i += 1) {
        fill_occs(wfn.nword, wfn.det_ptr(i), occs_up);
        fill_occs(wfn.nword, wfn.det_ptr(i) + wfn.nword, occs_dn);

        std::cout << "\nwfn.det_ptr(" << i << "): " << wfn.det_ptr(i)[0] << " " << wfn.det_ptr(i)[1] << std::endl;
        std::cout << "occs_up: " ;
        for (int l = 0; l < wfn.nocc_up; l++) {
            std::cout << occs_up[l] << " ";
        }
        std::cout << std::endl;
        std::cout << "occs_dn: " ;
        for (int l = 0; l < wfn.nocc_dn; l++) {
            std::cout << occs_dn[l] << " ";
        }
        std::cout << std::endl;

        for (j = 0; j < wfn.nocc_dn; ++j)
            occs_dn[j] += wfn.nbasis;

        std::cout << "updated occs_dn: ";
        for(int l = 0; l < wfn.nocc_dn; l++) {
            std::cout << occs_dn[l] << " ";
        }
        std::cout << std::endl;


        fill_det(wfn.nocc, occs_up, &dets[k]);
        dict[rank_det(&dets[k])] = i;
        std::cout << "det[" << k << "]: " << dets[k] << std::endl;
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
