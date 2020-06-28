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

#include <vector>

#include <pyci/pyci.h>


namespace pyci {


void DOCIWfn::compute_rdms_doci(const double *coeffs, double *d0, double *d2) const {
    int_t idet, jdet, i, j, k, l;
    double val1, val2;
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    for (idet = 0; idet < ndet; ++idet) {
        copy_det(idet, &det[0]);
        fill_occs(nword, &det[0], &occs[0]);
        fill_virs(nword, nbasis, &det[0], &virs[0]);
        // diagonal elements
        val1 = coeffs[idet] * coeffs[idet];
        for (i = 0; i < nocc; ++i) {
            k = occs[i];
            d0[k * (nbasis + 1)] += val1;
            for (j = i + 1; j < nocc; ++j) {
                l = occs[j];
                d2[nbasis * k + l] += val1;
                d2[nbasis * l + k] += val1;
            }
            // pair excitation elements
            for (j = 0; j < nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                jdet = index_det(&det[0]);
                excite_det(l, k, &det[0]);
                // check if excited determinant is in wfn
                if (jdet > idet) {
                    val2 = coeffs[idet] * coeffs[jdet];
                    d0[nbasis * k + l] += val2;
                    d0[nbasis * l + k] += val2;
                }
            }
        }
    }
}


void DOCIWfn::compute_rdms_genci(const double *coeffs, double *rdm1, double *rdm2) const {
    return; // TODO
}


void FullCIWfn::compute_rdms(const double *coeffs, double *rdm1, double *rdm2) const {
    return; // TODO
}


} // namespace pyci
