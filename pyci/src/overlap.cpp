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

template<class WfnType>
double compute_overlap_tmpl(const WfnType &wfn1, const WfnType &wfn2, const double *coeffs1,
                            const double *coeffs2) {
    if (wfn1.ndet > wfn2.ndet)
        return compute_overlap_tmpl<WfnType>(wfn2, wfn1, coeffs2, coeffs1);
    double olp = 0.0;
    int_t j;
    for (int_t i = 0; i < wfn1.ndet; ++i) {
        j = wfn2.index_det(wfn1.det_ptr(i));
        if (j != -1)
            olp += coeffs1[i] * coeffs2[j];
    }
    return olp;
}

double compute_overlap(const OneSpinWfn &wfn1, const OneSpinWfn &wfn2, const double *coeffs1,
                       const double *coeffs2) {
    return compute_overlap_tmpl<OneSpinWfn>(wfn1, wfn2, coeffs1, coeffs2);
}

double compute_overlap(const TwoSpinWfn &wfn1, const TwoSpinWfn &wfn2, const double *coeffs1,
                       const double *coeffs2) {
    return compute_overlap_tmpl<TwoSpinWfn>(wfn1, wfn2, coeffs1, coeffs2);
}

} // namespace pyci
