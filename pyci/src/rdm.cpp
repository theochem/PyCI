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

#include <cstring>

#include <vector>

#include <pyci.h>


namespace pyci {


void OneSpinWfn::compute_rdms_doci(const double *coeffs, double *d0, double *d2) const {
    // prepare working vectors
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    // fill rdms with zeros
    int_t i = nbasis * nbasis, j = 0;
    while (j < i) {
        d0[j] = 0;
        d2[j++] = 0;
    }
    // iterate over determinants
    int_t idet, jdet, k, l;
    double val1, val2;
    for (idet = 0; idet < ndet; ++idet) {
        // fill working vectors
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


void TwoSpinWfn::compute_rdms_fullci(const double *coeffs, double *aa, double *bb,
        double *aaaa, double *bbbb, double *abab) const {
    // prepare working vectors
    std::vector<uint_t> det(nword2);
    std::vector<int_t> occs_up(nocc_up);
    std::vector<int_t> occs_dn(nocc_dn);
    std::vector<int_t> virs_up(nvir_up);
    std::vector<int_t> virs_dn(nvir_dn);
    const uint_t *rdet_up, *rdet_dn;
    uint_t *det_up = &det[0], *det_dn = &det[nword];
    // fill rdms with zeros
    int_t i = nbasis * nbasis, j = 0;
    while (j < i) {
        aa[j] = 0;
        bb[j] = 0;
        aaaa[j] = 0;
        bbbb[j] = 0;
        abab[j++] = 0;
    }
    i *= nbasis * nbasis;
    while (j < i) {
        aaaa[j] = 0;
        bbbb[j] = 0;
        abab[j++] = 0;
    }
    // iterate over determinants
    int_t k, l, ii, jj, kk, ll, jdet, sign_up;
    int_t n1 = nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;
    for (int_t idet = 0; idet < ndet; ++idet) {
        // fill working vectors
        rdet_up = &dets[idet * nword2];
        rdet_dn = rdet_up + nword;
        std::memcpy(det_up, rdet_up, sizeof(uint_t) * nword2);
        fill_occs(nword, rdet_up, &occs_up[0]);
        fill_occs(nword, rdet_dn, &occs_dn[0]);
        fill_virs(nword, nbasis, rdet_up, &virs_up[0]);
        fill_virs(nword, nbasis, rdet_dn, &virs_dn[0]);
        val1 = coeffs[idet] * coeffs[idet];
        // loop over spin-up occupied indices
        for (i = 0; i < nocc_up; ++i) {
            ii = occs_up[i];
            // compute 0-0 terms
            //aa(ii, ii) += val1;
            aa[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < nocc_up; ++k) {
                kk = occs_up[k];
                //aaaa(ii, kk, ii, kk) += val1;
                aaaa[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                //aaaa(ii, kk, kk, ii) -= val1;
                aaaa[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                //aaaa(kk, ii, ii, kk) -= val1;
                aaaa[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                //rdm2(kk, ii, kk, ii) -= val1;
                aaaa[kk * n3 + ii * n2 + kk * n1 + ii] += val1;
            }
            for (k = 0; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                //abab(ii, kk, ii, kk) += val1;
                abab[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
            }
            // loop over spin-up virtual indices
            for (j = 0; j < nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                sign_up = phase_single_det(nword, ii, jj, rdet_up);
                jdet = index_det(det_up);
                // check if 1-0 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 1-0 terms
                    val2 = coeffs[idet] * coeffs[jdet] * sign_up;
                    //aa(ii, jj) += val2;
                    aa[ii * n1 + jj] += val2;
                    aa[jj * n1 + ii] += val2;
                    for (k = 0; k < nocc_up; ++k) {
                        if (i != k) {
                            kk = occs_up[k];
                            //aaaa(ii, kk, jj, kk) += val2;
                            aaaa[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            //aaaa(ii, kk, kk, jj) -= val2;
                            aaaa[ii * n3 + kk * n2 + kk * n1 + jj] -= val2;
                            // aaaa(kk, ii, kk, jj)
                            aaaa[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                            // aaaa(kk, ii, jj, kk)
                            aaaa[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            // switch particles
                            //aaaa(jj, kk, ii, kk)
                            aaaa[n3 * jj + n2 * kk + n1 * ii + kk] += val2;
                            //aaaa(jj, kk, kk, ii)
                            aaaa[n3 * jj + n2 * kk + n1 * kk + ii] -= val2;
                            // switch above
                            //aaaa(kk, jj, ii, kk)
                            aaaa[n3 * kk + n2 * jj + n1 * ii + kk] -= val2;
                            //aaaa(kk, jj, kk, ii)
                            aaaa[n3 * kk + n2 * jj + n1 * kk + ii] += val2;
                        }
                    }
                    for (k = 0; k < nocc_dn; ++k) {
                        kk = occs_dn[k];
                        //abab(ii, kk, jj, kk) += val2;
                        abab[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                        // TODO: check
                        //abab(jj, kk, ii, kk)
                        abab[n3 * jj + kk * n2 + ii * n1 + kk] += val2;
                    }
                }
                // loop over spin-down occupied indices
                for (k = 0; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    // loop over spin-down virtual indices
                    for (l = 0; l < nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 1-1 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = index_det(det_up);
                        // check if 1-1 excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 1-1 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * sign_up * phase_single_det(nword, kk, ll, rdet_dn);
                            //abab(ii, kk, jj, ll) += val2;
                            abab[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            //abab(jj, ll, ii, kk)
                            abab[n3 * jj + n2 * ll + n1 * ii + kk] += val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                // loop over spin-up occupied indices
                for (k = i + 1; k < nocc_up; ++k) {
                    kk = occs_up[k];
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        jdet = index_det(det_up);
                        // check if 2-0 excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * phase_double_det(nword, ii, kk, jj, ll, rdet_up);
                            //aaaa(ii, kk, jj, ll) += val2;
                            aaaa[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            //aaaa(ii, kk, ll, jj) -= val2;
                            aaaa[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                            //aaaa(kk, ii, jj, ll)
                            aaaa[n3 * kk + n2 * ii + n1 * jj + ll] -= val2;
                            //aaaa(kk, ii, ll, jj)
                            aaaa[n3 * kk + n2 * ii + n1 * ll + jj] += val2;
                            //aaaa(jj, ll, ii, kk) += val2;
                            aaaa[jj * n3 + ll * n2 + ii * n1 + kk] += val2;
                            //aaaa(jj, ll, kk, ii)
                            aaaa[jj * n3 + ll * n2 + kk * n1 + ii] -= val2;
                            //aaaa(ll, jj, ii, kk) -= val2;
                            aaaa[n3 * ll + n2 * jj + n1 * ii + kk] -= val2;
                            //aaaa(ll, jj, kk, ii) += val2;
                            aaaa[n3 * ll + n2 * jj + n1 * kk + ii] += val2;
                        }
                        excite_det(ll, kk, det_up);
                    }
                }
                excite_det(jj, ii, det_up);
            }
        }
        // loop over spin-down occupied indices
        for (i = 0; i < nocc_dn; ++i) {
            ii = occs_dn[i];
            // compute 0-0 terms
            //bb(ii, ii) += val1;
            bb[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < nocc_dn; ++k) {
                kk = occs_dn[k];
                //bbbb(ii, kk, ii, kk) += val1;
                bbbb[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                //bbbb(ii, kk, kk, ii) -= val1;
                bbbb[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                // TODO: Double check the indices work.
                bbbb[kk * n3 +  ii * n2 + ii * n1 + kk] -= val1;
                //rdm2(ii, kk, kk, ii) -= val1;
                bbbb[kk * n3 + ii * n2 + kk * n1 + ii] += val1;
            }
            // loop over spin-down virtual indices
            for (j = 0; j < nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                jdet = index_det(det_up);
                // check if 0-1 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 0-1 terms
                    val2 = coeffs[idet] * coeffs[jdet]
                         * phase_single_det(nword, ii, jj, rdet_dn);
                    //bb(ii, jj) += val2;
                    bb[ii * n1 + jj] += val2;
                    bb[jj * n1 + ii] += val2;
                    for (k = 0; k < nocc_up; ++k) {
                        kk = occs_up[k];
                        //abab(ii, kk, jj, kk) += val2;
                        abab[n3 * kk + n2 * ii + kk * n1 + jj] += val2;
                        //abab(kk, jj, kk, ii)
                        abab[n3 * kk + jj * n2 + kk * n1 + ii] += val2;
                    }
                    for (k = 0; k < nocc_dn; ++k) {
                        if (i != k){
                            kk = occs_dn[k];
                            //bbbb(ii, kk, jj, kk) += val2;
                            bbbb[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            //bbbb(ii, kk, kk, jj) -= val2;
                            bbbb[ii * n3 + kk * n2 + kk * n1 + jj] -= val2;
                            // bbbb(kk, ii, kk, jj)
                            bbbb[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                            // bbbb(kk, ii, jj, kk)
                            bbbb[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            // switch particles
                            //bbbb(jj, kk, ii, kk)
                            bbbb[n3 * jj + n2 * kk + n1 * ii + kk] += val2;
                            //bbbb(jj, kk, kk, ii)
                            bbbb[n3 * jj + n2 * kk + n1 * kk + ii] -= val2;
                            // switch above
                            //bbbb(kk, jj, ii, kk)
                            bbbb[n3 * kk + n2 * jj + n1 * ii + kk] -= val2;
                            //bbbb(kk, jj, kk, ii)
                            bbbb[n3 * kk + n2 * jj + n1 * kk + ii] += val2;
                        }
                    }
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < nocc_dn; ++k) {
                    kk = occs_dn[k];
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = index_det(det_up); // ALI I changed this to det_dn.
                        // check if excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * phase_double_det(nword, ii, kk, jj, ll, rdet_dn);
                            //bbbb(ii, kk, jj, ll) += val2;
                            bbbb[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            //bbbb(ii, kk, ll, jj) -= val2;
                            bbbb[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                            //bbbb(kk, ii, jj, ll)
                            bbbb[n3 * kk + n2 * ii + n1 * jj + ll] -= val2;
                            //bbbb(kk, ii, ll, jj)
                            bbbb[n3 * kk + n2 * ii + n1 * ll + jj] += val2;
                             //bbbb(jj, ll, ii, kk) += val2;
                            bbbb[jj * n3 + ll * n2 + ii * n1 + kk] += val2;
                            //bbbb(ll, jj, ii, kk) -= val2;
                            bbbb[n3 * ll + n2 * jj + n1 * ii + kk] -= val2;
                            //bbbb(jj, ll, kk, ii)
                            bbbb[jj * n3 + ll * n2 + kk * n1 + ii] -= val2;
                            //bbbb(ll, jj, kk, ii)
                            bbbb[n3 * ll + n2 * jj + n1 * kk + ii] += val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
    }
}



void OneSpinWfn::compute_rdms_genci(const double *coeffs, double *rdm1, double *rdm2) const {
    // prepare working vectors
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    const uint_t *rdet;
    // fill rdms with zeros
    int_t i = nbasis * nbasis, j = 0;
    while (j < i) {
        rdm1[j] = 0;
        rdm2[j++] = 0;
    }
    i *= nbasis * nbasis;
    while (j < i)
        rdm2[j++] = 0;
    // loop over determinants
    int_t k, l, ii, jj, kk, ll, jdet;
    int_t n1 = nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val1, val2;
    for (int_t idet = 0; idet < ndet; ++idet) {
        // fill working vectors
        rdet = &dets[idet * nword];
        std::memcpy(&det[0], rdet, sizeof(uint_t) * nword);
        fill_occs(nword, rdet, &occs[0]);
        fill_virs(nword, nbasis, rdet, &virs[0]);
        val1 = coeffs[idet] * coeffs[idet];
        // loop over occupied indices
        for (i = 0; i < nocc; ++i) {
            ii = occs[i];
            // compute diagonal terms
            //rdm1(ii, ii) += val1;
            rdm1[(n1 + 1) * ii] += val1;
            // k = i + 1; because symmetric matrix and that when k == i, it is zero
            for (k = i + 1; k < nocc; ++k) {
                kk = occs[k];
                //rdm2(ii, kk, ii, kk) += val1;
                rdm2[ii * n3 + kk * n2  + ii * n1 + kk] += val1;
                //rdm2(ii, kk, kk, ii) -= val1;
                rdm2[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                //rdm2(kk, ii, ii, kk) += val1;
                rdm2[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                //rdm2(kk, ii, kk, ii) -= val1;
                rdm2[kk * n3 + ii * n2 + kk * n1 + kk] += val1;
            }
            // loop over virtual indices
            for (j = 0; j < nvir; ++j) {
                jj = virs[j];
                // single excitation elements
                excite_det(ii, jj, &det[0]);
                jdet = index_det(&det[0]);
                // check if singly-excited determinant is in wfn
                if (jdet != -1) {
                    // compute single excitation terms
                    val2 = coeffs[idet] * coeffs[jdet] * phase_single_det(nword, ii, jj, rdet);
                    //rdm1(ii, jj) += val2;
                    rdm1[ii * n1 + jj] += val2;
                    for (k = 0; k < nocc; ++k) {
                        if (i != k) {
                            kk = occs[k];
                            //rdm2(ii, kk, jj, kk) += val2;
                            rdm2[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            //rdm2(ii, kk, kk, jj) -= val2;
                            rdm2[ii * n3 + kk * n2 + kk * n2 + jj] -= val2;
                            //rdm2(kk, ii, jj, kk) -= val2;
                            rdm2[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            //rdm2(kk, ii, kk, jj) -= val2;
                            rdm2[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                        }
                    }
                }
                // loop over occupied indices
                for (k = i + 1; k < nocc; ++k) {
                    kk = occs[k];
                    // loop over virtual indices
                    for (l = j + 1; l < nvir; ++l) {
                        ll = virs[l];
                        // double excitation elements
                        excite_det(kk, ll, &det[0]);
                        jdet = index_det(&det[0]);
                        // check if double excited determinant is in wfn
                        if (jdet != -1) {
                            // compute double excitation terms
                            val2 = coeffs[idet] * coeffs[jdet]
                                 * phase_double_det(nword, ii, kk, jj, ll, rdet);
                            //rdm2(ii, kk, jj, ll) += val2;
                            rdm2[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            //rdm2(ii, kk, ll, jj) -= val2;
                            rdm2[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                        }
                        excite_det(ll, kk, &det[0]);
                    }
                }
                excite_det(jj, ii, &det[0]);
            }
        }
    }
}


} // namespace pyci
