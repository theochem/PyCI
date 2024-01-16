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

void compute_rdms(const DOCIWfn &wfn, const double *coeffs, double *d0, double *d2) {
    // prepare working vectors
    AlignedVector<ulong> v_det(wfn.nword);
    AlignedVector<long> v_occs(wfn.nocc_up);
    AlignedVector<long> v_virs(wfn.nvir_up);
    ulong *det = &v_det[0];
    long *occs = &v_occs[0], *virs = &v_virs[0];
    // fill rdms with zeros
    long i = wfn.nbasis * wfn.nbasis, j = 0;
    while (j < i) {
        d0[j] = 0;
        d2[j++] = 0;
    }
    // iterate over determinants
    long idet, jdet, k, l;
    double val1, val2;
    for (idet = 0; idet < wfn.ndet; ++idet) {
        // fill working vectors
        wfn.copy_det(idet, det);
        fill_occs(wfn.nword, det, occs);
        fill_virs(wfn.nword, wfn.nbasis, det, virs);
        // diagonal elements
        val1 = coeffs[idet] * coeffs[idet];
        for (i = 0; i < wfn.nocc_up; ++i) {
            k = occs[i];
            d0[k * (wfn.nbasis + 1)] += val1;
            for (j = i + 1; j < wfn.nocc_up; ++j) {
                l = occs[j];
                d2[wfn.nbasis * k + l] += val1;
                d2[wfn.nbasis * l + k] += val1;
            }
            // pair excitation elements
            for (j = 0; j < wfn.nvir_up; ++j) {
                l = virs[j];
                excite_det(k, l, det);
                jdet = wfn.index_det(det);
                excite_det(l, k, det);
                // check if excited determinant is in wfn
                if (jdet > idet) {
                    val2 = coeffs[idet] * coeffs[jdet];
                    d0[wfn.nbasis * k + l] += val2;
                    d0[wfn.nbasis * l + k] += val2;
                }
            }
        }
    }
}

void compute_rdms(const FullCIWfn &wfn, const double *coeffs, double *rdm1, double *rdm2) {
    long n1 = wfn.nbasis;
    long n2 = wfn.nbasis * wfn.nbasis;
    long n3 = n1 * n2;
    long n4 = n2 * n2;
    double *aa = rdm1;
    double *bb = aa + n2;
    double *aaaa = rdm2;
    double *bbbb = aaaa + n4;
    double *abab = bbbb + n4;
    // prepare working vectors
    AlignedVector<ulong> v_det(wfn.nword2);
    AlignedVector<long> v_occs(wfn.nocc);
    AlignedVector<long> v_virs(wfn.nvir);
    const ulong *rdet_up, *rdet_dn;
    ulong *det_up = &v_det[0], *det_dn = &v_det[wfn.nword];
    long *occs_up = &v_occs[0], *occs_dn = &v_occs[wfn.nocc_up];
    long *virs_up = &v_virs[0], *virs_dn = &v_virs[wfn.nvir_up];
    // fill rdms with zeros
    long i = 2 * n2;
    long j = 0;
    while (j < i)
        rdm1[j++] = 0;
    i = 3 * n4;
    j = 0;
    while (j < i)
        rdm2[j++] = 0;
    // iterate over determinants
    long k, l, ii, jj, kk, ll, jdet, sign_up;
    double val1, val2;
    for (long idet = 0; idet < wfn.ndet; ++idet) {
        // fill working vectors
        rdet_up = wfn.det_ptr(idet);
        rdet_dn = rdet_up + wfn.nword;
        std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword2);
        fill_occs(wfn.nword, rdet_up, occs_up);
        fill_occs(wfn.nword, rdet_dn, occs_dn);
        fill_virs(wfn.nword, n1, rdet_up, virs_up);
        fill_virs(wfn.nword, n1, rdet_dn, virs_dn);
        val1 = coeffs[idet] * coeffs[idet];
        // loop over spin-up occupied indices
        for (i = 0; i < wfn.nocc_up; ++i) {
            ii = occs_up[i];
            // compute 0-0 terms
            // aa(ii, ii) += val1;
            aa[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < wfn.nocc_up; ++k) {
                kk = occs_up[k];
                // aaaa(ii, kk, ii, kk) += val1;
                aaaa[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                // aaaa(ii, kk, kk, ii) -= val1;
                aaaa[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                // aaaa(kk, ii, ii, kk) -= val1;
                aaaa[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                // rdm2(kk, ii, kk, ii) -= val1;
                aaaa[kk * n3 + ii * n2 + kk * n1 + ii] += val1;
            }
            for (k = 0; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                // abab(ii, kk, ii, kk) += val1;
                abab[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
            }
            // loop over spin-up virtual indices
            for (j = 0; j < wfn.nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
                jdet = wfn.index_det(det_up);
                // check if 1-0 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 1-0 terms
                    val2 = coeffs[idet] * coeffs[jdet] * sign_up;
                    // aa(ii, jj) += val2;
                    aa[ii * n1 + jj] += val2;
                    aa[jj * n1 + ii] += val2;
                    for (k = 0; k < wfn.nocc_up; ++k) {
                        if (i != k) {
                            kk = occs_up[k];
                            // aaaa(ii, kk, jj, kk) += val2;
                            aaaa[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            // aaaa(ii, kk, kk, jj) -= val2;
                            aaaa[ii * n3 + kk * n2 + kk * n1 + jj] -= val2;
                            // aaaa(kk, ii, kk, jj) += val2;
                            aaaa[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                            // aaaa(kk, ii, jj, kk) -= val2;
                            aaaa[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            // aaaa(jj, kk, ii, kk) += val2;
                            aaaa[n3 * jj + n2 * kk + n1 * ii + kk] += val2;
                            // aaaa(jj, kk, kk, ii) -= val2;
                            aaaa[n3 * jj + n2 * kk + n1 * kk + ii] -= val2;
                            // aaaa(kk, jj, ii, kk) -= val2;
                            aaaa[n3 * kk + n2 * jj + n1 * ii + kk] -= val2;
                            // aaaa(kk, jj, kk, ii) += val2;
                            aaaa[n3 * kk + n2 * jj + n1 * kk + ii] += val2;
                        }
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        kk = occs_dn[k];
                        // abab(ii, kk, jj, kk) += val2;
                        abab[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                        // abab(jj, kk, ii, kk) += val2;
                        abab[n3 * jj + kk * n2 + ii * n1 + kk] += val2;
                    }
                }
                // loop over spin-down occupied indices
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    // loop over spin-down virtual indices
                    for (l = 0; l < wfn.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 1-1 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = wfn.index_det(det_up);
                        // check if 1-1 excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 1-1 terms
                            val2 = coeffs[idet] * coeffs[jdet] * sign_up *
                                   phase_single_det(wfn.nword, kk, ll, rdet_dn);
                            // abab(ii, kk, jj, ll) += val2;
                            abab[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // abab(jj, ll, ii, kk) += val2;
                            abab[n3 * jj + n2 * ll + n1 * ii + kk] += val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                // loop over spin-up occupied indices
                for (k = i + 1; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < wfn.nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        jdet = wfn.index_det(det_up);
                        // check if 2-0 excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            val2 = coeffs[idet] * coeffs[jdet] *
                                   phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up);
                            // aaaa(ii, kk, jj, ll) += val2;
                            aaaa[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // aaaa(ii, kk, ll, jj) -= val2;
                            aaaa[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                            // aaaa(kk, ii, jj, ll) -= val2;
                            aaaa[n3 * kk + n2 * ii + n1 * jj + ll] -= val2;
                            // aaaa(kk, ii, ll, jj) += val2;
                            aaaa[n3 * kk + n2 * ii + n1 * ll + jj] += val2;
                            // aaaa(jj, ll, ii, kk) += val2;
                            aaaa[jj * n3 + ll * n2 + ii * n1 + kk] += val2;
                            // aaaa(jj, ll, kk, ii) -= val2;
                            aaaa[jj * n3 + ll * n2 + kk * n1 + ii] -= val2;
                            // aaaa(ll, jj, ii, kk) -= val2;
                            aaaa[n3 * ll + n2 * jj + n1 * ii + kk] -= val2;
                            // aaaa(ll, jj, kk, ii) += val2;
                            aaaa[n3 * ll + n2 * jj + n1 * kk + ii] += val2;
                        }
                        excite_det(ll, kk, det_up);
                    }
                }
                excite_det(jj, ii, det_up);
            }
        }
        // loop over spin-down occupied indices
        for (i = 0; i < wfn.nocc_dn; ++i) {
            ii = occs_dn[i];
            // compute 0-0 terms
            // bb(ii, ii) += val1;
            bb[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                // bbbb(ii, kk, ii, kk) += val1;
                bbbb[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                // bbbb(ii, kk, kk, ii) -= val1;
                bbbb[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                bbbb[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                // rdm2(ii, kk, kk, ii) -= val1;
                bbbb[kk * n3 + ii * n2 + kk * n1 + ii] += val1;
            }
            // loop over spin-down virtual indices
            for (j = 0; j < wfn.nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                jdet = wfn.index_det(det_up);
                // check if 0-1 excited determinant is in wfn
                if (jdet > idet) {
                    // compute 0-1 terms
                    val2 =
                        coeffs[idet] * coeffs[jdet] * phase_single_det(wfn.nword, ii, jj, rdet_dn);
                    // bb(ii, jj) += val2;
                    bb[ii * n1 + jj] += val2;
                    bb[jj * n1 + ii] += val2;
                    for (k = 0; k < wfn.nocc_up; ++k) {
                        kk = occs_up[k];
                        // abab(ii, kk, jj, kk) += val2;
                        abab[n3 * kk + n2 * ii + kk * n1 + jj] += val2;
                        // abab(kk, jj, kk, ii) += val2;
                        abab[n3 * kk + jj * n2 + kk * n1 + ii] += val2;
                    }
                    for (k = 0; k < wfn.nocc_dn; ++k) {
                        if (i != k) {
                            kk = occs_dn[k];
                            // bbbb(ii, kk, jj, kk) += val2;
                            bbbb[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            // bbbb(ii, kk, kk, jj) -= val2;
                            bbbb[ii * n3 + kk * n2 + kk * n1 + jj] -= val2;
                            // bbbb(kk, ii, kk, jj) += val2;
                            bbbb[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                            // bbbb(kk, ii, jj, kk) -= val2;
                            bbbb[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            // bbbb(jj, kk, ii, kk) += val2;
                            bbbb[n3 * jj + n2 * kk + n1 * ii + kk] += val2;
                            // bbbb(jj, kk, kk, ii) -= val2;
                            bbbb[n3 * jj + n2 * kk + n1 * kk + ii] -= val2;
                            // bbbb(kk, jj, ii, kk) -= val2;
                            bbbb[n3 * kk + n2 * jj + n1 * ii + kk] -= val2;
                            // bbbb(kk, jj, kk, ii) += val2;
                            bbbb[n3 * kk + n2 * jj + n1 * kk + ii] += val2;
                        }
                    }
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < wfn.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = wfn.index_det(det_up);
                        // check if excited determinant is in wfn
                        if (jdet > idet) {
                            // compute 2-0 terms
                            val2 = coeffs[idet] * coeffs[jdet] *
                                   phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn);
                            // bbbb(ii, kk, jj, ll) += val2;
                            bbbb[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // bbbb(ii, kk, ll, jj) -= val2;
                            bbbb[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                            // bbbb(kk, ii, jj, ll) -= val2;
                            bbbb[n3 * kk + n2 * ii + n1 * jj + ll] -= val2;
                            // bbbb(kk, ii, ll, jj) += val2;
                            bbbb[n3 * kk + n2 * ii + n1 * ll + jj] += val2;
                            // bbbb(jj, ll, ii, kk) += val2;
                            bbbb[jj * n3 + ll * n2 + ii * n1 + kk] += val2;
                            // bbbb(ll, jj, ii, kk) -= val2;
                            bbbb[n3 * ll + n2 * jj + n1 * ii + kk] -= val2;
                            // bbbb(jj, ll, kk, ii) -= val2;
                            bbbb[jj * n3 + ll * n2 + kk * n1 + ii] -= val2;
                            // bbbb(ll, jj, kk, ii) += val2;
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

void compute_rdms(const GenCIWfn &wfn, const double *coeffs, double *rdm1, double *rdm2) {
    long n1 = wfn.nbasis;
    long n2 = wfn.nbasis * wfn.nbasis;
    long n3 = n1 * n2;
    long n4 = n2 * n2;
    // prepare working vectors
    AlignedVector<ulong> v_det(wfn.nword);
    AlignedVector<long> v_occs(wfn.nocc);
    AlignedVector<long> v_virs(wfn.nvir);
    const ulong *rdet;
    ulong *det = &v_det[0];
    long *occs = &v_occs[0], *virs = &v_virs[0];
    // fill rdms with zeros
    long i = 2 * n2;
    long j = 0;
    while (j < i)
        rdm1[j++] = 0;
    i = 3 * n4;
    j = 0;
    while (j < i)
        rdm2[j++] = 0;
    // loop over determinants
    long k, l, ii, jj, kk, ll, jdet;
    double val1, val2;
    for (long idet = 0; idet < wfn.ndet; ++idet) {
        // fill working vectors
        rdet = wfn.det_ptr(idet);
        std::memcpy(det, rdet, sizeof(ulong) * wfn.nword);
        fill_occs(wfn.nword, rdet, occs);
        fill_virs(wfn.nword, n1, rdet, virs);
        val1 = coeffs[idet] * coeffs[idet];
        // loop over occupied indices
        for (i = 0; i < wfn.nocc; ++i) {
            ii = occs[i];
            // compute diagonal terms
            // rdm1(ii, ii) += val1;
            rdm1[(n1 + 1) * ii] += val1;
            // k = i + 1; because symmetric matrix and that when k == i, it is zero
            for (k = i + 1; k < wfn.nocc; ++k) {
                kk = occs[k];
                // rdm2(ii, kk, ii, kk) += val1;
                rdm2[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                // rdm2(ii, kk, kk, ii) -= val1;
                rdm2[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                // rdm2(kk, ii, ii, kk) += val1;
                rdm2[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                // rdm2(kk, ii, kk, ii) -= val1;
                rdm2[kk * n3 + ii * n2 + kk * n1 + kk] += val1;
            }
            // loop over virtual indices
            for (j = 0; j < wfn.nvir; ++j) {
                jj = virs[j];
                // single excitation elements
                excite_det(ii, jj, det);
                jdet = wfn.index_det(det);
                // check if singly-excited determinant is in wfn
                if (jdet != -1) {
                    // compute single excitation terms
                    val2 = coeffs[idet] * coeffs[jdet] * phase_single_det(wfn.nword, ii, jj, rdet);
                    // rdm1(ii, jj) += val2;
                    rdm1[ii * n1 + jj] += val2;
                    for (k = 0; k < wfn.nocc; ++k) {
                        if (i != k) {
                            kk = occs[k];
                            // rdm2(ii, kk, jj, kk) += val2;
                            rdm2[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            // rdm2(ii, kk, kk, jj) -= val2;
                            rdm2[ii * n3 + kk * n2 + kk * n2 + jj] -= val2;
                            // rdm2(kk, ii, jj, kk) -= val2;
                            rdm2[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            // rdm2(kk, ii, kk, jj) -= val2;
                            rdm2[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                        }
                    }
                }
                // loop over occupied indices
                for (k = i + 1; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    // loop over virtual indices
                    for (l = j + 1; l < wfn.nvir; ++l) {
                        ll = virs[l];
                        // double excitation elements
                        excite_det(kk, ll, det);
                        jdet = wfn.index_det(det);
                        // check if double excited determinant is in wfn
                        if (jdet != -1) {
                            // compute double excitation terms
                            val2 = coeffs[idet] * coeffs[jdet] *
                                   phase_double_det(wfn.nword, ii, kk, jj, ll, rdet);
                            // rdm2(ii, kk, jj, ll) += val2;
                            rdm2[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // rdm2(ii, kk, ll, jj) -= val2;
                            rdm2[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                        }
                        excite_det(ll, kk, det);
                    }
                }
                excite_det(jj, ii, det);
            }
        }
    }
}

void compute_transition_rdms(const DOCIWfn &wfn1, const DOCIWfn &wfn2, const double *coeffs1, const double *coeffs2, double *d0, double *d2) {
    // prepare working vectors
    AlignedVector<ulong> v_det(wfn1.nword);
    AlignedVector<long> v_occs(wfn1.nocc_up);
    AlignedVector<long> v_virs(wfn1.nvir_up);
    ulong *det = &v_det[0];
    long *occs = &v_occs[0], *virs = &v_virs[0];
    // fill rdms with zeros
    long i = wfn1.nbasis * wfn1.nbasis, j = 0;
    while (j < i) {
        d0[j] = 0;
        d2[j++] = 0;
    }
    // iterate over determinants
    long idet, jdet, k, l;
    double val1, val2;
    for (idet = 0; idet < wfn1.ndet; ++idet) {
        // fill working vectors
        wfn1.copy_det(idet, det);
        fill_occs(wfn1.nword, det, occs);
        fill_virs(wfn1.nword, wfn1.nbasis, det, virs);
        // diagonal elements
        jdet = wfn2.index_det(det);
        val1 = (jdet != -1) ? (coeffs1[idet] * coeffs2[jdet]) : 0.;
        for (i = 0; i < wfn1.nocc_up; ++i) {
            k = occs[i];
            d0[k * (wfn1.nbasis + 1)] += val1;
            for (j = i + 1; j < wfn1.nocc_up; ++j) {
                l = occs[j];
                d2[wfn1.nbasis * k + l] += val1;
                d2[wfn1.nbasis * l + k] += val1;
            }
            // pair excitation elements
            for (j = 0; j < wfn1.nvir_up; ++j) {
                l = virs[j];
                excite_det(k, l, det);
                jdet = wfn2.index_det(det);
                excite_det(l, k, det);
                // check if excited determinant is in wfn
                if (jdet != -1) {
                    val2 = coeffs1[idet] * coeffs2[jdet];
                    d0[wfn1.nbasis * k + l] += val2;
                }
            }
        }
    }
}

void compute_transition_rdms(const FullCIWfn &wfn1, const FullCIWfn &wfn2, const double *coeffs1, const double *coeffs2, double *rdm1, double *rdm2) {
    long n1 = wfn1.nbasis;
    long n2 = wfn1.nbasis * wfn1.nbasis;
    long n3 = n1 * n2;
    long n4 = n2 * n2;
    double *aa = rdm1;
    double *bb = aa + n2;
    double *aaaa = rdm2;
    double *bbbb = aaaa + n4;
    double *abab = bbbb + n4;
    // prepare working vectors
    AlignedVector<ulong> v_det(wfn1.nword2);
    AlignedVector<long> v_occs(wfn1.nocc);
    AlignedVector<long> v_virs(wfn1.nvir);
    const ulong *rdet_up, *rdet_dn;
    ulong *det_up = &v_det[0], *det_dn = &v_det[wfn1.nword];
    long *occs_up = &v_occs[0], *occs_dn = &v_occs[wfn1.nocc_up];
    long *virs_up = &v_virs[0], *virs_dn = &v_virs[wfn1.nvir_up];
    // fill rdms with zeros
    long i = 2 * n2;
    long j = 0;
    while (j < i)
        rdm1[j++] = 0;
    i = 3 * n4;
    j = 0;
    while (j < i)
        rdm2[j++] = 0;
    // iterate over determinants
    long k, l, ii, jj, kk, ll, jdet, sign_up;
    double val1, val2;
    for (long idet = 0; idet < wfn1.ndet; ++idet) {
        // fill working vectors
        rdet_up = wfn1.det_ptr(idet);
        rdet_dn = rdet_up + wfn1.nword;
        std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn1.nword2);
        fill_occs(wfn1.nword, rdet_up, occs_up);
        fill_occs(wfn1.nword, rdet_dn, occs_dn);
        fill_virs(wfn1.nword, n1, rdet_up, virs_up);
        fill_virs(wfn1.nword, n1, rdet_dn, virs_dn);
        jdet = wfn2.index_det(det_up);
        val1 = (jdet != -1) ? (coeffs1[idet] * coeffs2[jdet]) : 0.;
        // loop over spin-up occupied indices
        for (i = 0; i < wfn1.nocc_up; ++i) {
            ii = occs_up[i];
            // compute 0-0 terms
            // aa(ii, ii) += val1;
            aa[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < wfn1.nocc_up; ++k) {
                kk = occs_up[k];
                // aaaa(ii, kk, ii, kk) += val1;
                aaaa[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                // aaaa(ii, kk, kk, ii) -= val1;
                aaaa[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                // aaaa(kk, ii, ii, kk) -= val1;
                aaaa[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                // rdm2(kk, ii, kk, ii) -= val1;
                aaaa[kk * n3 + ii * n2 + kk * n1 + ii] += val1;
            }
            for (k = 0; k < wfn1.nocc_dn; ++k) {
                kk = occs_dn[k];
                // abab(ii, kk, ii, kk) += val1;
                abab[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
            }
            // loop over spin-up virtual indices
            for (j = 0; j < wfn1.nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                sign_up = phase_single_det(wfn1.nword, ii, jj, rdet_up);
                jdet = wfn2.index_det(det_up);
                // check if 1-0 excited determinant is in wfn
                if (jdet != -1) {
                    // compute 1-0 terms
                    val2 = coeffs1[idet] * coeffs2[jdet] * sign_up;
                    // aa(ii, jj) += val2;
                    aa[ii * n1 + jj] += val2;
                    for (k = 0; k < wfn1.nocc_up; ++k) {
                        if (i != k) {
                            kk = occs_up[k];
                            // aaaa(ii, kk, jj, kk) += val2;
                            aaaa[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            // aaaa(ii, kk, kk, jj) -= val2;
                            aaaa[ii * n3 + kk * n2 + kk * n1 + jj] -= val2;
                            // aaaa(kk, ii, kk, jj) += val2;
                            aaaa[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                            // aaaa(kk, ii, jj, kk) -= val2;
                            aaaa[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                        }
                    }
                    for (k = 0; k < wfn1.nocc_dn; ++k) {
                        kk = occs_dn[k];
                        // abab(ii, kk, jj, kk) += val2;
                        abab[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                    }
                }
                // loop over spin-down occupied indices
                for (k = 0; k < wfn1.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    // loop over spin-down virtual indices
                    for (l = 0; l < wfn1.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 1-1 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = wfn2.index_det(det_up);
                        // check if 1-1 excited determinant is in wfn
                        if (jdet != -1) {
                            // compute 1-1 terms
                            val2 = coeffs1[idet] * coeffs2[jdet] * sign_up *
                                   phase_single_det(wfn1.nword, kk, ll, rdet_dn);
                            // abab(ii, kk, jj, ll) += val2;
                            abab[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                // loop over spin-up occupied indices
                for (k = i + 1; k < wfn1.nocc_up; ++k) {
                    kk = occs_up[k];
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < wfn1.nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        jdet = wfn2.index_det(det_up);
                        // check if 2-0 excited determinant is in wfn
                        if (jdet != -1) {
                            // compute 2-0 terms
                            val2 = coeffs1[idet] * coeffs2[jdet] *
                                   phase_double_det(wfn1.nword, ii, kk, jj, ll, rdet_up);
                            // aaaa(ii, kk, jj, ll) += val2;
                            aaaa[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // aaaa(ii, kk, ll, jj) -= val2;
                            aaaa[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                            // aaaa(kk, ii, jj, ll) -= val2;
                            aaaa[n3 * kk + n2 * ii + n1 * jj + ll] -= val2;
                            // aaaa(kk, ii, ll, jj) += val2;
                            aaaa[n3 * kk + n2 * ii + n1 * ll + jj] += val2;
                        }
                        excite_det(ll, kk, det_up);
                    }
                }
                excite_det(jj, ii, det_up);
            }
        }
        // loop over spin-down occupied indices
        for (i = 0; i < wfn1.nocc_dn; ++i) {
            ii = occs_dn[i];
            // compute 0-0 terms
            // bb(ii, ii) += val1;
            bb[(n1 + 1) * ii] += val1;
            for (k = i + 1; k < wfn1.nocc_dn; ++k) {
                kk = occs_dn[k];
                // bbbb(ii, kk, ii, kk) += val1;
                bbbb[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                // bbbb(ii, kk, kk, ii) -= val1;
                bbbb[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                bbbb[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                // rdm2(ii, kk, kk, ii) -= val1;
                bbbb[kk * n3 + ii * n2 + kk * n1 + ii] += val1;
            }
            // loop over spin-down virtual indices
            for (j = 0; j < wfn1.nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                jdet = wfn2.index_det(det_up);
                // check if 0-1 excited determinant is in wfn
                if (jdet != -1) {
                    // compute 0-1 terms
                    val2 =
                        coeffs1[idet] * coeffs2[jdet] * phase_single_det(wfn1.nword, ii, jj, rdet_dn);
                    // bb(ii, jj) += val2;
                    bb[ii * n1 + jj] += val2;
                    for (k = 0; k < wfn1.nocc_up; ++k) {
                        kk = occs_up[k];
                        // abab(ii, kk, jj, kk) += val2;
                        abab[n3 * kk + n2 * ii + kk * n1 + jj] += val2;
                    }
                    for (k = 0; k < wfn1.nocc_dn; ++k) {
                        if (i != k) {
                            kk = occs_dn[k];
                            // bbbb(ii, kk, jj, kk) += val2;
                            bbbb[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            // bbbb(ii, kk, kk, jj) -= val2;
                            bbbb[ii * n3 + kk * n2 + kk * n1 + jj] -= val2;
                            // bbbb(kk, ii, kk, jj) += val2;
                            bbbb[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                            // bbbb(kk, ii, jj, kk) -= val2;
                            bbbb[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                        }
                    }
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < wfn1.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < wfn1.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        jdet = wfn2.index_det(det_up);
                        // check if excited determinant is in wfn
                        if (jdet != -1) {
                            // compute 2-0 terms
                            val2 = coeffs1[idet] * coeffs2[jdet] *
                                   phase_double_det(wfn1.nword, ii, kk, jj, ll, rdet_dn);
                            // bbbb(ii, kk, jj, ll) += val2;
                            bbbb[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // bbbb(ii, kk, ll, jj) -= val2;
                            bbbb[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                            // bbbb(kk, ii, jj, ll) -= val2;
                            bbbb[n3 * kk + n2 * ii + n1 * jj + ll] -= val2;
                            // bbbb(kk, ii, ll, jj) += val2;
                            bbbb[n3 * kk + n2 * ii + n1 * ll + jj] += val2;
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
    }
}

void compute_transition_rdms(const GenCIWfn &wfn1, const GenCIWfn &wfn2, const double *coeffs1, const double *coeffs2, double *rdm1, double *rdm2) {
    /* NOTE: I might not have accounted for double-counting here when translating this
     * transition-RDM code from the regular RDM code. I think it's correct, though.
     * GenCIWfn is still unused, so test this function when we start using it! */
    long n1 = wfn1.nbasis;
    long n2 = wfn1.nbasis * wfn1.nbasis;
    long n3 = n1 * n2;
    long n4 = n2 * n2;
    // prepare working vectors
    AlignedVector<ulong> v_det(wfn1.nword);
    AlignedVector<long> v_occs(wfn1.nocc);
    AlignedVector<long> v_virs(wfn1.nvir);
    const ulong *rdet;
    ulong *det = &v_det[0];
    long *occs = &v_occs[0], *virs = &v_virs[0];
    // fill rdms with zeros
    long i = 2 * n2;
    long j = 0;
    while (j < i)
        rdm1[j++] = 0;
    i = 3 * n4;
    j = 0;
    while (j < i)
        rdm2[j++] = 0;
    // loop over determinants
    long k, l, ii, jj, kk, ll, jdet;
    double val1, val2;
    for (long idet = 0; idet < wfn1.ndet; ++idet) {
        // fill working vectors
        rdet = wfn1.det_ptr(idet);
        std::memcpy(det, rdet, sizeof(ulong) * wfn1.nword);
        fill_occs(wfn1.nword, rdet, occs);
        fill_virs(wfn1.nword, n1, rdet, virs);
        jdet = wfn2.index_det(det);
        val1 = (jdet != -1) ? (coeffs1[idet] * coeffs2[idet]) : 0.;
        // loop over occupied indices
        for (i = 0; i < wfn1.nocc; ++i) {
            ii = occs[i];
            // compute diagonal terms
            // rdm1(ii, ii) += val1;
            rdm1[(n1 + 1) * ii] += val1;
            // k = i + 1; because symmetric matrix and that when k == i, it is zero
            for (k = i + 1; k < wfn1.nocc; ++k) {
                kk = occs[k];
                // rdm2(ii, kk, ii, kk) += val1;
                rdm2[ii * n3 + kk * n2 + ii * n1 + kk] += val1;
                // rdm2(ii, kk, kk, ii) -= val1;
                rdm2[ii * n3 + kk * n2 + kk * n1 + ii] -= val1;
                // rdm2(kk, ii, ii, kk) += val1;
                rdm2[kk * n3 + ii * n2 + ii * n1 + kk] -= val1;
                // rdm2(kk, ii, kk, ii) -= val1;
                rdm2[kk * n3 + ii * n2 + kk * n1 + kk] += val1;
            }
            // loop over virtual indices
            for (j = 0; j < wfn1.nvir; ++j) {
                jj = virs[j];
                // single excitation elements
                excite_det(ii, jj, det);
                jdet = wfn2.index_det(det);
                // check if singly-excited determinant is in wfn
                if (jdet != -1) {
                    // compute single excitation terms
                    val2 = coeffs1[idet] * coeffs2[jdet] * phase_single_det(wfn1.nword, ii, jj, rdet);
                    // rdm1(ii, jj) += val2;
                    rdm1[ii * n1 + jj] += val2;
                    for (k = 0; k < wfn1.nocc; ++k) {
                        if (i != k) {
                            kk = occs[k];
                            // rdm2(ii, kk, jj, kk) += val2;
                            rdm2[ii * n3 + kk * n2 + jj * n1 + kk] += val2;
                            // rdm2(ii, kk, kk, jj) -= val2;
                            rdm2[ii * n3 + kk * n2 + kk * n2 + jj] -= val2;
                            // rdm2(kk, ii, jj, kk) -= val2;
                            rdm2[kk * n3 + ii * n2 + jj * n1 + kk] -= val2;
                            // rdm2(kk, ii, kk, jj) -= val2;
                            rdm2[kk * n3 + ii * n2 + kk * n1 + jj] += val2;
                        }
                    }
                }
                // loop over occupied indices
                for (k = i + 1; k < wfn1.nocc; ++k) {
                    kk = occs[k];
                    // loop over virtual indices
                    for (l = j + 1; l < wfn1.nvir; ++l) {
                        ll = virs[l];
                        // double excitation elements
                        excite_det(kk, ll, det);
                        jdet = wfn2.index_det(det);
                        // check if double excited determinant is in wfn
                        if (jdet != -1) {
                            // compute double excitation terms
                            val2 = coeffs1[idet] * coeffs2[jdet] *
                                   phase_double_det(wfn1.nword, ii, kk, jj, ll, rdet);
                            // rdm2(ii, kk, jj, ll) += val2;
                            rdm2[ii * n3 + kk * n2 + jj * n1 + ll] += val2;
                            // rdm2(ii, kk, ll, jj) -= val2;
                            rdm2[ii * n3 + kk * n2 + ll * n1 + jj] -= val2;
                        }
                        excite_det(ll, kk, det);
                    }
                }
                excite_det(jj, ii, det);
            }
        }
    }
}

pybind11::tuple py_compute_rdms_doci(const DOCIWfn &wfn, const Array<double> coeffs) {
    Array<double> d0({wfn.nbasis, wfn.nbasis});
    Array<double> d2({wfn.nbasis, wfn.nbasis});
    compute_rdms(wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                 reinterpret_cast<double *>(d0.request().ptr),
                 reinterpret_cast<double *>(d2.request().ptr));
    return pybind11::make_tuple(d0, d2);
}

pybind11::tuple py_compute_rdms_fullci(const FullCIWfn &wfn, const Array<double> coeffs) {
    Array<double> rdm1({static_cast<long>(2), wfn.nbasis, wfn.nbasis});
    Array<double> rdm2({static_cast<long>(3), wfn.nbasis, wfn.nbasis, wfn.nbasis, wfn.nbasis});
    compute_rdms(wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                 reinterpret_cast<double *>(rdm1.request().ptr),
                 reinterpret_cast<double *>(rdm2.request().ptr));
    return pybind11::make_tuple(rdm1, rdm2);
}

pybind11::tuple py_compute_rdms_genci(const GenCIWfn &wfn, const Array<double> coeffs) {
    Array<double> rdm1({wfn.nbasis, wfn.nbasis});
    Array<double> rdm2({wfn.nbasis, wfn.nbasis, wfn.nbasis, wfn.nbasis});
    compute_rdms(wfn, reinterpret_cast<const double *>(coeffs.request().ptr),
                 reinterpret_cast<double *>(rdm1.request().ptr),
                 reinterpret_cast<double *>(rdm2.request().ptr));
    return pybind11::make_tuple(rdm1, rdm2);
}

pybind11::tuple py_compute_transition_rdms_doci(const DOCIWfn &wfn1, const DOCIWfn &wfn2, const Array<double> coeffs1, const Array<double> coeffs2) {
    Array<double> d0({wfn1.nbasis, wfn1.nbasis});
    Array<double> d2({wfn1.nbasis, wfn1.nbasis});
    compute_transition_rdms(wfn1, wfn2,
                 reinterpret_cast<const double *>(coeffs1.request().ptr),
                 reinterpret_cast<const double *>(coeffs2.request().ptr),
                 reinterpret_cast<double *>(d0.request().ptr),
                 reinterpret_cast<double *>(d2.request().ptr));
    return pybind11::make_tuple(d0, d2);
}

pybind11::tuple py_compute_transition_rdms_fullci(const FullCIWfn &wfn1, const FullCIWfn &wfn2, const Array<double> coeffs1, const Array<double> coeffs2) {
    Array<double> rdm1({static_cast<long>(2), wfn1.nbasis, wfn1.nbasis});
    Array<double> rdm2({static_cast<long>(3), wfn1.nbasis, wfn1.nbasis, wfn1.nbasis, wfn1.nbasis});
    compute_transition_rdms(wfn1, wfn2,
                 reinterpret_cast<const double *>(coeffs1.request().ptr),
                 reinterpret_cast<const double *>(coeffs2.request().ptr),
                 reinterpret_cast<double *>(rdm1.request().ptr),
                 reinterpret_cast<double *>(rdm2.request().ptr));
    return pybind11::make_tuple(rdm1, rdm2);
}

pybind11::tuple py_compute_transition_rdms_genci(const GenCIWfn &wfn1, const GenCIWfn &wfn2, const Array<double> coeffs1, const Array<double> coeffs2) {
    Array<double> rdm1({wfn1.nbasis, wfn1.nbasis});
    Array<double> rdm2({wfn1.nbasis, wfn1.nbasis, wfn1.nbasis, wfn1.nbasis});
    compute_transition_rdms(wfn1, wfn2,
                 reinterpret_cast<const double *>(coeffs1.request().ptr),
                 reinterpret_cast<const double *>(coeffs2.request().ptr),
                 reinterpret_cast<double *>(rdm1.request().ptr),
                 reinterpret_cast<double *>(rdm2.request().ptr));
    return pybind11::make_tuple(rdm1, rdm2);
}

} // namespace pyci
