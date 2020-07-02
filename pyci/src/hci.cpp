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

#include <cstdlib>
#include <cstring>

#include <vector>

#include <omp.h>

#include <pyci.h>


namespace pyci {


namespace { // anonymous


void hci_doci_run_thread(OneSpinWfn &wfn, OneSpinWfn &thread_wfn, const double *v, const double *coeffs,
    const double eps, const int_t istart, const int_t iend) {
    if (istart >= iend)
        return;
    // set attributes
    thread_wfn.nword = wfn.nword;
    thread_wfn.nbasis = wfn.nbasis;
    thread_wfn.nocc = wfn.nocc;
    thread_wfn.nvir = wfn.nvir;
    thread_wfn.ndet = 0;
    // prepare working vectors
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    // loop over determinants
    int_t i, j, k, l;
    uint_t rank;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        wfn.copy_det(idet, &det[0]);
        fill_occs(wfn.nword, &det[0], &occs[0]);
        fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
        // single/"pair"-excited elements elements
        for (i = 0; i < wfn.nocc; ++i) {
            k = occs[i];
            for (j = 0; j < wfn.nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                // add determinant if |H*c| > eps and not already in wfn
                if (std::abs(v[k * wfn.nbasis + l] * coeffs[idet]) > eps) {
                    rank = wfn.rank_det(&det[0]);
                    if (wfn.index_det_from_rank(rank) == -1)
                        thread_wfn.add_det_with_rank(&det[0], rank);
                }
                excite_det(l, k, &det[0]);
            }
        }
    }
}


void hci_genci_run_thread(OneSpinWfn &wfn, OneSpinWfn &thread_wfn, const double *one_mo, const double *two_mo,
    const double *coeffs, const double eps, const int_t istart, const int_t iend) {
    if (istart >= iend)
        return;
    // set attributes
    thread_wfn.nword = wfn.nword;
    thread_wfn.nbasis = wfn.nbasis;
    thread_wfn.nocc = wfn.nocc;
    thread_wfn.nvir = wfn.nvir;
    thread_wfn.ndet = 0;
    // prepare working vectors
    std::vector<uint_t> det(wfn.nword);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    uint_t rank;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        wfn.copy_det(idet, &det[0]);
        fill_occs(wfn.nword, &det[0], &occs[0]);
        fill_virs(wfn.nword, wfn.nbasis, &det[0], &virs[0]);
        // loop over occupied indices
        for (i = 0; i < wfn.nocc; ++i) {
            ii = occs[i];
            ioffset = n3 * ii;
            // loop over virtual indices
            for (j = 0; j < wfn.nvir; ++j) {
                jj = virs[j];
                // single excitation elements
                excite_det(ii, jj, &det[0]);
                val = one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    val += two_mo[koffset + n1 * jj + kk] - two_mo[koffset + n1 * kk + jj];
                }
                // add determinant if |H*c| > eps and not already in wfn
                if (std::abs(val * coeffs[idet]) > eps) {
                    rank = wfn.rank_det(&det[0]);
                    if (wfn.index_det_from_rank(rank) == -1)
                        thread_wfn.add_det_with_rank(&det[0], rank);
                }
                // loop over occupied indices
                for (k = i + 1; k < wfn.nocc; ++k) {
                    kk = occs[k];
                    koffset = ioffset + n2 * kk;
                    // loop over virtual indices
                    for (l = j + 1; l < wfn.nvir; ++l) {
                        ll = virs[l];
                        // double excitation elements
                        excite_det(kk, ll, &det[0]);
                        val = two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val * coeffs[idet]) > eps) {
                            rank = wfn.rank_det(&det[0]);
                            if (wfn.index_det_from_rank(rank) == -1)
                                thread_wfn.add_det_with_rank(&det[0], rank);
                        }
                        excite_det(ll, kk, &det[0]);
                    }
                }
                excite_det(jj, ii, &det[0]);
            }
        }
    }
}


void hci_fullci_run_thread(TwoSpinWfn &wfn, TwoSpinWfn &thread_wfn, const double *one_mo, const double *two_mo,
    const double *coeffs, const double eps, const int_t istart, const int_t iend) {
    if (istart >= iend)
        return;
    // set attributes
    thread_wfn.nword = wfn.nword;
    thread_wfn.nword2 = wfn.nword2;
    thread_wfn.nbasis = wfn.nbasis;
    thread_wfn.nocc_up = wfn.nocc_up;
    thread_wfn.nocc_dn = wfn.nocc_dn;
    thread_wfn.nvir_up = wfn.nvir_up;
    thread_wfn.nvir_dn = wfn.nvir_dn;
    thread_wfn.maxdet_up = wfn.maxdet_up;
    thread_wfn.maxdet_dn = wfn.maxdet_dn;
    thread_wfn.ndet = 0;
    // prepare working vectors
    std::vector<uint_t> det(wfn.nword2);
    std::vector<int_t> occs_up(wfn.nocc_up);
    std::vector<int_t> occs_dn(wfn.nocc_dn);
    std::vector<int_t> virs_up(wfn.nvir_up);
    std::vector<int_t> virs_dn(wfn.nvir_dn);
    const uint_t *rdet_up, *rdet_dn;
    uint_t *det_up = &det[0], *det_dn = &det[wfn.nword];
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    uint_t rank;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val;
    for (int_t idet = istart; idet < iend; ++idet) {
        // fill working vectors
        rdet_up = &wfn.dets[idet * wfn.nword2];
        rdet_dn = rdet_up + wfn.nword;
        std::memcpy(det_up, rdet_up, sizeof(uint_t) * wfn.nword2);
        fill_occs(wfn.nword, rdet_up, &occs_up[0]);
        fill_occs(wfn.nword, rdet_dn, &occs_dn[0]);
        fill_virs(wfn.nword, wfn.nbasis, rdet_up, &virs_up[0]);
        fill_virs(wfn.nword, wfn.nbasis, rdet_dn, &virs_dn[0]);
        // loop over spin-up occupied indices
        for (i = 0; i < wfn.nocc_up; ++i) {
            ii = occs_up[i];
            ioffset = n3 * ii;
            // loop over spin-up virtual indices
            for (j = 0; j < wfn.nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                val = one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    val += two_mo[koffset + n1 * jj + kk] - two_mo[koffset + n1 * kk + jj];
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    val += two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                // add determinant if |H*c| > eps and not already in wfn
                if (std::abs(val * coeffs[idet]) > eps) {
                    rank = wfn.rank_det(det_up);
                    if (wfn.index_det_from_rank(rank) == -1)
                        thread_wfn.add_det_with_rank(det_up, rank);
                }
                // loop over spin-down occupied indices
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = 0; l < wfn.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 1-1 excitation elements
                        excite_det(kk, ll, det_dn);
                        val = two_mo[koffset + n1 * jj + ll];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val * coeffs[idet]) > eps) {
                            rank = wfn.rank_det(det_up);
                            if (wfn.index_det_from_rank(rank) == -1)
                                thread_wfn.add_det_with_rank(det_up, rank);
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                // loop over spin-up occupied indices
                for (k = i + 1; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < wfn.nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        val = two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val * coeffs[idet]) > eps) {
                            rank = wfn.rank_det(det_up);
                            if (wfn.index_det_from_rank(rank) == -1)
                                thread_wfn.add_det_with_rank(det_up, rank);
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
            ioffset = n3 * ii;
            // loop over spin-down virtual indices
            for (j = 0; j < wfn.nvir_dn; ++j) {
                jj = virs_dn[j];
                // 0-1 excitation elements
                excite_det(ii, jj, det_dn);
                val = one_mo[n1 * ii + jj];
                for (k = 0; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    val += two_mo[ioffset + n2 * kk + n1 * jj + kk];
                }
                for (k = 0; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    val += two_mo[koffset + n1 * jj + kk] - two_mo[koffset + n1 * kk + jj];
                }
                // add determinant if |H*c| > eps and not already in wfn
                if (std::abs(val * coeffs[idet]) > eps) {
                    rank = wfn.rank_det(det_up);
                    if (wfn.index_det_from_rank(rank) == -1)
                        thread_wfn.add_det_with_rank(det_up, rank);
                }
                // loop over spin-down occupied indices
                for (k = i + 1; k < wfn.nocc_dn; ++k) {
                    kk = occs_dn[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-down virtual indices
                    for (l = j + 1; l < wfn.nvir_dn; ++l) {
                        ll = virs_dn[l];
                        // 0-2 excitation elements
                        excite_det(kk, ll, det_dn);
                        val = two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val * coeffs[idet]) > eps) {
                            rank = wfn.rank_det(det_up);
                            if (wfn.index_det_from_rank(rank) == -1)
                                thread_wfn.add_det_with_rank(det_up, rank);
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
    }
}


template<class WFN>
void hci_condense_thread(WFN &wfn, WFN &thread_wfn) {
    wfn.add_dets_from_wfn(thread_wfn);
    thread_wfn.clear();
}


} // namespace // anonymous


int_t OneSpinWfn::run_hci_doci(const double *v, const double *coeffs, const double eps) {
    // save ndet as ndet_old
    int_t ndet_old = ndet;
    // do computation in chunks by making smaller OneSpinWfns in parallel
    int_t nthread = omp_get_max_threads();
    if (nthread == 1) {
        hci_doci_run_thread(*this, *this, v, coeffs, eps, 0, ndet_old);
        return ndet - ndet_old;
    }
    int_t chunksize = ndet_old / nthread + ((ndet_old % nthread) ? 1 : 0);
    std::vector<OneSpinWfn> wfns(nthread);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < ndet_old) ? istart + chunksize : ndet_old;
        hci_doci_run_thread(*this, wfns[ithread], v, coeffs, eps, istart, iend);
    }
    // fill original OneSpinWfn (this instance)
    for (int_t t = 0; t < nthread; ++t)
        hci_condense_thread(*this, wfns[t]);
    // return number of determinants added
    return ndet - ndet_old;
}


int_t OneSpinWfn::run_hci_genci(const double *one_mo, const double *two_mo, const double *coeffs, const double eps) {
    // save ndet as ndet_old
    int_t ndet_old = ndet;
    // do computation in chunks by making smaller OneSpinWfns in parallel
    int_t nthread = omp_get_max_threads();
    if (nthread == 1) {
        hci_genci_run_thread(*this, *this, one_mo, two_mo, coeffs, eps, 0, ndet_old);
        return ndet - ndet_old;
    }
    int_t chunksize = ndet_old / nthread + ((ndet_old % nthread) ? 1 : 0);
    std::vector<OneSpinWfn> wfns(nthread);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < ndet_old) ? istart + chunksize : ndet_old;
        hci_genci_run_thread(*this, wfns[ithread], one_mo, two_mo, coeffs, eps, istart, iend);
    }
    // fill original OneSpinWfn (this instance)
    for (int_t t = 0; t < nthread; ++t)
        hci_condense_thread(*this, wfns[t]);
    // return number of determinants added
    return ndet - ndet_old;
}


int_t TwoSpinWfn::run_hci_fullci(const double *one_mo, const double *two_mo, const double *coeffs, const double eps) {
    // save ndet as ndet_old
    int_t ndet_old = ndet;
    // do computation in chunks by making smaller OneSpinWfns in parallel
    int_t nthread = omp_get_max_threads();
    if (nthread == 1) {
        hci_fullci_run_thread(*this, *this, one_mo, two_mo, coeffs, eps, 0, ndet_old);
        return ndet - ndet_old;
    }
    int_t chunksize = ndet_old / nthread + ((ndet_old % nthread) ? 1 : 0);
    std::vector<TwoSpinWfn> wfns(nthread);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < ndet_old) ? istart + chunksize : ndet_old;
        hci_fullci_run_thread(*this, wfns[ithread], one_mo, two_mo, coeffs, eps, istart, iend);
    }
    // fill original OneSpinWfn (this instance)
    for (int_t t = 0; t < nthread; ++t)
        hci_condense_thread(*this, wfns[t]);
    // return number of determinants added
    return ndet - ndet_old;
}


} // namespace pyci
