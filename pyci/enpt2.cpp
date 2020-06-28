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

#include <utility>
#include <vector>

#include <omp.h>

#include <parallel_hashmap/phmap.h>

#include <pyci/fullci.h>


namespace pyci {


typedef hashmap<int_t, std::pair<double, double>> phashmap;


namespace { // anonymous


void enpt2_fullci_add_terms(const FullCIWfn &wfn, const double *one_mo, const double *two_mo,
    std::pair<double, double> &term, const double val, const int_t n3, const int_t n2,
    const int_t *occs_up, const int_t *occs_dn) {
    // add enpt2 term to terms
    term.first += val;
    // check if diagonal element is already computed (i.e. not zero from initialization)
    if (term.second != (double)0.0)
        return;
    // compute diagonal element
    int_t i, j, k, l, ioffset, koffset;
    double diag = 0.0;
    for (i = 0; i < wfn.nocc_up; ++i) {
        j = occs_up[i];
        ioffset = n3 * j;
        diag += one_mo[(wfn.nbasis + 1) * j];
        for (k = i + 1; k < wfn.nocc_up; ++k) {
            l = occs_up[k];
            koffset = ioffset + n2 * l;
            diag += two_mo[koffset + wfn.nbasis * j + l] - two_mo[koffset + wfn.nbasis * l + j];
        }
        for (k = 0; k < wfn.nocc_dn; ++k) {
            l = occs_dn[k];
            diag += two_mo[ioffset + n2 * l + wfn.nbasis * j + l];
        }
    }
    for (i = 0; i < wfn.nocc_dn; ++i) {
        j = occs_dn[i];
        ioffset = n3 * j;
        diag += one_mo[(wfn.nbasis + 1) * j];
        for (k = i + 1; k < wfn.nocc_dn; ++k) {
            l = occs_dn[k];
            koffset = ioffset + n2 * l;
            diag += two_mo[koffset + wfn.nbasis * j + l] - two_mo[koffset + wfn.nbasis * l + j];
        }
    }
    term.second = diag;
}


void enpt2_fullci_run_thread(const FullCIWfn &wfn, phashmap &terms, const double *one_mo,
    const double *two_mo, const double *coeffs, const double eps, const int_t istart, const int_t iend) {
    if (istart >= iend) return;
    // prepare working vectors
    std::vector<uint_t> det(wfn.nword2);
    std::vector<int_t> occs_up(wfn.nocc_up);
    std::vector<int_t> occs_dn(wfn.nocc_dn);
    std::vector<int_t> virs_up(wfn.nvir_up);
    std::vector<int_t> virs_dn(wfn.nvir_dn);
    std::vector<int_t> tmps_up(wfn.nocc_up);
    std::vector<int_t> tmps_dn(wfn.nocc_dn);
    const uint_t *rdet_up, *rdet_dn;
    uint_t *det_up = &det[0], *det_dn = &det[wfn.nword];
    int_t *t_up = &tmps_up[0], *t_dn = &tmps_dn[0];
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    int_t rank_up_ref, rank_dn_ref, sign_up, rank_up, rank;
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
        std::memcpy(t_up, &occs_up[0], sizeof(int_t) * wfn.nocc_up);
        std::memcpy(t_dn, &occs_dn[0], sizeof(int_t) * wfn.nocc_dn);
        rank_up_ref = rank_det(n1, wfn.nocc_up, rdet_up) * wfn.maxdet_dn;
        rank_dn_ref = rank_det(n1, wfn.nocc_dn, rdet_dn);
        // loop over spin-up occupied indices
        for (i = 0; i < wfn.nocc_up; ++i) {
            ii = occs_up[i];
            ioffset = n3 * ii;
            // loop over spin-up virtual indices
            for (j = 0; j < wfn.nvir_up; ++j) {
                jj = virs_up[j];
                // 1-0 excitation elements
                excite_det(ii, jj, det_up);
                sign_up = phase_single_det(wfn.nword, ii, jj, rdet_up);
                rank_up = rank_det(n1, wfn.nocc_up, det_up) * wfn.maxdet_dn;
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
                val *= coeffs[idet];
                // add determinant if |H*c| > eps and not already in wfn
                if (std::abs(val) > eps) {
                    rank = rank_up + rank_dn_ref;
                    if (wfn.index_det_from_rank(rank) == -1) {
                        val *= sign_up;
                        fill_occs(wfn.nword, det_up, t_up);
                        enpt2_fullci_add_terms(wfn, one_mo, two_mo, terms[rank], val, n3, n2, t_up, t_dn);
                    }
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
                        val = two_mo[koffset + n1 * jj + ll] * coeffs[idet];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val) > eps) {
                            rank = rank_up + rank_det(n1, wfn.nocc_dn, det_dn);
                            if (wfn.index_det_from_rank(rank) == -1) {
                                val *= sign_up * phase_single_det(wfn.nword, kk, ll, rdet_dn);
                                fill_occs(wfn.nword, det_dn, t_dn);
                                enpt2_fullci_add_terms(wfn, one_mo, two_mo, terms[rank], val, n3, n2, t_up, t_dn);
                            }
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                std::memcpy(t_dn, &occs_dn[0], sizeof(int_t) * wfn.nocc_dn);
                // loop over spin-up occupied indices
                for (k = i + 1; k < wfn.nocc_up; ++k) {
                    kk = occs_up[k];
                    koffset = ioffset + n2 * kk;
                    // loop over spin-up virtual indices
                    for (l = j + 1; l < wfn.nvir_up; ++l) {
                        ll = virs_up[l];
                        // 2-0 excitation elements
                        excite_det(kk, ll, det_up);
                        val = (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj]) * coeffs[idet];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val) > eps) {
                            rank = rank_det(n1, wfn.nocc_up, det_up) * wfn.maxdet_dn + rank_dn_ref;
                            if (wfn.index_det_from_rank(rank) == -1) {
                                val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up);
                                fill_occs(wfn.nword, det_up, t_up);
                                enpt2_fullci_add_terms(wfn, one_mo, two_mo, terms[rank], val, n3, n2, t_up, t_dn);
                            }
                        }
                        excite_det(ll, kk, det_up);
                    }
                }
                excite_det(jj, ii, det_up);
            }
        }
        std::memcpy(t_up, &occs_up[0], sizeof(int_t) * wfn.nocc_up);
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
                val *= coeffs[idet];
                // add determinant if |H*c| > eps and not already in wfn
                if (std::abs(val) > eps) {
                    rank = rank_up_ref + rank_det(n1, wfn.nocc_dn, det_dn);
                    if (wfn.index_det_from_rank(rank) == -1) {
                        val *= phase_single_det(wfn.nword, ii, jj, rdet_dn);
                        fill_occs(wfn.nword, det_dn, t_dn);
                        enpt2_fullci_add_terms(wfn, one_mo, two_mo, terms[rank], val, n3, n2, t_up, t_dn);
                    }
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
                        val = (two_mo[koffset + n1 * jj + ll] - two_mo[koffset + n1 * ll + jj]) * coeffs[idet];
                        // add determinant if |H*c| > eps and not already in wfn
                        if (std::abs(val) > eps) {
                            rank = rank_up_ref + rank_det(n1, wfn.nocc_dn, det_dn);
                            if (wfn.index_det_from_rank(rank) == -1) {
                                val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn); 
                                fill_occs(wfn.nword, det_dn, t_dn);
                                enpt2_fullci_add_terms(wfn, one_mo, two_mo, terms[rank], val, n3, n2, t_up, t_dn);
                            }
                        }
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
    }
}


void enpt2_condense_thread(phashmap &terms, phashmap &thread_terms, const int_t ithread) {
    if (ithread == 0) {
        std::swap(terms, thread_terms);
        return;
    }
    std::pair<double, double> *pair;
    for (auto& keyval : thread_terms) {
        pair = &terms[keyval.first];
        pair->first += keyval.second.first;
        pair->second = keyval.second.second;
    }
    thread_terms.clear();
}


} // namespace // anonymous


double FullCIWfn::compute_enpt2(const double *one_mo, const double *two_mo, const double *coeffs,
    const double energy, const double eps) const {
    // do computation in chunks by making smaller hashmaps in parallel
    int_t nthread = omp_get_max_threads();
    hashmap<int_t, std::pair<double, double>> terms;
    std::vector<phashmap> vterms(nthread);
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < ndet) ? istart + chunksize : ndet;
        enpt2_fullci_run_thread(*this, vterms[ithread], one_mo, two_mo, coeffs, eps, istart, iend);
        // combine hashmaps into larger terms hashmap
        #pragma omp for ordered schedule(static,1)
        for (int_t t = 0; t < nthread; ++t)
            #pragma omp ordered
            enpt2_condense_thread(terms, vterms[t], t);
    }
    // compute enpt2 correction
    double result = 0.0;
    for (auto& keyval : terms)
        result += keyval.second.first * keyval.second.first / (energy - keyval.second.second);
    return result;
}


} // namespace pyci
