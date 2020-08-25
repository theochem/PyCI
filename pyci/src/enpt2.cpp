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

#include <omp.h>
#include <parallel_hashmap/phmap.h>
#include <pyci.h>

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace pyci {

typedef hashmap<uint_t, std::pair<double, double>> p_hashmap;

namespace {

template<class WfnType>
double compute_enpt2_tmpl(const Ham &, const WfnType &, const double *, const double, const double);

void compute_enpt2_thread_condense(p_hashmap &, p_hashmap &, const int_t);

void compute_enpt2_thread_gather(const FullCIWfn &, const double *, const double *,
                                 std::pair<double, double> &, const double, const int_t,
                                 const int_t, const int_t *);

void compute_enpt2_thread_gather(const GenCIWfn &, const double *, const double *,
                                 std::pair<double, double> &, const double, const int_t,
                                 const int_t, const int_t *);

void compute_enpt2_thread_terms(const Ham &, const FullCIWfn &, p_hashmap &, const double *,
                                const double, const int_t, uint_t *, int_t *, int_t *, int_t *);

void compute_enpt2_thread_terms(const Ham &, const GenCIWfn &, p_hashmap &, const double *,
                                const double, const int_t, uint_t *, int_t *, int_t *, int_t *);

} // namespace

double compute_enpt2(const Ham &ham, const DOCIWfn &wfn, const double *coeffs, const double energy,
                     const double eps) {
    return compute_enpt2_tmpl<FullCIWfn>(ham, FullCIWfn(wfn), coeffs, energy, eps);
}

double compute_enpt2(const Ham &ham, const FullCIWfn &wfn, const double *coeffs,
                     const double energy, const double eps) {
    return compute_enpt2_tmpl<FullCIWfn>(ham, wfn, coeffs, energy, eps);
}

double compute_enpt2(const Ham &ham, const GenCIWfn &wfn, const double *coeffs, const double energy,
                     const double eps) {
    return compute_enpt2_tmpl<GenCIWfn>(ham, wfn, coeffs, energy, eps);
}

namespace {

template<class WfnType>
double compute_enpt2_tmpl(const Ham &ham, const WfnType &wfn, const double *coeffs,
                          const double energy, const double eps) {
    p_hashmap terms;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = wfn.ndet / nthread + ((wfn.ndet % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t start = ithread * chunksize;
        int_t end = (start + chunksize < wfn.ndet) ? start + chunksize : wfn.ndet;
        p_hashmap t_terms;
        std::vector<uint_t> det(wfn.nword2);
        std::vector<int_t> occs(wfn.nocc);
        std::vector<int_t> virs(wfn.nvir);
        std::vector<int_t> tmps(wfn.nocc);
        for (int_t i = start; i < end; ++i)
            compute_enpt2_thread_terms(ham, wfn, t_terms, coeffs, eps, i, &det[0], &occs[0],
                                       &virs[0], &tmps[0]);
#pragma omp for ordered schedule(static, 1)
        for (int_t i = 0; i < nthread; ++i)
#pragma omp ordered
            compute_enpt2_thread_condense(terms, t_terms, i);
    }
    // compute enpt2 correction
    double result = 0.0;
    for (const auto &keyval : terms)
        result += keyval.second.first * keyval.second.first / (energy - keyval.second.second);
    return result;
}

void compute_enpt2_thread_condense(p_hashmap &terms, p_hashmap &t_terms, const int_t ithread) {
    std::pair<double, double> *pair;
    if (ithread == 0) {
        terms.swap(t_terms);
    } else {
        for (auto &keyval : t_terms) {
            pair = &terms[keyval.first];
            pair->first += keyval.second.first;
            pair->second = keyval.second.second;
        }
        t_terms.clear();
    }
}

void compute_enpt2_thread_gather(const FullCIWfn &wfn, const double *one_mo, const double *two_mo,
                                 std::pair<double, double> &term, const double val, const int_t n2,
                                 const int_t n3, const int_t *occs_up) {
    const int_t *occs_dn = occs_up + wfn.nocc_up;
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

void compute_enpt2_thread_terms(const Ham &ham, const FullCIWfn &wfn, p_hashmap &terms,
                                const double *coeffs, const double eps, const int_t idet,
                                uint_t *det_up, int_t *occs_up, int_t *virs_up, int_t *t_up) {
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset, sign_up;
    uint_t rank;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val;
    const uint_t *rdet_up = wfn.det_ptr(idet);
    const uint_t *rdet_dn = rdet_up + wfn.nword;
    uint_t *det_dn = det_up + wfn.nword;
    int_t *occs_dn = occs_up + wfn.nocc_up;
    int_t *virs_dn = virs_up + wfn.nvir_up;
    int_t *t_dn = t_up + wfn.nocc_up;
    std::memcpy(det_up, rdet_up, sizeof(uint_t) * wfn.nword2);
    fill_occs(wfn.nword, rdet_up, occs_up);
    fill_occs(wfn.nword, rdet_dn, occs_dn);
    fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
    fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
    std::memcpy(t_up, occs_up, sizeof(int_t) * wfn.nocc_up);
    std::memcpy(t_dn, occs_dn, sizeof(int_t) * wfn.nocc_dn);
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
            val = ham.one_mo[n1 * ii + jj];
            for (k = 0; k < wfn.nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                val += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
            }
            for (k = 0; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                val += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
            }
            val *= coeffs[idet];
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val) > eps) {
                rank = wfn.rank_det(det_up);
                if (wfn.index_det_from_rank(rank) == -1) {
                    val *= sign_up;
                    fill_occs(wfn.nword, det_up, t_up);
                    compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank], val, n2,
                                                n3, t_up);
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
                    val = ham.two_mo[koffset + n1 * jj + ll] * coeffs[idet];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val) > eps) {
                        rank = wfn.rank_det(det_up);
                        if (wfn.index_det_from_rank(rank) == -1) {
                            val *= sign_up * phase_single_det(wfn.nword, kk, ll, rdet_dn);
                            fill_occs(wfn.nword, det_dn, t_dn);
                            compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank],
                                                        val, n2, n3, t_up);
                        }
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            std::memcpy(t_dn, occs_dn, sizeof(int_t) * wfn.nocc_dn);
            // loop over spin-up occupied indices
            for (k = i + 1; k < wfn.nocc_up; ++k) {
                kk = occs_up[k];
                koffset = ioffset + n2 * kk;
                // loop over spin-up virtual indices
                for (l = j + 1; l < wfn.nvir_up; ++l) {
                    ll = virs_up[l];
                    // 2-0 excitation elements
                    excite_det(kk, ll, det_up);
                    val =
                        (ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj]) *
                        coeffs[idet];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val) > eps) {
                        rank = wfn.rank_det(det_up);
                        if (wfn.index_det_from_rank(rank) == -1) {
                            val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_up);
                            fill_occs(wfn.nword, det_up, t_up);
                            compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank],
                                                        val, n2, n3, t_up);
                        }
                    }
                    excite_det(ll, kk, det_up);
                }
            }
            excite_det(jj, ii, det_up);
        }
    }
    std::memcpy(t_up, occs_up, sizeof(int_t) * wfn.nocc_up);
    // loop over spin-down occupied indices
    for (i = 0; i < wfn.nocc_dn; ++i) {
        ii = occs_dn[i];
        ioffset = n3 * ii;
        // loop over spin-down virtual indices
        for (j = 0; j < wfn.nvir_dn; ++j) {
            jj = virs_dn[j];
            // 0-1 excitation elements
            excite_det(ii, jj, det_dn);
            val = ham.one_mo[n1 * ii + jj];
            for (k = 0; k < wfn.nocc_up; ++k) {
                kk = occs_up[k];
                val += ham.two_mo[ioffset + n2 * kk + n1 * jj + kk];
            }
            for (k = 0; k < wfn.nocc_dn; ++k) {
                kk = occs_dn[k];
                koffset = ioffset + n2 * kk;
                val += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
            }
            val *= coeffs[idet];
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val) > eps) {
                rank = wfn.rank_det(det_up);
                if (wfn.index_det_from_rank(rank) == -1) {
                    val *= phase_single_det(wfn.nword, ii, jj, rdet_dn);
                    fill_occs(wfn.nword, det_dn, t_dn);
                    compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank], val, n2,
                                                n3, t_up);
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
                    val =
                        (ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj]) *
                        coeffs[idet];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val) > eps) {
                        rank = wfn.rank_det(det_up);
                        if (wfn.index_det_from_rank(rank) == -1) {
                            val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet_dn);
                            fill_occs(wfn.nword, det_dn, t_dn);
                            compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank],
                                                        val, n2, n3, t_up);
                        }
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            excite_det(jj, ii, det_dn);
        }
    }
}

void compute_enpt2_thread_gather(const GenCIWfn &wfn, const double *one_mo, const double *two_mo,
                                 std::pair<double, double> &term, const double val, const int_t n2,
                                 const int_t n3, const int_t *occs) {
    // add enpt2 term to terms
    term.first += val;
    // check if diagonal element is already computed (i.e. not zero from initialization)
    if (term.second != (double)0.0)
        return;
    // compute diagonal element
    int_t i, j, k, l, ioffset, koffset;
    double diag = 0.0;
    for (i = 0; i < wfn.nocc; ++i) {
        j = occs[i];
        ioffset = n3 * j;
        diag += one_mo[(wfn.nbasis + 1) * j];
        for (k = i + 1; k < wfn.nocc; ++k) {
            l = occs[k];
            koffset = ioffset + n2 * l;
            diag += two_mo[koffset + wfn.nbasis * j + l] - two_mo[koffset + wfn.nbasis * l + j];
        }
    }
    term.second = diag;
}

void compute_enpt2_thread_terms(const Ham &ham, const GenCIWfn &wfn, p_hashmap &terms,
                                const double *coeffs, const double eps, const int_t idet,
                                uint_t *det, int_t *occs, int_t *virs, int_t *tmps) {
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    uint_t rank;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val;
    const uint_t *rdet = wfn.det_ptr(idet);
    std::memcpy(&det[0], rdet, sizeof(uint_t) * wfn.nword);
    fill_occs(wfn.nword, rdet, &occs[0]);
    fill_virs(wfn.nword, wfn.nbasis, rdet, &virs[0]);
    std::memcpy(&tmps[0], &occs[0], sizeof(int_t) * wfn.nocc);
    // loop over occupied indices
    for (i = 0; i < wfn.nocc; ++i) {
        ii = occs[i];
        ioffset = n3 * ii;
        // loop over virtual indices
        for (j = 0; j < wfn.nvir; ++j) {
            jj = virs[j];
            // single excitation elements
            excite_det(ii, jj, &det[0]);
            val = ham.one_mo[n1 * ii + jj];
            for (k = 0; k < wfn.nocc; ++k) {
                kk = occs[k];
                koffset = ioffset + n2 * kk;
                val += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
            }
            val *= coeffs[idet];
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val) > eps) {
                rank = wfn.rank_det(&det[0]);
                if (wfn.index_det_from_rank(rank) == -1) {
                    val *= phase_single_det(wfn.nword, ii, jj, rdet);
                    fill_occs(wfn.nword, &det[0], &tmps[0]);
                    compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank], val, n2,
                                                n3, tmps);
                }
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
                    val =
                        (ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj]) *
                        coeffs[idet];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val) > eps) {
                        rank = wfn.rank_det(&det[0]);
                        if (wfn.index_det_from_rank(rank) == -1) {
                            val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet);
                            fill_occs(wfn.nword, &det[0], &tmps[0]);
                            compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank],
                                                        val, n2, n3, tmps);
                        }
                    }
                    excite_det(ll, kk, &det[0]);
                }
            }
            excite_det(jj, ii, &det[0]);
        }
    }
}

} // namespace

} // namespace pyci
