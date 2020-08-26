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

typedef HashMap<ulong, std::pair<double, double>> PairHashMap;

namespace {

template<class WfnType>
double compute_enpt2_tmpl(const Ham &, const WfnType &, const double *, const double, const double);

}

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

void compute_enpt2_thread_condense(PairHashMap &terms, PairHashMap &t_terms, const long ithread) {
    std::pair<double, double> *pair;
    for (auto &keyval : t_terms) {
        pair = &terms[keyval.first];
        pair->first += keyval.second.first;
        pair->second = keyval.second.second;
    }
    PairHashMap().swap(t_terms);
}

void compute_enpt2_thread_gather(const FullCIWfn &wfn, const double *one_mo, const double *two_mo,
                                 std::pair<double, double> &term, const double val, const long n2,
                                 const long n3, const long *occs_up) {
    const long *occs_dn = occs_up + wfn.nocc_up;
    // add enpt2 term to terms
    term.first += val;
    // check if diagonal element is already computed (i.e. not zero from initialization)
    if (term.second != (double)0.0)
        return;
    // compute diagonal element
    long i, j, k, l, ioffset, koffset;
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

void compute_enpt2_thread_terms(const Ham &ham, const FullCIWfn &wfn, PairHashMap &terms,
                                const double *coeffs, const double eps, const long idet,
                                ulong *det_up, long *occs_up, long *virs_up, long *t_up) {
    long i, j, k, l, ii, jj, kk, ll, ioffset, koffset, sign_up;
    ulong rank;
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val;
    const ulong *rdet_up = wfn.det_ptr(idet);
    const ulong *rdet_dn = rdet_up + wfn.nword;
    ulong *det_dn = det_up + wfn.nword;
    long *occs_dn = occs_up + wfn.nocc_up;
    long *virs_dn = virs_up + wfn.nvir_up;
    long *t_dn = t_up + wfn.nocc_up;
    std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword2);
    fill_occs(wfn.nword, rdet_up, occs_up);
    fill_occs(wfn.nword, rdet_dn, occs_dn);
    fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
    fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
    std::memcpy(t_up, occs_up, sizeof(long) * wfn.nocc_up);
    std::memcpy(t_dn, occs_dn, sizeof(long) * wfn.nocc_dn);
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
            std::memcpy(t_dn, occs_dn, sizeof(long) * wfn.nocc_dn);
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
    std::memcpy(t_up, occs_up, sizeof(long) * wfn.nocc_up);
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
                                 std::pair<double, double> &term, const double val, const long n2,
                                 const long n3, const long *occs) {
    // add enpt2 term to terms
    term.first += val;
    // check if diagonal element is already computed (i.e. not zero from initialization)
    if (term.second != (double)0.0)
        return;
    // compute diagonal element
    long i, j, k, l, ioffset, koffset;
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

void compute_enpt2_thread_terms(const Ham &ham, const GenCIWfn &wfn, PairHashMap &terms,
                                const double *coeffs, const double eps, const long idet, ulong *det,
                                long *occs, long *virs, long *tmps) {
    long i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    ulong rank;
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val;
    const ulong *rdet = wfn.det_ptr(idet);
    std::memcpy(det, rdet, sizeof(ulong) * wfn.nword);
    fill_occs(wfn.nword, rdet, occs);
    fill_virs(wfn.nword, wfn.nbasis, rdet, virs);
    std::memcpy(tmps, occs, sizeof(long) * wfn.nocc);
    // loop over occupied indices
    for (i = 0; i < wfn.nocc; ++i) {
        ii = occs[i];
        ioffset = n3 * ii;
        // loop over virtual indices
        for (j = 0; j < wfn.nvir; ++j) {
            jj = virs[j];
            // single excitation elements
            excite_det(ii, jj, det);
            val = ham.one_mo[n1 * ii + jj];
            for (k = 0; k < wfn.nocc; ++k) {
                kk = occs[k];
                koffset = ioffset + n2 * kk;
                val += ham.two_mo[koffset + n1 * jj + kk] - ham.two_mo[koffset + n1 * kk + jj];
            }
            val *= coeffs[idet];
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val) > eps) {
                rank = wfn.rank_det(det);
                if (wfn.index_det_from_rank(rank) == -1) {
                    val *= phase_single_det(wfn.nword, ii, jj, rdet);
                    fill_occs(wfn.nword, det, tmps);
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
                    excite_det(kk, ll, det);
                    val =
                        (ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj]) *
                        coeffs[idet];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val) > eps) {
                        rank = wfn.rank_det(det);
                        if (wfn.index_det_from_rank(rank) == -1) {
                            val *= phase_double_det(wfn.nword, ii, kk, jj, ll, rdet);
                            fill_occs(wfn.nword, det, tmps);
                            compute_enpt2_thread_gather(wfn, ham.one_mo, ham.two_mo, terms[rank],
                                                        val, n2, n3, tmps);
                        }
                    }
                    excite_det(ll, kk, det);
                }
            }
            excite_det(jj, ii, det);
        }
    }
}

template<class WfnType>
void compute_enpt2_thread(const Ham &ham, const WfnType &wfn, PairHashMap &terms,
                          const double *coeffs, const double eps, const long start,
                          const long end) {
    std::vector<ulong> det(wfn.nword2);
    std::vector<long> occs(wfn.nocc);
    std::vector<long> virs(wfn.nvir);
    std::vector<long> tmps(wfn.nocc);
    for (long i = start; i < end; ++i)
        compute_enpt2_thread_terms(ham, wfn, terms, coeffs, eps, i, &det[0], &occs[0], &virs[0],
                                   &tmps[0]);
}

template<class WfnType>
double compute_enpt2_tmpl(const Ham &ham, const WfnType &wfn, const double *coeffs,
                          const double energy, const double eps) {
    long nthread = get_num_threads(), start, end;
    long chunksize = wfn.ndet / nthread + ((wfn.ndet % nthread) ? 1 : 0);
    PairHashMap terms;
    std::vector<PairHashMap> v_terms(nthread);
    std::vector<std::thread> v_threads(nthread);
    for (long i = 0; i < nthread; ++i) {
        start = i * chunksize;
        end = (start + chunksize < wfn.ndet) ? start + chunksize : wfn.ndet;
        v_threads.emplace_back(&compute_enpt2_thread<WfnType>, std::ref(ham), std::ref(wfn),
                               std::ref(v_terms[i]), coeffs, eps, start, end);
    }
    long n = 0;
    for (auto &thread : v_threads) {
        thread.join();
        compute_enpt2_thread_condense(terms, v_terms[n], n);
        ++n;
    }
    // compute enpt2 correction
    double result = 0.0;
    for (const auto &keyval : terms)
        result += keyval.second.first * keyval.second.first / (energy - keyval.second.second);
    return result;
}

} // namespace

} // namespace pyci
