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

#include <cstdlib>
#include <cstring>
#include <thread>

namespace pyci {

namespace {

template<class WfnType>
int_t add_hci_tmpl(const Ham &, WfnType &, const double *, const double);

}

int_t add_hci(const Ham &ham, DOCIWfn &wfn, const double *coeffs, const double eps) {
    return add_hci_tmpl<DOCIWfn>(ham, wfn, coeffs, eps);
}

int_t add_hci(const Ham &ham, FullCIWfn &wfn, const double *coeffs, const double eps) {
    return add_hci_tmpl<FullCIWfn>(ham, wfn, coeffs, eps);
}

int_t add_hci(const Ham &ham, GenCIWfn &wfn, const double *coeffs, const double eps) {
    return add_hci_tmpl<GenCIWfn>(ham, wfn, coeffs, eps);
}

namespace {

void hci_thread_add_dets(const Ham &ham, const DOCIWfn &wfn, DOCIWfn &t_wfn, const double *coeffs,
                         const double eps, const int_t idet, uint_t *det, int_t *occs,
                         int_t *virs) {
    int_t i, j, k, l;
    uint_t rank;
    // fill working vectors
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
    // single/"pair"-excited elements elements
    for (i = 0; i < wfn.nocc_up; ++i) {
        k = occs[i];
        for (j = 0; j < wfn.nvir_up; ++j) {
            l = virs[j];
            excite_det(k, l, det);
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(ham.v[k * wfn.nbasis + l] * coeffs[idet]) > eps) {
                rank = wfn.rank_det(det);
                if (wfn.index_det_from_rank(rank) == -1)
                    t_wfn.add_det_with_rank(det, rank);
            }
            excite_det(l, k, det);
        }
    }
}

void hci_thread_add_dets(const Ham &ham, const FullCIWfn &wfn, FullCIWfn &t_wfn,
                         const double *coeffs, const double eps, const int_t idet, uint_t *det_up,
                         int_t *occs_up, int_t *virs_up) {
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
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
    std::memcpy(det_up, rdet_up, sizeof(uint_t) * wfn.nword2);
    fill_occs(wfn.nword, rdet_up, occs_up);
    fill_occs(wfn.nword, rdet_dn, occs_dn);
    fill_virs(wfn.nword, wfn.nbasis, rdet_up, virs_up);
    fill_virs(wfn.nword, wfn.nbasis, rdet_dn, virs_dn);
    // loop over spin-up occupied indices
    for (i = 0; i < wfn.nocc_up; ++i) {
        ii = occs_up[i];
        ioffset = n3 * ii;
        // loop over spin-up virtual indices
        for (j = 0; j < wfn.nvir_up; ++j) {
            jj = virs_up[j];
            // 1-0 excitation elements
            excite_det(ii, jj, det_up);
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
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val * coeffs[idet]) > eps) {
                rank = wfn.rank_det(det_up);
                if (wfn.index_det_from_rank(rank) == -1)
                    t_wfn.add_det_with_rank(det_up, rank);
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
                    val = ham.two_mo[koffset + n1 * jj + ll];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val * coeffs[idet]) > eps) {
                        rank = wfn.rank_det(det_up);
                        if (wfn.index_det_from_rank(rank) == -1)
                            t_wfn.add_det_with_rank(det_up, rank);
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
                    val = ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val * coeffs[idet]) > eps) {
                        rank = wfn.rank_det(det_up);
                        if (wfn.index_det_from_rank(rank) == -1)
                            t_wfn.add_det_with_rank(det_up, rank);
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
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val * coeffs[idet]) > eps) {
                rank = wfn.rank_det(det_up);
                if (wfn.index_det_from_rank(rank) == -1)
                    t_wfn.add_det_with_rank(det_up, rank);
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
                    val = ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val * coeffs[idet]) > eps) {
                        rank = wfn.rank_det(det_up);
                        if (wfn.index_det_from_rank(rank) == -1)
                            t_wfn.add_det_with_rank(det_up, rank);
                    }
                    excite_det(ll, kk, det_dn);
                }
            }
            excite_det(jj, ii, det_dn);
        }
    }
}

void hci_thread_add_dets(const Ham &ham, const GenCIWfn &wfn, GenCIWfn &t_wfn, const double *coeffs,
                         const double eps, const int_t idet, uint_t *det, int_t *occs,
                         int_t *virs) {
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    uint_t rank;
    int_t n1 = wfn.nbasis;
    int_t n2 = n1 * n1;
    int_t n3 = n1 * n2;
    double val;
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
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
            // add determinant if |H*c| > eps and not already in wfn
            if (std::abs(val * coeffs[idet]) > eps) {
                rank = wfn.rank_det(det);
                if (wfn.index_det_from_rank(rank) == -1)
                    t_wfn.add_det_with_rank(det, rank);
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
                    val = ham.two_mo[koffset + n1 * jj + ll] - ham.two_mo[koffset + n1 * ll + jj];
                    // add determinant if |H*c| > eps and not already in wfn
                    if (std::abs(val * coeffs[idet]) > eps) {
                        rank = wfn.rank_det(det);
                        if (wfn.index_det_from_rank(rank) == -1)
                            t_wfn.add_det_with_rank(det, rank);
                    }
                    excite_det(ll, kk, det);
                }
            }
            excite_det(jj, ii, det);
        }
    }
}

template<class WfnType>
void hci_thread(const Ham &ham, const WfnType &wfn, WfnType &t_wfn, const double *coeffs,
                const double eps, const int_t start, const int_t end) {
    std::vector<uint_t> det(wfn.nword2);
    std::vector<int_t> occs(wfn.nocc);
    std::vector<int_t> virs(wfn.nvir);
    for (int_t i = start; i < end; ++i)
        hci_thread_add_dets(ham, wfn, t_wfn, coeffs, eps, i, &det[0], &occs[0], &virs[0]);
};

template void hci_thread(const Ham &, const DOCIWfn &, DOCIWfn &, const double *, const double,
                         const int_t, const int_t);

template void hci_thread(const Ham &, const FullCIWfn &, FullCIWfn &, const double *, const double,
                         const int_t, const int_t);

template void hci_thread(const Ham &, const GenCIWfn &, GenCIWfn &, const double *, const double,
                         const int_t, const int_t);

template<class WfnType>
int_t add_hci_tmpl(const Ham &ham, WfnType &wfn, const double *coeffs, const double eps) {
    int_t ndet_old = wfn.ndet;
    int_t nthread = get_num_threads(), start, end;
    int_t chunksize = ndet_old / nthread + ((ndet_old % nthread) ? 1 : 0);
    std::vector<std::thread> v_threads;
    std::vector<WfnType> v_wfns;
    v_threads.reserve(nthread);
    v_wfns.reserve(nthread);
    for (int_t i = 0; i < nthread; ++i) {
        start = i * chunksize;
        end = (start + chunksize < ndet_old) ? start + chunksize : ndet_old;
        v_wfns.emplace_back(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn);
        v_threads.emplace_back(&hci_thread<WfnType>, std::ref(ham), std::ref(wfn),
                               std::ref(v_wfns.back()), coeffs, eps, start, end);
    }
    int_t n = 0;
    for (auto &thread : v_threads) {
        thread.join();
        wfn.add_dets_from_wfn(v_wfns[n++]);
    }
    return wfn.ndet - ndet_old;
}

} // namespace

} // namespace pyci
