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

namespace {

void hci_thread_add_dets(const SQuantOp &ham, const DOCIWfn &wfn, DOCIWfn &t_wfn, const double *coeffs,
                         const double eps, const long idet, ulong *det, long *occs, long *virs) {
    ulong rank;
    // fill working vectors
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
    // single/"pair"-excited elements elements
    for (long i = 0, k; i < wfn.nocc_up; ++i) {
        k = occs[i];
        for (long j = 0, l; j < wfn.nvir_up; ++j) {
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

void hci_thread_add_dets(const SQuantOp &ham, const FullCIWfn &wfn, FullCIWfn &t_wfn,
                         const double *coeffs, const double eps, const long idet, ulong *det_up,
                         long *occs_up, long *virs_up) {
    long i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
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
    std::memcpy(det_up, rdet_up, sizeof(ulong) * wfn.nword2);
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

void hci_thread_add_dets(const SQuantOp &ham, const GenCIWfn &wfn, GenCIWfn &t_wfn, const double *coeffs,
                         const double eps, const long idet, ulong *det, long *occs, long *virs) {
    ulong rank;
    long n1 = wfn.nbasis;
    long n2 = n1 * n1;
    long n3 = n1 * n2;
    double val;
    wfn.copy_det(idet, det);
    fill_occs(wfn.nword, det, occs);
    fill_virs(wfn.nword, wfn.nbasis, det, virs);
    // loop over occupied indices
    for (long i = 0, ii, ioffset, koffset; i < wfn.nocc; ++i) {
        ii = occs[i];
        ioffset = n3 * ii;
        // loop over virtual indices
        for (long j = 0, jj, k, kk; j < wfn.nvir; ++j) {
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
                for (long l = j + 1, ll; l < wfn.nvir; ++l) {
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
void hci_thread(const SQuantOp &ham, const WfnType &wfn, WfnType &t_wfn, const double *coeffs,
                const double eps, const long start, const long end) {
    AlignedVector<ulong> det(wfn.nword2);
    AlignedVector<long> occs(wfn.nocc);
    AlignedVector<long> virs(wfn.nvir);
    for (long i = start; i < end; ++i)
        hci_thread_add_dets(ham, wfn, t_wfn, coeffs, eps, i, &det[0], &occs[0], &virs[0]);
};

} // namespace

template<class WfnType>
long add_hci(const SQuantOp &ham, WfnType &wfn, const double *coeffs, const double eps, long nthread) {
    long ndet_old = wfn.ndet;
    if (nthread == -1)
        nthread = get_num_threads();
    long chunksize = ndet_old / nthread + static_cast<bool>(ndet_old % nthread);

    while (nthread > 1 && chunksize < PYCI_CHUNKSIZE_MIN) {
        nthread /= 2;
        chunksize = ndet_old / nthread + static_cast<bool>(ndet_old % nthread);
    }
    Vector<std::thread> v_threads;
    Vector<WfnType> v_wfns;
    v_threads.reserve(nthread);
    v_wfns.reserve(nthread);
    for (long i = 0, start, end = 0; i < nthread; ++i) {
        start = end;
        end = std::min(start + chunksize, ndet_old);
        v_wfns.emplace_back(wfn.nbasis, wfn.nocc_up, wfn.nocc_dn);
        v_threads.emplace_back(&hci_thread<WfnType>, std::ref(ham), std::ref(wfn),
                               std::ref(v_wfns.back()), coeffs, eps, start, end);
    }
    long n = 0;
    for (auto &thread : v_threads) {
        thread.join();
        wfn.add_dets_from_wfn(v_wfns[n++]);
    }
    return wfn.ndet - ndet_old;
}

template long add_hci<DOCIWfn>(const SQuantOp &, DOCIWfn &, const double *, const double, long);

template long add_hci<FullCIWfn>(const SQuantOp &, FullCIWfn &, const double *, const double, long);

template long add_hci<GenCIWfn>(const SQuantOp &, GenCIWfn &, const double *, const double, long);

template<class WfnType>
long py_add_hci(const SQuantOp &ham, WfnType &wfn, const Array<double> coeffs, const double eps,
                const long nthread) {
    return add_hci<WfnType>(ham, wfn, reinterpret_cast<const double *>(coeffs.request().ptr), eps,
                            nthread);
}

template long py_add_hci<DOCIWfn>(const SQuantOp &, DOCIWfn &, const Array<double>, const double,
                                  const long);

template long py_add_hci<FullCIWfn>(const SQuantOp &, FullCIWfn &, const Array<double>, const double,
                                    const long);

template long py_add_hci<GenCIWfn>(const SQuantOp &, GenCIWfn &, const Array<double>, const double,
                                   const long);

} // namespace pyci
