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

#include <algorithm>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include <parallel_hashmap/phmap.h>

#include <pyci/doci.h>
#include <pyci/fullci.h>


namespace pyci {


FullCIWfn::FullCIWfn() : nword(1), nbasis(2), nocc_up(1), nocc_dn(1), nvir_up(1), nvir_dn(1),
    ndet(0), maxdet_up(1), maxdet_dn(1) {
    return;
};


FullCIWfn::FullCIWfn(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_) {
    init(nbasis_, nocc_up_, nocc_dn_);
}


FullCIWfn::FullCIWfn(const FullCIWfn &wfn) {
    from_fullciwfn(wfn);
};


FullCIWfn::FullCIWfn(const char *filename) {
    from_file(filename);
}


FullCIWfn::FullCIWfn(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_, const int_t n,
    const uint_t *dets_) {
    from_det_array(nbasis_, nocc_up_, nocc_dn_, n, dets_);
}


FullCIWfn::FullCIWfn(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_, const int_t n,
    const int_t *occs) {
    from_occs_array(nbasis_, nocc_up_, nocc_dn_, n, occs);
}


FullCIWfn::~FullCIWfn() {
    return;
}


void FullCIWfn::init(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_) {
    // check that determinants of this wave function can be hashed
    int_t maxdet_up_ = binomial(nbasis_, nocc_up_);
    int_t maxdet_dn_ = binomial(nbasis_, nocc_dn_);
    if (maxdet_up_ * maxdet_dn_ >= PYCI_INT_MAX)
        throw std::runtime_error("nbasis, nocc_up, nocc_dn too large for hash type");
    // set attributes
    nword = nword_det(nbasis_);
    nword2 = nword * 2;
    nbasis = nbasis_;
    nocc_up = nocc_up_;
    nocc_dn = nocc_dn_;
    nvir_up = nbasis_ - nocc_up_;
    nvir_dn = nbasis_ - nocc_dn_;
    ndet = 0;
    maxdet_up = maxdet_up_;
    maxdet_dn = maxdet_dn_;
    // prepare determinant array and hashmap
    dets.resize(0);
    dict.clear();
}


void FullCIWfn::from_fullciwfn(const FullCIWfn &wfn) {
    nword = wfn.nword;
    nword2 = wfn.nword2;
    nbasis = wfn.nbasis;
    nocc_up = wfn.nocc_up;
    nocc_dn = wfn.nocc_dn;
    nvir_up = wfn.nvir_up;
    nvir_dn = wfn.nvir_dn;
    ndet = wfn.ndet;
    maxdet_up = wfn.maxdet_up;
    maxdet_dn = wfn.maxdet_dn;
    dets = wfn.dets;
    dict = wfn.dict;
}


void FullCIWfn::from_file(const char *filename) {
    // read file
    bool success = false;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file.read((char *)&ndet, sizeof(int_t))    &&
        file.read((char *)&nbasis, sizeof(int_t))  &&
        file.read((char *)&nocc_up, sizeof(int_t)) &&
        file.read((char *)&nocc_dn, sizeof(int_t))) {
        // set attributes
        nword = nword_det(nbasis);
        nword2 = nword * 2;
        nvir_up = nbasis - nocc_up;
        nvir_dn = nbasis - nocc_dn;
        maxdet_up = binomial(nbasis, nocc_up);
        maxdet_dn = binomial(nbasis, nocc_dn);
        // prepare determinant array and hashmap
        dets.resize(0);
        dict.clear();
        dets.resize(nword2 * ndet);
        dict.reserve(ndet);
        if (file.read((char *)&dets[0], sizeof(uint_t) * nword2 * ndet)) success = true;
    }
    file.close();
    // populate hashmap
    int_t j = 0;
    if (success)
        for (int_t i = 0; i < ndet; ++i) {
            dict[rank_det(nbasis, nocc_up, &dets[j]) * maxdet_dn
               + rank_det(nbasis, nocc_dn, &dets[j + nword])] = i;
            j += nword2;
        }
    else throw std::runtime_error("Error in file");
}


void FullCIWfn::from_det_array(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_,
    const int_t n, const uint_t *dets_) {
    init(nbasis_, nocc_up_, nocc_dn_);
    ndet = n;
    dets.resize(n * nword2);
    std::memcpy(&dets[0], dets_, sizeof(uint_t) * n * nword2);
    int_t j = 0;
    for (int_t i = 0; i < n; ++i) {
        dict[rank_det(nbasis_, nocc_up_, &dets_[j]) * maxdet_dn
           + rank_det(nbasis_, nocc_dn_, &dets_[j + nword])] = i;
        j += nword2;
    }
}


void FullCIWfn::from_occs_array(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_,
    const int_t n, const int_t *occs) {
    init(nbasis_, nocc_up_, nocc_dn_);
    ndet = n;
    dets.resize(n * nword2);
    int_t nthread = omp_get_max_threads();
    int_t chunksize = n / nthread + ((n % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < n) ? start + chunksize : n;
        int_t j = start * nocc_up_ * 2;
        int_t k = start * nword2;
        for (int_t i = start; i < end; ++i) {
            fill_det(nocc_up_, &occs[j], &dets[k]);
            j += nocc_up_;
            k += nword;
            fill_det(nocc_dn_, &occs[j], &dets[k]);
            j += nocc_up_;
            k += nword;
        }
    }
    int_t j = 0;
    for (int_t i = 0; i < n; ++i) {
        dict[rank_det(nbasis_, nocc_up_, &dets[j]) * maxdet_dn
           + rank_det(nbasis_, nocc_dn_, &dets[j + nword])] = i;
        j += nword2;
    }
}


void FullCIWfn::to_file(const char *filename) const {
    bool success = false;
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (file.write((char *)&ndet, sizeof(int_t))    &&
        file.write((char *)&nbasis, sizeof(int_t))  &&
        file.write((char *)&nocc_up, sizeof(int_t)) &&
        file.write((char *)&nocc_dn, sizeof(int_t)) &&
        file.write((char *)&dets[0], sizeof(uint_t) * nword2 * ndet)) success = true;
    file.close();
    if (!success) throw std::runtime_error("Error writing file");
}


void FullCIWfn::to_occs_array(const int_t low_ind, const int_t high_ind, int_t *occs) const {
    if (low_ind == high_ind) return;
    int_t range = high_ind - low_ind;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = range / nthread + ((range % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < range) ? start + chunksize : range;
        int_t j = (low_ind + start) * nword2;
        int_t k = start * nocc_up * 2;
        for (int_t i = start; i < end; ++i) {
            fill_occs(nword, &dets[j], &occs[k]);
            j += nword;
            k += nocc_up;
            fill_occs(nword, &dets[j], &occs[k]);
            j += nword;
            k += nocc_up;
        }
    }
}


int_t FullCIWfn::index_det(const uint_t *det) const {
    FullCIWfn::hashmap_type::const_iterator search = dict.find(
        rank_det(nbasis, nocc_up, &det[0]) * maxdet_dn + rank_det(nbasis, nocc_dn, &det[nword])
    );
    return (search == dict.end()) ? -1 : search->second;
}


int_t FullCIWfn::index_det_from_rank(const int_t rank) const {
    FullCIWfn::hashmap_type::const_iterator search = dict.find(rank);
    return (search == dict.end()) ? -1 : search->second;
}


void FullCIWfn::copy_det(const int_t i, uint_t *det) const {
    std::memcpy(det, &dets[i * nword2], sizeof(uint_t) * nword2);
}


const uint_t * FullCIWfn::det_ptr(const int_t i) const {
    return &dets[i * nword2];
}


int_t FullCIWfn::add_det(const uint_t *det) {
    if (dict.insert(std::make_pair(
            rank_det(nbasis, nocc_up, &det[0]) * maxdet_dn + rank_det(nbasis, nocc_dn, &det[nword]),
            ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(uint_t) * nword2);
        return ndet++;
    }
    return -1;
}


int_t FullCIWfn::add_det_from_occs(const int_t *occs) {
    std::vector<uint_t> det(nword2);
    fill_det(nocc_up, &occs[0], &det[0]);
    fill_det(nocc_dn, &occs[nocc_up], &det[nword]);
    return add_det(&det[0]);
}


void FullCIWfn::add_all_dets() {
    // prepare determinant array and hashmap
    ndet = maxdet_up * maxdet_dn;
    dets.resize(0);
    dict.clear();
    dets.resize(ndet * nword2);
    dict.reserve(ndet);
    for (int_t idet = 0; idet < ndet; ++idet)
        dict[idet] = idet;
    // add spin-up determinants to array
    int_t nthread = omp_get_max_threads();
    int_t chunksize = maxdet_up / nthread + ((maxdet_up % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < maxdet_up) ? start + chunksize : maxdet_up;
        std::vector<int_t> occs(nocc_up + 1);
        std::vector<uint_t> det(nword);
        unrank_indices(nbasis, nocc_up, start, &occs[0]);
        occs[nocc_up] = nbasis + 1;
        int_t j = start * maxdet_dn, k;
        for (int_t idet = start; idet < end; ++idet) {
            fill_det(nocc_up, &occs[0], &det[0]);
            for (k = 0; k < maxdet_dn; ++k)
                std::memcpy(&dets[nword2 * j++], &det[0], sizeof(uint_t) * nword);
            std::fill(det.begin(), det.end(), PYCI_UINT_ZERO);
            next_colex(&occs[0]);
        }
    }
    // add spin-down determinants to array
    chunksize = maxdet_dn / nthread + ((maxdet_dn % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < maxdet_dn) ? start + chunksize : maxdet_dn;
        std::vector<int_t> occs(nocc_dn + 1);
        std::vector<uint_t> det(nword);
        unrank_indices(nbasis, nocc_dn, start, &occs[0]);
        occs[nocc_dn] = nbasis + 1;
        int_t j, k;
        for (int_t idet = start; idet < end; ++idet) {
            fill_det(nocc_dn, &occs[0], &det[0]);
            j = idet;
            for (k = 0; k < maxdet_up; ++k) {
                std::memcpy(&dets[nword2 * j + nword], &det[0], sizeof(uint_t) * nword);
                j += maxdet_dn;
            }
            std::fill(det.begin(), det.end(), PYCI_UINT_ZERO);
            next_colex(&occs[0]);
        }
    }
}


void FullCIWfn::add_excited_dets(const uint_t *rdet, const int_t e_up, const int_t e_dn) {
    // handle trivial case
    if ((e_up == 0) && (e_dn == 0)) {
        add_det(rdet);
        return;
    }
    // make spin-up and spin-down parts
    DOCIWfn wfn_up(nbasis, nocc_up);
    DOCIWfn wfn_dn(nbasis, nocc_dn);
    wfn_up.add_excited_dets(&rdet[0], e_up);
    wfn_dn.add_excited_dets(&rdet[nword], e_dn);
    // add determinants
    int_t i, j;
    std::vector<uint_t> det(nword2);
    uint_t *dets_up = &wfn_up.dets[0];
    uint_t *dets_dn = &wfn_dn.dets[0];
    for (i = 0; i < wfn_up.ndet; ++i) {
        std::memcpy(&det[0], &dets_up[i * nword], sizeof(uint_t) * nword);
        for (j = 0; j < wfn_dn.ndet; ++j) {
            std::memcpy(&det[nword], &dets_dn[j * nword], sizeof(uint_t) * nword);
            add_det(&det[0]);
        }
    }
}


void FullCIWfn::reserve(const int_t n) {
    dets.reserve(n * nword2);
    dict.reserve(n);
}


void FullCIWfn::squeeze() {
    dets.shrink_to_fit();
}


int_t FullCIWfn::run_hci(const double *one_mo, const double *two_mo, const double *coeffs, const double eps) {
    // save ndet as ndet_old
    int_t ndet_old = ndet;
    // do computation in chunks by making smaller FullCIWfns in parallel
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    std::vector<FullCIWfn> wfns(nthread);
    #pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < ndet_old) ? istart + chunksize : ndet_old;
        wfns[ithread].run_hci_run_thread(*this, one_mo, two_mo, coeffs, eps, istart, iend);
    }
    // fill original FullCIWfn (this object)
    for (int_t t = 0; t < nthread; ++t)
        run_hci_condense_thread(wfns[t]);
    // return number of determinants added
    return ndet - ndet_old;
}


void FullCIWfn::run_hci_run_thread(const FullCIWfn &wfn, const double *one_mo, const double *two_mo,
    const double *coeffs, const double eps, const int_t istart, const int_t iend) {
    if (istart >= iend) return;
    // set attributes
    nword = wfn.nword;
    nword2 = wfn.nword2;
    nbasis = wfn.nbasis;
    nocc_up = wfn.nocc_up;
    nocc_dn = wfn.nocc_dn;
    nvir_up = wfn.nvir_up;
    nvir_dn = wfn.nvir_dn;
    maxdet_up = wfn.maxdet_up;
    maxdet_dn = wfn.maxdet_dn;
    // prepare working vectors
    std::vector<uint_t> det(wfn.nword2);
    std::vector<int_t> occs_up(wfn.nocc_up), occs_dn(wfn.nocc_dn);
    std::vector<int_t> virs_up(wfn.nvir_up), virs_dn(wfn.nvir_dn);
    const uint_t *rdet_up, *rdet_dn;
    uint_t *det_up = &det[0], *det_dn = &det[nword];
    // loop over determinants
    int_t i, j, k, l, ii, jj, kk, ll, ioffset, koffset;
    int_t rank_up_ref, rank_dn_ref, rank_up;
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
                // add determinant if |H*c| > eps and not already in wfn
                if ((std::abs(val * coeffs[idet]) > eps) &&
                    (wfn.index_det_from_rank(rank_up + rank_dn_ref) == -1))
                    add_det(det_up);
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
                        if ((std::abs(val * coeffs[idet]) > eps) &&
                            (wfn.index_det_from_rank(rank_up + rank_det(n1, wfn.nocc_dn, det_dn)) == -1))
                            add_det(det_up);
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
                        if ((std::abs(val * coeffs[idet]) > eps) &&
                            (wfn.index_det_from_rank(rank_det(n1, wfn.nocc_up, det_up)
                            /*------------------*/ * wfn.maxdet_dn + rank_dn_ref) == -1))
                            add_det(det_up);
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
                if ((std::abs(val * coeffs[idet]) > eps) &&
                    (wfn.index_det_from_rank(rank_up_ref + rank_det(n1, wfn.nocc_dn, det_dn)) == -1))
                    add_det(det_up);
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
                        if ((std::abs(val * coeffs[idet]) > eps) &&
                            (wfn.index_det_from_rank(rank_up_ref + rank_det(n1, wfn.nocc_dn, det_dn)) == -1))
                            add_det(det_up);
                        excite_det(ll, kk, det_dn);
                    }
                }
                excite_det(jj, ii, det_dn);
            }
        }
    }
}


void FullCIWfn::run_hci_condense_thread(FullCIWfn &wfn) {
    if (!(wfn.ndet)) return;
    for (int_t idet = 0; idet < wfn.ndet; ++idet)
        add_det(&wfn.dets[idet * nword2]);
    wfn.dets.resize(0);
    wfn.dict.clear();
}


} // namespace pyci
