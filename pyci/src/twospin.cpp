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

#ifndef PYCI_EXACT_HASH
#include <SpookyV2.h>
#endif

#include <pyci.h>


namespace pyci {


TwoSpinWfn::TwoSpinWfn(void) {
    return;
}


TwoSpinWfn::TwoSpinWfn(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_) {
    init(nbasis_, nocc_up_, nocc_dn_);
}


TwoSpinWfn::TwoSpinWfn(const OneSpinWfn &wfn) {
    from_onespinwfn(wfn);
}


TwoSpinWfn::TwoSpinWfn(const TwoSpinWfn &wfn) {
    from_twospinwfn(wfn);
}


TwoSpinWfn::TwoSpinWfn(const char *filename) {
    from_file(filename);
}


TwoSpinWfn::TwoSpinWfn(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_, const int_t n,
    const uint_t *dets_) {
    from_det_array(nbasis_, nocc_up_, nocc_dn_, n, dets_);
}


TwoSpinWfn::TwoSpinWfn(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_, const int_t n,
    const int_t *occs) {
    from_occs_array(nbasis_, nocc_up_, nocc_dn_, n, occs);
}


TwoSpinWfn::~TwoSpinWfn(void) {
    return;
}


void TwoSpinWfn::init(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_) {
    if ((nocc_dn_ < 0) || (nocc_dn_ > nocc_up_) || (nocc_up_ > nbasis_))
        throw std::domain_error("nocc_up cannot be > nbasis and nocc_dn cannot be > nocc_up");
    int_t maxdet_up_ = binomial(nbasis_, nocc_up_);
    int_t maxdet_dn_ = binomial(nbasis_, nocc_dn_);
#ifdef PYCI_EXACT_HASH
    if (binomial_raises(nbasis_, nocc_up_) || (maxdet_up_ * maxdet_dn_ >= PYCI_INT_MAX))
        throw std::domain_error("nbasis, nocc_up, nocc_dn too large for hash type");
#else
    if (binomial_raises(nbasis_, nocc_up_)) {
        maxdet_up_ = PYCI_INT_MAX;
        maxdet_dn_ = PYCI_INT_MAX;
    }
#endif
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
}


void TwoSpinWfn::from_onespinwfn(const OneSpinWfn &wfn) {
    int_t maxdet = binomial(wfn.nbasis, wfn.nocc);
    if (maxdet * maxdet >= PYCI_INT_MAX)
#ifdef PYCI_EXACT_HASH
        throw std::domain_error("nbasis, nocc_up, nocc_dn too large for hash type");
#else
        maxdet = PYCI_INT_MAX;
#endif
    nword = wfn.nword;
    nword2 = wfn.nword * 2;
    nbasis = wfn.nbasis;
    nocc_up = wfn.nocc;
    nocc_dn = wfn.nocc;
    nvir_up = wfn.nvir;
    nvir_dn = wfn.nvir;
    ndet = wfn.ndet;
    maxdet_up = maxdet;
    maxdet_dn = maxdet;
    dets.resize(wfn.ndet * nword2);
    dict.clear();
    for (int_t idet = 0; idet < wfn.ndet; ++idet) {
        std::memcpy(&dets[idet * nword2], &wfn.dets[idet * nword], sizeof(uint_t) * wfn.nword);
        std::memcpy(&dets[idet * nword2 + nword], &wfn.dets[idet * nword], sizeof(uint_t) * wfn.nword);
        dict[rank_det(&dets[idet * nword2])] = idet;
    }
}


void TwoSpinWfn::from_twospinwfn(const TwoSpinWfn &wfn) {
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


void TwoSpinWfn::from_file(const char *filename) {
    bool success = false;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    do {
        if (!(file.read((char *)&ndet, sizeof(int_t))    &&
                    file.read((char *)&nbasis, sizeof(int_t))  &&
                    file.read((char *)&nocc_up, sizeof(int_t)) &&
                    file.read((char *)&nocc_dn, sizeof(int_t))))
            break;
        nword = nword_det(nbasis);
        nword2 = nword * 2;
        nvir_up = nbasis - nocc_up;
        nvir_dn = nbasis - nocc_dn;
        maxdet_up = binomial(nbasis, nocc_up);
        maxdet_dn = binomial(nbasis, nocc_dn);
        std::fill(dets.begin(), dets.end(), PYCI_UINT_ZERO);
        dets.resize(nword2 * ndet);
        dict.clear();
        dict.reserve(ndet);
        if (file.read((char *)&dets[0], sizeof(uint_t) * nword2 * ndet))
            success = true;
    } while (false);
    file.close();
    if (success)
        for (int_t idet = 0; idet < ndet; ++idet)
            dict[rank_det(&dets[nword2 * idet])] = idet;
    else
        throw std::ios_base::failure("error in file");
}


void TwoSpinWfn::from_det_array(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_,
    const int_t n, const uint_t *dets_) {
    init(nbasis_, nocc_up_, nocc_dn_);
    ndet = n;
    dets.resize(n * nword2);
    std::memcpy(&dets[0], dets_, sizeof(uint_t) * n * nword2);
    for (int_t idet = 0; idet < n; ++idet)
        dict[rank_det(&dets[nword2 * idet])] = idet;
}


void TwoSpinWfn::from_occs_array(const int_t nbasis_, const int_t nocc_up_, const int_t nocc_dn_,
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
    for (int_t idet = 0; idet < n; ++idet)
        dict[rank_det(&dets[nword2 * idet])] = idet;
}


void TwoSpinWfn::to_file(const char *filename) const {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    bool success = file.write((char *)&ndet, sizeof(int_t))
        && file.write((char *)&nbasis, sizeof(int_t))
        && file.write((char *)&nocc_up, sizeof(int_t))
        && file.write((char *)&nocc_dn, sizeof(int_t))
        && file.write((char *)&dets[0], sizeof(uint_t) * nword2 * ndet);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}


void TwoSpinWfn::to_occs_array(const int_t low_ind, const int_t high_ind, int_t *occs) const {
    if (low_ind == high_ind)
        return;
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


int_t TwoSpinWfn::index_det(const uint_t *det) const {
    const auto &search = dict.find(rank_det(det));
    return (search == dict.end()) ? -1 : search->second;
}


int_t TwoSpinWfn::index_det_from_rank(const uint_t rank) const {
    const auto &search = dict.find(rank);
    return (search == dict.end()) ? -1 : search->second;
}


void TwoSpinWfn::copy_det(const int_t i, uint_t *det) const {
    std::memcpy(det, &dets[i * nword2], sizeof(uint_t) * nword2);
}


const uint_t * TwoSpinWfn::det_ptr(const int_t i) const {
    return &dets[i * nword2];
}


uint_t TwoSpinWfn::rank_det(const uint_t *det) const {
#ifdef PYCI_EXACT_HASH
    return (uint_t)(rank_colex(nbasis, nocc_up, det) * maxdet_dn + rank_colex(nbasis, nocc_dn, &det[nword]));
#else
    return (uint_t)SpookyHash::Hash64((void *)det, sizeof(uint_t) * nword2, PYCI_SPOOKYHASH_SEED);
#endif
}


int_t TwoSpinWfn::add_det(const uint_t *det) {
    if (dict.insert(std::make_pair(rank_det(det), ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(uint_t) * nword2);
        return ndet++;
    }
    return -1;
}


int_t TwoSpinWfn::add_det_with_rank(const uint_t *det, const uint_t rank) {
    if (dict.insert(std::make_pair(rank, ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(uint_t) * nword2);
        return ndet++;
    }
    return -1;
}


int_t TwoSpinWfn::add_det_from_occs(const int_t *occs) {
    std::vector<uint_t> det(nword2);
    fill_det(nocc_up, &occs[0], &det[0]);
    fill_det(nocc_dn, &occs[nocc_up], &det[nword]);
    return add_det(&det[0]);
}


void TwoSpinWfn::add_all_dets(void) {
    if ((maxdet_up == PYCI_INT_MAX) || (maxdet_dn == PYCI_INT_MAX))
        throw std::domain_error("cannot generate > 2 ** 63 determinants");
    ndet = maxdet_up * maxdet_dn;
    std::fill(dets.begin(), dets.end(), PYCI_UINT_ZERO);
    dets.resize(ndet * nword2);
    dict.clear();
    dict.reserve(ndet);
    // add spin-up determinants to array
    int_t nthread = omp_get_max_threads();
    int_t chunksize = maxdet_up / nthread + ((maxdet_up % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < maxdet_up) ? start + chunksize : maxdet_up;
        std::vector<int_t> occs(nocc_up + 1);
        std::vector<uint_t> det(nword);
        unrank_colex(nbasis, nocc_up, start, &occs[0]);
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
        unrank_colex(nbasis, nocc_dn, start, &occs[0]);
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
    for (int_t idet = 0; idet < ndet; ++idet)
        dict[rank_det(&dets[nword2 * idet])] = idet;
}


void TwoSpinWfn::add_excited_dets(const uint_t *rdet, const int_t e_up, const int_t e_dn) {
    if ((e_up == 0) && (e_dn == 0)) {
        add_det(rdet);
        return;
    }
    OneSpinWfn wfn_up(nbasis, nocc_up);
    OneSpinWfn wfn_dn(nbasis, nocc_dn);
#pragma omp parallel sections
    {
#pragma omp section
        wfn_up.add_excited_dets(&rdet[0], e_up);
#pragma omp section
        wfn_dn.add_excited_dets(&rdet[nword], e_dn);
    }
    std::vector<uint_t> det(nword2);
    int_t j;
    for (int_t i = 0; i < wfn_up.ndet; ++i) {
        std::memcpy(&det[0], &wfn_up.dets[i * nword], sizeof(uint_t) * nword);
        for (j = 0; j < wfn_dn.ndet; ++j) {
            std::memcpy(&det[nword], &wfn_dn.dets[j * nword], sizeof(uint_t) * nword);
            add_det(&det[0]);
        }
    }
}


void TwoSpinWfn::add_dets_from_wfn(const TwoSpinWfn &wfn) {
    for (const auto &keyval : wfn.dict)
        add_det_with_rank(&wfn.dets[keyval.second * nword2], keyval.first);
}


void TwoSpinWfn::reserve(const int_t n) {
    dets.reserve(n * nword2);
    dict.reserve(n);
}


void TwoSpinWfn::squeeze(void) {
    dets.shrink_to_fit();
}


void TwoSpinWfn::clear(void) {
    dets.resize(0);
    dets.shrink_to_fit();
    dict.clear();
    ndet = 0;
}


double TwoSpinWfn::compute_overlap(const double *coeffs, const TwoSpinWfn &wfn, const double *w_coeffs) const {
    // run this function for the smaller wfn
    if (ndet > wfn.ndet)
        return wfn.compute_overlap(w_coeffs, *this, coeffs);
    // iterate over this instance's determinants in parallel
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    double olp = 0.0;
#pragma omp parallel reduction(+:olp)
    {
        int_t istart = omp_get_thread_num() * chunksize;
        int_t iend = (istart + chunksize < ndet) ? istart + chunksize : ndet;
        int_t jdet;
        for (int_t idet = istart; idet < iend; ++idet) {
            // add c[idet] * c[jdet] if both wfns constain determinant idet
            jdet = wfn.index_det(&dets[idet * nword2]);
            if (jdet != -1) olp += coeffs[idet] * w_coeffs[jdet];
        }
    }
    return olp;
}


} // namespace pyci
