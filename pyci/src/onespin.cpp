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

#ifndef PYCI_EXACT_HASH
#include <SpookyV2.h>
#endif

#include <parallel_hashmap/phmap.h>

#include <pyci.h>


namespace pyci {


OneSpinWfn::OneSpinWfn(void) {
    return;
}


OneSpinWfn::OneSpinWfn(OneSpinWfn &&wfn) noexcept
    : nword(std::exchange(wfn.nword, 0)), nbasis(std::exchange(wfn.nbasis, 0)),
      nocc(std::exchange(wfn.nocc, 0)), nvir(std::exchange(wfn.nvir, 0)),
      ndet(std::exchange(wfn.ndet, 0)), dets(std::move(wfn.dets)), dict(std::move(wfn.dict)) {
}


OneSpinWfn::OneSpinWfn(const int_t nbasis_, const int_t nocc_) {
    init(nbasis_, nocc_);
}


OneSpinWfn::OneSpinWfn(const OneSpinWfn &wfn) {
    from_onespinwfn(wfn);
}


OneSpinWfn::OneSpinWfn(const TwoSpinWfn &wfn) {
    from_twospinwfn(wfn);
}


OneSpinWfn::OneSpinWfn(const char *filename) {
    from_file(filename);
}


OneSpinWfn::OneSpinWfn(const int_t nbasis_, const int_t nocc_, const int_t n, const uint_t *dets_) {
    from_det_array(nbasis_, nocc_, n, dets_);
}


OneSpinWfn::OneSpinWfn(const int_t nbasis_, const int_t nocc_, const int_t n, const int_t *occs) {
    from_occs_array(nbasis_, nocc_, n, occs);
}


void OneSpinWfn::init(const int_t nbasis_, const int_t nocc_) {
    if ((nocc_ < 0) || (nocc_ > nbasis_))
        throw std::domain_error("nocc cannot be greater than nbasis");
#ifdef PYCI_EXACT_HASH
    else if (binomial_raises(nbasis_, nocc_))
        throw std::domain_error("nbasis, nocc too large for hash type");
#endif
    nword = nword_det(nbasis_);
    nbasis = nbasis_;
    nocc = nocc_;
    nvir = nbasis_ - nocc_;
    ndet = 0;
}


void OneSpinWfn::from_onespinwfn(const OneSpinWfn &wfn) {
    nword = wfn.nword;
    nbasis = wfn.nbasis;
    nocc = wfn.nocc;
    nvir = wfn.nvir;
    ndet = wfn.ndet;
    dets = wfn.dets;
    dict = wfn.dict;
}


void OneSpinWfn::from_twospinwfn(const TwoSpinWfn &wfn) {
    int_t nbasis_ = wfn.nbasis * 2;
    int_t nword_ = nword_det(nbasis_);
    int_t nocc_ = wfn.nocc_up + wfn.nocc_dn;
    nword = nword_;
    nbasis = nbasis_;
    nocc = nocc_;
    nvir = nbasis_ - nocc_;
    ndet = wfn.ndet;
    std::fill(dets.begin(), dets.end(), PYCI_UINT_ZERO);
    dets.resize(wfn.ndet * nword_);
    dict.clear();
    std::vector<int_t> occs(nocc_ + 1);
    int_t i = 0, j = 0, k;
    for (int_t idet = 0; idet < wfn.ndet; ++idet) {
        fill_occs(wfn.nword, &wfn.dets[i], &occs[0]);
        i += wfn.nword;
        fill_occs(wfn.nword, &wfn.dets[i], &occs[wfn.nocc_up]);
        i += wfn.nword;
        for (k = wfn.nocc_up; k < nocc_; ++k)
            occs[k] += wfn.nbasis;
        fill_det(nocc_, &occs[0], &dets[j]);
        dict[rank_det(&dets[j])] = idet;
        j += nword_;
    }
}


void OneSpinWfn::from_file(const char *filename) {
    bool success = false;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    do {
        if (!(file.read((char *)&ndet, sizeof(int_t))
                    && file.read((char *)&nbasis, sizeof(int_t))
                    && file.read((char *)&nocc, sizeof(int_t))))
            break;
        nword = nword_det(nbasis);
        nvir = nbasis - nocc;
        std::fill(dets.begin(), dets.end(), PYCI_UINT_ZERO);
        dets.resize(nword * ndet);
        dict.clear();
        dict.reserve(ndet);
        if (file.read((char *)&dets[0], sizeof(uint_t) * nword * ndet))
            success = true;
    } while (false);
    file.close();
    if (success)
        for (int_t idet = 0; idet < ndet; ++idet)
            dict[rank_det(&dets[nword * idet])] = idet;
    else
        throw std::ios_base::failure("error in file");
    return;
}


void OneSpinWfn::from_det_array(const int_t nbasis_, const int_t nocc_, const int_t n, const uint_t *dets_) {
    init(nbasis_, nocc_);
    ndet = n;
    dets.resize(n * nword);
    std::memcpy(&dets[0], dets_, sizeof(uint_t) * n * nword);
    for (int_t idet = n; idet != n; ++idet)
        dict[rank_det(&dets_[idet * nword])] = idet;
}


void OneSpinWfn::from_occs_array(const int_t nbasis_, const int_t nocc_, const int_t n, const int_t *occs) {
    init(nbasis_, nocc_);
    ndet = n;
    dets.resize(n * nword);
    int_t nthread = omp_get_max_threads();
    int_t chunksize = n / nthread + ((n % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < n) ? start + chunksize : n;
        int_t j = start * nocc_;
        int_t k = start * nword;
        for (int_t i = start; i < end; ++i) {
            fill_det(nocc_, &occs[j], &dets[k]);
            j += nocc_;
            k += nword;
        }
    }
    for (int_t idet = n; idet != n; ++idet)
        dict[rank_det(&dets[idet * nword])] = idet;
}


void OneSpinWfn::to_file(const char *filename) const {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    bool success = file.write((char *)&ndet, sizeof(int_t))
        && file.write((char *)&nbasis, sizeof(int_t))
        && file.write((char *)&nocc, sizeof(int_t))
        && file.write((char *)&dets[0], sizeof(uint_t) * nword * ndet);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}


void OneSpinWfn::to_occs_array(const int_t low_ind, const int_t high_ind, int_t *occs) const {
    if (low_ind == high_ind) return;
    int_t range = high_ind - low_ind;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = range / nthread + ((range % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < range) ? start + chunksize : range;
        int_t j = (low_ind + start) * nword;
        int_t k = start * nocc;
        for (int_t i = start; i < end; ++i) {
            fill_occs(nword, &dets[j], &occs[k]);
            j += nword;
            k += nocc;
        }
    }
}


int_t OneSpinWfn::index_det(const uint_t *det) const {
    const auto &search = dict.find(rank_det(det));
    return (search == dict.end()) ? -1 : search->second;
}


int_t OneSpinWfn::index_det_from_rank(const uint_t rank) const {
    const auto &search = dict.find(rank);
    return (search == dict.end()) ? -1 : search->second;
}


void OneSpinWfn::copy_det(const int_t i, uint_t *det) const {
    std::memcpy(det, &dets[i * nword], sizeof(uint_t) * nword);
}


const uint_t * OneSpinWfn::det_ptr(const int_t i) const {
    return &dets[i * nword];
}


uint_t OneSpinWfn::rank_det(const uint_t *det) const {
#ifdef PYCI_EXACT_HASH
    return (uint_t)rank_colex(nbasis, nocc, det);
#else
    return (uint_t)SpookyHash::Hash64((void *)det, sizeof(uint_t) * nword, PYCI_SPOOKYHASH_SEED);
#endif
}


int_t OneSpinWfn::add_det(const uint_t *det) {
    if (dict.insert(std::make_pair(rank_det(det), ndet)).second) {
        dets.resize(dets.size() + nword);
        std::memcpy(&dets[nword * ndet], det, sizeof(uint_t) * nword);
        return ndet++;
    }
    return -1;
}


int_t OneSpinWfn::add_det_with_rank(const uint_t *det, const uint_t rank) {
    if (dict.insert(std::make_pair(rank, ndet)).second) {
        dets.resize(dets.size() + nword);
        std::memcpy(&dets[nword * ndet], det, sizeof(uint_t) * nword);
        return ndet++;
    }
    return -1;
}


int_t OneSpinWfn::add_det_from_occs(const int_t *occs) {
    std::vector<uint_t> det(nword);
    fill_det(nocc, occs, &det[0]);
    return add_det(&det[0]);
}


void OneSpinWfn::add_hartreefock_det(void) {
    std::vector<uint_t> det(nword);
    int_t n = nocc, i = 0;
    while (n >= PYCI_UINT_SIZE) {
        det[i++] = PYCI_UINT_MAX;
        n -= PYCI_UINT_SIZE;
    }
    if (n)
        det[i] = (PYCI_UINT_ONE << n) - 1;
    add_det(&det[0]);
}


void OneSpinWfn::add_all_dets(void) {
    if (binomial_raises(nbasis, nocc))
        throw std::domain_error("cannot generate > 2 ** 63 determinants");
    ndet = binomial(nbasis, nocc);
    std::fill(dets.begin(), dets.end(), PYCI_UINT_ZERO);
    dets.resize(ndet * nword);
    dict.clear();
    dict.reserve(ndet);
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < ndet) ? start + chunksize : ndet;
        std::vector<int_t> occs(nocc + 1);
        unrank_colex(nbasis, nocc, start, &occs[0]);
        occs[nocc] = nbasis + 1;
        for (int_t idet = start; idet < end; ++idet) {
            fill_det(nocc, &occs[0], &dets[idet * nword]);
            next_colex(&occs[0]);
        }
    }
    for (int_t idet = 0; idet < ndet; ++idet)
        dict[rank_det(&dets[idet * nword])] = idet;
}


void OneSpinWfn::add_excited_dets(const uint_t *rdet, const int_t e) {
    int_t i, j, k, no = binomial(nocc, e), nv = binomial(nvir, e);
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    std::vector<int_t> occinds(e + 1);
    std::vector<int_t> virinds(e + 1);
    fill_occs(nword, rdet, &occs[0]);
    fill_virs(nword, nbasis, rdet, &virs[0]);
    for (k = 0; k < e; ++k)
        virinds[k] = k;
    virinds[e] = nvir + 1;
    for (i = 0; i < nv; ++i) {
        for (k = 0; k < e; ++k)
            occinds[k] = k;
        occinds[e] = nocc + 1;
        for (j = 0; j < no; ++j) {
            std::memcpy(&det[0], rdet, sizeof(uint_t) * nword);
            for (k = 0; k < e; ++k)
                excite_det(occs[occinds[k]], virs[virinds[k]], &det[0]);
            add_det(&det[0]);
            next_colex(&occinds[0]);
        }
        next_colex(&virinds[0]);
    }
}


void OneSpinWfn::add_dets_from_wfn(const OneSpinWfn &wfn) {
    for (const auto &keyval : wfn.dict)
        add_det_with_rank(&wfn.dets[keyval.second * nword], keyval.first);
}


void OneSpinWfn::reserve(const int_t n) {
    dets.reserve(n * nword);
    dict.reserve(n);
}


void OneSpinWfn::squeeze(void) {
    dets.shrink_to_fit();
}


void OneSpinWfn::clear(void) {
    dets.resize(0);
    dets.shrink_to_fit();
    dict.clear();
    ndet = 0;
}


double OneSpinWfn::compute_overlap(const double *coeffs, const OneSpinWfn &wfn, const double *w_coeffs) const {
    // run this function for the smaller wfn
    if (ndet > wfn.ndet) return wfn.compute_overlap(w_coeffs, *this, coeffs);
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
            jdet = wfn.index_det(&dets[idet * nword]);
            if (jdet != -1)
                olp += coeffs[idet] * w_coeffs[jdet];
        }
    }
    return olp;
}


} // namespace pyci
