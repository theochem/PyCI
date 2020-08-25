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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include <parallel_hashmap/phmap.h>

#include <SpookyV2.h>

#include <pyci.h>

namespace pyci {

TwoSpinWfn::TwoSpinWfn(const TwoSpinWfn &wfn) : Wfn(wfn) {
}

TwoSpinWfn::TwoSpinWfn(TwoSpinWfn &&wfn) noexcept : Wfn(wfn) {
}

TwoSpinWfn::TwoSpinWfn(const std::string &filename) {
    int_t n, nb, nu, nd;
    bool failed = true;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    do {
        if (!(file.read((char *)&n, sizeof(int_t)) && file.read((char *)&nb, sizeof(int_t)) &&
              file.read((char *)&nu, sizeof(int_t)) && file.read((char *)&nd, sizeof(int_t))))
            break;
        nword2 = nword_det(nb) * 2;
        dets.resize(nword2 * n);
        if (file.read((char *)&dets[0], sizeof(uint_t) * nword2 * n))
            failed = false;
    } while (false);
    file.close();
    if (failed)
        throw std::ios_base::failure("error in file");
    Wfn::init(nb, nu, nd);
    ndet = n;
    dict.reserve(n);
    for (int_t idet = 0; idet < ndet; ++idet)
        dict[rank_det(&dets[nword2 * idet])] = idet;
}

TwoSpinWfn::TwoSpinWfn(const int_t nb, const int_t nu, const int_t nd) : Wfn(nb, nu, nd) {
}

TwoSpinWfn::TwoSpinWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n,
                       const uint_t *ptr)
    : TwoSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword2);
    std::memcpy(&dets[0], ptr, sizeof(uint_t) * n * nword2);
    for (int_t i = 0; i < n; ++i)
        dict[rank_det(ptr + i * nword2)] = i;
}

TwoSpinWfn::TwoSpinWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n,
                       const int_t *ptr)
    : TwoSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword2);
    int_t nthread = omp_get_max_threads();
    int_t chunksize = n / nthread + ((n % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < n) ? start + chunksize : n;
        int_t j = start * nu * 2;
        int_t k = start * nword2;
        for (int_t i = start; i < end; ++i) {
            fill_det(nu, ptr + j, &dets[k]);
            j += nu;
            k += nword;
            fill_det(nd, ptr + j, &dets[k]);
            j += nu;
            k += nword;
        }
    }
    for (int_t i = 0; i < n; ++i)
        dict[rank_det(&dets[i * nword2])] = i;
}

void TwoSpinWfn::to_file(const std::string &filename) const {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    bool success = file.write((char *)&ndet, sizeof(int_t)) &&
                   file.write((char *)&nbasis, sizeof(int_t)) &&
                   file.write((char *)&nocc_up, sizeof(int_t)) &&
                   file.write((char *)&nocc_dn, sizeof(int_t)) &&
                   file.write((char *)&dets[0], sizeof(uint_t) * nword2 * ndet);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}

void TwoSpinWfn::to_det_array(const int_t low, const int_t high, uint_t *ptr) const {
    if (low >= high)
        return;
    std::memcpy(ptr, &dets[low * nword], (high - low) * nword2 * sizeof(uint_t));
}

void TwoSpinWfn::to_occ_array(const int_t low, const int_t high, int_t *ptr) const {
    if (low == high)
        return;
    int_t j = low * nword2, k = 0;
    for (int_t i = low; i < high; ++i) {
        fill_occs(nword, &dets[j], ptr + k);
        j += nword;
        k += nocc_up;
        fill_occs(nword, &dets[j], ptr + k);
        j += nword;
        k += nocc_up;
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

uint_t TwoSpinWfn::rank_det(const uint_t *det) const {
    return SpookyHash::Hash64((void *)det, sizeof(uint_t) * nword2, PYCI_SPOOKYHASH_SEED);
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

void TwoSpinWfn::add_hartreefock_det(void) {
    std::vector<uint_t> det(nword2);
    int_t n = nocc_up, i = 0;
    while (n >= PYCI_UINT_SIZE) {
        det[i++] = PYCI_UINT_MAX;
        n -= PYCI_UINT_SIZE;
    }
    if (n)
        det[i] = (PYCI_UINT_ONE << n) - 1;
    n = nocc_dn;
    i = nword;
    while (n >= PYCI_UINT_SIZE) {
        det[i++] = PYCI_UINT_MAX;
        n -= PYCI_UINT_SIZE;
    }
    if (n)
        det[i] = (PYCI_UINT_ONE << n) - 1;
    add_det(&det[0]);
}

void TwoSpinWfn::add_all_dets(void) {
    if ((maxrank_up == PYCI_INT_MAX) || (maxrank_dn == PYCI_INT_MAX))
        throw std::domain_error("cannot generate > 2 ** 63 determinants");
    ndet = maxrank_up * maxrank_dn;
    std::fill(dets.begin(), dets.end(), PYCI_UINT_ZERO);
    dets.resize(ndet * nword2);
    dict.clear();
    dict.reserve(ndet);
    // add spin-up determinants to array
    int_t nthread = omp_get_max_threads();
    int_t chunksize = maxrank_up / nthread + ((maxrank_up % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < maxrank_up) ? start + chunksize : maxrank_up;
        std::vector<int_t> occs(nocc_up + 1);
        std::vector<uint_t> det(nword);
        unrank_colex(nbasis, nocc_up, start, &occs[0]);
        occs[nocc_up] = nbasis + 1;
        int_t j = start * maxrank_dn, k;
        for (int_t i = start; i < end; ++i) {
            fill_det(nocc_up, &occs[0], &det[0]);
            for (k = 0; k < maxrank_dn; ++k)
                std::memcpy(&dets[j++ * nword2], &det[0], sizeof(uint_t) * nword);
            std::fill(det.begin(), det.end(), PYCI_UINT_ZERO);
            next_colex(&occs[0]);
        }
    }
    // add spin-down determinants to array
    chunksize = maxrank_dn / nthread + ((maxrank_dn % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < maxrank_dn) ? start + chunksize : maxrank_dn;
        std::vector<int_t> occs(nocc_dn + 1);
        std::vector<uint_t> det(nword);
        unrank_colex(nbasis, nocc_dn, start, &occs[0]);
        occs[nocc_dn] = nbasis + 1;
        int_t j, k;
        for (int_t i = start; i < end; ++i) {
            fill_det(nocc_dn, &occs[0], &det[0]);
            j = i;
            for (k = 0; k < maxrank_up; ++k) {
                std::memcpy(&dets[j * nword2 + nword], &det[0], sizeof(uint_t) * nword);
                j += maxrank_dn;
            }
            std::fill(det.begin(), det.end(), PYCI_UINT_ZERO);
            next_colex(&occs[0]);
        }
    }
    for (int_t i = 0; i < ndet; ++i)
        dict[rank_det(&dets[i * nword2])] = i;
}

void TwoSpinWfn::add_excited_dets(const uint_t *rdet, const int_t e_up, const int_t e_dn) {
    if ((e_up == 0) && (e_dn == 0)) {
        add_det(rdet);
        return;
    }
    OneSpinWfn wfn_up(nbasis, nocc_up, nocc_up);
    OneSpinWfn wfn_dn(nbasis, nocc_dn, nocc_dn);
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
        std::memcpy(&det[0], wfn_up.det_ptr(i), sizeof(uint_t) * nword);
        for (j = 0; j < wfn_dn.ndet; ++j) {
            std::memcpy(&det[nword], wfn_dn.det_ptr(j), sizeof(uint_t) * nword);
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

} // namespace pyci
