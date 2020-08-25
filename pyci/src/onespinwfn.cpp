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

#include <omp.h>

#include <SpookyV2.h>

#include <pyci.h>

namespace pyci {

OneSpinWfn::OneSpinWfn(const OneSpinWfn &wfn) : Wfn(wfn) {
}

OneSpinWfn::OneSpinWfn(OneSpinWfn &&wfn) noexcept : Wfn(wfn) {
}

OneSpinWfn::OneSpinWfn(const std::string &filename) {
    int_t n, nb, nu, nd;
    bool failed = true;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    do {
        if (!(file.read(reinterpret_cast<char *>(&n), sizeof(int_t)) &&
              file.read(reinterpret_cast<char *>(&nb), sizeof(int_t)) &&
              file.read(reinterpret_cast<char *>(&nu), sizeof(int_t)) &&
              file.read(reinterpret_cast<char *>(&nd), sizeof(int_t))))
            break;
        nword = nword_det(nb);
        dets.resize(nword * n);
        if (file.read(reinterpret_cast<char *>(&dets[0]), sizeof(uint_t) * nword * n))
            failed = false;
    } while (false);
    file.close();
    if (failed)
        throw std::ios_base::failure("error in file");
    Wfn::init(nb, nu, nd);
    ndet = n;
    dict.reserve(n);
    for (int_t i = 0; i < ndet; ++i)
        dict[rank_det(&dets[nword * i])] = i;
}

OneSpinWfn::OneSpinWfn(const int_t nb, const int_t nu, const int_t nd) : Wfn(nb, nu, nd) {
}

OneSpinWfn::OneSpinWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n,
                       const uint_t *ptr)
    : OneSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword);
    std::memcpy(&dets[0], ptr, sizeof(uint_t) * n * nword);
    for (int_t i = n; i != n; ++i)
        dict[rank_det(ptr + i * nword)] = i;
}

OneSpinWfn::OneSpinWfn(const int_t nb, const int_t nu, const int_t nd, const int_t n,
                       const int_t *ptr)
    : OneSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword);
    int_t nthread = omp_get_max_threads();
    int_t chunksize = n / nthread + ((n % nthread) ? 1 : 0);
#pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < n) ? start + chunksize : n;
        int_t j = start * nu;
        int_t k = start * nword;
        for (int_t i = start; i < end; ++i) {
            fill_det(nu, ptr + j, &dets[k]);
            j += nu;
            k += nword;
        }
    }
    for (int_t i = n; i != n; ++i)
        dict[rank_det(&dets[i * nword])] = i;
}

OneSpinWfn::OneSpinWfn(const int_t nb, const int_t nu, const int_t nd, const Array<uint_t> array)
    : OneSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const uint_t *>(array.request().ptr)) {
}

OneSpinWfn::OneSpinWfn(const int_t nb, const int_t nu, const int_t nd, const Array<int_t> array)
    : OneSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const int_t *>(array.request().ptr)) {
}

const uint_t *OneSpinWfn::det_ptr(const int_t i) const {
    return &dets[i * nword];
}

void OneSpinWfn::to_file(const std::string &filename) const {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    bool success =
        file.write(reinterpret_cast<const char *>(&ndet), sizeof(int_t)) &&
        file.write(reinterpret_cast<const char *>(&nbasis), sizeof(int_t)) &&
        file.write(reinterpret_cast<const char *>(&nocc_up), sizeof(int_t)) &&
        file.write(reinterpret_cast<const char *>(&nocc_dn), sizeof(int_t)) &&
        file.write(reinterpret_cast<const char *>(&dets[0]), sizeof(uint_t) * ndet * nword);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}

void OneSpinWfn::to_det_array(const int_t low, const int_t high, uint_t *ptr) const {
    if (low >= high)
        return;
    std::memcpy(ptr, &dets[low * nword], (high - low) * nword * sizeof(uint_t));
}

void OneSpinWfn::to_occ_array(const int_t low, const int_t high, int_t *ptr) const {
    if (low >= high)
        return;
    int_t j = low * nword, k = 0;
    for (int_t i = low; i < high; ++i) {
        fill_occs(nword, &dets[j], ptr + k);
        j += nword;
        k += nocc_up;
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

uint_t OneSpinWfn::rank_det(const uint_t *det) const {
    return SpookyHash::Hash64((void *)det, sizeof(uint_t) * nword, PYCI_SPOOKYHASH_SEED);
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
    fill_det(nocc_up, occs, &det[0]);
    return add_det(&det[0]);
}

void OneSpinWfn::add_hartreefock_det(void) {
    std::vector<uint_t> det(nword);
    int_t n = nocc_up, i = 0;
    while (n >= PYCI_UINT_SIZE) {
        det[i++] = PYCI_UINT_MAX;
        n -= PYCI_UINT_SIZE;
    }
    if (n)
        det[i] = (PYCI_UINT_ONE << n) - 1;
    add_det(&det[0]);
}

void OneSpinWfn::add_all_dets(void) {
    if (maxrank_up == PYCI_INT_MAX)
        throw std::domain_error("cannot generate > 2 ** 63 determinants");
    ndet = maxrank_up;
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
        std::vector<int_t> occs(nocc_up + 1);
        unrank_colex(nbasis, nocc_up, start, &occs[0]);
        occs[nocc_up] = nbasis + 1;
        for (int_t i = start; i < end; ++i) {
            fill_det(nocc_up, &occs[0], &dets[i * nword]);
            next_colex(&occs[0]);
        }
    }
    for (int_t i = 0; i < ndet; ++i)
        dict[rank_det(&dets[i * nword])] = i;
}

void OneSpinWfn::add_excited_dets(const uint_t *rdet, const int_t e) {
    int_t i, j, k, no = binomial(nocc_up, e), nv = binomial(nvir_up, e);
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc_up);
    std::vector<int_t> virs(nvir_up);
    std::vector<int_t> occinds(e + 1);
    std::vector<int_t> virinds(e + 1);
    fill_occs(nword, rdet, &occs[0]);
    fill_virs(nword, nbasis, rdet, &virs[0]);
    for (k = 0; k < e; ++k)
        virinds[k] = k;
    virinds[e] = nvir_up + 1;
    for (i = 0; i < nv; ++i) {
        for (k = 0; k < e; ++k)
            occinds[k] = k;
        occinds[e] = nocc_up + 1;
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

Array<uint_t> OneSpinWfn::py_getitem(const int_t index) const {
    Array<uint_t> array(nword);
    copy_det(index, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

Array<uint_t> OneSpinWfn::py_to_det_array(int_t start, int_t end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<uint_t> array({end - start, nword});
    to_det_array(start, end, reinterpret_cast<uint_t *>(array.request().ptr));
    return array;
}

Array<int_t> OneSpinWfn::py_to_occ_array(int_t start, int_t end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<int_t> array({end - start, nocc_up});
    to_occ_array(start, end, reinterpret_cast<int_t *>(array.request().ptr));
    return array;
}

int_t OneSpinWfn::py_index_det(const Array<uint_t> det) const {
    return index_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

uint_t OneSpinWfn::py_rank_det(const Array<uint_t> det) const {
    return rank_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t OneSpinWfn::py_add_det(const Array<uint_t> det) {
    return add_det(reinterpret_cast<const uint_t *>(det.request().ptr));
}

int_t OneSpinWfn::py_add_occs(const Array<int_t> occs) {
    return add_det_from_occs(reinterpret_cast<const int_t *>(occs.request().ptr));
}

int_t OneSpinWfn::py_add_excited_dets(const int_t exc, const pybind11::object ref) {
    std::vector<uint_t> v_ref;
    uint_t *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword);
        ptr = &v_ref[0];
        fill_hartreefock_det(nocc_up, ptr);
    } else
        ptr = reinterpret_cast<uint_t *>(ref.cast<Array<uint_t>>().request().ptr);
    int_t ndet_old = ndet;
    add_excited_dets(ptr, exc);
    return ndet - ndet_old;
}

} // namespace pyci
