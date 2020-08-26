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

TwoSpinWfn::TwoSpinWfn(const TwoSpinWfn &wfn) : Wfn(wfn) {
}

TwoSpinWfn::TwoSpinWfn(TwoSpinWfn &&wfn) noexcept : Wfn(wfn) {
}

TwoSpinWfn::TwoSpinWfn(const std::string &filename) {
    long n, nb, nu, nd;
    bool failed = true;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    do {
        if (!(file.read(reinterpret_cast<char *>(&n), sizeof(long)) &&
              file.read(reinterpret_cast<char *>(&nb), sizeof(long)) &&
              file.read(reinterpret_cast<char *>(&nu), sizeof(long)) &&
              file.read(reinterpret_cast<char *>(&nd), sizeof(long))))
            break;
        nword2 = nword_det(nb) * 2;
        dets.resize(nword2 * n);
        if (file.read(reinterpret_cast<char *>(&dets[0]), sizeof(unsigned long) * nword2 * n))
            failed = false;
    } while (false);
    file.close();
    if (failed)
        throw std::ios_base::failure("error in file");
    Wfn::init(nb, nu, nd);
    ndet = n;
    dict.reserve(n);
    for (long idet = 0; idet < ndet; ++idet)
        dict[rank_det(&dets[nword2 * idet])] = idet;
}

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd) : Wfn(nb, nu, nd) {
}

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd, const long n,
                       const unsigned long *ptr)
    : TwoSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword2);
    std::memcpy(&dets[0], ptr, sizeof(unsigned long) * n * nword2);
    for (long i = 0; i < n; ++i)
        dict[rank_det(ptr + i * nword2)] = i;
}

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : TwoSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword2);
    long j = 0, k = 0;
    for (long i = 0; i < n; ++i) {
        fill_det(nu, ptr + j, &dets[k]);
        j += nu;
        k += nword;
        fill_det(nd, ptr + j, &dets[k]);
        j += nu;
        k += nword;
    }
    for (long i = 0; i < n; ++i)
        dict[rank_det(&dets[i * nword2])] = i;
}

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd,
                       const Array<unsigned long> array)
    : TwoSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const unsigned long *>(array.request().ptr)) {
}

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd, const Array<long> array)
    : TwoSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const long *>(array.request().ptr)) {
}

const unsigned long *TwoSpinWfn::det_ptr(const long i) const {
    return &dets[i * nword2];
}

void TwoSpinWfn::to_file(const std::string &filename) const {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    bool success =
        file.write(reinterpret_cast<const char *>(&ndet), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&nbasis), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&nocc_up), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&nocc_dn), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&dets[0]), sizeof(unsigned long) * nword2 * ndet);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}

void TwoSpinWfn::to_det_array(const long low, const long high, unsigned long *ptr) const {
    if (low >= high)
        return;
    std::memcpy(ptr, &dets[low * nword], (high - low) * nword2 * sizeof(unsigned long));
}

void TwoSpinWfn::to_occ_array(const long low, const long high, long *ptr) const {
    if (low == high)
        return;
    long j = low * nword2, k = 0;
    for (long i = low; i < high; ++i) {
        fill_occs(nword, &dets[j], ptr + k);
        j += nword;
        k += nocc_up;
        fill_occs(nword, &dets[j], ptr + k);
        j += nword;
        k += nocc_up;
    }
}

long TwoSpinWfn::index_det(const unsigned long *det) const {
    const auto &search = dict.find(rank_det(det));
    return (search == dict.end()) ? -1 : search->second;
}

long TwoSpinWfn::index_det_from_rank(const unsigned long rank) const {
    const auto &search = dict.find(rank);
    return (search == dict.end()) ? -1 : search->second;
}

void TwoSpinWfn::copy_det(const long i, unsigned long *det) const {
    std::memcpy(det, &dets[i * nword2], sizeof(unsigned long) * nword2);
}

unsigned long TwoSpinWfn::rank_det(const unsigned long *det) const {
    return SpookyHash::Hash64((void *)det, sizeof(unsigned long) * nword2, PYCI_SPOOKYHASH_SEED);
}

long TwoSpinWfn::add_det(const unsigned long *det) {
    if (dict.insert(std::make_pair(rank_det(det), ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(unsigned long) * nword2);
        return ndet++;
    }
    return -1;
}

long TwoSpinWfn::add_det_with_rank(const unsigned long *det, const unsigned long rank) {
    if (dict.insert(std::make_pair(rank, ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(unsigned long) * nword2);
        return ndet++;
    }
    return -1;
}

long TwoSpinWfn::add_det_from_occs(const long *occs) {
    std::vector<unsigned long> det(nword2);
    fill_det(nocc_up, &occs[0], &det[0]);
    fill_det(nocc_dn, &occs[nocc_up], &det[nword]);
    return add_det(&det[0]);
}

void TwoSpinWfn::add_hartreefock_det(void) {
    std::vector<unsigned long> det(nword2);
    fill_hartreefock_det(nocc_up, &det[0]);
    fill_hartreefock_det(nocc_dn, &det[nword]);
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
    std::vector<long> occs(nocc_up + 1);
    std::vector<unsigned long> det(nword);
    unrank_colex(nbasis, nocc_up, 0, &occs[0]);
    occs[nocc_up] = nbasis + 1;
    long j = 0, k;
    for (long i = 0; i < maxrank_up; ++i) {
        fill_det(nocc_up, &occs[0], &det[0]);
        for (k = 0; k < maxrank_dn; ++k)
            std::memcpy(&dets[j++ * nword2], &det[0], sizeof(unsigned long) * nword);
        std::fill(det.begin(), det.end(), PYCI_UINT_ZERO);
        next_colex(&occs[0]);
    }
    // add spin-down determinants to array
    unrank_colex(nbasis, nocc_dn, 0, &occs[0]);
    occs[nocc_dn] = nbasis + 1;
    j = 0;
    for (long i = 0; i < maxrank_dn; ++i) {
        fill_det(nocc_dn, &occs[0], &det[0]);
        j = i;
        for (k = 0; k < maxrank_up; ++k) {
            std::memcpy(&dets[j * nword2 + nword], &det[0], sizeof(unsigned long) * nword);
            j += maxrank_dn;
        }
        std::fill(det.begin(), det.end(), PYCI_UINT_ZERO);
        next_colex(&occs[0]);
    }
    for (long i = 0; i < ndet; ++i)
        dict[rank_det(&dets[i * nword2])] = i;
}

void TwoSpinWfn::add_excited_dets(const unsigned long *rdet, const long e_up, const long e_dn) {
    if ((e_up == 0) && (e_dn == 0)) {
        add_det(rdet);
        return;
    }
    OneSpinWfn wfn_up(nbasis, nocc_up, nocc_up);
    wfn_up.add_excited_dets(&rdet[0], e_up);
    OneSpinWfn wfn_dn(nbasis, nocc_dn, nocc_dn);
    wfn_dn.add_excited_dets(&rdet[nword], e_dn);
    std::vector<unsigned long> det(nword2);
    long j;
    for (long i = 0; i < wfn_up.ndet; ++i) {
        std::memcpy(&det[0], wfn_up.det_ptr(i), sizeof(unsigned long) * nword);
        for (j = 0; j < wfn_dn.ndet; ++j) {
            std::memcpy(&det[nword], wfn_dn.det_ptr(j), sizeof(unsigned long) * nword);
            add_det(&det[0]);
        }
    }
}

void TwoSpinWfn::add_dets_from_wfn(const TwoSpinWfn &wfn) {
    for (const auto &keyval : wfn.dict)
        add_det_with_rank(&wfn.dets[keyval.second * nword2], keyval.first);
}

void TwoSpinWfn::reserve(const long n) {
    dets.reserve(n * nword2);
    dict.reserve(n);
}

Array<unsigned long> TwoSpinWfn::py_getitem(const long index) const {
    Array<unsigned long> array(nword);
    copy_det(index, reinterpret_cast<unsigned long *>(array.request().ptr));
    return array;
}

Array<unsigned long> TwoSpinWfn::py_to_det_array(long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<unsigned long> array({end - start, static_cast<long>(2), nword});
    to_det_array(start, end, reinterpret_cast<unsigned long *>(array.request().ptr));
    return array;
}

Array<long> TwoSpinWfn::py_to_occ_array(long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<long> array({end - start, static_cast<long>(2), nocc_up});
    to_occ_array(start, end, reinterpret_cast<long *>(array.request().ptr));
    return array;
}

long TwoSpinWfn::py_index_det(const Array<unsigned long> det) const {
    return index_det(reinterpret_cast<const unsigned long *>(det.request().ptr));
}

unsigned long TwoSpinWfn::py_rank_det(const Array<unsigned long> det) const {
    return rank_det(reinterpret_cast<const unsigned long *>(det.request().ptr));
}

long TwoSpinWfn::py_add_det(const Array<unsigned long> det) {
    return add_det(reinterpret_cast<const unsigned long *>(det.request().ptr));
}

long TwoSpinWfn::py_add_occs(const Array<long> occs) {
    return add_det_from_occs(reinterpret_cast<const long *>(occs.request().ptr));
}

long TwoSpinWfn::py_add_excited_dets(const long exc, const pybind11::object ref) {
    std::vector<unsigned long> v_ref;
    unsigned long *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword2);
        ptr = &v_ref[0];
        fill_hartreefock_det(nocc_up, ptr);
        fill_hartreefock_det(nocc_dn, ptr + nword);
    } else
        ptr = reinterpret_cast<unsigned long *>(ref.cast<Array<unsigned long>>().request().ptr);
    long ndet_old = ndet;
    long maxup = (nocc_up < nvir_up) ? nocc_up : nvir_up;
    long maxdn = (nocc_dn < nvir_dn) ? nocc_dn : nvir_dn;
    long a = (exc < maxup) ? exc : maxup;
    long b = exc - a;
    while ((a >= 0) && (b <= maxdn))
        add_excited_dets(ptr, a--, b++);
    return ndet - ndet_old;
}

} // namespace pyci
