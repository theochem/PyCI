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
        if (file.read(reinterpret_cast<char *>(&dets[0]), sizeof(ulong) * nword2 * n))
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

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : TwoSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword2);
    std::memcpy(&dets[0], ptr, sizeof(ulong) * n * nword2);
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

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd, const Array<ulong> array)
    : TwoSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const ulong *>(array.request().ptr)) {
}

TwoSpinWfn::TwoSpinWfn(const long nb, const long nu, const long nd, const Array<long> array)
    : TwoSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const long *>(array.request().ptr)) {
}

const ulong *TwoSpinWfn::det_ptr(const long i) const {
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
        file.write(reinterpret_cast<const char *>(&dets[0]), sizeof(ulong) * nword2 * ndet);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}

void TwoSpinWfn::to_det_array(const long low, const long high, ulong *ptr) const {
    if (low >= high)
        return;
    std::memcpy(ptr, &dets[low * nword], (high - low) * nword2 * sizeof(ulong));
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

long TwoSpinWfn::index_det(const ulong *det) const {
    const auto &search = dict.find(rank_det(det));
    return (search == dict.end()) ? -1 : search->second;
}

long TwoSpinWfn::index_det_from_rank(const Hash rank) const {
    const auto &search = dict.find(rank);
    return (search == dict.end()) ? -1 : search->second;
}

void TwoSpinWfn::copy_det(const long i, ulong *det) const {
    std::memcpy(det, &dets[i * nword2], sizeof(ulong) * nword2);
}

Hash TwoSpinWfn::rank_det(const ulong *det) const {
    return spookyhash(nword2, det);
}

long TwoSpinWfn::add_det(const ulong *det) {
    if (dict.insert(std::make_pair(rank_det(det), ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(ulong) * nword2);
        return ndet++;
    }
    return -1;
}

long TwoSpinWfn::add_det_with_rank(const ulong *det, const Hash rank) {
    if (dict.insert(std::make_pair(rank, ndet)).second) {
        dets.resize(dets.size() + nword2);
        std::memcpy(&dets[nword2 * ndet], det, sizeof(ulong) * nword2);
        return ndet++;
    }
    return -1;
}

long TwoSpinWfn::add_det_from_occs(const long *occs) {
    AlignedVector<ulong> det(nword2);
    fill_det(nocc_up, &occs[0], &det[0]);
    fill_det(nocc_dn, &occs[nocc_up], &det[nword]);
    return add_det(&det[0]);
}

void TwoSpinWfn::add_hartreefock_det(void) {
    AlignedVector<ulong> det(nword2);
    fill_hartreefock_det(nocc_up, &det[0]);
    fill_hartreefock_det(nocc_dn, &det[nword]);
    add_det(&det[0]);
}

namespace {

void twospinwfn_add_all_dets_thread(const long nword, const long nbasis, const long nocc_up,
                                    const long nocc_dn, const long maxrank_up,
                                    const long maxrank_dn, ulong *dets, const long ithread,
                                    const long nthread) {
    AlignedVector<long> v_occs(nocc_up + 1);
    AlignedVector<ulong> v_det(nword);
    long *occs = &v_occs[0];
    ulong *det = &v_det[0];
    long nword2 = nword * 2;
    long start = end_chunk_idx(ithread, nthread, maxrank_up);
    long end = std::min(end_chunk_idx(ithread + 1, nthread, maxrank_up), maxrank_up);
    long j, k;
    unrank_colex(nbasis, nocc_up, start, occs);
    occs[nocc_up] = nbasis + 1;
    k = start * maxrank_dn * nword2;
    for (long i = start; i < end; ++i) {
        fill_det(nocc_up, occs, det);
        for (j = 0; j < maxrank_dn; ++j) {
            std::memcpy(dets + k, det, sizeof(ulong) * nword);
            k += nword2;
        }
        std::fill(v_det.begin(), v_det.end(), 0UL);
        next_colex(occs);
    }
    start = end_chunk_idx(ithread, nthread, maxrank_dn);
    end = std::min(end_chunk_idx(ithread + 1, nthread, maxrank_dn), maxrank_dn);
    unrank_colex(nbasis, nocc_dn, start, occs);
    occs[nocc_dn] = nbasis + 1;
    for (long i = start; i < end; ++i) {
        fill_det(nocc_dn, occs, det);
        k = i * nword2 + nword;
        for (j = 0; j < maxrank_up; ++j) {
            std::memcpy(dets + k, det, sizeof(ulong) * nword);
            k += maxrank_dn * nword2;
        }
        std::fill(v_det.begin(), v_det.end(), 0UL);
        next_colex(occs);
    }
}

} // namespace

void TwoSpinWfn::add_all_dets(long nthread) {
    if (nthread == -1)
        nthread = get_num_threads();
    ndet = maxrank_up * maxrank_dn;
    long chunksize = ndet / nthread + static_cast<bool>(ndet % nthread);
    while (nthread > 1 && chunksize < PYCI_CHUNKSIZE_MIN) {
        nthread /= 2;
        chunksize = ndet / nthread + static_cast<bool>(ndet % nthread);
    }
    std::fill(dets.begin(), dets.end(), 0UL);
    dets.resize(ndet * nword2);
    dict.clear();
    dict.reserve(ndet);
    Vector<std::thread> v_threads;
    v_threads.reserve(nthread);
    for (long i = 0; i < nthread; ++i)
        v_threads.emplace_back(twospinwfn_add_all_dets_thread, nword, nbasis, nocc_up, nocc_dn,
                               maxrank_up, maxrank_dn, &dets[0], i, nthread);
    for (auto &thread : v_threads)
        thread.join();
    for (long i = 0; i < ndet; ++i)
        dict[rank_det(&dets[i * nword2])] = i;
}

void TwoSpinWfn::add_excited_dets(const ulong *rdet, const long e_up, const long e_dn) {
    if ((e_up == 0) && (e_dn == 0)) {
        add_det(rdet);
        return;
    }
    OneSpinWfn wfn_up(nbasis, nocc_up, nocc_up);
    wfn_up.add_excited_dets(&rdet[0], e_up);
    OneSpinWfn wfn_dn(nbasis, nocc_dn, nocc_dn);
    wfn_dn.add_excited_dets(&rdet[nword], e_dn);
    AlignedVector<ulong> det(nword2);
    long j;
    for (long i = 0; i < wfn_up.ndet; ++i) {
        std::memcpy(&det[0], wfn_up.det_ptr(i), sizeof(ulong) * nword);
        for (j = 0; j < wfn_dn.ndet; ++j) {
            std::memcpy(&det[nword], wfn_dn.det_ptr(j), sizeof(ulong) * nword);
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

Array<ulong> TwoSpinWfn::py_getitem(const long index) const {
    return Array<const ulong>({2L, nword}, {nword * sizeof(ulong), sizeof(ulong)}, det_ptr(index));
}

Array<ulong> TwoSpinWfn::py_to_det_array(long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<ulong> array({end - start, static_cast<long>(2), nword});
    to_det_array(start, end, reinterpret_cast<ulong *>(array.request().ptr));
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

long TwoSpinWfn::py_index_det(const Array<ulong> det) const {
    return index_det(reinterpret_cast<const ulong *>(det.request().ptr));
}

Hash TwoSpinWfn::py_rank_det(const Array<ulong> det) const {
    return rank_det(reinterpret_cast<const ulong *>(det.request().ptr));
}

long TwoSpinWfn::py_add_det(const Array<ulong> det) {
    return add_det(reinterpret_cast<const ulong *>(det.request().ptr));
}

long TwoSpinWfn::py_add_occs(const Array<long> occs) {
    return add_det_from_occs(reinterpret_cast<const long *>(occs.request().ptr));
}

long TwoSpinWfn::py_add_excited_dets(const long exc, const pybind11::object ref) {
    AlignedVector<ulong> v_ref;
    ulong *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword2);
        ptr = &v_ref[0];
        fill_hartreefock_det(nocc_up, ptr);
        fill_hartreefock_det(nocc_dn, ptr + nword);
    } else
        ptr = reinterpret_cast<ulong *>(ref.cast<Array<ulong>>().request().ptr);
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
