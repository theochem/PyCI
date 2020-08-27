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

OneSpinWfn::OneSpinWfn(const OneSpinWfn &wfn) : Wfn(wfn) {
}

OneSpinWfn::OneSpinWfn(OneSpinWfn &&wfn) noexcept : Wfn(wfn) {
}

OneSpinWfn::OneSpinWfn(const std::string &filename) {
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
        nword = nword_det(nb);
        dets.resize(nword * n);
        if (file.read(reinterpret_cast<char *>(&dets[0]), sizeof(ulong) * nword * n))
            failed = false;
    } while (false);
    file.close();
    if (failed)
        throw std::ios_base::failure("error in file");
    Wfn::init(nb, nu, nd);
    ndet = n;
    dict.reserve(n);
    for (long i = 0; i < ndet; ++i)
        dict[rank_det(&dets[nword * i])] = i;
}

OneSpinWfn::OneSpinWfn(const long nb, const long nu, const long nd) : Wfn(nb, nu, nd) {
}

OneSpinWfn::OneSpinWfn(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : OneSpinWfn(nb, nu, nd) {
    ndet = n;
    dets.resize(n * nword);
    std::memcpy(&dets[0], ptr, sizeof(ulong) * n * nword);
    for (long i = 0; i < n; ++i)
        dict[rank_det(ptr + i * nword)] = i;
}

OneSpinWfn::OneSpinWfn(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : OneSpinWfn(nb, nu, nd) {
    long j = 0, k = 0;
    ndet = n;
    dets.resize(n * nword);
    for (long i = 0; i < n; ++i) {
        fill_det(nu, ptr + j, &dets[k]);
        j += nu;
        k += nword;
    }
    for (long i = 0; i < n; ++i)
        dict[rank_det(&dets[i * nword])] = i;
}

OneSpinWfn::OneSpinWfn(const long nb, const long nu, const long nd, const Array<ulong> array)
    : OneSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const ulong *>(array.request().ptr)) {
}

OneSpinWfn::OneSpinWfn(const long nb, const long nu, const long nd, const Array<long> array)
    : OneSpinWfn(nb, nu, nd, array.request().shape[0],
                 reinterpret_cast<const long *>(array.request().ptr)) {
}

const ulong *OneSpinWfn::det_ptr(const long i) const {
    return &dets[i * nword];
}

void OneSpinWfn::to_file(const std::string &filename) const {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    bool success =
        file.write(reinterpret_cast<const char *>(&ndet), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&nbasis), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&nocc_up), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&nocc_dn), sizeof(long)) &&
        file.write(reinterpret_cast<const char *>(&dets[0]), sizeof(ulong) * ndet * nword);
    file.close();
    if (!success)
        throw std::ios_base::failure("error writing file");
}

void OneSpinWfn::to_det_array(const long low, const long high, ulong *ptr) const {
    if (low >= high)
        return;
    std::memcpy(ptr, &dets[low * nword], (high - low) * nword * sizeof(ulong));
}

void OneSpinWfn::to_occ_array(const long low, const long high, long *ptr) const {
    if (low >= high)
        return;
    long j = low * nword, k = 0;
    for (long i = low; i < high; ++i) {
        fill_occs(nword, &dets[j], ptr + k);
        j += nword;
        k += nocc_up;
    }
}

long OneSpinWfn::index_det(const ulong *det) const {
    const auto &search = dict.find(rank_det(det));
    return (search == dict.end()) ? -1 : search->second;
}

long OneSpinWfn::index_det_from_rank(const ulong rank) const {
    const auto &search = dict.find(rank);
    return (search == dict.end()) ? -1 : search->second;
}

void OneSpinWfn::copy_det(const long i, ulong *det) const {
    std::memcpy(det, &dets[i * nword], sizeof(ulong) * nword);
}

ulong OneSpinWfn::rank_det(const ulong *det) const {
    return SpookyHash::Hash64((void *)det, sizeof(ulong) * nword, PYCI_SPOOKYHASH_SEED);
}

long OneSpinWfn::add_det(const ulong *det) {
    if (dict.insert(std::make_pair(rank_det(det), ndet)).second) {
        dets.resize(dets.size() + nword);
        std::memcpy(&dets[nword * ndet], det, sizeof(ulong) * nword);
        return ndet++;
    }
    return -1;
}

long OneSpinWfn::add_det_with_rank(const ulong *det, const ulong rank) {
    if (dict.insert(std::make_pair(rank, ndet)).second) {
        dets.resize(dets.size() + nword);
        std::memcpy(&dets[nword * ndet], det, sizeof(ulong) * nword);
        return ndet++;
    }
    return -1;
}

long OneSpinWfn::add_det_from_occs(const long *occs) {
    AlignedVector<ulong> det(nword);
    fill_det(nocc_up, occs, &det[0]);
    return add_det(&det[0]);
}

void OneSpinWfn::add_hartreefock_det(void) {
    AlignedVector<ulong> det(nword);
    fill_hartreefock_det(nocc_up, &det[0]);
    add_det(&det[0]);
}

void OneSpinWfn::add_all_dets(void) {
    if (maxrank_up == Max<long>())
        throw std::domain_error("cannot generate > 2 ** 63 determinants");
    ndet = maxrank_up;
    std::fill(dets.begin(), dets.end(), 0UL);
    dets.resize(ndet * nword);
    dict.clear();
    dict.reserve(ndet);
    AlignedVector<long> occs(nocc_up + 1);
    unrank_colex(nbasis, nocc_up, 0, &occs[0]);
    occs[nocc_up] = nbasis + 1;
    for (long i = 0; i < ndet; ++i) {
        fill_det(nocc_up, &occs[0], &dets[i * nword]);
        next_colex(&occs[0]);
    }
    for (long i = 0; i < ndet; ++i)
        dict[rank_det(&dets[i * nword])] = i;
}

void OneSpinWfn::add_excited_dets(const ulong *rdet, const long e) {
    long i, j, k, no = binomial(nocc_up, e), nv = binomial(nvir_up, e);
    AlignedVector<ulong> det(nword);
    AlignedVector<long> occs(nocc_up);
    AlignedVector<long> virs(nvir_up);
    AlignedVector<long> occinds(e + 1);
    AlignedVector<long> virinds(e + 1);
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
            std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
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

void OneSpinWfn::reserve(const long n) {
    dets.reserve(n * nword);
    dict.reserve(n);
}

Array<ulong> OneSpinWfn::py_getitem(const long index) const {
    Array<ulong> array(nword);
    copy_det(index, reinterpret_cast<ulong *>(array.request().ptr));
    return array;
}

Array<ulong> OneSpinWfn::py_to_det_array(long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<ulong> array({end - start, nword});
    to_det_array(start, end, reinterpret_cast<ulong *>(array.request().ptr));
    return array;
}

Array<long> OneSpinWfn::py_to_occ_array(long start, long end) const {
    if (start == -1) {
        start = 0;
        if (end == -1)
            end = ndet;
    } else if (end == -1) {
        end = start;
        start = 0;
    }
    Array<long> array({end - start, nocc_up});
    to_occ_array(start, end, reinterpret_cast<long *>(array.request().ptr));
    return array;
}

long OneSpinWfn::py_index_det(const Array<ulong> det) const {
    return index_det(reinterpret_cast<const ulong *>(det.request().ptr));
}

ulong OneSpinWfn::py_rank_det(const Array<ulong> det) const {
    return rank_det(reinterpret_cast<const ulong *>(det.request().ptr));
}

long OneSpinWfn::py_add_det(const Array<ulong> det) {
    return add_det(reinterpret_cast<const ulong *>(det.request().ptr));
}

long OneSpinWfn::py_add_occs(const Array<long> occs) {
    return add_det_from_occs(reinterpret_cast<const long *>(occs.request().ptr));
}

long OneSpinWfn::py_add_excited_dets(const long exc, const pybind11::object ref) {
    AlignedVector<ulong> v_ref;
    ulong *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword);
        ptr = &v_ref[0];
        fill_hartreefock_det(nocc_up, ptr);
    } else
        ptr = reinterpret_cast<ulong *>(ref.cast<Array<ulong>>().request().ptr);
    long ndet_old = ndet;
    add_excited_dets(ptr, exc);
    return ndet - ndet_old;
}

} // namespace pyci
