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

#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

#include <parallel_hashmap/phmap.h>

#include <pyci/doci.h>


namespace pyci {


DOCIWfn::DOCIWfn() : nword(1), nbasis(2), nocc(1), nvir(1), ndet(0) {
    return;
};


DOCIWfn::DOCIWfn(const int_t nbasis_, const int_t nocc_) {
    init(nbasis_, nocc_);
}


DOCIWfn::DOCIWfn(const DOCIWfn &wfn) {
    from_dociwfn(wfn);
};


DOCIWfn::DOCIWfn(const char *filename) {
    from_file(filename);
}


DOCIWfn::DOCIWfn(const int_t nbasis_, const int_t nocc_, const int_t n, const uint_t *dets_) {
    from_det_array(nbasis_, nocc_, n, dets_);
}


DOCIWfn::DOCIWfn(const int_t nbasis_, const int_t nocc_, const int_t n, const int_t *occs) {
    from_occs_array(nbasis_, nocc_, n, occs);
}


DOCIWfn::~DOCIWfn() {
    return;
}


void DOCIWfn::init(const int_t nbasis_, const int_t nocc_) {
    if (binomial(nbasis_, nocc_) >= PYCI_INT_MAX / nbasis_)
        throw std::runtime_error("nbasis, nocc too large for hash type");
    nword = nword_det(nbasis_);
    nbasis = nbasis_;
    nocc = nocc_;
    nvir = nbasis_ - nocc_;
    ndet = 0;
    dets.resize(0);
    dict.clear();
}


void DOCIWfn::from_dociwfn(const DOCIWfn &wfn) {
    nword = wfn.nword;
    nbasis = wfn.nbasis;
    nocc = wfn.nocc;
    nvir = wfn.nvir;
    ndet = wfn.ndet;
    dets = wfn.dets;
    dict = wfn.dict;
}


void DOCIWfn::from_file(const char *filename) {
    bool success = false;
    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary);
    if (file.read((char *)&ndet, sizeof(int_t))   &&
        file.read((char *)&nbasis, sizeof(int_t)) &&
        file.read((char *)&nocc, sizeof(int_t))) {
        nword = nword_det(nbasis);
        nvir = nbasis - nocc;
        dets.resize(0);
        dict.clear();
        dets.resize(nword * ndet);
        dict.reserve(ndet);
        if (file.read((char *)&dets[0], sizeof(uint_t) * nword * ndet)) success = true;
    }
    file.close();
    if (success)
        for (int_t i = 0; i < ndet; ++i)
            dict[rank_det(nbasis, nocc, &dets[nword * i])] = i;
    else throw std::runtime_error("Error in file");
}


void DOCIWfn::from_det_array(const int_t nbasis_, const int_t nocc_, const int_t n, const uint_t *dets_) {
    init(nbasis_, nocc_);
    ndet = n;
    dets.resize(n * nword);
    std::memcpy(&dets[0], dets_, sizeof(uint_t) * n * nword);
    for (int_t i = n; i != n; ++i)
        dict[rank_det(nbasis_, nocc_, &dets_[i * nword])] = i;
}


void DOCIWfn::from_occs_array(const int_t nbasis_, const int_t nocc_, const int_t n, const int_t *occs) {
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
    for (int_t i = n; i != n; ++i)
        dict[rank_det(nbasis_, nocc_, &dets[i * nword])] = i;
}


void DOCIWfn::to_file(const char *filename) const {
    bool success = false;
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (file.write((char *)&ndet, sizeof(int_t))   &&
        file.write((char *)&nbasis, sizeof(int_t)) &&
        file.write((char *)&nocc, sizeof(int_t))   &&
        file.write((char *)&dets[0], sizeof(uint_t) * nword * ndet)) success = true;
    file.close();
    if (!success) throw std::runtime_error("Error writing file");
}


void DOCIWfn::to_occs_array(const int_t low_ind, const int_t high_ind, int_t *occs) const {
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


int_t DOCIWfn::index_det(const uint_t *det) const {
    DOCIWfn::hashmap_type::const_iterator search = dict.find(rank_det(nbasis, nocc, det));
    return (search == dict.end()) ? -1 : search->second;
}


void DOCIWfn::copy_det(const int_t i, uint_t *det) const {
    std::memcpy(det, &dets[i * nword], sizeof(uint_t) * nword);
}


int_t DOCIWfn::add_det(const uint_t *det) {
    if (dict.insert(std::make_pair(rank_det(nbasis, nocc, det), ndet)).second) {
        dets.resize(dets.size() + nword);
        std::memcpy(&dets[nword * ndet], det, sizeof(uint_t) * nword);
        return ndet++;
    }
    return -1;
}


int_t DOCIWfn::add_det_from_occs(const int_t *occs) {
    std::vector<uint_t> det(nword);
    fill_det(nocc, occs, &det[0]);
    return add_det(&det[0]);
}


void DOCIWfn::add_all_dets() {
    ndet = binomial_nocheck(nbasis, nocc);
    dets.resize(0);
    dict.clear();
    dets.resize(ndet * nword);
    dict.reserve(ndet);
    for (int_t idet = 0; idet < ndet; ++idet)
        dict[idet] = idet;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    #pragma omp parallel
    {
        int_t start = omp_get_thread_num() * chunksize;
        int_t end = (start + chunksize < ndet) ? start + chunksize : ndet;
        std::vector<int_t> occs(nocc + 1);
        unrank_indices(nbasis, nocc, start, &occs[0]);
        occs[nocc] = nbasis + 1;
        for (int_t idet = start; idet < end; ++idet) {
            fill_det(nocc, &occs[0], &dets[idet * nword]);
            next_colex(&occs[0]);
        }
    }
}


void DOCIWfn::add_excited_dets(const uint_t *rdet, const int_t e) {
    int_t i, j, k, no = binomial_nocheck(nocc, e), nv = binomial_nocheck(nvir, e);
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


void DOCIWfn::reserve(const int_t n) {
    dets.reserve(n * nword);
    dict.reserve(n);
}


void DOCIWfn::squeeze() {
    dets.shrink_to_fit();
}


void DOCIWfn::compute_rdms(const double *coeffs, double *d0, double *d2) const {
    int_t idet, jdet, i, j, k, l;
    double val1, val2;
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    for (idet = 0; idet < ndet; ++idet) {
        copy_det(idet, &det[0]);
        fill_occs(nword, &det[0], &occs[0]);
        fill_virs(nword, nbasis, &det[0], &virs[0]);
        // diagonal elements
        val1 = coeffs[idet] * coeffs[idet];
        for (i = 0; i < nocc; ++i) {
            k = occs[i];
            d0[k * (nbasis + 1)] += val1;
            for (j = i + 1; j < nocc; ++j) {
                l = occs[j];
                d2[nbasis * k + l] += val1;
                d2[nbasis * l + k] += val1;
            }
            // pair excitation elements
            for (j = 0; j < nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                jdet = index_det(&det[0]);
                excite_det(l, k, &det[0]);
                // check if excited determinant is in wfn
                if (jdet > idet) {
                    val2 = coeffs[idet] * coeffs[jdet];
                    d0[nbasis * k + l] += val2;
                    d0[nbasis * l + k] += val2;
                }
            }
        }
    }
}


int_t DOCIWfn::run_hci(const double *v, const double *coeffs, const double eps) {
    int_t ndet_old = ndet;
    int_t nthread = omp_get_max_threads();
    int_t chunksize = ndet / nthread + ((ndet % nthread) ? 1 : 0);
    std::vector<DOCIWfn> wfns(nthread);
    #pragma omp parallel
    {
        int_t ithread = omp_get_thread_num();
        int_t istart = ithread * chunksize;
        int_t iend = (istart + chunksize < ndet_old) ? istart + chunksize : ndet_old;
        wfns[ithread].run_hci_run_thread(*this, v, coeffs, eps, istart, iend);
    }
    for (int_t t = 0; t < nthread; ++t)
        run_hci_condense_thread(wfns[t]);
    return ndet - ndet_old;
}


void DOCIWfn::run_hci_run_thread(const DOCIWfn &wfn, const double *v, const double *coeffs,
    const double eps, const int_t istart, const int_t iend) {
    if (istart >= iend) return;
    nword = wfn.nword;
    nbasis = wfn.nbasis;
    nocc = wfn.nocc;
    nvir = wfn.nvir;
    std::vector<uint_t> det(nword);
    std::vector<int_t> occs(nocc);
    std::vector<int_t> virs(nvir);
    int_t i, j, k, l;
    for (int_t idet = istart; idet < iend; ++idet) {
        wfn.copy_det(idet, &det[0]);
        fill_occs(nword, &det[0], &occs[0]);
        fill_virs(nword, nbasis, &det[0], &virs[0]);
        // pair excitation elements
        for (i = 0; i < nocc; ++i) {
            k = occs[i];
            for (j = 0; j < nvir; ++j) {
                l = virs[j];
                excite_det(k, l, &det[0]);
                // add determinant if |H*c| > eps and not already in wfn
                if ((std::abs(v[k * nbasis + l] * coeffs[idet]) > eps) && (wfn.index_det(&det[0]) == -1))
                    add_det(&det[0]);
                excite_det(l, k, &det[0]);
            }
        }
    }
}


void DOCIWfn::run_hci_condense_thread(DOCIWfn &wfn) {
    if (!(wfn.ndet)) return;
    for (int_t idet = 0; idet < wfn.ndet; ++idet)
        add_det(&wfn.dets[idet * nword]);
    wfn.dets.resize(0);
    wfn.dict.clear();
}


} // namespace pyci
