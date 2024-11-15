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

NonSingletCI::NonSingletCI(const NonSingletCI &wfn) : GenCIWfn(wfn) {
}

NonSingletCI::NonSingletCI(NonSingletCI &&wfn) noexcept : GenCIWfn(wfn) {
}

NonSingletCI::NonSingletCI(const DOCIWfn &wfn) : GenCIWfn(wfn){
}

NonSingletCI::NonSingletCI(const FullCIWfn &wfn) : GenCIWfn(wfn){
}

NonSingletCI::NonSingletCI(const std::string &filename) : GenCIWfn(filename) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd) : GenCIWfn(nb, nu, nd) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const long n, const ulong *ptr)
    : GenCIWfn(nb, nu, nd, n, ptr) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const long n, const long *ptr)
    : GenCIWfn(nb, nu, nd, n, ptr) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const Array<ulong> array)
    : NonSingletCI(nb, nu, nd, array.request().shape[0],
                   reinterpret_cast<const ulong *>(array.request().ptr)) {
}

NonSingletCI::NonSingletCI(const long nb, const long nu, const long nd, const Array<long> array)
    : NonSingletCI(nb, nu, nd, array.request().shape[0],
                   reinterpret_cast<const long *>(array.request().ptr)) {
}

 
void NonSingletCI:add_excited_dets(const ulong *rdet, const long e){
    long i, j, k, no = binomial(nocc_up, e), nv = binomial(nvir_up, e);
    AlignedVector<ulong> det(nword);

    AlignedVector<long> occs(nocc);
    AlignedVector<long> occs_up(nocc_up);
    AlignedVector<long> occs_dn(nocc_dn);
    AlignedVector<long> occs_pairs(nocc_up);

    AlignedVector<long> virs(nvir);
    AlignedVector<long> virs_up(nvir_up);
    AlignedVector<long> virs_dn(nvir_dn);
    AlignedVector<long> virs_pairs(nocc_up);

    AlignedVector<long> occinds(e + 1);
    AlignedVector<long> virinds(e + 1);

}


void NonSingletCI::fill_hartreefock_det(long nb2, long nocc, ulong *det) {
    /* GenCIWfn build using FullCIWfn initializes the OneSpinWfn with nbasis * 2, so we are calling it nb2 here*/
    long i = 0;
    long nb = nb/2;
    long nocc_beta = std::min(nocc, nb);
    long nocc_alpha = std::min(0L, nocc - nb);

    // First, handle beta spins
    while (nocc_beta >= Size<ulong>()){
        det[i++] = Max<ulong>();
        nocc_beta -= Size<ulong>();
    }

    if (nocc_beta) {
        det[i] = (1UL << nocc_beta) -1;
        i++;
    }
     
    // Fill alpha spins (second half)
    while (nocc_alpha >= Size<ulong>()){
        det[i++] = Max<ulong>();
        nocc_alpha -= Size<ulong>();
    }

    if (nocc_alpha) {
        det[i] = (1UL << nocc) - 1;
    }

} //namespace pyci



