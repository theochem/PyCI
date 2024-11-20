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
    //long i, j, k, no = binomial(nocc_up, e), nv = binomial(nvir_up, e);
    AlignedVector<ulong> det(nword);

    AlignedVector<long> occs(nocc);
    AlignedVector<long> occs_up(nocc_up);
    AlignedVector<long> occs_dn(nocc_dn);
    AlignedVector<std::pair<int,int>> occs_pairs;

    AlignedVector<long> virs(nvir);
    AlignedVector<long> virs_up(nvir_up);
    AlignedVector<long> virs_dn(nvir_dn);
    AlignedVector<std::pair<int,int>> virs_pairs;

    AlignedVector<long> occinds(e + 1);
    AlignedVector<long> virinds(e + 1);
    fill_occs(nword, rdet, &occs[0]);
    fill_virs(nword, nbasis, rdet, &virs[0]);
     
    

    // Generate list of up & dn indices for ocs & virs
    for (int i : occs) {
        if (i < nbasis) occ_up.push_back(i);
        else occ_dn.push_back(i);
    }
    for (int a : virs) {
        if (a < nbasis) virs_up.push_back(i);
        else virs_dn.push_back(i);
    }
    // Create an unordered set for fast lookup of occupied down-orbitals
    std::unordered_set<int> occ_dn_set(occ_dn.begin(), occ_dn.end()); 
    // Generate occ_pairs
    for (int i : occ_up) {
        if (occ_dn_set.find(i + nbasis) != occ_dn_set.end()) {
            occ_pairs.push_back({i, i + nbasis});   
        }
    }

    // Create an unordered set for fast looup of virtual orbitals
    std::unordered_set<int> virs_set(virs.begin(), virs.end());
    // form virtual pairs
     for (int a : virs) {
         if (virs_set.find(a + nbasis) != virs_set.end()) {
             virs_pairs.push_back({a, a + nbasis});   
         }
     }

     // Handle excitation order 1
     if (n == 1) {
         for (long occ : occ_up) {
             for (long vir : vir_up) {
                 std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                 excite_det(occ, vir, &det[0])
                 add_det(&det[0]) 
             }
         }
         for (long occ : occ_dn) {
             for (long vir : vir_up) {
                 std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                 excite_det(occ, vir, &det[0])
                 add_det(&det[0]) 
             }
         }
         return ;
     }



    //// Handle excitation order 2 
    //if (e == 2){
    //    for (const auto& occ_pair : occ_pairs){
    //        for (const auto& vir_pair : vir_pairs){
    //            long o_up = occ_pair.first;
    //            long o_dn = occ_pair.second;
    //            long v_up = vir_pair.first;
    //            long v_dn = vir_pair.second;
    //         
    //            std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
    //            excite_det(o_up, v_up, &det[0])
    //            excite_det(o_dn, v_dn, &det[0])
    //            add_det(&det[0]) 
    //           
    //        }
    //    }
    //}
    
    // Handle excitation orders >= 2
    // Loop over possible values of d (number of pair excitations)
    for (long d = 0; d <= std::min(e/2, static_cast<long>(occ_pairs.size())); ++d){
        // Number of single excitations
        long num_singles = e - 2 * d;
        
        // Apply d pair excitations 
        if (d > 0) {
            for (long i = 0; i < d; ++i) {
                const auto& occ_pair = occ_pairs[i];
                const auto& vir_pair = vir_pairs[i];

                long o_up = occ_pair.first;
                long o_dn = occ_pair.second;
                long v_up = vir_pair.first;
                long v_dn = vir_pair.second;

                std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                     
                excite_det(o_up, v_up, &det[0])
                excite_det(o_dn, v_dn, &det[0])
                add_det(&det[0])
            } 
        }
            // Apply num_singles single excitations
        if (num_singles > 0) {
            for (long i = 0; i < num_singles; ++i){
                long occ = (i % 2 == 0) ? occ_up[i % occ_up.size()] : occ_dn[i % occ_dn.size()];
                for (long vir : virs) {
                    std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                    excite_det(occ, vir, det);
                }
            }
        }
    }
    
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


void NonSingletCI::py_add_excited_dets(const long exc, const pybind11::object ref) {
    AlignedVector<ulong> v_ref;
    ulong *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword)
        ptr = &v_ref[0]
        fill_hartreefock_det(nbasis,nocc, ptr);
    } else
        ptr = reinterpret_cast<ulong *>(ref.cast<Array<ulong>>().request().ptr);
    long ndet_old = ndet;
    add_excited_dets(ptr, exc);
    return ndet - ndet_old;
}
} //namespace pyci



