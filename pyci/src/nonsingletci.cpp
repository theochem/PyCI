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
#include <iostream>

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


void NonSingletCI::add_excited_dets(const ulong *rdet, const long e){
    std::cout << "Inside nonsingletci/add_excited_dets" << std::endl;
    //long i, j, k, no = binomial(nocc_up, e), nv = binomial(nvirs_up, e);
    AlignedVector<ulong> det(nword);

    AlignedVector<long> occs(nocc);
    AlignedVector<long> occs_up(occs.size());
    AlignedVector<long> occs_dn(occs.size());
    AlignedVector<std::pair<int,int>> occ_pairs;

    AlignedVector<long> virs(nbasis - nocc);
    AlignedVector<long> virs_up(virs.size());
    AlignedVector<long> virs_dn(virs.size());
    AlignedVector<std::pair<int,int>> vir_pairs;

    AlignedVector<long> occinds(e + 1);
    AlignedVector<long> virinds(e + 1);
    fill_occs(nword, rdet, &occs[0]);
    fill_virs(nword, nbasis, rdet, &virs[0]);

    int up_idx = 0, dn_idx = 0;
    std::cout << "nocc_up: " << nocc_up << ", nvir_up: " << nvir_up << std::endl;
    std::cout << "nocc: " << nocc << ", nvir: " << nvir << std::endl;
    std::cout << "e: " << e << std::endl;
    std::cout << "rdet: " << *rdet << std::endl;
    std::cout << "occs: "; 
    for (const auto& elem : occs) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    std::cout << "virs: ";
    for (const auto& elem : virs) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;   

    // Generate list of up & dn indices for ocs & virs
    for (int i : occs) {
        if (i < nbasis/2) occs_up[up_idx++] = i;
        else occs_dn[dn_idx++] = i;
    }
    
    // Resize vectors to actual size
    occs_up.resize(up_idx);
    occs_dn.resize(dn_idx);

    std::cout << "occs_up: ";
    for (const auto& elem : occs_up) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    std::cout << "occs_dn: ";
    for (const auto& elem : occs_dn) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    
    up_idx = 0;
    dn_idx = 0;
    for (int a : virs) {
        if (a < nbasis/2) virs_up[up_idx++] = a;
        else virs_dn[dn_idx++] = a;
    }
    // Resize vectors to actual size
    virs_up.resize(up_idx);
    virs_dn.resize(dn_idx);

    std::cout << "virs_up: ";
    for (const auto& elem : virs_up) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    std::cout << "virs_dn: ";
    for (const auto& elem : virs_dn) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    // Create an unordered set for fast lookup of occupied down-orbitals
    std::unordered_set<int> occ_dn_set(occs_dn.begin(), occs_dn.end()); 
    // Generate occ_pairs
    for (int i : occs_up) {
        if (occ_dn_set.find(i + nbasis/2) != occ_dn_set.end()) {
            occ_pairs.push_back({i, i + nbasis/2});   
        }
    }
    std::cout << "occ_pairs: ";
    for (const auto& elem : occ_pairs) {
        std::cout << elem.first << " " << elem.second << std::endl;
    }
    // Create an unordered set for fast looup of virtual orbitals
    std::unordered_set<int> virs_set(virs.begin(), virs.end());
    // form virtual pairs
    for (int a : virs) {
        if (virs_set.find(a + nbasis/2) != virs_set.end()) {
            vir_pairs.push_back({a, a + nbasis/2});   
        }
    }
    std::cout << "vir_pairs: ";
    for (const auto& elem : vir_pairs) {
        std::cout << elem.first << " " << elem.second << std::endl;
    }
    
     // Handle excitation order 1
    if (e == 1) {
        std::cout << "Handling excitation order 1" << std::endl;
        std::cout << "Determinants of excitation order 1" << std::endl;
        for (long occ : occs) {
            for (long vir : virs) {
                std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                excite_det(occ, vir, &det[0]);
                add_det(&det[0]); 
                // Print determinant after excite_det loop
                
                for (int k = 0; k < nword; ++k) {
                    std::cout << det[k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl; 
        return ;
    }

    // Handle excitation orders >= 2
    // Iterate over possible (d,s) pairs: d pair excitations and s single excitations
    std::cout << "Handling excitation order >= 2" << std::endl;
    for (long d = 0; d <= std::min(e/2, static_cast<long>(occ_pairs.size())); ++d){
        long s = e - 2 * d;
        std::cout << "d: " << d << ", num_singles" << s << std::endl;
        if (s > static_cast<long>(occ_pairs.size()) - d) continue; // Not enough pairs for singles
        
        // Generate all combinations of d pairs
        std::vector<std::vector<long>> pair_combinations;
        std::vector<long> pair_indices(occ_pairs.size());
        std::iota(pair_indices.begin(), pair_indices.end(), 0);
        
        do {
            std::vector<long> combination(pair_indices.begin(), pair_indices.begin() + d);
            pair_combinations.push_back(combination);
        } while (std::next_permutation(pair_indices.begin(), pair_indices.end()));
        
        // Process each combination of d pairs
        for (const auto& pair_comb: pair_combinations) {
            std::vector<std::pair<int,int>> used_occ_pairs;
            std::vector<long> used_virtuals;

            // Apply pair excitations
            std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
            for (long idx : pair_comb) {
                const auto& occ_pair = occ_pairs[idx];
                const auto& vir_pair = vir_pairs[idx];

                excite_det(occ_pair.first, vir_pair.first, &det[0]);
                excite_det(occ_pair.second, vir_pair.second, &det[0]);

                used_occ_pairs.push_back(occ_pair);
                used_virtuals.push_back(vir_pair.first);
                used_virtuals.push_back(vir_pair.second);
            }

            //Exclude used pairs and virtual orbitals
            std::vector<long> remaining_occ_indices;
            for (std::vector<std::pair<int, int>>::size_type i = 0; i < occ_pairs.size(); ++i) {
                if (std::find(pair_comb.begin(), pair_comb.end(), i) == pair_comb.end()) {
                    remaining_occ_indices.push_back(static_cast<int>(i));
                }
            }

            // Generate all combinations of s singles
            std::vector<std::vector<long>> single_combinations;
            std::vector<long> single_indices(remaining_occ_indices.size());
            std::iota(single_indices.begin(), single_indices.end(), 0);

            do {
                std::vector<long> combination(single_indices.begin(), single_indices.begin() + s);
                single_combinations.push_back(combination);
            } while (std::next_permutation(single_indices.begin(), single_indices.end()));

            // Process each combination of s singles
            for (const auto& single_comb : single_combinations) {
                std::memcpy(&det[0], rdet, sizeof(ulong) * nword);

                // Apply single excitations
                for (long idx : single_comb) {
                    long occ_idx = remaining_occ_indices[idx];
                    long occ = occ_pairs[occ_idx].first; // Use the first of the remaining occ_pair
                    for (long vir: virs) {
                        if (std::find(used_virtuals.begin(), used_virtuals.end(), vir) != used_virtuals.end()) continue; 
                        excite_det(occ, vir, &det[0]);
                    }
                }
                add_det(&det[0]);
            }
        }


    }    
}
 


void NonSingletCI::fill_hartreefock_det(long nb2, long nocc, ulong *det) {
   /* GenCIWfn build using FullCIWfn initializes the OneSpinWfn with nbasis * 2, so we are calling it nb2 here*/
    long nb = nb2/2;
    // FIXME: The code is assuming nocc is even
    long nocc_beta = nocc/2; //std::min(nocc, nb);
    long nocc_alpha = nocc/2; //std::min(0L, nocc - nb);
    long num_ulongs = (nb2 + Size<ulong>() - 1) / Size<ulong>();
    

    std::cout << "Inside nonsingletci/fill_hartreefock_det" << std::endl;
    std::cout << "nb: " << nb << std::endl;
    std::cout << "nocc: " << nocc << std::endl;
    std::cout << "nocc_beta: " << nocc_beta << std::endl;
    std::cout << "nocc_alpha: " << nocc_alpha << std::endl;

    for (long i = 0; i < nocc_beta; ++i) {
        long bit_index = nb - nocc_beta + i;
        det[bit_index / Size<ulong>()] |= 1UL << (bit_index % Size<ulong>());
    }
    
    for (long i = 0; i < nocc_alpha; ++i) {
        long bit_index = i;
        det[bit_index / Size<ulong>()] |= 1UL << (bit_index % Size<ulong>());
    }
    std::cout << "det: ";
    for (int i = 0; i < num_ulongs; ++i) {
        std::cout << det[i] << " ";
    }
    std::cout << std::endl;

}

long NonSingletCI::py_add_excited_dets(const long exc, const pybind11::object ref) {
    AlignedVector<ulong> v_ref;
    ulong *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword);
        ptr = &v_ref[0];
        fill_hartreefock_det(nbasis,nocc, ptr);
    } else
        ptr = reinterpret_cast<ulong *>(ref.cast<Array<ulong>>().request().ptr);
    long ndet_old = ndet;
   add_excited_dets(ptr, exc);
    return ndet - ndet_old;
}
} //namespace pyci



