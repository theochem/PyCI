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


// Function to generate cartesian product from a vector of pairs
// In our case: this function is used to generate all possible combinations of
// occupied orbitals to excite from for singles given pair of occupied orbitals in ref det
std::vector<std::vector<long>> NonSingletCI::generate_cartesian_product(
    const std::vector<std::pair<int,int>>& pairs, long k) {
    std::vector<std::vector<long>> result;
    std::cout << "Inside nonsingletci/generate_cartesian_product" << std::endl;
    std::cout <<   "pairs.size(): " << pairs.size() << std::endl;
    std::cout << "k: " << k << std::endl;

    long nocc_pairs = pairs.size();
    if (k > nocc_pairs) return result;
    std::vector<std::vector<long>> temp_combinations = {{}};

    
    for (const auto& pair: pairs) {
        std::vector<std::vector<long>> new_result;

        // For each combination in the current result, 
        // extend it with elements of the pair
        for (const auto& combination: temp_combinations) {
            if (combination.size() < k){
                for (auto elem : {pair.first, pair.second}) {
                    auto new_combination = combination;
                    new_combination.push_back(elem);
                    new_result.push_back(new_combination);
                }
            }
        }
        // Move the new combination into the result
        temp_combinations = std::move(new_result);  
        std::cout << "temp_combinations: ";
    }
    // Filter out combinations that are not of size k
    for (const auto& combination: temp_combinations) {
        if (combination.size() == k) {
            result.push_back(combination);
        }
    
    }
    // Debug output to ensure combinations are being generated
    std::cout << "Generated " << result.size() << " combinations" << std::endl;
    
    return result;
}


// Function to generate combinations, based on indices
std::vector<std::vector<long>> NonSingletCI::generate_combinations(long n, long k) {
    std::vector<std::vector<long>> combinations;

    if (k > n) return combinations;

    std::vector<long> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<bool> mask(n, false);
    std::fill(mask.begin(), mask.begin() + k, true);

    do {
        std::vector<long> combination;
        for (long i = 0; i < n; ++i) {
            if (mask[i]) combination.push_back(indices[i]);
        }
        combinations.push_back(combination);
    } while (std::prev_permutation(mask.begin(), mask.end()));
    
    return combinations;
}

void NonSingletCI::add_excited_dets(const ulong *rdet, const long e){
    std::cout << "Inside nonsingletci/add_excited_dets" << std::endl;
    //long i, j, k, no = binomial(nocc_up, e), nv = binomial(nvirs_up, e);
    AlignedVector<ulong> det(nword);

    AlignedVector<long> occs(nocc);
    AlignedVector<long> virs(nbasis - nocc);
    AlignedVector<long> occs_up, occs_dn, virs_up, virs_dn;
    occs_up.reserve(occs.size());
    occs_dn.reserve(occs.size());
    virs_up.reserve(occs.size());
    virs_dn.reserve(occs.size());

    AlignedVector<std::pair<int,int>> occ_pairs;
    AlignedVector<std::pair<int,int>> vir_pairs;

    AlignedVector<long> occinds(e + 1);
    AlignedVector<long> virinds(e + 1);
    fill_occs(nword, rdet, &occs[0]);
    fill_virs(nword, nbasis, rdet, &virs[0]);

    for (int i : occs) {
        (i < nbasis / 2 ? occs_up : occs_dn).push_back(i);
    }

    for (int a : virs) {
        (a < nbasis / 2 ? virs_up : virs_dn).push_back(a);
    }

    // Efficient resizing
    occs_up.shrink_to_fit();
    occs_dn.shrink_to_fit();
    virs_up.shrink_to_fit();
    virs_dn.shrink_to_fit();

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
    

    // Handle excitation order 0
    if (e == 0) {
        std::cout << "Handling excitation order 0" << std::endl;
        std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
        add_det(&det[0]);
        return ;
    }
    // Handle excitation order 1
    if (e == 1) {
        std::cout << "-----Handling excitation order 1-----" << std::endl;
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
    long nocc_pairs = occ_pairs.size();
    long nvir_pairs = vir_pairs.size();
    std::cout << "nocc_pairs: " << nocc_pairs << std::endl;
    if (e >= 2) {
        // Iterate over possible (d,s) pairs: d pair excitations and s single excitations
        std::cout << "--------Handling excitation order >= 2--------" << std::endl;
        
        for (long d = 0; d <= std::min(e/2, nocc_pairs); ++d){
            long num_singles = e - 2 * d;

            std::cout << "d: " << d << ", num_singles: " << num_singles << std::endl;
            
            // Not enough pairs for singles
            if (num_singles > (nocc_pairs - d)) {
                std::cout << "Not enough pairs for singles" << std::endl;
                continue; 
            }

            // Generate occ pair combinations
            std::vector<std::vector<std::vector<long>>> opair_combinations;
            opair_combinations.push_back(generate_combinations(nocc_pairs, d));
            std::cout << "Generated opair_combinations: " << std::endl;
            for (const auto& opair_comb : opair_combinations[0]) {
                for (const auto& idx : opair_comb) {
                    std::cout << "(" << occ_pairs[idx].first << ", " << occ_pairs[idx].second << ") ";
                }                
            }
            std::cout << std::endl;

            std::vector<std::vector<std::vector<long>>> vpair_combinations;
            vpair_combinations.push_back(generate_combinations(nvir_pairs, d));
            std::cout << "Generated vpair_combinations: " << std::endl;
            for (const auto& pair_comb : vpair_combinations[0]) {
                for (const auto& idx : pair_comb) {
                    std::cout << "(" << vir_pairs[idx].first << ", " << vir_pairs[idx].second << ") ";
                }                
            }
            std::cout << std::endl;

            // Process pair combinations for current d
            for (const auto& opair_comb : opair_combinations[0]) {
                for (const auto& vpair_comb : vpair_combinations[0]) {
                    std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                    
                    
                    std::vector<long> used_virs;
                    for (long idx = 0; idx < d; ++idx) {
                        const auto& occ_pair = occ_pairs[opair_comb[idx]];
                        const auto& vir_pair = vir_pairs[vpair_comb[idx]];

                        excite_det(occ_pair.first, vir_pair.first, &det[0]);
                        excite_det(occ_pair.second, vir_pair.second, &det[0]);

                        used_virs.push_back(vir_pair.first);
                        used_virs.push_back(vir_pair.second);
                        
                        std::cout << "Determinant after pair excitation of" << std::endl;
                        std::cout << "occ_pair: " << occ_pair.first << " " << occ_pair.second << std::endl;
                        std::cout << "vir_pair: " << vir_pair.first << " " << vir_pair.second << std::endl;
                        for (int k = 0; k < nword; ++k) {
                            std::cout << det[k] << " ";
                        }
                        std::cout << std::endl;
                    }

                    // // Determine remaining occupied indices
                    // std::vector<long> remaining_occ_indices;
                    // for (long i = 0; i < nocc_pairs; ++i) {
                    //     if (std::find(opair_comb.begin(), opair_comb.end(), i) == opair_comb.end()) {
                    //         remaining_occ_indices.push_back(i);
                    //     }
                    // }
                    // std::cout << "Remaining occ_indices: ";
                    // for (const auto& idx : remaining_occ_indices) {
                    //     std::cout << "(" << occ_pairs[idx].first << ", " << occ_pairs[idx].second << ") ";
                    // }
                    // std::cout << std::endl;
                    if (num_singles > 0) {
                        // Determine remaining occupied pairs
                        std::vector<std::pair<int,int>> remaining_occ_pairs;
                        for (long i = 0; i < nocc_pairs; ++i) {
                            if (std::find(opair_comb.begin(), opair_comb.end(), i) == opair_comb.end()) {
                                remaining_occ_pairs.push_back(occ_pairs[i]);
                            }
                        }

                        //Print remaining occ_pairs for debugging
                        std::cout << "Remaining occ_pairs: ";
                        for (const auto& occ_pair : remaining_occ_pairs) {
                            std::cout << "(" << occ_pair.first << ", " << occ_pair.second << ") ";
                        }
                        std::cout << std::endl;

                        std::vector<long> remaining_virs;
                        for (long i = 0; i < virs.size(); ++i) {
                            if (std::find(used_virs.begin(), used_virs.end(), virs[i]) == used_virs.end()) {
                                remaining_virs.push_back(virs[i]);
                            }
                        }
                        std::cout << "Remaining virs: ";
                        for (const auto& vir : remaining_virs) {
                            std::cout << vir << " ";
                        }
                        std::cout << std::endl;

                        // Process single combinations for current num_singles
                        auto occ_combinations = generate_cartesian_product(remaining_occ_pairs, num_singles);
                        auto vir_combinations = generate_combinations(remaining_virs.size(), num_singles);
                        
                        // Print occ_combinations and vir_combinations for debugging
                        // std::cout << "Available occ_pairs for singles: ";
                        // for (const auto& idx : remaining_occ_indices) {
                        //     std::cout << "(" << occ_pairs[idx].first << ", " << occ_pairs[idx].second << ") ";
                        // }
                        // std::cout << std::endl;

                        std::cout << "Available virs for singles: ";
                        for (const auto& vir : remaining_virs) {
                            std::cout << vir << " ";
                        }
                        std::cout << std::endl;

                        // std::cout << "Generated occ_combinations: " << std::endl;
                        // for (const auto& occ_comb : occ_combinations) {
                        //     std::cout << "(";
                        //     for (const auto& occ_idx : occ_comb) {
                        //         std::cout << occ_pairs[remaining_occ_indices[occ_idx]].first << " ";
                        //     }
                        //     std::cout << ")" << std::endl;
                        // }

                        std::cout << "Generated occ_combinations: " << std::endl;
                        for (const auto& occ_comb : occ_combinations) {
                            for (const auto& elem: occ_comb) {
                                std::cout << "(" << elem << ") ";
                            }
                            std::cout << std::endl;
                        }
                        
                        std::cout << "Generated vir_combinations: " << std::endl;
                        for (const auto& vir_comb : vir_combinations) {
                            std::cout << "(";
                            for (const auto& vir_idx : vir_comb) {
                                std::cout << remaining_virs[vir_idx] << " ";
                            }
                            std::cout << ")" << std::endl;
                        }

                        
                        for (const auto& occ_comb : occ_combinations) {
                            for (const auto& vir_comb : vir_combinations) {
                                // Do NOT reset det here; use the alredy existed det from pair excitations
                                AlignedVector<ulong> temp_det(nword);
                                std::memcpy(&temp_det[0], det.data(), sizeof(ulong) * nword);
                                
                                // Apply single excitations
                                for (long idx = 0; idx < num_singles; ++idx){
                                    long occ = occ_comb[idx];
                                    long vir = remaining_virs[vir_comb[idx]];
                                    excite_det(occ, vir, &temp_det[0]);
                                    std::cout << "Exciting occ: " << occ << " to vir: " << vir << std::endl;
                                }
                                add_det(&temp_det[0]);
                                // Print determinant after single excitations
                                std::cout << "Determinant for single combination" << std::endl;
                                for (int k = 0; k < nword; ++k) {
                                    std::cout << temp_det[k] << " ";
                                }
                                std::cout << std::endl;
                            }
                            
                        }
                    }   
                    else if (num_singles == 0){
                        // If num_singles == 0, directly add determinant after pair excitations
                        add_det(&det[0]);
                        // Print determinant after single excitations
                        std::cout << "Determinant for s=0" << std::endl;
                        for (int k = 0; k < nword; ++k) {
                            std::cout << det[k] << " ";
                        }
                        std::cout << std::endl;
                    }
                }    
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

    std::cout << "Filling beta spin orbitals" << std::endl;
    for (long i = 0; i < nocc_beta; ++i) {
        long bit_index = nb + i;
        std::cout << "bit_index: " << bit_index << std::endl;
        det[bit_index / Size<ulong>()] |= 1UL << (bit_index % Size<ulong>());
    }
    
    std::cout << "Filling alpha spin orbitals" << std::endl;
    for (long i = 0; i < nocc_alpha; ++i) {
        std::cout << "bit_index: " << i << std::endl;
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

        is_hf_det = true;
    } else
        ptr = reinterpret_cast<ulong *>(ref.cast<Array<ulong>>().request().ptr);
        is_hf_det = false;
    long ndet_old = ndet;
    add_excited_dets(ptr, exc);
    return ndet - ndet_old;
}
} //namespace pyci



// //    // Generate all combinations of d pairs
//             std::vector<std::vector<long>> opair_combinations;
//             std::vector<long> pair_indices(occ_pairs.size());
//             std::iota(pair_indices.begin(), pair_indices.end(), 0);
            
//             do {
//                 std::vector<long> combination(pair_indices.begin(), pair_indices.begin() + d);
//                 opair_combinations.push_back(combination);
//             } while (std::next_permutation(pair_indices.begin(), pair_indices.end()));
            
//             // Process each combination of d pairs
//             for (const auto& opair_comb: opair_combinations) {
//                 std::vector<std::pair<int,int>> used_occ_pairs;
//                 std::vector<long> used_virtuals;

//                 // Apply pair excitations
//                 std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
//                 for (long idx : opair_comb) {
//                     const auto& occ_pair = occ_pairs[idx];
//                     const auto& vir_pair = vir_pairs[idx];

//                     excite_det(occ_pair.first, vir_pair.first, &det[0]);
//                     excite_det(occ_pair.second, vir_pair.second, &det[0]);

//                     used_occ_pairs.push_back(occ_pair);
//                     used_virtuals.push_back(vir_pair.first);
//                     used_virtuals.push_back(vir_pair.second);
//                 }

//                 //Exclude used pairs and virtual orbitals
//                 std::vector<long> remaining_occ_indices;
//                 for (std::vector<std::pair<int, int>>::size_type i = 0; i < occ_pairs.size(); ++i) {
//                     if (std::find(opair_comb.begin(), opair_comb.end(), i) == opair_comb.end()) {
//                         remaining_occ_indices.push_back(static_cast<int>(i));
//                     }
//                 }

//                 // Generate all combinations of s singles
//                 std::vector<std::vector<long>> single_combinations;
//                 std::vector<long> single_indices(remaining_occ_indices.size());
//                 std::iota(single_indices.begin(), single_indices.end(), 0);

//                 do {
//                     std::vector<long> combination(single_indices.begin(), single_indices.begin() + s);
//                     single_combinations.push_back(combination);
//                 } while (std::next_permutation(single_indices.begin(), single_indices.end()));

//                 // Process each combination of s singles
//                 for (const auto& single_comb : single_combinations) {
//                     std::memcpy(&det[0], rdet, sizeof(ulong) * nword);

//                     // Apply single excitations
//                     for (long idx : single_comb) {
//                         long occ_idx = remaining_occ_indices[idx];
//                         long occ = occ_pairs[occ_idx].first; // Use the first of the remaining occ_pair
//                         for (long vir: virs) {
//                             if (std::find(used_virtuals.begin(), used_virtuals.end(), vir) != used_virtuals.end()) continue; 
//                             excite_det(occ, vir, &det[0]);
//                         }
//                     }
//                     add_det(&det[0]);
//                 }
//             }
