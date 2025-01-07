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
    const AlignedVector<std::pair<int,int>>& pairs, std::size_t k) {
    std::vector<std::vector<long>> result;
    // std::cout << "Inside nonsingletci/generate_cartesian_product" << std::endl;
    // std::cout <<   "pairs.size(): " << pairs.size() << std::endl;
    // std::cout << "k: " << k << std::endl;

    std::size_t nocc_pairs = pairs.size();
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
    }
    // Filter out combinations that are not of size k
    for (const auto& combination: temp_combinations) {
        if (combination.size() == k) {
            result.push_back(combination);
        }
    
    }
    // Debug output to ensure combinations are being generated
    // std::cout << "Generated " << result.size() << " combinations" << std::endl;
    
    return result;
}


// Function to generate combinations, based on indices
std::vector<std::vector<long>> NonSingletCI::generate_combinations(std::size_t n, std::size_t k) {
    std::vector<std::vector<long>> combinations;

    if (k > n) return combinations;

    std::vector<long> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<bool> mask(n, false);
    std::fill(mask.begin(), mask.begin() + k, true);

    do {
        std::vector<long> combination;
        for (std::size_t i = 0; i < n; ++i) {
            if (mask[i]) combination.push_back(indices[i]);
        }
        combinations.push_back(combination);
    } while (std::prev_permutation(mask.begin(), mask.end()));
    
    return combinations;
}

template <typename T>
void NonSingletCI::print_vector(const std::string& name, const AlignedVector<T>& vec) {
    std::cout << name << ": ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}
void NonSingletCI::print_pairs(const std::string& label, const AlignedVector<std::pair<int, int>>& pairs) {
    std::cout << label << ": ";
    for (const auto& elem : pairs) {
        std::cout << elem.first << " " << elem.second << " ";
    }
    std::cout << std::endl;
}

long NonSingletCI::calc_sindex(const long occ, const long vir) const {
    long o, v;
    long nb = nbasis / 2;
    if (vir < nb) {
        v = vir - nocc / 2;
    } else {
        v = vir - nocc;
    }                  
    if (occ >= nb) {
        o = occ - (nb - nocc / 2);
    } else{
        o = occ;
    }
    // std::cout << "\nocc: " << occ << ", vir: " << vir << std::endl;
    // std::cout << "o: " << o << ", v: " << v << std::endl;
    // std::cout << "nocc: " << nocc << ", nb: " << nb << std::endl;

    long idx = nocc / 2 * (nbasis - nocc) / 2 +  o * (nbasis - nocc) + v;
    // std::cout << "idx: " << idx << std::endl;
    return idx;
}

long NonSingletCI::calc_pindex(const long occ, const long vir) const {
    long o, v;
    // Assuming that only alpha orbitals are provided in the input
    if (vir < nbasis / 2) {
        v = vir - nocc / 2;
    } else {
        v = vir - nocc;
    }                  
    if (occ >= nbasis / 2) {
        o = occ - (nbasis / 2 - nocc / 2);
    } else {
        o = occ;
    }
    long idx = (nbasis / 2 - nocc / 2) * o + v;
    return idx;
}

void NonSingletCI::add_excited_dets(const ulong *rdet, const long e){
    // std::cout << "Inside nonsingletci/add_excited_dets" << std::endl;
    // //long i, j, k, no = binomial(nocc_up, e), nv = binomial(nvirs_up, e);
    // std::cout << "nocc: " << nocc << ", nvir: " << nvir << std::endl;
    // std::cout << "nocc_up: " << nocc_up << ", nvir_up: " << nvir_up << std::endl;
    // std::cout << "nocc_dn: " << nocc_dn << ", nvir_dn: " << nvir_dn << std::endl;
    // std::cout << "nbasis: " << nbasis << std::endl;

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
    occ_pairs.reserve(occs_up.size());
    vir_pairs.reserve(virs.size());

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

    // std::cout << "nocc_up: " << nocc_up << ", nvir_up: " << nvir_up << std::endl;
    // std::cout << "nocc: " << nocc << ", nvir: " << nvir << std::endl;
    // std::cout << "e: " << e << std::endl;
    // std::cout << "rdet: " << *rdet << std::endl;
    // print_vector("occs", occs);
    // print_vector("virs", virs);
    // print_vector("occs_up", occs_up);
    // print_vector("occs_dn", occs_dn);
    // print_vector("virs_up", virs_up);
    // print_vector("virs_dn", virs_dn);

    // Create an unordered set for fast lookup of occupied down-orbitals
    std::unordered_set<int> occ_dn_set(occs_dn.begin(), occs_dn.end()); 
    std::unordered_set<int> virs_set(virs.begin(), virs.end());

    // Generate occ_pairs and vir_pairs
    for (int i : occs_up) {
        if (occ_dn_set.find(i + nbasis/2) != occ_dn_set.end()) {
            occ_pairs.push_back({i, i + nbasis/2});   
        }
    }

    for (int a : virs) {
        if (virs_set.find(a + nbasis/2) != virs_set.end()) {
            vir_pairs.push_back({a, a + nbasis/2});   
        }
    }
    
    // print_pairs("occ_pairs", occ_pairs);
    // print_pairs("vir_pairs", vir_pairs);
    

    // Handle excitation order 0
    if (e == 0) {
        // std::cout << "Handling excitation order 0" << std::endl;
        
        std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
        add_det(&det[0]);
        
        return ;
    }
    // Handle excitation order 1
    if (e == 1) {
        // Flatten occ_pairs into a single-dimensional array
        std::vector<long> flattened_occs;
        for (const auto& pair : occ_pairs) {
            flattened_occs.push_back(pair.first);
            flattened_occs.push_back(pair.second);
        }
        // std::cout << "-----Handling excitation order 1-----" << std::endl;
        // std::cout << "Determinants of excitation order 1" << std::endl;

        for (long occ : flattened_occs) {
            for (long vir : virs) {
                std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                excite_det(occ, vir, &det[0]);
                add_det(&det[0]);
                
                // std::cout << "Determinant after excitation of " << occ << " " << vir << std::endl;
                // for (int k = 0; k < nword; ++k) {
                //     std::cout << det[k] << " ";
                // }
                // std::cout << std::endl;
            }
            // std::cout << std::endl;
        }
        // std::cout << std::endl; 
        return ;


        // //-----------------------pCCDSspin test-----------------------
        // // std::cout << "-----Handling excitation order 1-----" << std::endl;
        // for (long occ : occs_up) {
        //     for (long vir : virs_up) {
        //         std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
        //         excite_det(occ, vir, &det[0]);
        //         add_det(&det[0]);
        //     }
        // }

        // for (long occ : occs_dn) {
        //     for (long vir : virs_dn) {
        //         std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
        //         excite_det(occ, vir, &det[0]);
        //         add_det(&det[0]);
        //     }
        // }
        // return;
        // //-----------------------pCCDSspin test-----------------------
    }

    // Handle excitation orders >= 2
    std::size_t nocc_pairs = occ_pairs.size();
    std::size_t nvir_pairs = vir_pairs.size();

    // std::cout << "nocc_pairs: " << nocc_pairs << std::endl;
    
    if (e >= 2) {
        // Iterate over possible (d,s) pairs: d pair excitations and s single excitations
        // std::cout << "--------Handling excitation order >= 2--------" << std::endl;
        
        for (std::size_t d = 0; d <= std::min(static_cast<std::size_t>(e/2), nocc_pairs); ++d){
            std::size_t num_singles = e - 2 * d;

            // std::cout << "d: " << d << ", num_singles: " << num_singles << std::endl;
            
            // Not enough pairs for singles
            if (num_singles > (nocc_pairs - d)) {
                // std::cout << "Not enough pairs for singles" << std::endl;
                continue; 
            }

            // Generate occ pair and vir pair combinations
            std::vector<std::vector<std::vector<long>>> opair_combinations;
            std::vector<std::vector<std::vector<long>>> vpair_combinations;
            opair_combinations.push_back(generate_combinations(nocc_pairs, d));            
            vpair_combinations.push_back(generate_combinations(nvir_pairs, d));


            // std::cout << "Generated opair_combinations: " << std::endl;
            // for (const auto& opair_comb : opair_combinations[0]) {
            //     for (const auto& idx : opair_comb) {
            //         std::cout << "(" << occ_pairs[idx].first << ", " << occ_pairs[idx].second << ") ";
            //     }                
            // }
            // std::cout << std::endl;

            
            // std::cout << "Generated vpair_combinations: " << std::endl;
            // for (const auto& pair_comb : vpair_combinations[0]) {
            //     for (const auto& idx : pair_comb) {
            //         std::cout << "(" << vir_pairs[idx].first << ", " << vir_pairs[idx].second << ") ";
            //     }                
            // }
            // std::cout << std::endl;

            // Process pair combinations for current d
            for (const auto& opair_comb : opair_combinations[0]) {
                for (const auto& vpair_comb : vpair_combinations[0]) {
                    // if (opair_comb.empty() || vpair_comb.empty()) {
                        // std::cout << "Error: Empty combination detected" << std::endl;
                        // continue;
                    // }

                    std::memcpy(&det[0], rdet, sizeof(ulong) * nword);
                    std::vector<long> used_virs;
                    std::vector<long> used_occs;
                    // DetExcParamIndx det_exc;
                    
                    for (std::size_t idx = 0; idx < d; ++idx) {
                        const auto& occ_pair = occ_pairs[opair_comb[idx]];
                        const auto& vir_pair = vir_pairs[vpair_comb[idx]];

                        excite_det(occ_pair.first, vir_pair.first, &det[0]);
                        excite_det(occ_pair.second, vir_pair.second, &det[0]);


                        used_occs.push_back(occ_pair.first);
                        used_occs.push_back(occ_pair.second);
                        used_virs.push_back(vir_pair.first);
                        used_virs.push_back(vir_pair.second);
                        
                        // std::cout << "Determinant after pair excitation of" << std::endl;
                        // std::cout << "occ_pair: " << occ_pair.first << " " << occ_pair.second << std::endl;
                        // std::cout << "vir_pair: " << vir_pair.first << " " << vir_pair.second << std::endl;
                        // for (int k = 0; k < nword; ++k) {
                        //     std::cout << det[k] << " ";
                        // }
                        // std::cout << std::endl;
                        
                    }

   
                    if (num_singles > 0) {
                        // Determine remaining occupied pairs
                        AlignedVector<std::pair<int,int>> remaining_occ_pairs;
                        for (std::size_t i = 0; i < nocc_pairs; ++i) {
                            if (std::find(opair_comb.begin(), opair_comb.end(), i) == opair_comb.end()) {
                                remaining_occ_pairs.push_back(occ_pairs[i]);
                            }
                        }
                        // print_pairs("remaining_occ_pairs", remaining_occ_pairs);


                        AlignedVector<long> remaining_virs;
                        for (std::size_t i = 0; i < virs.size(); ++i) {
                            if (std::find(used_virs.begin(), used_virs.end(), virs[i]) == used_virs.end()) {
                                remaining_virs.push_back(virs[i]);
                            }
                        }
                        // print_vector("remaining_virs", remaining_virs);


                        // Process single combinations for current num_singles
                        auto occ_combinations = generate_cartesian_product(remaining_occ_pairs, num_singles);
                        auto vir_combinations = generate_combinations(remaining_virs.size(), num_singles);

                        // std::cout << "Generated occ_combinations: " << std::endl;
                        // for (const auto& occ_comb : occ_combinations) {
                        //     for (const auto& elem: occ_comb) {
                        //         std::cout << "(" << elem << ") ";
                        //     }
                        //     std::cout << std::endl;
                        // }
                        
                        // std::cout << "Generated vir_combinations: " << std::endl;
                        // for (const auto& vir_comb : vir_combinations) {
                        //     std::cout << "(";
                        //     for (const auto& vir_idx : vir_comb) {
                        //         std::cout << remaining_virs[vir_idx] << " ";
                        //     }
                        //     std::cout << ")" << std::endl;
                        // }

                        
                        for (const auto& occ_comb : occ_combinations) {
                            for (const auto& vir_comb : vir_combinations) {
                                // Do NOT reset det here; use the alredy existed det from pair excitations
                                AlignedVector<ulong> temp_det(nword);
                                std::memcpy(&temp_det[0], det.data(), sizeof(ulong) * nword);
                                
                                // Apply single excitations
                                for (std::size_t idx = 0; idx < num_singles; ++idx){
                                    long occ = occ_comb[idx];
                                    long vir = remaining_virs[vir_comb[idx]];
                                    excite_det(occ, vir, &temp_det[0]);
                                    // std::cout << "Exciting occ: " << occ << " to vir: " << vir << std::endl;
                                    
                                }
                                add_det(&temp_det[0]);
                                // Print determinant after single excitations
                                // std::cout << "Determinant for single combination" << std::endl;
                                // for (int k = 0; k < nword; ++k) {
                                //     std::cout << temp_det[k] << " ";
                                // }
                                // std::cout << std::endl;
                            }
                            
                        }

                        // //-----------------------pCCDSspin test-----------------------
                        // AlignedVector<long> remaining_occs;
                        // AlignedVector<long> remaining_virs;
                        // for (long occ : occs) {
                        //     if (std::find(used_occs.begin(), used_occs.end(), occ) == used_occs.end()) {
                        //         remaining_occs.push_back(occ);
                        //     }
                        // }

                        // for (long vir : virs) {
                        //     if (std::find(used_virs.begin(), used_virs.end(), vir) == used_virs.end()) {
                        //         remaining_virs.push_back(vir);
                        //     }
                        // }

                        // AlignedVector<long> rem_occ_up;
                        // AlignedVector<long> rem_occ_dn;
                        // for (long occ : remaining_occs) {
                        //     if (occ < nbasis / 2) {
                        //         rem_occ_up.push_back(occ);
                        //     } else {
                        //         rem_occ_dn.push_back(occ);
                        //     }
                        // }

                        // AlignedVector<long> rem_vir_up;
                        // AlignedVector<long> rem_vir_dn;
                        // for (long vir : remaining_virs) {
                        //     if (vir < nbasis / 2) {
                        //         rem_vir_up.push_back(vir);
                        //     } else {
                        //         rem_vir_dn.push_back(vir);
                        //     }
                        // }
                   
                        // auto occ_combs = generate_combinations(rem_occ_up.size(), num_singles);
                        // auto vir_combs = generate_combinations(rem_vir_up.size(), num_singles);

                        
                        // for (const auto& occ_comb : occ_combs) {
                        //     for (const auto& vir_comb : vir_combs) {
                        //         AlignedVector<ulong> temp_det(nword);
                        //         std::memcpy(&temp_det[0], det.data(), sizeof(ulong) * nword);
                        //         for (std::size_t idx = 0; idx < num_singles; ++idx) {
                        //             long occ = rem_occ_up[occ_comb[idx]];
                        //             long vir = rem_vir_up[vir_comb[idx]];
                        //             excite_det(occ, vir, &temp_det[0]);
                        //         }
                        //         add_det(&temp_det[0]);
                        //     }
                        // }

                        // auto occ_combs_dn = generate_combinations(rem_occ_dn.size(), num_singles);
                        // auto vir_combs_dn = generate_combinations(rem_vir_dn.size(), num_singles);

                        // for (const auto& occ_comb : occ_combs_dn) {
                        //     for (const auto& vir_comb : vir_combs_dn) {
                        //         AlignedVector<ulong> temp_det(nword);
                        //         std::memcpy(&temp_det[0], det.data(), sizeof(ulong) * nword);
                        //         for (std::size_t idx = 0; idx < num_singles; ++idx) {
                        //             long occ = rem_occ_dn[occ_comb[idx]];
                        //             long vir = rem_vir_dn[vir_comb[idx]];
                        //             excite_det(occ, vir, &temp_det[0]);
                        //         }
                        //         add_det(&temp_det[0]);
                        //     }
                        // }

                        // //-----------------------pCCDSspin test-----------------------
                    }   
                    else if (num_singles == 0){
                        // If num_singles == 0, directly add determinant after pair excitations
                        add_det(&det[0]);
                        // Print determinant after single excitations
                        // std::cout << "Determinant for s=0" << std::endl;
                        // for (int k = 0; k < nword; ++k) {
                        //     std::cout << det[k] << " ";
                        // }
                        // std::cout << std::endl; 
                    }
                }    
            }
        }
        return ;
    }
}

 


void NonSingletCI::fill_hartreefock_det(long nb2, long nocc, ulong *det) const {
   /* GenCIWfn build using FullCIWfn initializes the OneSpinWfn with nbasis * 2, so we are calling it nb2 here*/
    long nb = nb2/2;
    // Ensure nocc is even
    if (nocc % 2 != 0) {
        throw std::invalid_argument("Number of occupied orbitals (nocc) must be even.");
    }
    long nocc_beta = nocc / 2;
    long nocc_alpha = nocc / 2;
    // long num_ulongs = (nb2 + Size<ulong>() - 1) / Size<ulong>();
    

    // std::cout << "Inside nonsingletci/fill_hartreefock_det" << std::endl;
    // std::cout << "nb: " << nb << std::endl;
    // std::cout << "nocc: " << nocc << std::endl;
    // std::cout << "nocc_beta: " << nocc_beta << std::endl;
    // std::cout << "nocc_alpha: " << nocc_alpha << std::endl;

    // std::cout << "Filling beta spin orbitals" << std::endl;
    for (long i = 0; i < nocc_beta; ++i) {
        long bit_index = nb + i;
        // std::cout << "bit_index: " << bit_index << std::endl;
        det[bit_index / Size<ulong>()] |= 1UL << (bit_index % Size<ulong>());
    }
    
    // std::cout << "Filling alpha spin orbitals" << std::endl;
    for (long i = 0; i < nocc_alpha; ++i) {
        // std::cout << "bit_index: " << i << std::endl;
        long bit_index = i;
        det[bit_index / Size<ulong>()] |= 1UL << (bit_index % Size<ulong>());
    }
    // std::cout << "det: ";
    // for (int i = 0; i < num_ulongs; ++i) {
    //     std::cout << det[i] << " ";
    // }
    // std::cout << std::endl;
}



/**
 * @brief Adds excited determinants to the current wavefunction.
 *
 * This function generates excited determinants based on the given excitation level
 * and reference determinant. If no reference determinant is provided, it uses the
 * Hartree-Fock determinant.
 *
 * @param exc The excitation level.
 * @param ref The reference determinant as a pybind11 object. If none, Hartree-Fock determinant is used.
 * @return The number of new determinants added.
 */
long NonSingletCI::py_add_excited_dets(const long exc, const pybind11::object ref) {
    AlignedVector<ulong> v_ref;
    ulong *ptr;
    if (ref.is(pybind11::none())) {
        v_ref.resize(nword);
        ptr = &v_ref[0];
        fill_hartreefock_det(nbasis, nocc, ptr);
    } else {
        ptr = reinterpret_cast<ulong *>(ref.cast<Array<ulong>>().request().ptr);
    }
    long ndet_old = ndet;
    add_excited_dets(ptr, exc);
    return ndet - ndet_old;
}
} //namespace pyci
