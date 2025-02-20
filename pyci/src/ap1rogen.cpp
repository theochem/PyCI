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
#include <set>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>

namespace pyci {

// Constructor with Raw Pointers

// call to the base class constructor fullciwfn with the provided parameters
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(const SparseOp &op_, const NonSingletCI &wfn_,
                                 const std::size_t n_detcons_,
                                 const long *idx_detcons_,
                                 const double *val_detcons_,
                                 const std::size_t n_paramcons_,
                                 const long *idx_paramcons_,
                                 const double *val_paramcons_)
: Objective<NonSingletCI>::Objective(op_, wfn_, n_detcons_, idx_detcons_, val_detcons_, n_paramcons_, idx_paramcons_, val_paramcons_), wfn_(wfn_)
{
    // init_overlap(wfn_); 
    ranks = {1, 2};
    set_attributes_from_wfn(wfn_);
    exops = assign_exops(); //wfn_.nbasis, wfn_.nocc, wfn_.nword);
    // assign_and_sort_exops(ranks, wfn_.nbasis);
    nparam = nocc / 2 * (nbasis/2 - nocc / 2); //paired-doubles
    nparam += nocc * (nbasis - nocc); // alpha, beta singles

    ovlp.resize(nconn);
    d_ovlp.resize(nconn * nparam);
}
// call to initizlize the overlap related data

// Constructor with pybind11 objects
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(const SparseOp &op_, const NonSingletCI &wfn_,
                                 const pybind11::object idx_detcons_,
                                 const pybind11::object val_detcons_,
                                 const pybind11::object idx_paramcons_,
                                 const pybind11::object val_paramcons_)
: Objective<NonSingletCI>::Objective(op_, wfn_, idx_detcons_, val_detcons_, idx_paramcons_, val_paramcons_), wfn_(wfn_)
{
    // init_overlap(wfn_);
    ranks = {1, 2};
    set_attributes_from_wfn(wfn_);
    exops = assign_exops(); //wfn_.nbasis, wfn_.nocc, wfn_.nword);
    // assign_and_sort_exops(ranks, wfn_.nbasis);
    nparam = nocc / 2 * (nbasis/2 - nocc / 2); //paired-doubles
    nparam += nocc * (nbasis - nocc); // alpha, beta singles

    ovlp.resize(nconn);
    d_ovlp.resize(nconn * nparam);
}
// Copy Constructor
// obj is the constant reference to another object to be copied
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(const AP1roGeneralizedSenoObjective &obj)
: Objective<NonSingletCI>::Objective(obj), exops(obj.exops), ind_exops(obj.ind_exops), ranks(obj.ranks), det_exc_param_indx(obj.det_exc_param_indx), nexc_list(obj.nexc_list), wfn_(obj.wfn_)
{
    return;
}

// Move constructor
// obj is the rvalue reference to another object to be moved
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(AP1roGeneralizedSenoObjective &&obj) noexcept
: Objective<NonSingletCI>::Objective(obj), exops(std::move(obj.exops)), ind_exops(std::move(obj.ind_exops)), ranks(std::move(obj.ranks)), det_exc_param_indx(obj.det_exc_param_indx), nexc_list(std::move(obj.nexc_list)), wfn_(obj.wfn_)
{
    return;
}

// Function to set attributes from wfn
void AP1roGeneralizedSenoObjective::set_attributes_from_wfn(const NonSingletCI & wfn) {
    nbasis = wfn.nbasis;
    nocc = wfn.nocc;
    ndet = wfn.ndet;
    nword = wfn.nword;

    // std::cout << "nbasis: " << nbasis << ", nocc: " << nocc << ", ndet: " << ndet << ", nword: " << nword << std::endl;
    // // Print the types of the attributes
    // std::cout << "Type of wfn.nbasis: " << typeid(decltype(wfn.nbasis)).name() << std::endl;
    // std::cout << "Type of wfn.nocc: " << typeid(decltype(wfn.nocc)).name() << std::endl;
    // std::cout << "Type of wfn.ndet: " << typeid(decltype(wfn.ndet)).name() << std::endl;
    // std::cout << "Type of wfn.nword: " << typeid(decltype(wfn.nword)).name() << std::endl;
}

std::unordered_map<std::vector<long>, int, AP1roGeneralizedSenoObjective::HashVector, 
        AP1roGeneralizedSenoObjective::VectorEqual> AP1roGeneralizedSenoObjective::assign_exops(){
    //const long nbasis, const long nocc, const long nword){
    
    int counter = 0;
    // std::cout << "nocc: " << nocc << ", nbasis: " << nbasis << std::endl;
    //  Occupied indices of HF reference det
    AlignedVector<ulong> rdet(nword);
    wfn_.fill_hartreefock_det(nbasis, nocc, &rdet[0]);
    AlignedVector<long> ex_from(nocc);
    fill_occs(nword, rdet.data(), &ex_from[0]);

    AlignedVector<long> ex_to;
    for (std::size_t i = 0; i < nbasis; ++i) {
        if (std::find(ex_from.begin(), ex_from.end(), i) == ex_from.end()) {
            ex_to.push_back(i);
        }
    }

    // std::cout << "****Ex_from: ";
    // for (long i : ex_from) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "****Ex_to: ";
    // for (long i : ex_to) {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    // Generate Single excitations: non-spin-preserving
    for (size_t i = 0; i < ex_from.size(); ++i) {
        long occ_alpha = ex_from[i];
        for (long vir : ex_to) {
            std::vector<long> exop = {occ_alpha, vir};
            exops[exop] = counter++;
        }
    }

    long nspatial = nbasis / 2;

    // Generate paired double excitations
    for (size_t i = 0; i < ex_from.size() / 2; ++i) {
        long occ_alpha = ex_from[i];
        for (size_t a = 0; a < ex_to.size() / 2; ++a) {
            long vir_alpha = ex_to[a];
            std::vector<long> exop = {occ_alpha, occ_alpha + nspatial, vir_alpha, vir_alpha + nspatial};
            exops[exop] = counter++;            
        }
    }

    // std::cout << "****Exops: ";
    // for (const auto& exop : exops) {
    //     std::cout << "{ ";
    //     for (long i : exop.first) {
    //         std::cout << i << " ";
    //     }
    //     std::cout << "} : " << exop.second << std::endl;
    // }


    // Create a vector of keys from the exops map
    for (const auto& pair : exops) {
        ind_exops.push_back(pair.first);
    }

    // Sort the vector of keys based on the values in the exops map
    std::sort(ind_exops.begin(), ind_exops.end(), [this](const std::vector<long>& a, const std::vector<long>& b) {
        return exops[a] < exops[b];
    });

    return exops;
}





// Helper function to print the result
void AP1roGeneralizedSenoObjective::printPartitions(const std::vector<Partition>& partitions) {
    for (const auto& partition : partitions) {
        std::cout << "{ ";
        for (int coin : partition) {
            std::cout << coin << " ";
        }
        std::cout << "}\n";
    }   
}

// Recursive function to find all partitions
std::vector<AP1roGeneralizedSenoObjective::Partition> AP1roGeneralizedSenoObjective::findPartitions(const std::vector<int>& coins, int numCoinTypes, int total, Cache& cache) {
    // std::cout << "\nCoins.size(): " << coins.size() << std::endl;
    // std::cout << "numCoinTypes: " << numCoinTypes << std::endl;
    // std::cout << "total: " << total << std::endl;
    
    if (total == 0) {
        return {{}};
    }   

    if (total < 0 || numCoinTypes <= 0) {
        return {}; 
    }   

    CacheKey key = {numCoinTypes, total};
    if (cache.find(key) != cache.end()) {
        return cache[key];
    }   

    std::vector<Partition> result;

    // Include the last coin
    for (auto& partition : findPartitions(coins, numCoinTypes, total - coins[numCoinTypes - 1], cache)) {
        partition.push_back(coins[numCoinTypes - 1]);
        result.push_back(partition);
    }   

    // Exclude the last coin
    auto excludeLast = findPartitions(coins, numCoinTypes - 1, total, cache);
    result.insert(result.end(), excludeLast.begin(), excludeLast.end());

    cache[key] = result;
    // std::cout << "Cache size: " << cache.size() << std::endl;
    // std::cout << "Key: " << key.first << " " << key.second << std::endl;
    // std::cout << "Result size: " << result.size() << std::endl;
    return result;
}

// Main function to handle the partitions
std::vector<AP1roGeneralizedSenoObjective::Partition> AP1roGeneralizedSenoObjective::intPartitionRecursive(const std::vector<int>& coins, int numCoinTypes, int total) {
    // std::cout << "IntPartitionRecur: \nCoins: ";
    // for (int coin : coins) {
    //     std::cout << coin << " ";
    // }
    // std::cout << std::endl;

    Cache cache;
    auto result = findPartitions(coins, numCoinTypes, total, cache);

    // Sort each partition to ensure consistency
    for (auto& partition : result) {
        std::sort(partition.begin(), partition.end());
    }

    // Remove duplicates (optional if input guarantees no duplicates)
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    // std::cout << "Number of partitions: " << result.size() << std::endl;
    // std::cout << "Partitions: ";
    // printPartitions(result);
    
    return result;
}



void AP1roGeneralizedSenoObjective::generateCombinations(const std::vector<std::size_t> collection, std::vector<int>::size_type bin_size, int start, 
    std::vector<int>& current_comb, std::vector<std::vector<int>>& results) {
    if (current_comb.size() == bin_size) {
        results.push_back(current_comb);
        return;
    }

    for (std::vector<int>::size_type i = start; i < collection.size(); ++i) {
        current_comb.push_back(collection[i]);
        generateCombinations(collection, bin_size, i + 1, current_comb, results);
        current_comb.pop_back();
    }
}


// Function to generate unordered partitions
void AP1roGeneralizedSenoObjective::get_unordered_partition(std::vector<int>& collection, std::vector<std::pair<int, int>>& bin_size_num, 
                                  std::vector<std::vector<int>>& current_partition, int index, 
                                  std::vector<std::vector<std::vector<int>>>& results) {
   
    if (index == 0) {
        results.push_back(current_partition);
        return;
    }

    int last = collection[index-1];
    // std::cout << "Last element from collection: " << last << std::endl;

    std::vector<std::vector<std::vector<int>>> prev_partitions;
    get_unordered_partition(collection, bin_size_num, current_partition, index - 1, prev_partitions);

    for (const auto& prev_partition : prev_partitions) {
        // std::cout << "Previous partition: ";
        // print_current_partition(prev_partition);

        int ind_bin = -1;
        int ind_size = 0;
        int ind_bin_size = -1;

        while (ind_size < static_cast<int>(bin_size_num.size())) {
            ind_bin += 1;
            ind_bin_size += 1;

            if (ind_bin_size == bin_size_num[ind_size].second) {
                ind_size += 1;
                ind_bin_size = 0;
                if (ind_size == static_cast<int>(bin_size_num.size())) {
                    break;
                }
            }

            std::vector<int> subset = prev_partition[ind_bin];
            // std::cout << "Selected subset: ";
            // for (int num : subset) std::cout << num << " ";
            // std::cout << std::endl;

            if (subset.empty()) {
                std::vector<std::vector<int>> new_partition = prev_partition;
                new_partition[ind_bin].push_back(last);
                results.push_back(new_partition);

                if (bin_size_num[ind_size].second > 1) {
                    ind_bin += bin_size_num[ind_size].second - ind_bin_size - 1;
                    ind_size += 1;
                    ind_bin_size = -1;
                }
                continue;
            }

            if (!(subset.back() < last)) {
                // std::cout << "Skipping because " << subset.back() << " is not less than " << last << std::endl;
                continue;
            }

            if (!(static_cast<int>(subset.size()) < bin_size_num[ind_size].first)) {
                // std::cout << "Skipping because subset length " << subset.size() << " exceeds limit " << bin_size_num[ind_size].first << std::endl;
                continue;
            }

            std::vector<std::vector<int>> new_partition = prev_partition;
            new_partition[ind_bin].push_back(last);
            results.push_back(new_partition);
        }
    }
}


std::vector<std::vector<std::vector<int>>> AP1roGeneralizedSenoObjective::generate_unordered_partitions(std::vector<int> collection, 
    std::vector<std::pair<int, int>> bin_size_num) {
    // std::cout << "Entering in generate unordered partition\n" ;
    std::vector<std::vector<std::vector<int>>> results;

    // Compute total number of bins
    int total_bins = 0;
    for (auto& bin : bin_size_num) total_bins += bin.second;
    // std::cout << "Total bins: " << total_bins << std::endl;

    std::vector<std::vector<int>> current_partition(total_bins); // Initialize correct number of bins
    std::sort(collection.begin(), collection.end()); // Ensure input order
    get_unordered_partition(collection, bin_size_num, current_partition, collection.size(), results);

    return results;
}



// Grouping By Size
std::unordered_map<int, std::vector<std::vector<int>>> AP1roGeneralizedSenoObjective::group_by_size(
    const std::vector<std::vector<int>>& partitions) {
    // std::cout << "\nInside group_by_size\n";
    std::unordered_map<int, std::vector<std::vector<int>>> grouped_exops;
    for (const auto& part : partitions) {
        // std::cout << "Part size: " << part.size() << std::endl;
        grouped_exops[part.size()].push_back(part);
    }
    // print_grouped_exops(grouped_exops);
    // std::cout << "groups by size: " ;
    // for (const auto& group : grouped_exops) {
    //     std::cout << "Group size: " << group.first << std::endl;
    //     for (const auto& part : group.second) {
    //         std::cout << "{ ";
    //         for (int i : part) {
    //             std::cout << i << " ";
    //         }
    //         std::cout << "}\n";
    //     }
    // }

    return grouped_exops;
}

// // Generating Permutations
// template <typename T>
// std::vector<std::vector<T>> AP1roGeneralizedSenoObjective::generate_perms(
//     const std::vector<T>& elements, 
//     std::vector<T>& current, 
//     std::vector<bool>& used) {
//     std::vector<std::vector<T>> result;
//     if (current.size() == elements.size()) {
//         result.push_back(current);
//         return result;
//     }
//     for (size_t i = 0; i < elements.size(); ++i) {
//         if (used[i]) continue;
//         used[i] = true;
//         current.push_back(elements[i]);
//         auto perms = generate_perms(elements, current, used);
//         result.insert(result.end(), perms.begin(), perms.end());
//         current.pop_back();
//         used[i] = false;
//     }

//     return result;
// }

// Helper function to generate permutations
template <typename T>
std::vector<std::vector<T>> AP1roGeneralizedSenoObjective::generate_perms(const std::vector<T> & vec) {
    std::vector<std::vector<T>> perms;
    std::vector<T> temp = vec;
    do {
        perms.push_back(temp);
    } while (std::next_permutation(temp.begin(), temp.end()));
    
    // std::cout << "Permutations: ";
    // for (const auto& perm : perms) {
    //     std::cout << "{ ";
    //     for (T i : perm) {
    //         std::cout << i << " ";
    //     }
    //     std::cout << "}\n";
    // }

    return perms;
}

// Combining Paired Permutations
// combine annihilation and creation operators of the same size
// std::vector<std::pair<std::pair<std::vector<int>, std::vector<int>>> 
std::vector<std::pair<std::vector<int>, std::vector<int>>> AP1roGeneralizedSenoObjective::combine_pairs(
                                            const std::vector<std::vector<int>>& annhs, 
                                            const std::vector<std::vector<int>>& creas) {
    std::vector<std::pair<std::vector<int>, std::vector<int>>> combined_perms;
    for (const auto& annh : annhs) {
        for (const auto& crea : creas) {
            combined_perms.emplace_back(annh, crea);
        }
    }
    return combined_perms;
}


// Computes the sign of a permutation
int AP1roGeneralizedSenoObjective::sign_perm(std::vector<std::size_t> jumbled_set, const std::vector<std::size_t> ordered_set, bool is_decreasing = false) {
    if (is_decreasing) {
        std::reverse(jumbled_set.begin(), jumbled_set.end());
    }

    // Determine the ordered set if not provided
    std::vector<std::size_t> sorted_set = ordered_set.empty() ? jumbled_set : ordered_set;

    // Ensure the ordered set is strictly increasing
    if (!std::is_sorted(sorted_set.begin(), sorted_set.end())) {
        throw std::invalid_argument("ordered_set must be strictly increasing.");
    }

    int sign = 1;

    // Count transpositions needed to sort
    for (size_t i = 0; i < sorted_set.size(); ++i) {
        for (size_t j = 0; j < jumbled_set.size(); ++j) {
            if (jumbled_set[j] > sorted_set[i]) {
                sign *= -1;
            } else if (jumbled_set[j] == sorted_set[i]) {
                break;
            }
        }
    }

    return sign;
}

// Helper function to determine the smallest integer type to hold indices
template <typename T>
T AP1roGeneralizedSenoObjective::choose_dtype(size_t nparams) {
    size_t two_power = static_cast<size_t>(std::ceil(std::log2(nparams)));
    if (two_power <= 8) {
        return uint8_t();
    } else if (two_power <= 16) {
        return uint16_t();
    } else if (two_power <= 32) {
        return uint32_t();
    } else if (two_power <= 64) {
        return uint64_t();
    } else {
        throw std::runtime_error("Can only support up to 2**63 number of parameters");
    }
}

// Helper function to generate the Cartesian product of vectors
template <typename T>
std::vector<std::vector<std::pair<std::vector<T>, std::vector<T>>>> AP1roGeneralizedSenoObjective::cartesian_product(const std::vector<std::pair<std::vector<std::vector<T>>, std::vector<std::vector<T>>>>& v) {
    std::vector<std::vector<std::pair<std::vector<T>, std::vector<T>>>> s = {{}};
    for (const auto& u : v) {
        std::vector<std::vector<std::pair<std::vector<T>, std::vector<T>>>> r;
        for (const auto& x : s) {
            for (const auto& y1 : u.first) {
                for (const auto& y2 : u.second) {
                    r.push_back(x);
                    r.back().emplace_back(y1, y2);
                }
            }
        }
        s = std::move(r);
    }
    return s;
}

// Explicit instantiation of the cartesian_product template for the required type
// template std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> cartesian_product(const std::vector<std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>>&);

// std::vector<std::vector<T>> AP1roGeneralizedSenoObjective::cartesian_product(const std::vector<std::vector<T>>& v) {
//     std::vector<std::vector<T>> result;
//     if (v.empty()) return result;

//     // Initialize result with the first vector
//     result.push_back({});
//     for (const auto& vec : v) {
//         std::vector<std::vector<T>> temp;
//         for (const auto& res : result) {
//             for (const auto& elem : vec) {
//                 temp.push_back(res);
//                 temp.back().push_back(elem);
//             }
//         }
//         result = std::move(temp);
//     }
//     return result;
// }

// Ensure the template function is instantiated for the required types

// template std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> AP1roGeneralizedSenoObjective::cartesian_product(const std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>>&);

// template class cartesian_product<std::pair<std::vector<int>, std::vector<int>>>;


// Function to generate all possible excitation operators
void AP1roGeneralizedSenoObjective::generate_possible_exops(
    const std::vector<std::size_t>& a_inds, const std::vector<std::size_t>& c_inds) 
{
    // std::cout << "\nGenerating possible excitation operators...\n";

    // const std::unordered_set<std::vector<int>>& valid_exops;
    std::pair<std::vector<std::size_t>, std::vector<std::size_t>> key = std::make_pair(a_inds, c_inds);
    // std::unordered_map<std::vector<std::vector<std::size_t>>, std::size_t> inds_multi;
    std::unordered_map<int, std::vector<std::vector<int>>> inds_multi;
    
    // Step 1: Get partitoins of a_inds size
    int exrank = a_inds.size();
    int nranks = ranks.size();
    auto partitions = intPartitionRecursive(ranks, nranks, exrank);

    for (const auto& partition : partitions) {
        // std::cout << "Partition: ";
        // for (int size : partition) {
        //     std::cout << size << " ";
        // }
        // std::cout << "\n";
        // Step 2: Convert partition into bin size and count format
        std::unordered_map<int, int> reduced_partition;
        for (int size : partition) {
            reduced_partition[size]++;
        }

        //bin_size_num = exc_op_rank, counter of the number of that excitation operator
        std::vector<std::pair<int, int>> bin_size_num;
        // for (const auto& [bin_size, count] : reduced_partition) {
        for (const auto& bins : reduced_partition) {
            const auto& bin_size = bins.first;
            const auto& count = bins.second;
            bin_size_num.emplace_back(bin_size, count);
            // std::cout << "Bin size: " << bin_size << " Count: " << count << "\n";
        }

        // Step 3: Generate all unordered partitions of a_inds and c_inds
        // convert a_inds from size_t to int
        std::vector<int> a_inds_int(a_inds.begin(), a_inds.end());
        std::vector<int> c_inds_int(c_inds.begin(), c_inds.end());
        auto annhs_partitions = generate_unordered_partitions(a_inds_int, bin_size_num);
        auto creas_partitions = generate_unordered_partitions(c_inds_int, bin_size_num);

        // std::cout << "annhs_partitions:\n";
        // for (const auto& annhs : annhs_partitions) {
        //     std::cout << "{ ";
        //     for (const auto& annh : annhs) {
        //         std::cout << "{ ";
        //         for (int i : annh) {
        //             std::cout << i << " ";
        //         }
        //         std::cout << "} ";
        //     }
        //     std::cout << "}\n";
        // }
        // std::cout << "creas_partitions:\n";
        // for (const auto& creas : creas_partitions) {
        //     std::cout << "{ ";
        //     for (const auto& crea : creas) {
        //         std::cout << "{ ";
        //         for (int i : crea) {
        //             std::cout << i << " ";
        //         }
        //         std::cout << "} ";
        //     }
        //     std::cout << "}\n";
        // }


        for (const auto& annhs : annhs_partitions) {
            // auto annhs_grouped = group_by_size(annhs);
            std::unordered_map<int, std::vector<std::vector<int>>> annhs_grouped;
            for (const auto& annh : annhs) {
                annhs_grouped[annh.size()].push_back(annh);
            }

            // std::cout << "annh_grouped:\n";
            // for (const auto& pair : annhs_grouped) {
            //     std::cout << pair.first << ": ";
            //     for (const auto& vec : pair.second) {
            //         std::cout << "{ ";
            //         for (int i : vec) {
            //             std::cout << i << " ";
            //         }
            //         std::cout << "} ";
            //     }
            //     std::cout << std::endl;
            // }
            
            for (const auto& creas: creas_partitions) {
                // auto creas_grouped = group_by_size(creas);
                std::unordered_map<int, std::vector<std::vector<int>>> creas_grouped;
                for (const auto& crea : creas) {
                    creas_grouped[crea.size()].push_back(crea);
                }

                // std::cout << "creas_grouped:\n";
                // for (const auto& pair : creas_grouped) {
                //     std::cout << pair.first << ": ";
                //     for (const auto& vec : pair.second) {
                //         std::cout << "{ ";
                //         for (int i : vec) {
                //             std::cout << i << " ";
                //         }
                //         std::cout << "} ";
                //     }
                //     std::cout << std::endl;
                // }

                // Step 3: Generate permutations for each group
                std::unordered_map<int, std::vector<std::vector<std::vector<int>>>> creas_perms;
                // // --------------------------------------------------------------------
                // for (const auto& pair : creas_grouped) {
                //     std::cout << "Generating perms:\n";
                //     std::cout << pair.second.size() << std::endl;
                //     if (pair.first == 1 && pair.second.size() > 1) {
                //         // Permute all bins together
                //         std::vector<int> all_bins;
                //         for (const auto& vec : pair.second) {
                //             all_bins.insert(all_bins.end(), vec.begin(), vec.end());
                //         }
                //         auto perms = generate_perms(all_bins);
                //         for (const auto& perm : perms) {
                //             std::vector<std::vector<int>> split_perm;
                //             for (size_t i = 0; i < perm.size(); i ++) {
                //                 split_perm.push_back(std::vector<int>(perm.begin() + i, perm.begin() + i + pair.first));
                //             }
                //             creas_perms[pair.first].insert(creas_perms[pair.first].end(), split_perm.begin(), split_perm.end());
                //         }
                //     } else {
                //         const auto& group = pair.second;
                //         std::vector<std::vector<std::vector<int>>> perms;
                //         if (group.size() > 1) {
                //             // Generate permutations of the vectors in the group
                //             std::vector<std::vector<int>> group_copy = group;
                //             do {
                //                 perms.push_back(group_copy);
                //             } while (std::next_permutation(group_copy.begin(), group_copy.end()));
                //         } else {
                //             perms.push_back(group);
                //         }
                //         creas_perms[pair.first] = perms;
                //         // for (const auto& vec : pair.second){
                //         //     std::cout << "vec: ";
                //         //     for (int i : vec) {
                //         //         std::cout << i << " ";
                //         //     }
                //         //     std::cout << std::endl;                            
                //         //     creas_perms[pair.first] = generate_perms(vec);
                //         // }
                //     }
                    
                // }
                // // --------------------------------------------------------------------
                    // const auto& size = size_group.first;
                    // const auto& group = size_group.second;
                    // std::vector<std::vector<int>> perms;
                    // for (const auto& crea: group) {
                    //     std::vector<int> temp;
                    //     std::vector<bool> used(crea.size(), false);
                    //     std::cout << "genreate_perms\n";
                    //     auto current_perms = generate_perms(crea, temp, used);
                    //     perms.insert(perms.end(), current_perms.begin(), current_perms.end());
                    // }
                    // creas_perms[size] = perms;
                    // std::cout << "creas_perms:\n";
                    // for (const auto& perm : perms) {
                    //     std::cout << "{ ";
                    //     for (int i : perm) {
                    //         std::cout << i << " ";
                    //     }
                    //     std::cout << "}\n";
                    // }
                // }

                for (const auto& pair : creas_grouped) {
                    int size = pair.first;
                    const auto& group = pair.second;
                    std::vector<std::vector<std::vector<int>>> perms;

                    // Generate permutations of the vectors in group
                    std::vector<std::vector<int>> group_copy = group;
                    do {
                        perms.push_back(group_copy);
                    } while (std::next_permutation(group_copy.begin(), group_copy.end()));
                    creas_perms[size] = perms;
                }

                // Print creas_perms for debugging
                // std::cout << "creas_perms:\n";
                // for (const auto& pair : creas_perms) {
                //     std::cout << pair.first << ": ";
                //     for (const auto& vecs : pair.second) {
                //         std::cout << "{ ";
                //         for (const auto& vec : vecs) {
                //             std::cout << "{ ";
                //             for (int i : vec) {
                //                 std::cout << i << " ";
                //             }
                //             std::cout << "} ";
                //         }
                //         std::cout << "} ";
                //     }
                //     std::cout << std::endl;
                // }

                // Combine permutations of each annihilation and creation pair (of same size)
                // std::vector<std::vector<std::pair<std::vector<int>, std::vector<int>>>> exc_perms;
                std::vector<std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>> exc_perms;
                for (const auto& size_num : bin_size_num) {
                    int size = size_num.first;
                    // if (annhs_grouped.find(size) != annhs_grouped.end() && creas_perms.find(size) != creas_perms.end()) {
                    //     if (size == 1 && size_num.second > 1) {
                    //         /// Special case: handle permutations of entire group
                    //         std::size_t nperms = creas_perms[size].size() / creas_grouped[size].size();
                    //         std::cout << "nperms: " << nperms << std::endl;
                    //         for (std::size_t i = 0; i < creas_perms[size].size(); i += nperms) {
                    //             std::vector<std::vector<int>> split_crea;
                    //             for (std::size_t j = 0; j < nperms ; ++j) {
                    //                 split_crea.push_back(creas_perms[size][i + j]);
                    //             }
                    //             exc_perms.push_back(std::make_pair(annhs_grouped[size], split_crea));
                    //         }
                            
                    //     } else {
                    //         exc_perms.push_back(std::make_pair(annhs_grouped[size], creas_perms[size]));
                    //     }
                    // }
                    if (annhs_grouped.find(size) != annhs_grouped.end() && creas_perms.find(size) != creas_perms.end()) {
                        for (const auto& crea_perm : creas_perms[size]){
                            exc_perms.push_back(std::make_pair(annhs_grouped[size], crea_perm));
                        }
                    }
                }
                    // auto annh_group = annhs_grouped[size];
                    // auto crea_perm = creas_perms[size];
                    // std::cout << "Printing crea_perm\n";
                    // for (const auto& crea : crea_perm) {
                    //     std::cout << "{ ";
                    //     for (int i : crea) {
                    //         std::cout << i << " ";
                    //     }
                    //     std::cout << "}\n";
                    // }
                    // std::vector<std::pair<std::vector<int>, std::vector<int>>> combined;
                    // for (const auto& annh : annh_group) {
                    //     for (const auto& crea : crea_perm) {

                            // std::cout << "Annihilation: ";
                            // for (int i : annh) {
                            //     std::cout << i << " ";
                            // }
                            // std::cout << std::endl;
                            // std::cout << "Creation: ";
                            // for (int i : crea) {
                            //     std::cout << i << " ";
                            // }
                            // std::cout << std::endl;
                            // combined.emplace_back(std::make_pair(annh, crea));
                           
                            // combined.push_back(std::make_pair(annh, crea));
                            // std::cout << "Combined: {";
                            // for (const auto& elem : combined.back().first) {
                            //     std::cout << elem << " ";
                            // }
                            // std::cout << "-->";
                            // for (const auto& elem : combined.back().second) {
                            //     std::cout << elem << " ";
                            // }
                            // std::cout << "}" << std::endl;
                        // }
                    // }
                    // exc_perms.push_back(combined);

                // }

                // Print exc_perms
                // std::cout << "\nexc_perms: ";
                // for (const auto& pair : exc_perms) {
                //     for (const auto& annh : pair.first) {
                //         std::cout << "{ ";
                //         for (int i : annh) {
                //             std::cout << i << " ";
                //         }
                //     }
                //     std::cout << "} -> { ";
                //     for (const auto& crea : pair.second) {
                //         for (int i : crea) {
                //             std::cout << i << " ";
                //         }
                //         std::cout << "} ";
                //     }    
                //     std::cout << std::endl;
                // }

                
                // for (const auto& excs : cartesian_product(exc_perms)) {
                for (const auto& excs : exc_perms) {
                    // std::cout << "***Cartesian product***\n";
                    std::vector<std::vector<long>> combs;
                    // std::vector<std::pair<std::vector<long>, std::vector<long>>> combs;
                    bool is_continue = false;
                    // for (std::size_t i = 0; i < excs.first.size(); ++i) {
                    //     // auto [annh, crea] = exc;
                    auto annhs = excs.first;
                    auto creas = excs.second;

                    // std::cout << "Annihilation: ";
                    // for (const auto& vec : annhs) {
                    //     std::cout << "{ ";
                    //     for (int elem : vec) {
                    //         std::cout << elem << " ";
                    //     }
                    //     std::cout << "} ";
                    // }
                    // std::cout << std::endl;
                    // std::cout << "Creation: ";
                    // for (const auto& vec : creas) {
                    //     std::cout << "{ ";
                    //     for (int elem : vec) {
                    //         std::cout << elem << " ";
                    //     }
                    //     std::cout << "} ";
                    // }
                    // std::cout << std::endl;
                        //(annh);
                        // op.insert(op.end(), crea.begin(), crea.end());
                        // op.reserve(annh.size() + crea.size()); // Reserve space for efficiency
                        
                        // std::vector<long> op;
                        // for (int a : annh) {
                        //     std::cout << "a: " << a << std::endl;
                        //     op.push_back(static_cast<long>(a)); // Convert each element to long
                        // }
                        // for (int c : crea) {
                        //     std::cout << "c: " << c << std::endl;
                        //     op.push_back(static_cast<long>(c)); // Convert each element to long
                        // }
                    for (std::size_t j = 0; j < annhs.size(); ++j) {
                        std::vector<long> annh(annhs[j].begin(), annhs[j].end()); // Convert each element to long
                        std::vector<long> crea(creas[j].begin(), creas[j].end()); // Convert each element to long
                        std::vector<long> op(annh.begin(), annh.end());
                        op.insert(op.end(), crea.begin(), crea.end());
                        // std::cout << "op: ";
                        // for (long elem : op) {
                        //     std::cout << elem << " ";
                        // }
                        // std::cout << std::endl;

                        // std::cout << "Size of exops: " << exops.size() << std::endl;
            
                        // std::csout<< "searching for op in exops\n";
                        // if (exops.find(op) != exops.end()) {
                        //     std::cout << "op found in exops\n";
                        // }
                        // else if (exops.find(op) == exops.end()) {
                        //     std::cout << "op not found in exops\n";
                        //     is_continue = false;
                        //     break;
                        // }
                        // try {
                        //     combs.push_back(op);
                        //     std::cout << "Successfully pushed op to combs" << std::endl;
                        //     } catch (const std::exception& e) {
                        //         std::cerr << "Exception caught during push_back: " << e.what() << std::endl;
                        //     } catch (...) {
                        //         std::cerr << "Unknown exception caught during push_back" << std::endl;
                        // }

                        if (std::find_if(exops.begin(), exops.end(), [&op](const auto& pair) {
                            return pair.first == op;
                        }) == exops.end()) {
                            // std::cout << "op not found in exops\n";
                            is_continue = true;
                            break;
                        }
                        combs.push_back(op);
                        
                        // }
                    
                    }
                    if (is_continue) break;
                    // if (is_continue) continue;

                    int num_hops = 0, prev_hurdles = 0;
                    //std::cout << "combs size: " << combs.size() << std::endl;
            
                    std::vector<std::size_t> jumbled_a_inds, jumbled_c_inds;
                    for (const auto& exop : combs) {
                        size_t num_inds = exop.size() / 2;
                        num_hops += prev_hurdles * num_inds;
                        prev_hurdles += num_inds;
                        
                        //std::cout << "num_hops: " << num_hops << ", prev_hurdles: " << prev_hurdles << std::endl;

                        jumbled_a_inds.insert(jumbled_a_inds.end(), exop.begin(), exop.begin() + num_inds);
                        jumbled_c_inds.insert(jumbled_c_inds.end(), exop.begin() + num_inds, exop.end());
                    }

                    int sign = (num_hops % 2 == 0) ? 1 : -1;
                    
                    

                    sign *= sign_perm(jumbled_a_inds, a_inds, false);
                    sign *= sign_perm(jumbled_c_inds, c_inds, false);

                    std::vector<int> inds;
                    for (const auto& exop : combs) {
                        inds.push_back(exops[exop]);
                    }
                    inds.push_back(sign);
                    //std::cout << "Sign: " << sign << std::endl;

                    
                    //std::cout << "Inds: ";
                    // for (int i : inds) {
                    //     std::cout << i << " ";
                    //     if (i > 10000) {
                    //         std::cout << "i > 10000\n";
                    //     }
                    // }
                    // std::cout << std::endl;

                    if (inds_multi.find(inds.size() - 1) == inds_multi.end()) {
                        //std::cout << "Inds_multi not found\n";
                        inds_multi[static_cast<int>(inds.size() - 1)] = {};
                    }
                    inds_multi[static_cast<int>(inds.size() - 1)].push_back(inds);
                }
            }
        
        } 
    }
    //std::cout << "addign inds_multi exop_combs: ";
    // exop_combs[key].push_back(inds_multi);
    try {
        exop_combs[key] = inds_multi;
        //std::cout << "Successfully assigned inds_multi to exop_combs" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught during exop_combs assignment: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught during exop_combs assignment" << std::endl;
    }

    // std::cout << "exop_combs done";
    // Print the contents of exop_combs
    // std::cout << "Contents of exop_combs:" << std::endl;
    // for (const auto& pair : exop_combs) {
    //     std::cout << "\n{";
    //     for (const auto& elem : pair.first.first) {
    //         std::cout << elem << ", ";
    //     }
    //     for (const auto& elem : pair.first.second) {
    //         std::cout << elem << ", ";
    //     }
    //     std::cout << "} : ";
    //     for (const auto& vec : pair.second) {
    //         std::cout << "{";
    //         const auto& rank = vec.first;
    //         const auto& inds = vec.second;
    //         std::cout << rank << ":" ;
    //         for (const auto& ind : inds) {
    //             std::cout << "{";
    //             for (int i : ind) {
    //                 std::cout << i << " ";
    //             }
    //             std::cout << "}";
    //         }
    //         std::cout << "}, ";
    //     }
    //     std::cout << std::endl;
    // }



    // Determine dtype
    // uint64_t dtype = choose_dtype<uint64_t>(nparam);

//     // Prepare reshaped and finalized inds_multi
//     for (auto& [rank, inds] : inds_multi) {
//         // Reshape to (-1, rank + 1) and store in uint64_t format
//         size_t row_size = rank + 1;
//         size_t nrows = inds.size() / row_size;
//         std::vector<uint64_t> reshaped_inds;
//         std::pair<std::vector<std::size_t>, std::vector<std::size_t>> key = std::make_pair(a_inds, c_inds);

//         for (size_t i = 0; i < nrows; ++i) {
//             uint64_t packed_row = 0;            
//             for (size_t j = 0; j < row_size; ++j) {
//                 packed_row = (packed_row << 16) | static_cast<uint64_t>(inds[i * row_size + j][0]);
//             }
//             reshaped_inds.push_back(packed_row);   
//         }
//         // Save reshaped indices in the exop_combs structure
//         exop_combs[key][rank] = reshaped_inds;
//     }
}


// std::variant<double, std::vector<double>> AP1roGeneralizedSenoObjective::product_amplitudes_multi(
double AP1roGeneralizedSenoObjective::product_amplitudes_multi(
    const std::unordered_map<int, std::vector<std::vector<int>>>& inds_multi,
    bool deriv, const double* x){
    // std::cout << "deriv: " << deriv << std::endl;
    double output = 0.0;
    if (deriv) {
        throw std::invalid_argument("Derivative computation is not supported.");
    } else {
        // for (const auto& [exc_order, inds_sign] : inds_multi) {
        // std::cout << "In product_amplitudes_multi\n";
        // std::cout << "inds_multi size: " << inds_multi.size() << std::endl;

        for (const auto& exc_inds : inds_multi) {
            // std::cout << "Processing exc_inds\n";
        
            // std::size_t exc_order = exc_inds.first;
            const std::vector<std::vector<int>> inds_sign = exc_inds.second;
            
            // std::cout << "Exc_order: " << exc_order << std::endl;
            // std::cout << "inds_sign size: " << inds_sign.size() << std::endl;
        
            // for (const auto& row : inds_sign) {    
            //     std::cout << "Row: ";
            //     for (int i : row) {
            //         std::cout << i << " ";
            //     }
            //     std::cout << std::endl;
            // }

            std::vector<std::vector<int>> indices;
            std::vector<int> signs;
            for (const auto& row : inds_sign) {

                // std::cout << "Row: ";
                // for (int i : row) {
                //     std::cout << i << " ";
                // }
                // std::cout << std::endl;

                if (row.empty()){
                    std::cerr << "Error: row in empty\n";
                }

                

                std::vector<int> ind(row.begin(), row.end() - 1);
                indices.push_back(ind);
                
                // std::cout << "Indices: ";
                // for (int i : ind) {
                //     std::cout << i << " ";
                // }
                // std::cout << std::endl;

                int sign = row.back();
                signs.push_back (sign > 1 ? -1 :  sign);
            }

            for (size_t i = 0; i < indices.size(); ++i) {
                double product = 1.0;
                for (const auto& idx : indices[i]) {
                    // std::cout << "x[idx]: " << x[idx] << std::endl;
                    product *= x[idx];
                }
                output += product * signs[i];
            }
        }
        
    }
    return output;

    // // Derivative computation
    // std::vector<double> output(nparam, 0.0);
    // for (const auto& [exc_order, inds_sign] : inds_multi) {
    //     std::vector<std::vector<int>> indices;
    //     std::vector<int> signs;
    //     for (const auto& row : inds_sign) {
    //         std::vector<int> ind(row.begin(), row.end() - 1);
    //         indices.push_back(ind);
    //         int sign = row.back();
    //         signs.push_back(sign > 1 ? -1 : sign);
    //     }

    //     std::unordered_set<int> unique_inds;
    //     for (const auto& row : indices) {
    //         unique_inds.insert(row.begin(), row.end());
    //     }
        
    //     for (const auto& ind : unique_inds) {
    //         std::vector<bool> bool_inds;
    //         for (const auto& row : indices) {
    //             bool_inds.push_back(std::find(row.begin(), row.end(), ind) != row.end());
    //         }

    //         std::vector<bool> row_inds;
    //         for (const auto& bi : bool_inds) {
    //             row_inds.push_back(bi);
    //         }

    //         std::vector<std::vector<double>> selected_params;
    //         for (const auto& row : indices) {
    //             std::vector<double> temp;
    //             for (const auto& idx : row) {
    //                 temp.push_back(x[idx]);
    //             }
    //             selected_params.push_back(temp);
    //         }

    //         std::vector<double> old_params;
    //         for (size_t i = 0; i < selected_params.size(); ++i) {
    //             if (bool_inds[i]) {
    //                 old_params.push_back(selected_params[i][ind]);
    //                 selected_params[i][ind] = 1.0;
    //             }
    //         }

    //         for (size_t i = 0; i < selected_params.size(); ++i) {
    //             if (row_inds[i]) {
    //                 double prod = 1.0;
    //                 for (const auto& val : selected_params[i]) {
    //                     prod *= val;
    //                 }
    //                 output[ind] += prod * signs[i];
    //             }
    //         }

    //         for (size_t i = 0; i < selected_params.size(); ++i) {
    //             if (bool_inds[i]) {
    //                 selected_params[i][ind] = old_params[i];
    //             }
    //         }
    //     }
    // }    
    // return output;
}

std::vector<double> AP1roGeneralizedSenoObjective::product_ampli_multi_deriv(
    const std::unordered_map<int, std::vector<std::vector<int>>>& indices_multi,
    bool deriv, const double* params){
    std::vector<double> output(nparam, 0.0);
    if (!deriv) {
        throw std::invalid_argument("Deriv must be enabled for this function.");
    }

    for (const auto& indices_sign : indices_multi) {
        const auto& indices = indices_sign.second;
        std::vector<int> signs(indices.size());
        std::transform(indices.begin(), indices.end(), signs.begin(), [](const std::vector<int>& row) {
            return row.back() > 1 ? -1 : row.back();
        });

        std::set<int> unique_indices;
        for (const auto& row : indices) {
            unique_indices.insert(row.begin(), row.end() - 1);
        }

        for (int ind : unique_indices) {
            std::vector<bool> bool_indices(indices.size());
            std::transform(indices.begin(), indices.end(), bool_indices.begin(), [ind](const std::vector<int>& row) {
                return std::find(row.begin(), row.end() - 1, ind) != row.end() - 1;
            });

            std::vector<bool> row_inds(indices.size());
            std::transform(bool_indices.begin(), bool_indices.end(), row_inds.begin(), [](bool b) { return b; });

            std::vector<std::vector<double>> selected_params(indices.size());
            for (std::size_t i = 0; i < indices.size(); ++i) {
                selected_params[i].resize(indices[i].size() - 1);
                for (std::size_t j = 0; j < indices[i].size() - 1; ++j) {
                    selected_params[i][j] = params[indices[i][j]];
                }
            }

            std::vector<double> old_params(selected_params.size());
            for (std::size_t i = 0; i < selected_params.size(); ++i) {
                if (bool_indices[i]) {
                    old_params[i] = selected_params[i][0];
                    selected_params[i][0] = 1.0;
                }
            }

            for (std::size_t i = 0; i < row_inds.size(); ++i) {
                if (row_inds[i]) {
                    double prod = 1.0;
                    for (std::size_t j = 0; j < selected_params[i].size(); ++j) {
                        prod *= selected_params[i][j];
                    }
                    output[ind] += prod * signs[i];
                }
            }

            for (std::size_t i = 0; i < selected_params.size(); ++i) {
                if (bool_indices[i]) {
                    selected_params[i][0] = old_params[i];
                }
            }
        }
    }
    return output;
}

int AP1roGeneralizedSenoObjective::sign_swap(AlignedVector<ulong> sd, long pos_current, long pos_future){
    // Return the signature of applying annihilators then creators to the Slater determinant.
    if (pos_current < 0 || pos_future < 0) {
        throw std::invalid_argument("The current and future positions must be positive integers.");
    }

    // if (!((sd >> pos_current) & 1)) {
    if (!((sd[pos_current / Size<ulong>()] >> (pos_current % Size<ulong>())) & 1)) {
        throw std::invalid_argument("Given orbital is not occupied in the Slater determinant.");
    }

    int num_trans = 0;
    if (pos_current < pos_future) {
        // Count the number of set bits between pos_current and pos
        for (long i = pos_current + 1; i <= pos_future; ++i) {
            if ((sd[i / Size<ulong>()] >> (i % Size<ulong>())) & 1) {
                ++num_trans;
            }
        }
    } else {
        for (long i = pos_future; i < pos_current; ++i) {
            if ((sd[i / Size<ulong>()] >> (i % Size<ulong>())) & 1) {
                ++num_trans;
            }
        }
    }
    return (num_trans % 2 == 0) ? 1 : -1;

}

int AP1roGeneralizedSenoObjective::sign_excite(AlignedVector<ulong> sd,
    const std::vector<std::size_t>& annihilators, const std::vector<std::size_t>& creators) {
    // Return the sign of the Slater determinant after applying excitation operators.
    int sign = 1;
    for (std::size_t i : annihilators) {
        // if (!((sd >> i) & 1)) {
        if (!((sd[i / Size<ulong>()] >> (i % Size<ulong>())) & 1)) {
            //print sd
            std::cout << "Annihilator: " << i << std::endl;
            for (std::size_t k = 0; k < sd.size(); ++k) {
                std::cout << sd[k] << " ";
            }
            std::cout << std::endl;
            throw std::invalid_argument("Given Slater determinant cannot be excited using the given creators and annihilators.");
        }
        sign *= sign_swap(sd, i, 0);
        // sd = sd & ~(1UL << i); // annihilate ith orbital
        sd[i / Size<ulong>()] &= ~(1UL << (i % Size<ulong>())); // annihilate ith orbital
    }

    for (std::size_t a : creators) {
        // sd = sd | (1UL << a); // create ath orbital
        sd[a / Size<ulong>()] |= (1UL << (a % Size<ulong>())); // create ath orbital
        // if (sd == 0) {
        if (!((sd[a / Size<ulong>()] >> (a % Size<ulong>())) & 1)) {
            //print sd
            std::cout << "Creator: " << a << std::endl;
            for (std::size_t k = 0; k < sd.size(); ++k) {
                std::cout << sd[k] << " ";
            }
            std::cout << std::endl;
            throw std::invalid_argument("Given Slater determinant cannot be excited using the given creators and annihilators.");
        }
        sign *= sign_swap(sd, a, 0);
    }
    return sign;
}


void AP1roGeneralizedSenoObjective::overlap(std::size_t ndet, const double* x, double* y)
{
    std::cout << "-------Computing Overlap\n nbasis: " << nbasis << ", ndet: " << ndet << "\n" ;
    // long nocc_up = nocc / 2; 
    // long nb = nbasis / 2;

    // // nparam = nocc_up * (nb - nocc_up); //paired-doubles
    // // nparam += nocc * (nbasis - nocc); // alpha, beta singles

    // // ovlp.resize(nconn);
    // // d_ovlp.resize(nconn * nparam);
    std::cout << "nconn: " << nconn << ", nparam: " << nparam << "\n";

    AlignedVector<ulong> rdet(nword);
    wfn_.fill_hartreefock_det(nbasis, nocc, &rdet[0]);
    std::cout << "rdet: " ;
    for (std::size_t k = 0; k < nword; k++) {
        std::cout << rdet[k]; }
    std::cout << std::endl;

    if (y == nullptr) {
        std::cerr << "Error: y is a null pointer" << std::endl;
        // return;
    }

    det_ac_inds.resize(nconn);

    for (std::size_t idet = 0; idet != ndet; ++idet) {
    //  std::cout << "\n-----Processing idet = " << idet << "\n";
        y[idet] = 0.0;
        // std::cout << "Accessing det_ptr" << std::endl;

        if (idet >= static_cast<std::size_t>(wfn_.ndet)) {
            std::cerr << "Error: idet (" << idet << ") is out of bounds (ndet: " << wfn_.ndet << ")" << std::endl;
            continue;
        }

        const ulong *det = wfn_.det_ptr(idet);
        if (det == nullptr) {
            std::cerr << "Error: det_ptr returned nullptr for idet = " << idet << std::endl;
            // continue;
        }

        // std::cout << "det: " ;
        for (std::size_t k = 0; k < nword; k++) {
            std::cout << det[k]; }
        std::cout << ",,";

        // Check if given det is the same as the reference det
        bool are_same = true;
        for (std::size_t k = 0; k < nword; ++k) {
            // std::cout << det[k] << " ";
            if (rdet[k] != det[k]) {
                are_same = false;
                break;
            }
        }
        // std::cout << "are_same: " << are_same << std::endl;

        ulong word, hword, pword;
        std::size_t h, p, nexc = 0;

        std::vector<std::size_t> holes;
        std::vector<std::size_t> particles;

        // Container to save annhiliation and creation indices, and sign
        DetExcParamIndx ac_info;
        ac_info.det.resize(nword);
        ac_info.pair_inds.resize(1); // for annhilation indices
        ac_info.single_inds.resize(1); // for creation indices

        // Default values for indices
        ac_info.pair_inds[0] = -1;
        ac_info.single_inds[0] = -1;
        ac_info.sign = 1;

        // Storing current det
        std::memcpy(&ac_info.det[0], &det[0], sizeof(ulong) * nword);

        // Collect holes and particles
        for (std::size_t iword = 0; iword != nword; ++iword)
        {
            word = rdet[iword] ^ det[iword]; //str for excitation
            hword = word & rdet[iword]; //str for hole
            pword = word & det[iword]; //str for particle

            while(hword){
                h = Ctz(hword);
                p = Ctz(pword);
                
                std::size_t hole_idx = h + iword * Size<ulong>();
                std::size_t part_idx = p + iword * Size<ulong>(); // - nocc_up;
                
                holes.push_back(hole_idx);
                particles.push_back(part_idx);

                hword &= ~(1UL << h);
                pword &= ~(1UL << p);

                ++nexc;
            }
        }


        // Sen-o processing
        // Check  if annhiliation and creation operators are in the exop_combinations
        std::pair<std::vector<std::size_t>, std::vector<std::size_t>> key = std::make_pair(holes, particles);
        // std::cout << "key: " << key.first.size() << " " << key.second.size() << std::endl;
        if (!are_same){
            // std::cout << "holes: ";
            // for (std::size_t i : holes) {
            //         std::cout << i << " ";}
            // std::cout << std::endl;

            if (exop_combs.find(key) == exop_combs.end()) {
                
                // std::cout << "particles: ";
                // for (std::size_t i : particles) {
                //     std::cout << i << " ";
                // }
                // std::cout << std::endl;
 
                std::vector<std::size_t> holes_int(holes.begin(), holes.end());
                std::vector<std::size_t> particles_int(particles.begin(), particles.end());
                generate_possible_exops(holes_int, particles_int);
                // std::cout << "\nGenerated possible exops\n";

                std::unordered_map<int, std::vector<std::vector<int>>> inds_multi = exop_combs[key];
                
                // std::cout <<"inds_multi size: " << inds_multi.size() << std::endl;
                // std::cout << "**Inds_multi: ";
                // for (const auto& rank_inds: inds_multi) {
                //     const auto& rank = rank_inds.first;
                //     const std::vector<std::vector<int>> inds = rank_inds.second;
                //     std::cout << rank << ":" << std::endl;
                //     for (const auto& ind : inds) {
                //         std::cout << "{";
                //         for (int i : ind) {
                //             std::cout << i << " ";
                //         }
                //         std::cout << "}\n";
                //     }
                // }
                

                // Occupied indices of HF reference det
                AlignedVector<long> occs(nocc);
                fill_occs(nword, rdet.data(), &occs[0]);

                // Processing sen-o
                for (auto& pair : inds_multi){  
                    auto& exc_order = pair.first;
                    auto& inds_sign_ = pair.second;

                    // std::cout << "exc_order: " << exc_order << std::endl;
                    // std::cout << "inds_sign_ size: " << inds_sign_.size() << std::endl;
                    
                    // last element of inds_sign is the sign of the excitation
                    // auto& sign_ = inds_sign_.back();

                    // std::cout << "Params: ";
                    // for (int i = 0; i < nparam; i++) {
                    //     std::cout << x[i] << " ";
                    // }

                    std::vector<std::vector<int>> selected_rows;
                    // for (auto it = inds_sign_.begin(); it != inds_sign_.end() - 1; ++it) {
                    //     const auto& row = *it;                        
                    for (const auto& row : inds_sign_) {
                        // std::cout << "Processing Row: ";
                        // for (int i : row) {
                        //     std::cout << i << " ";
                        // }
                        // std::cout << std::endl;

                        std::unordered_set<int> trash;
                        bool skip_row = false;
                      
                        // for (const auto& exop_indx : row) {
                        for (auto it = row.begin(); it != row.end() - 1; ++it) {
                            const auto& exop_indx = *it;

                            // Check if exop_indx is a valid index
                            if (exop_indx < 0 || static_cast<std::size_t>(exop_indx) >= ind_exops.size()) {
                                std::cout << "ind_exops.size(): " << ind_exops.size() << std::endl;
                                std::cerr << "Error: exop_indx " << exop_indx << " is out of bounds" << std::endl;
                                continue;
                            }
                            const auto& exop = ind_exops[exop_indx];
                            
                            // std::cout << "exop_indx: " << exop_indx << std::endl;
                            // std::cout << "exop: ";
                            // for (long i : exop) {
                            //     std::cout << i << " ";
                            // }
                            // std::cout << std::endl;

                            if (exop.size() == 2) {
                                // std::cout << "exop size: 2\n";
                                if (trash.find(exop[0]) != trash.end()) {
                                    skip_row = true;
                                    break;
                                }
                                if (exop[0] < static_cast<int>(nbasis / 2)) {
                                    if (std::find(occs.begin(), occs.end(), exop[0] + nbasis / 2) == occs.end()) {
                                        skip_row = true;
                                        break;
                                    } else {
                                        trash.insert(exop[0]);
                                        trash.insert(exop[0] + nbasis / 2);
                                    }
                                } else {
                                    if (std::find(occs.begin(), occs.end(), exop[0] - nbasis / 2) == occs.end()) {
                                        skip_row = true;
                                        break;
                                    } else {
                                        trash.insert(exop[0]);
                                        trash.insert(exop[0] - nbasis / 2);
                                    }
                                }
                            } else {
                                // std::cout << "exop size: != 2\n";
                                for (size_t j = 0; j < exop.size() / 2; ++j) {
                                    if (trash.find(exop[j]) != trash.end()) {
                                        skip_row = true;
                                        break;
                                    } else {
                                        trash.insert(exop[j]);
                                    }
                                }
                                if (skip_row) break;
                            }
                        }
                        if (!skip_row) {
                            selected_rows.push_back(row);
                        }
                    }
                    inds_multi[exc_order] = selected_rows;
                    
                }

                // std::cout << "Inds_multi: ";
                // for (const auto& rank_inds: inds_multi) {
                //     const auto& rank = rank_inds.first;
                //     const std::vector<std::vector<int>> inds = rank_inds.second;
                //     std::cout << rank << ":" << std::endl;
                //     for (const auto& ind : inds) {
                //         std::cout << "{";
                //         for (int i : ind) {
                //             std::cout << i << " ";
                //         }
                //         std::cout << "}\n";
                //     }
                // }
                

                double amplitudes = 0.0;
                amplitudes = product_amplitudes_multi(inds_multi, false, x);
                // double amplitudes = std::get<double>(amplitudes_variant);
                int sign = sign_excite(rdet, holes, particles);
                ac_info.sign = sign;
                y[idet] = sign * amplitudes;

                // std::cout << "Amplitudes: " ;
                std::cout << amplitudes << ",," << sign << std::endl;
                // std::cout << "Sign: " <<;
                


            } else{
                std::unordered_map<int, std::vector<std::vector<int>>> inds_multi = exop_combs[key];
                auto amplitudes = product_amplitudes_multi(inds_multi, false, x);
                int sign = sign_excite(rdet, holes, particles);
                ac_info.sign = sign;
                std::cout << amplitudes << ",," << sign << std::endl;
                y[idet] = sign * amplitudes;
            }

            // Store the annihilation and creation indices and sign
            if (holes.size() > 0 && particles.size() > 0) {
                // std::cout << "Storing annihilation and creation indices\n";
                std::vector<long> holes_long(holes.begin(), holes.end());
                std::vector<long> particles_long(particles.begin(), particles.end());
                ac_info.pair_inds.clear();
                ac_info.single_inds.clear();
                ac_info.pair_inds = holes_long;
                ac_info.single_inds = particles_long;
                
                // Ensure det_ac_inds has enough space
                ensure_struct_size(det_ac_inds, idet+1);
                det_ac_inds[idet] = ac_info;
                // std::cout << "storing done\n";
            }

        } else if (are_same) {
            y[idet] = 1.0;
            det_ac_inds[idet] = ac_info;
        } else {
            y[idet] = 0.0;
            det_ac_inds[idet] = ac_info;
        }
        
        
    }
    
}



    


//---------------------------------------------------------------------------------------------------------------------------------------------------
template <typename T>
void AP1roGeneralizedSenoObjective::generate_combinations(const std::vector<T>& elems, int k, std::vector<std::vector<T>>& result, long nbasis) {
    std::vector<bool> mask(elems.size());
    std::fill(mask.end() - k, mask.end() + k, true);
    do {
        std::vector<T> combination;
        for (std::size_t i = 0; i < elems.size(); ++i) {
            if (mask[i]) combination.push_back(elems[i]);
        }
        if (k == 2) {
            if (combination.size() == 2 && (combination[1] - combination[0] == static_cast<unsigned long> (nbasis))) {
                result.push_back(combination);
            }
        } else {
            result.push_back(combination);
        }
    } while (std::next_permutation(mask.begin(), mask.end()));
}

std::vector<std::pair<int, int>> AP1roGeneralizedSenoObjective::generate_partitions(int e, int max_opairs, bool nvir_pairs) {
    std::vector<std::pair<int, int>> partitions;
    for (int p = 0; p <= std::min(e / 2 , max_opairs); ++p) {
        int s = e - 2 * p;
        if (max_opairs > 0 && nvir_pairs && p) {
            if (e % 2 !=0) { // if e is odd
                partitions.emplace_back(p, s); 
            }
            else if (e %2 == 0 && s <= (max_opairs - p)) {
                partitions.emplace_back(p, s);
            }
        } else if (max_opairs == 0) {
            // if (e % 2 ==0 && s <= max_opairs) {
            partitions.emplace_back(0, s);
            // } else if 
        
        }
    }
    return partitions;
}

void AP1roGeneralizedSenoObjective::generate_excitations(const std::vector<std::size_t>& holes,
    const std::vector<std::size_t>& particles, int excitation_order, std::vector<long>& pair_inds,
    std::vector<long>& single_inds, long nbasis, const NonSingletCI &wfn_) {

    AlignedVector<std::pair<int,int>> occ_pairs, vir_pairs;
    AlignedVector<long> occs_up, occs_dn, virs_up, virs_dn; // , temp_occs;
    for (int i : holes) {
        (i < nbasis ? occs_up : occs_dn).push_back(i);
    }
    for (int a : particles) {
        (a < nbasis ? virs_up : virs_dn).push_back(a);
    }

    // Create an unordered set for fast lookup of occupied down-orbitals
    std::unordered_set<int> occ_dn_set(occs_dn.begin(), occs_dn.end()); 
    std::unordered_set<int> virs_set(particles.begin(), particles.end());    

    // Generate occ_pairs
    for (long i : occs_up) {
        if (occ_dn_set.find(i + nbasis) != occ_dn_set.end()) {
            occ_pairs.push_back({i, i + nbasis});   
            // temp_occs.push_back(i);
            // temp_occs.push_back(i + nbasis);
        }
    }
    // Generate vir_pairs
    for (long a : particles) {
        if (virs_set.find(a + nbasis) != virs_set.end()) {
            vir_pairs.push_back({a, a + nbasis});   
        }
    }

    std::vector<std::pair<int,int>>::size_type max_pairs = occ_pairs.size();
    bool nvir_pairs = false;
    if (max_pairs == vir_pairs.size()) {
        nvir_pairs = true;
    }
    
    auto partitions = generate_partitions(excitation_order, max_pairs, nvir_pairs);

    for (const auto& pair : partitions) {
        const auto& num_pairs = pair.first;
        const auto& num_singles = pair.second; 
        // Step 2: Generate combinations of pairs and singles
        std::vector<std::vector<std::size_t>> hole_pairs, hole_singles;
        std::vector<std::vector<std::size_t>> part_pairs, part_singles;

        // Iterate over all unique combintaions of pairs and singles
        std::vector<std::size_t> used_holes, used_parts;

        if (num_pairs > 0) {
            generate_combinations(holes, 2, hole_pairs, nbasis);
            generate_combinations(particles, 2, part_pairs, nbasis);
            
            for (const auto& hpair_comb : hole_pairs) {
                for (const auto& ppair_comb : part_pairs){
                    if (!hpair_comb.empty() || !ppair_comb.empty()) {
                        long pindx = wfn_.calc_pindex(hpair_comb[0], ppair_comb[0]);

                        // Clear the default values
                        if (pair_inds[0] == -1) {
                            pair_inds.clear();
                        }
                        pair_inds.push_back(pindx);
                        used_holes.insert(used_holes.end(), hpair_comb.begin(), hpair_comb.end());
                        used_parts.insert(used_parts.end(), ppair_comb.begin(), ppair_comb.end());
                    }
                }
            }
        }

        // Now handle single excitations
        if (num_singles  > 0) {
            // Filter holes and particles to exclude used ones
            std::vector<std::size_t> remaining_holes, remaining_particles;

            // Exclude used holes
            std::copy_if(holes.begin(), holes.end(), std::back_inserter(remaining_holes),
                        [&](std::size_t h) { return std::find(used_holes.begin(), used_holes.end(), h) == used_holes.end(); });

            // Exclude used particles
            std::copy_if(particles.begin(), particles.end(), std::back_inserter(remaining_particles),
                        [&](std::size_t p) { return std::find(used_parts.begin(), used_parts.end(), p) == used_parts.end(); });

            generate_combinations(remaining_holes, 1, hole_singles, nbasis);
            generate_combinations(remaining_particles, 1, part_singles, nbasis);

            for (const auto& hsingle_comb : hole_singles) {
                for (const auto& psingle_comb : part_singles) {
                    if (std::find(used_holes.begin(), used_holes.end(), hsingle_comb[0]) == used_holes.end() &&
                        std::find(used_parts.begin(), used_parts.end(), psingle_comb[0]) == used_parts.end()) {
                        // Calculate the index of the single excitation
                        long sindx = wfn_.calc_sindex(hsingle_comb[0], psingle_comb[0]);
                        // Clear the default values
                        if (single_inds[0] == -1) {
                            single_inds.clear();
                        }
                        single_inds.push_back(sindx);
                    }
                }
            }
        }
    
    }
}





void AP1roGeneralizedSenoObjective::init_overlap(const NonSingletCI &wfn_)
{
    long nocc_up = wfn_.nocc / 2; 
    long nbasis = wfn_.nbasis / 2;

    nparam = nocc_up * (nbasis - nocc_up); //paired-doubles
    nparam += wfn_.nocc * (wfn_.nbasis - wfn_.nocc); // beta singles

    // ovlp.resize(nconn);
    // d_ovlp.resize(nconn * nparam);
    det_exc_param_indx.resize(nconn);

    // std::size_t nword = (ulong)wfn_.nword;
    long nb = wfn_.nbasis;
    long nocc = wfn_.nocc;

    for (std::size_t idet = 0; idet != nconn; ++idet)
    {
        AlignedVector<ulong> rdet(nword);
        wfn_.fill_hartreefock_det(nb, nocc, &rdet[0]);
        const ulong *det = wfn_.det_ptr(idet);
        ensure_struct_size(det_exc_param_indx, idet+1);

        // Check if given det is the same as the reference det
        bool are_same = true;
        for (std::size_t k = 0; k < nword; ++k) {
            // std::cout << det[k] << " ";
            if (rdet[k] != det[k]) {
                are_same = false;
                break;
            }
        }
        
        ulong word, hword, pword;
        std::size_t h, p, nexc = 0;

        std::vector<std::size_t> holes;
        std::vector<std::size_t> particles;

        // Collect holes and particles
        for (std::size_t  iword = 0; iword != nword; ++iword)
        {
            word = rdet[iword] ^ det[iword]; //str for excitation
            hword = word & rdet[iword]; //str for hole
            pword = word & det[iword]; //str for particle

            while(hword){
                h = Ctz(hword);
                p = Ctz(pword);
                
                std::size_t hole_idx = h + iword * Size<ulong>();
                std::size_t part_idx = p + iword * Size<ulong>(); // - nocc_up;
                
                holes.push_back(hole_idx);
                particles.push_back(part_idx);

                hword &= ~(1UL << h);
                pword &= ~(1UL << p);

                ++nexc;
            }
        }
        // New container for the excitation information
        DetExcParamIndx exc_info;
        exc_info.det.resize(nword);
        exc_info.pair_inds.resize(1);
        exc_info.single_inds.resize(1);

        // Default values for indices
        exc_info.pair_inds[0] = -1;
        exc_info.single_inds[0] = -1;

        std::memcpy(&exc_info.det[0], &det[0], sizeof(ulong) * nword);

        // Generate excitations if the given det is not the same as the reference det
        if (!are_same) generate_excitations(holes, particles, nexc, exc_info.pair_inds, exc_info.single_inds, nbasis, wfn_);
        
        // Sort the indices for single and pair excitations
        std::sort(exc_info.pair_inds.begin(), exc_info.pair_inds.end());
        std::sort(exc_info.single_inds.begin(), exc_info.single_inds.end());

        // Store the sign of the overall excitation
        int sign = 1;
        if (exc_info.pair_inds[0] != -1 || exc_info.single_inds[0] != -1) {
            sign = sign_excite(rdet, holes, particles);
        }
        exc_info.sign = sign;
        
        // store the excitation information
        det_exc_param_indx[idet] = exc_info;
    }
}


bool AP1roGeneralizedSenoObjective::permanent_calculation(const std::vector<long>& excitation_inds, const double* x, double& permanent) {
    // Ryser's Algorithm
    std::size_t n = static_cast<std::size_t>(std::sqrt(excitation_inds.size()));
    if (n == 0) {permanent = 1.0; return true;}
    if (n == 1) {
        permanent = x[excitation_inds[0]]; 
        return true;
    }
    permanent = 0.0;
    std::size_t subset_count = 1UL << n; // 2^n subsets
    for (std::size_t subset = 0; subset < subset_count; ++subset) {
        double rowsumprod = 1.0;

        for (std::size_t  i = 0; i < n; ++i) {
            double rowsum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                if (subset & (1UL << j)) {
                    // std::cout << "\n rowsum: " << rowsum << std::endl;
                    // std::cout << "excitation_inds[i * n + j]: " << excitation_inds[i * n + j] << std::endl;
                    // std::cout << "x[excitation_inds[i * n + j]]: " << x[excitation_inds[i * n + j]] << std::endl;
                    rowsum += x[excitation_inds[i * n + j]];
                    // std::cout << "rowsum: " << rowsum << std::endl;
                }
            }
            if (std::isnan(rowsum) || std::isinf(rowsum)) {
                std::cerr << "Error: rowsum is invalid (NaN or Inf) at subset " << subset << std::endl;
                return false;    
            }
            rowsumprod *= rowsum;
        }
        if (std::isnan(rowsumprod) || std::isinf(rowsumprod)) {
            std::cerr << "Error: rowsumprod is invalid (NaN or Inf) at subset " << subset << std::endl;
            return false;
        }

        // multiply by the parity of the subset
        permanent += rowsumprod * (1 - ((__builtin_popcount(subset) & 1) << 1));
    }
    // If n (matrix size) is odd, multiply by -1
    permanent *= ((n % 2 == 1) ? -1 : 1);
    
    if (std::isnan(permanent) || std::isinf(permanent)) {
        std::cerr << "Error: permanent is invalid (NaN or Inf)" << std::endl;
        return false;
    }
    return true;
}

double AP1roGeneralizedSenoObjective::compute_derivative(const std::vector<long> excitation_inds, 
    const double* x, std::size_t iparam) {

    // Check if excitation_inds is empty or first element is -1
    if (excitation_inds.empty() || excitation_inds[0] == -1) {return 0.0;}

    // Check if iparam is within excitation_inds
    auto it = std::lower_bound(excitation_inds.begin(), excitation_inds.end(), iparam);
    if (it == excitation_inds.end()) {return 0.0;}


    std::size_t n = static_cast<std::size_t>(std::sqrt(excitation_inds.size()));
    if (n == 0) {return 0.0;}
    if (n == 1) {
        return (excitation_inds[0] == static_cast<double>(iparam)) ? 1.0 : 0.0;
    }

    double derivative = 0.0;
    std::size_t subset_count = 1UL << n; // 2^n subsets
    //Find the position of iparam in excitation_inds
    std::size_t pos = std::distance(excitation_inds.begin(), it);
    // Derive i and j from pos of iparam
    std::size_t i = pos / n;
    std::size_t j = pos % n;

    for (std::size_t subset = 0; subset < subset_count; ++subset) {
        // skip subsets where column j is not included
        if (!(subset & (1UL << j))) continue;

        double rowsumprod = 1.0;

        // Compute product of row sums for rows k != i
        for (std::size_t k = 0; k < n; ++k) {
            if (k == i) continue;

            double rowsum = 0.0;
            
            for (std::size_t l = 0; l < n; ++l) {
                if (subset & (1UL << l)) {
                    rowsum += x[excitation_inds[k * n + l]];                    
                }
            }            
            rowsumprod *= rowsum;
        }
        // Parity adjustment
        int subset_parity = (__builtin_popcount(subset) % 2 == 0) ? 1 : -1;
        derivative += rowsumprod * subset_parity;
    }
    // Final adjustment for the parity of the matrix
    derivative *= ((n % 2 == 1) ? -1 : 1);

    if (std::isnan(derivative) || std::isinf(derivative)) {
        std::cerr << "Error: permanent is invalid (NaN or Inf)" << std::endl;
    }
    return derivative;

}

void AP1roGeneralizedSenoObjective::d_overlap(std::size_t ndet, const double* x, double* y) {
    std::cout << "-------Computing Derivative\n nbasis: " << nbasis << ", ndet: " << ndet << "\n" ;
    for (std::size_t idet = 0; idet< ndet; idet++) {
        // Ensure we have annhiliation and creation indices for this determinant
        if (idet < det_ac_inds.size()) {
            std::cout << "\n**Processing idet: " << idet ;
            const DetExcParamIndx ac_info = det_ac_inds[idet];
            
            std::cout << " " ;
            for (std::size_t k = 0; k < nword; k++) {
                std::cout << ac_info.det[k]; }
            std::cout << std::endl;

            // if reference determinant, set the derivative to zero
            if (ac_info.pair_inds[0] == -1 && ac_info.single_inds[0] == -1) {
                std::cout << "Processing reference determinant\n";
                for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
                    y[ndet * iparam + idet] = 0.0;
                }
                continue;
            } else {
                std::vector<std::size_t> a_inds(ac_info.pair_inds.begin(), ac_info.pair_inds.end()); //annihilation indices
                std::vector<std::size_t> c_inds(ac_info.single_inds.begin(), ac_info.single_inds.end()); //creation indices
                std::pair<std::vector<std::size_t>, std::vector<std::size_t>> key = std::make_pair(a_inds, c_inds);

                // combined_inds.insert(combined_inds.end(), c_inds.begin(), c_inds.end());
                if (exop_combs.find(key) == exop_combs.end()) {
                    generate_possible_exops(a_inds, c_inds);
                }

                auto& inds_multi = exop_combs[key];
                AlignedVector<ulong> det = ac_info.det;
                const int sign = ac_info.sign;
                std::vector<long> occs(nocc);
                fill_occs(nword, det.data(), &occs[0]);

                for (auto& pair : inds_multi) {
                    auto& indices_sign = pair.second;
                    std::vector<std::vector<int>> selected_rows;

                    // std::cout << "exc_order: " << pair.first << std::endl;

                    for (const auto& row : indices_sign) {
                        std::cout << "Row: ";
                        std::cout << "[";
                        for (int i : row) {
                            std::cout << i << " ";
                        }
                        std::cout << "]" << std::endl;

                        std::unordered_set<int> trash;
                        bool skip_row = false;
                      
                        // for (const auto& exop_indx : row) {
                        for (auto it = row.begin(); it != row.end() - 1; ++it) {
                            const auto& exop_index = *it;

                            // Check if exop_indx is a valid index
                            if (exop_index < 0 || static_cast<std::size_t>(exop_index) >= ind_exops.size()) {
                                std::cout << "ind_exops.size(): " << ind_exops.size() << std::endl;
                                std::cerr << "Error: exop_index " << exop_index << " is out of bounds" << std::endl;
                                continue;
                            }
                            const auto& exop = ind_exops[exop_index];

                            // std::cout << "exop_index: " << exop_index << std::endl;
                            // std::cout << "exop: ";
                            // for (long i : exop) {
                            //     std::cout << i << " ";
                            // }
                            // std::cout << std::endl;

                            if (exop.size() == 2) {
                                if (trash.find(exop[0]) != trash.end()) {
                                    // std::cout << "exop[0] already in trash\n";
                                    skip_row = true;
                                    break;
                                }
                                if (exop[0] < static_cast<int>(nbasis / 2)) {
                                    if (std::find(occs.begin(),  occs.end(), exop[0] + static_cast<int>(nbasis / 2)) == occs.end()) {
                                        skip_row = true;
                                        // std::cout << "exop[0] + nbasis / 2 not in occs\n";
                                        break;
                                    } else {
                                        // std::cout << "exop[0] + nbasis / 2 in occs\n";
                                        trash.insert(exop[0]);
                                        trash.insert(exop[0] + static_cast<int>(nbasis / 2));
                                    }
                                } else {
                                    if (std::find(occs.begin(), occs.end(), exop[0] - static_cast<int>(nbasis / 2)) == occs.end()) {
                                        skip_row = true;
                                        // std::cout << "exop[0] - nbasis / 2 not in occs\n";
                                        break;
                                    } else {
                                        // std::cout << "exop[0] - nbasis / 2 in occ\n";
                                        trash.insert(exop[0]);
                                        trash.insert(exop[0] - static_cast<int>(nbasis / 2));
                                    }
                                }
                            } else {
                                for (std::size_t j = 0; j < exop.size() / 2; ++j) {
                                    if (trash.find(exop[j]) != trash.end()) {
                                        std::cout << "exop[j] already in trash\n";
                                        skip_row = true;
                                        break;
                                    } else {
                                        trash.insert(exop[j]);
                                    }
                                }
                                if (skip_row) break;
                            }
                        }
                        if (!skip_row) {
                            selected_rows.push_back(row);
                        }
                    }
                    
                    std::cout << "Selected: ";
                    for (const auto& row : selected_rows) {
                        std::cout << "[";
                        for (int i : row) {
                            std::cout << i << " ";
                        }
                        std::cout << "]" << std::endl;
                    }
                    std::cout << std::endl;

                    indices_sign = selected_rows;
                    inds_multi[pair.first] = indices_sign;
                }
                std::vector<double> damplitudes = product_ampli_multi_deriv(inds_multi, true, x);
                for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
                    y[ndet * iparam + idet] = sign * damplitudes[iparam];
                    if (damplitudes[iparam] != 0.0) {
                        std::cout <<  "iparam: " << iparam << ", dampli: " << damplitudes[iparam] << ", sign: " << sign << std::endl;
                    }
                }  

            }
        }
    }
}


//-------------------------------------FIRST APPROACH BEGIN-------------------------------------------------------------------------
// void AP1roGeneralizedSenoObjective::overlap(std::size_t ndet, const double *x, double *y) {
//     p_permanent.resize(nconn);
//     s_permanent.resize(nconn);
//     //print x
//     for (std::size_t i = 0; i < nparam; ++i) {
//         std::cout << x[i] << " ";
//     }
//     std::cout << "\n" << std::endl;


//     for (std::size_t idet = 0; idet != ndet; ++idet) {
    
//         if (idet < det_exc_param_indx.size()) {
//             // Access the excitation parameter indices
//             const DetExcParamIndx& exc_info = det_exc_param_indx[idet];
//             double pair_permanent = 1.0;
//             double single_permanent = 1.0;

//             if (exc_info.pair_inds[0] != -1) {
//                 if (!permanent_calculation(exc_info.pair_inds, x, pair_permanent)) {
//                     std::cerr << "Error calculating pair_permanent for idet" << idet << "\n" << std::endl;
//                 }
//             }
            
//             if (exc_info.single_inds[0] != -1) {
//                 if (!permanent_calculation(exc_info.single_inds, x, single_permanent)) {
//                     std::cerr << "Error calculating single_permanent for idet " << idet << "\n" << std::endl;
//                 }
//             }
//             // If given det is not allowed in the wfn, set the permanents to zero
//             // idet = 0 is the reference HF determinant 
//             if (exc_info.pair_inds[0] == -1 && exc_info.single_inds[0] == -1 && idet != 0) {
//                 pair_permanent = 0.0;
//                 single_permanent = 0.0;
//             }

//             p_permanent[idet] = pair_permanent;
//             s_permanent[idet] = single_permanent;


//             if (y != nullptr && idet < ndet) {
//                 y[idet] = exc_info.sign * pair_permanent * single_permanent;
//                 if (idet < 225) {
//                     std::cout << idet << " " << exc_info.det[0] << " " << y[idet] << " " << exc_info.sign << std::endl;
//                 }
//             }else {
//                 std::cerr << "y is nullptr or idet is out of bounds" << std::endl;
//             }
//         } else {
//             std::cout << "idet" <<  idet << " not found in storage" << std::endl;
//             y[idet] = 0.0;
//             s_permanent[idet] = 0.0;
//             p_permanent[idet] = 0.0;
//         }
//     }
// }


// void AP1roGeneralizedSenoObjective::d_overlap(const size_t ndet, const double *x, double *y){

//     for (std::size_t idet = 0; idet != ndet; ++idet)   {
//         // Ensure we have the excitation parameters for this determinant
//         if (idet < det_exc_param_indx.size()) {
//             const DetExcParamIndx exc_info = det_exc_param_indx[idet];
//             double pair_permanent = 1.0;
//             double single_permanent = 1.0;
//             if (idet < s_permanent.size() && idet < p_permanent.size()) {
//                 pair_permanent = p_permanent[idet];
//                 single_permanent = s_permanent[idet];
//             }
//             // else {
//             //     if (exc_info.pair_inds[0] != -1) {
//             //         if (!permanent_calculation(exc_info.pair_inds, x, pair_permanent)) {
//             //             std::cerr << "Error calculating pair_permanent for idet" << idet << std::endl;
//             //         }
//             //     }

//             //     if (exc_info.single_inds[0] != -1) {
//             //         if (!permanent_calculation(exc_info.single_inds, x, single_permanent)) {
//             //             std::cerr << "Error calculating single_permanent for idet " << idet << std::endl;
//             //         }
//             //     }
                
//             // }
            
//             for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
//                 double dpair = 0.0;
//                 double dsingle = 0.0;
//                 if (exc_info.single_inds[0] != -1) {
//                     dsingle = compute_derivative(exc_info.single_inds, x, iparam);
//                 }
                
//                 if (exc_info.pair_inds[0] != -1) {
//                     dpair = compute_derivative(exc_info.pair_inds, x, iparam);
//                 }
//                 // std::cout << "\nidet: " << idet << ", det: " << exc_info.det[0] << std::endl;
//                 // std::cout << "single_inds: " << exc_info.single_inds[0] << ", pair_inds: " << exc_info.pair_inds[0] << std::endl;
//                 // std::cout << "single_permanent: " << single_permanent << ", pair_permanent: " << pair_permanent << std::endl;
//                 // std::cout <<  "wrt iparam: " << iparam << ", dpair: " << dpair << ", dsingle: " << dsingle << std::endl;
//                 // std::cout << "idet: " << idet << " deriv: " << dpair * single_permanent + dsingle * pair_permanent << std::endl;
//                 y[ndet * iparam + idet] = dpair * single_permanent + dsingle * pair_permanent;
//             }
//         }
//         else {
//             for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
//                 y[ndet * iparam + idet] = 0.0;
//             }
//         }
//     }
// }
//-------------------------------------FIRST APPROACH END-------------------------------------------------------------------------


} // namespace pyci
