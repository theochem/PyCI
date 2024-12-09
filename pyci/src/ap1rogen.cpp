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
#include <unordered_map>
#include <iostream>
namespace pyci {

// See apig.cpp for reference

// Constructor with Raw Pointers

// call to the base class constructor fullciwfn with the provided parameters
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(const SparseOp &op_, const NonSingletCI &wfn_,
                                 const std::size_t n_detcons_,
                                 const long *idx_detcons_,
                                 const double *val_detcons_,
                                 const std::size_t n_paramcons_,
                                 const long *idx_paramcons_,
                                 const double *val_paramcons_)
: Objective<NonSingletCI>::Objective(op_, wfn_, n_detcons_, idx_detcons_, val_detcons_, n_paramcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_); 
}
// call to initizlize the overlap related data

// Constructor with pybind11 objects
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(const SparseOp &op_, const NonSingletCI &wfn_,
                                 const pybind11::object idx_detcons_,
                                 const pybind11::object val_detcons_,
                                 const pybind11::object idx_paramcons_,
                                 const pybind11::object val_paramcons_)
: Objective<NonSingletCI>::Objective(op_, wfn_, idx_detcons_, val_detcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_);
}
// Copy Constructor
// obj is the constant reference to another object to be copied
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(const AP1roGeneralizedSenoObjective &obj)
: Objective<NonSingletCI>::Objective(obj), nexc_list(obj.nexc_list), det_exc_param_indx(obj.det_exc_param_indx)
{
    return;
}

// Move constructor
// obj is the rvalue reference to another object to be moved
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(AP1roGeneralizedSenoObjective &&obj) noexcept
: Objective<NonSingletCI>::Objective(obj), nexc_list(std::move(obj.nexc_list)), det_exc_param_indx(std::move(obj.det_exc_param_indx))
{
    return;
}

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
        // std::cout << "p, s: " << p << " " << s << std::endl;
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
    std::cout << "Inside generate_excitations" << std::endl;
    AlignedVector<std::pair<int,int>> occ_pairs, vir_pairs;
    AlignedVector<long> occs_up, occs_dn, virs_up, virs_dn, temp_occs;
    for (int i : holes) {
        // std::cout << "i: " << i << ", nbasis: " << nbasis << std::endl;
        (i < nbasis ? occs_up : occs_dn).push_back(i);
    }
    for (int a : particles) {
        // std::cout << "a: " << a << ", nbasis: " << nbasis << std::endl;
        (a < nbasis ? virs_up : virs_dn).push_back(a);
    }

    std::cout << "up dn set created\n";
    // Create an unordered set for fast lookup of occupied down-orbitals
    std::unordered_set<int> occ_dn_set(occs_dn.begin(), occs_dn.end()); 
    std::unordered_set<int> virs_set(particles.begin(), particles.end());    
    // Generate occ_pairs and vir_pairs
    for (long i : occs_up) {
        if (occ_dn_set.find(i + nbasis) != occ_dn_set.end()) {
            occ_pairs.push_back({i, i + nbasis});   
            temp_occs.push_back(i);
            temp_occs.push_back(i + nbasis);
        }
    }
    // std::cout << "occ_pairs created\n";
    for (long a : particles) {
        if (virs_set.find(a + nbasis) != virs_set.end()) {
            vir_pairs.push_back({a, a + nbasis});   
        }
    }
    // std::cout << "vir_pairs created\n";
        
    std::cout << "\nInside generate_excitations" << std::endl;
    std::cout << "holes:" ;
    for (const auto& hole : holes) {
        std::cout << hole << " ";
    }
    std::cout << std::endl;
    std::cout << "particles:" ;
    for (const auto& particle : particles) {
        std::cout << particle << " ";
    }
    std::cout << std::endl;
    int max_pairs = occ_pairs.size();
    bool nvir_pairs = false;
    if (max_pairs == vir_pairs.size()) {
        nvir_pairs = true;
        std::cout << "nvir_pairs: " << nvir_pairs << std::endl;
    }
    
    std::cout << "exci order: " << excitation_order << ", max_pairs: " << max_pairs << std::endl;
    auto partitions = generate_partitions(excitation_order, max_pairs, nvir_pairs);
    std::cout << "Generated partitions" << std::endl;
    
    std::cout << "Partitions: " << std::endl;
    for (const auto& [num_pairs, num_singles] : partitions) {
        std::cout << num_pairs << " " << num_singles << std::endl;
    }

    for (const auto& [num_pairs, num_singles] : partitions) {
        // Step 2: Generate combinations of pairs and singles
        std::cout << "Generating combinations of pairs and singles" << std::endl;
        std::cout << "num_pairs: " << num_pairs << ", num_singles: " << num_singles << std::endl;
        std::vector<std::vector<std::size_t>> hole_pairs, hole_singles;
        std::vector<std::vector<std::size_t>> part_pairs, part_singles;

        generate_combinations(holes, 2, hole_pairs, nbasis);
        generate_combinations(particles, 2, part_pairs, nbasis);



        // Iterate over all unique combintaions of pairs and singles
        std::vector<std::size_t> used_holes, used_parts;
        for (const auto& hpair_comb : hole_pairs) {
            for (const auto& ppair_comb : part_pairs){
                // if (used_holes.empty() || std::none_of(hpair_comb.begin(), hpair_comb.end(),
                //         [&](std::size_t h) { return std::find(used_holes.begin(), used_holes.end(), h) != used_holes.end(); })) {
                if (!hpair_comb.empty() || !ppair_comb.empty()) {
                    long pindx = wfn_.calc_pindex(hpair_comb[0], ppair_comb[0]);
                    std::cout << "Pair index: " << pindx << std::endl;
                    std::cout << "hpair_comb: " << hpair_comb[0] << " " << hpair_comb[1] << std::endl;
                    std::cout << "ppair_comb: " << ppair_comb[0] << " " << ppair_comb[1] << std::endl;
                    pair_inds.push_back(pindx);
                    used_holes.insert(used_holes.end(), hpair_comb.begin(), hpair_comb.end());
                    used_parts.insert(used_parts.end(), ppair_comb.begin(), ppair_comb.end());
                } else {
                    pair_inds.clear();
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
                    // if (used_holes.empty() || std::none_of(hsingle_comb.begin(), hsingle_comb.end(),
                    //         [&](std::size_t h) { return std::find(used_holes.begin(), used_holes.end(), h) != used_holes.end(); })) {
                    // Ensure the selected single excitations do not reuse already used indices
                    if (std::find(used_holes.begin(), used_holes.end(), hsingle_comb[0]) == used_holes.end() &&
                        std::find(used_parts.begin(), used_parts.end(), psingle_comb[0]) == used_parts.end()) {
                    long sindx = wfn_.calc_sindex(hsingle_comb[0], psingle_comb[0]);
                    single_inds.push_back(sindx);
                    std::cout << "Single index: " << sindx << std::endl;
                    std::cout << "h: " << hsingle_comb[0] <<  "p: " << psingle_comb[0] << std::endl;
                        // used_holes.push_back(hsingle_comb[0]);
                        // used_parts.push_back(psingle_comb[0]);
                    }
                }
            }
        }
        else {
            single_inds.clear();
        }
    
    }
}

void AP1roGeneralizedSenoObjective::init_overlap(const NonSingletCI &wfn_)
{
    std::cout << "Inside init_overlap" << std::endl;
    // Initialize your class-specific variables here
    // init_Overlap objective for the AP1roGSDspin_sen-o 
    std::cout << "wfn_.nocc_up: " << wfn_.nocc_up << "wfn_.nvir_up" << wfn_.nvir_up << std::endl;
    std::cout << "wfn_.nocc: " << wfn_.nocc << "wfn_.nvir" << wfn_.nvir << std::endl;
    long nocc_up = wfn_.nocc / 2; 
    long nbasis = wfn_.nbasis / 2;
    // long nvir_up = nbasis - nocc_up;

    nparam = nocc_up * (nbasis - nocc_up); //paired-doubles
    std::cout << "nparam (doubles): " << nparam << std::endl;
    nparam += wfn_.nocc * (2* nbasis - wfn_.nocc); // beta singles
    std::cout << "nparam (doubles + S_alpha + S_beta): " << nparam << std::endl;
    det_exc_param_indx = wfn_.det_exc_param_indx;
    std::cout << "Size of det_exc_param_indx: " << det_exc_param_indx.size() << std::endl;

    // ovlp.resize(wfn_.ndet);
    // d_ovlp.resize(wfn_.ndet * nparam);

    std::size_t nword = (ulong)wfn_.nword;
    long nb = wfn_.nbasis;
    long nocc = wfn_.nocc;
    // std::unordered_map<std::vector<ulong>, DetExcParamIndx> det_map;

    // Populate the hash map (assume det_exc_param_indx is iterable)
    // for (const auto& exc_info : det_exc_param_indx) {
    //     det_map[exc_info.det] = exc_info; // Use exc_info.det as the key
    // }

    for (std::size_t idet = 0; idet != nconn; ++idet)
    {
        AlignedVector<ulong> rdet(nword);
        wfn_.fill_hartreefock_det(nb, nocc, &rdet[0]);

        // std::cout << "After fill_hartreefock_det rdet:" ;
        // for (std::size_t k = 0; k < nword; ++k) {
        //         std::cout << rdet[k] << " ";
        // }
        // std::cout << std::endl;
        //wfn_.print_vector("rdet", rdet);

        const ulong *det = wfn_.det_ptr(idet);

        // std::vector<ulong> det_vector(det, det + nword);
        // auto it = det_map.find(det_vector);
        // if (it != det_map.end()) {
        //     std::cout << "Found det in det_map" << std::endl;
        //     // std::cout << "Det: " << det_vector << std::endl;
        //     // std::cout << "DetExcParamIndx: " << it->second << std::endl;
        // std::cout << "from storage: " << det_exc_param_indx[idet].det << std::endl;
        ensure_struct_size(det_exc_param_indx, idet+1);

        if (!det_exc_param_indx[idet].det.empty()) {
            std::cout << "\nidet: " << idet <<  " found in det_map" << std::endl; 
            std::cout << "Det from wfn: ";
            for (std::size_t k = 0; k < nword; ++k) {
                std::cout << det[k] << " ";
            }
            std::cout << std::endl;
            std::cout << "det_exc_param_indx[idet].det: ";
            for (std::size_t k = 0; k < nword; ++k) {
                std::cout << det_exc_param_indx[idet].det[k] << " ";
            }
            std::cout << std::endl;
            
        } else {
            std::cout << "\nDet: " ;
            for (std::size_t k = 0; k < nword; ++k) {
                std::cout << det[k] << " ";
            }
            std::cout << ", at iDet: " << idet << ", not found in det_exc_param_indx" << std::endl;
            
            ulong word, hword, pword;
            std::size_t h, p, nexc = 0;

            std::vector<std::size_t> holes;
            std::vector<std::size_t> particles;

            // Collect holes and particles
            for (std::size_t iword = 0; iword != nword; ++iword)
            {
                word = rdet[iword] ^ det[iword]; //str for excitation
                hword = word & rdet[iword]; //str for hole
                pword = word & det[iword]; //str for particle
                std::cout << "word: " << word << std::endl;
                std::cout << "hword: " << hword << std::endl;
                std::cout << "pword: " << pword << std::endl;
                while(hword){
                    h = Ctz(hword);
                    p = Ctz(pword);
                    
                    std::size_t hole_idx = h + iword * Size<ulong>();
                    std::size_t part_idx = p + iword * Size<ulong>(); // - nocc_up;
                    
                    holes.push_back(hole_idx);
                    particles.push_back(part_idx);

                    hword &= ~(1UL << h);
                    pword &= ~(1UL << p);
                    // std::cout << "word: " << word << std::endl;
                    // std::cout << "hword: " << hword << std::endl;
                    // std::cout << "pword: " << pword << std::endl;
                    // std::cout << "h: " << h << ", hole_idx: " << hole_idx << std::endl;
                    // std::cout << "p: " << p << ", part_idx: " << part_idx << std::endl;
                    ++nexc;
                }
            }
            //nexc_list[idet] = nexc;
            std::cout << "nexc: " << nexc << std::endl;
            DetExcParamIndx exc_info;
            exc_info.det.resize(nword);
            std::memcpy(&exc_info.det[0], &det[0], sizeof(ulong) * nword);
            // std::cout << "\nCopied det" << std::endl;
            generate_excitations(holes, particles, nexc, exc_info.pair_inds, exc_info.single_inds, nbasis, wfn_);
            std::cout << "Generated excitations" << std::endl;
            std::cout << "size of det_exc_param_indx: " << det_exc_param_indx.size() << std::endl;
            det_exc_param_indx[idet] = exc_info;
        }
    }
}


double AP1roGeneralizedSenoObjective::permanent_calculation(const std::vector<long>& excitation_inds, const double* x) {
    std::size_t num_excitations = excitation_inds.size();
    if (num_excitations == 0) return 1.0;
    
    double permanent = 0.0;
    std::size_t subset_count = 1UL << num_excitations;

    for (std::size_t subset = 0; subset < subset_count; ++subset) {
        double rowsumprod = 1.0;

        for (std::size_t  j = 0; j < num_excitations; ++j) {
            double rowsum = 0.0;
            for (std::size_t j = 0; j < num_excitations; ++j) {
                if (subset & (1UL << j)) {
                    rowsum += x[excitation_inds[j]];
                }
            }
            rowsumprod *= rowsum;
        }
        permanent += rowsumprod * (1 - ((__builtin_popcount(subset) & 1) << 1));
    }
    permanent *= ((num_excitations % 2 == 1) ? -1 : 1);
    return permanent;
}

// void AP1roGeneralizedSenoObjective::overlap(const NonSingletCI &wfn_, const double *x, double *y) {
void AP1roGeneralizedSenoObjective::overlap(std::size_t ndet, const double *x, double *y) {

    for (std::size_t idet = 0; idet != ndet; ++idet) {
        
        //Retrieve the DetExcParamIndx object from the hash map
        // const ulong* det = det_ptr(idet);

        // // std::vector<ulong> det_vector(det, det + wfn_.nword);
        // // Find corresponding DetExcParamIndx for the current determinant
        // std::unordered_map<std::vector<ulong>, DetExcParamIndx> det_map;
        // // Populate the hash map (assume det_exc_param_indx is iterable)
        // for (const auto& exc_info : det_exc_param_indx) {
        //     det_map[exc_info.det] = exc_info; // Use exc_info.det as the key
        // }

        // auto it = det_map.find(det_vector);

        if (idet < det_exc_param_indx.size()) {
            // Access the excitation parameter indices
            const DetExcParamIndx& exc_info = det_exc_param_indx[idet];

            double pair_permanent = permanent_calculation(exc_info.pair_inds, x);
            double single_permanent = permanent_calculation(exc_info.single_inds, x);

            y[idet] = pair_permanent * single_permanent;
        } else {
            // std::cout << "Det" << det_vector << " not found in det_map" << std::endl;
            y[idet] = 0.0;

        }
    }
}


double AP1roGeneralizedSenoObjective::compute_derivative(
    const std::vector<long>& excitation_inds, 
    const double* x,
    std::size_t excitation_idx) {

    double derivative = 0.0;

    std::vector<double> modified_x(x, x + nparam);
    modified_x[excitation_inds[excitation_idx]] = 1.0;
    derivative = permanent_calculation(excitation_inds, modified_x.data());

    return derivative;
}


// void AP1roGeneralizedSenoObjective::d_overlap(const NonSingletCI &wfn_, const size_t ndet, const double *x, double *y){
void AP1roGeneralizedSenoObjective::d_overlap(const size_t ndet, const double *x, double *y){

    // Loop over each determinant
    for (std::size_t idet = 0; idet != ndet; ++idet)
    {
        // Retrieve the corresponding determinant
        // const ulong* det = det_ptr(idet);

        // std::vector<ulong> det_vector(det, det + wfn_.nword);
        // Find corresponding DetExcParamIndx for the current determinant
        // std::unordered_map<std::vector<ulong>, DetExcParamIndx> det_map;
        // // Populate the hash map (assume det_exc_param_indx is iterable)
        // for (const auto& exc_info : det_exc_param_indx) {
        //     det_map[exc_info.det] = exc_info; // Use exc_info.det as the key
        // }
        
        // Find corresponding DetExcParamIndx for the current determinant
        // auto it = det_map.find(det_vector);
        
        // Ensure we have the excitation parameters for this determinant
        if (idet < det_exc_param_indx.size()) {
            const DetExcParamIndx& exc_info = det_exc_param_indx[idet];
            
            // Loop over each parameter (paired and single excitations)
            std::size_t param_index = 0;

            // Derivative for paired excitations
            for (std::size_t i = 0; i < exc_info.pair_inds.size(); ++i) {
                // Get the excitation index for the pair
                const std::size_t excitation_idx = exc_info.pair_inds[i];

                // Compute the derivative of the permanent with respect to this excitation
                double derivative = compute_derivative(exc_info.pair_inds, x, excitation_idx);
                
                // Store the result in the output vector d_ovlp (size ndet * nparam)
                y[idet * nparam + param_index] = derivative;

                ++param_index;
            }

            // Derivative for single excitations
            for (std::size_t i = 0; i < exc_info.single_inds.size(); ++i) {
                // Get the excitation index for the single
                const std::size_t excitation_idx = exc_info.single_inds[i];

                // Compute the derivative of the permanent with respect to this excitation
                double derivative = compute_derivative(exc_info.single_inds, x, excitation_idx);

                // Store the result in the output vector d_ovlp (size ndet * nparam)
                y[idet * nparam + param_index] = derivative;

                ++param_index;
            }
        }
        else {
            std::cout << "Determinant " << idet << " not found in det_map" << std::endl;
            // Set all derivatives to zero if determinant is not found
            for (std::size_t i = 0; i < nparam; ++i) {
                y[idet * nparam + i] = 0.0;
            }
        }
    }
}


} // namespace pyci



// std::vector<std::pair<std::size_t, std::size_t>> occ_pairs;
//             for (std::size_t hole in holes) {
//                 std::size_t conjugate = hole + wfn_.nbasis / 2;
//                 if(std::find(holes.begin(), holes.end(), conjugate) != holes.end()) {
//                     occ_pairs.push_back(std::make_pair(hole, conjugate));
//                     // exc_info.pair_inds.push_back(wfn_.nvir_up * hole);
//                 }
//             }

//             std::vector<std::size_t, std::size_t> vir_pairs;
//             for (std::size_t part in particles) {
//                 std::size_t conjugate = part + wfn_.nbasis / 2;
//                 if(std::find(particles.begin(), particles.end(), conjugate) != particles.end()) {
//                     vir_pairs.push_back(std::make_pair(part, conjugate));
//                     // exc_info.pair_inds.push_back(wfn_.nvir_up * part);
//                 }
//             }

//             for (const auto& pair : occ_pairs) {
//                for (const auto& vir_pair : vir_pairs) {
//                    exc_info.pair_inds.push_back(wfn_.nvir_up * pair.first + vir_pair.first);
//                    exc_info.pair_inds.push_back(wfn_.nvir_up * pair.second + vir_pair.second);
//                }
//             }



        // std::cout << "Generated hole_pairs: " << std::endl; 
        // for (const auto& hole_pair : hole_pairs) {
        //     for (const auto& elem: hole_pair) {
        //         std::cout << elem << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Limit the number of pairs and singles to the requested partition
        // hole_pairs.resize(std::min(hole_pairs.size(), static_cast<std::size_t>(num_pairs)));
        // hole_singles.resize(std::min(hole_singles.size(), static_cast<std::size_t>(num_singles)));
        // part_pairs.resize(std::min(part_pairs.size(), static_cast<std::size_t>(num_pairs)));
        // part_singles.resize(std::min(part_singles.size(), static_cast<std::size_t>(num_singles)));

        // Match pairs and singles
        // for (const auto& hole_pair : hole_pairs) {
        //     for (const auto& part_pair : part_pairs) {
        //         std::cout << "hole_pair: " << hole_pair[0] << " " << hole_pair[1] << std::endl;
        //         std::cout << "part_pair: " << part_pair[0] << " " << part_pair[1] << std::endl;
        //         // Check constraints
        //         pair_inds.push_back(nvir_up * hole_pair[0] + part_pair[0]);
        //         //pair_inds.push_back(wfn_.nvir_up * hole_pair[1] + part_pair[1]);
        //     }
        // }

        // for (const auto& hole_single : hole_singles) {
        //     for (const auto& part_single : part_singles) {
        //         // Check constraints
        //         std::cout << "h: " << hole_single[0] << ", p: " << part_single[0] << std::endl;
        //         single_inds.push_back(nvir_up * nocc / 2 + hole_single[0] * nvir + part_single[0]);
        //     }
        // }
