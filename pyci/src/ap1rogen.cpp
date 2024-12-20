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
#include <limits>
#include <cmath>
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
: Objective<NonSingletCI>::Objective(obj), det_exc_param_indx(obj.det_exc_param_indx), nexc_list(obj.nexc_list)
{
    return;
}

// Move constructor
// obj is the rvalue reference to another object to be moved
AP1roGeneralizedSenoObjective::AP1roGeneralizedSenoObjective(AP1roGeneralizedSenoObjective &&obj) noexcept
: Objective<NonSingletCI>::Objective(obj), det_exc_param_indx(obj.det_exc_param_indx), nexc_list(std::move(obj.nexc_list))
{
    return;
}

template <typename T>
void AP1roGeneralizedSenoObjective::generate_combinations(const std::vector<T>& elems, int k, std::vector<std::vector<T>>& result, long nbasis) {
    std::vector<bool> mask(elems.size());
    std::fill(mask.end() - k, mask.end() + k, true);
    // std::fill(mask.begin(), mask.begin() + k, true);
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
        // //std::cout << "p, s: " << p << " " << s << std::endl;
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
    //std::cout << "Inside generate_excitations" << std::endl;
    AlignedVector<std::pair<int,int>> occ_pairs, vir_pairs;
    AlignedVector<long> occs_up, occs_dn, virs_up, virs_dn, temp_occs;
    for (int i : holes) {
        // //std::cout << "i: " << i << ", nbasis: " << nbasis << std::endl;
        (i < nbasis ? occs_up : occs_dn).push_back(i);
    }
    for (int a : particles) {
        // std::cout << "a: " << a << ", nbasis: " << nbasis << std::endl;
        (a < nbasis ? virs_up : virs_dn).push_back(a);
    }

    //std::cout << "up dn set created\n";
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
        
    // std::cout << "\nInside generate_excitations" << std::endl;
    // std::cout << "holes:" ;
    // for (const auto& hole : holes) {
    //     std::cout << hole << " ";
    // // }
    // std::cout << std::endl;
    // std::cout << "particles:" ;
    // for (const auto& particle : particles) {
    //     std::cout << particle << " ";
    // }
    // std::cout << std::endl;

    std::vector<std::pair<int,int>>::size_type max_pairs = occ_pairs.size();
    bool nvir_pairs = false;
    if (max_pairs == vir_pairs.size()) {
        nvir_pairs = true;
        std::cout << "nvir_pairs: " << nvir_pairs << std::endl;
    }
    
    // std::cout << "exci order: " << excitation_order << ", max_pairs: " << max_pairs << std::endl;
    auto partitions = generate_partitions(excitation_order, max_pairs, nvir_pairs);
    // std::cout << "Generated partitions" << std::endl;
    
    // std::cout << "Partitions: " << std::endl;
    // for (const auto& pair : partitions) {
    //     std::cout << "num_pairs: " << pair.first << ", num_singles: " << pair.second << std::endl;
    // }

    for (const auto& pair : partitions) {
        const auto& num_pairs = pair.first;
        const auto& num_singles = pair.second; 
        // Step 2: Generate combinations of pairs and singles
        // std::cout << "Generating combinations of pairs and singles" << std::endl;
        // std::cout << "num_pairs: " << num_pairs << ", num_singles: " << num_singles << std::endl;
        std::vector<std::vector<std::size_t>> hole_pairs, hole_singles;
        std::vector<std::vector<std::size_t>> part_pairs, part_singles;

        // Iterate over all unique combintaions of pairs and singles
        std::vector<std::size_t> used_holes, used_parts;

        if (num_pairs > 0) {
            generate_combinations(holes, 2, hole_pairs, nbasis);
            generate_combinations(particles, 2, part_pairs, nbasis);
            // std::cout << "Generated hole_pairs" << std::endl;


            
            for (const auto& hpair_comb : hole_pairs) {
                for (const auto& ppair_comb : part_pairs){
                    // std::cout << "Handling pair excitations" << std::endl;
                    // if (used_holes.empty() || std::none_of(hpair_comb.begin(), hpair_comb.end(),
                    //         [&](std::size_t h) { return std::find(used_holes.begin(), used_holes.end(), h) != used_holes.end(); })) {
                    if (!hpair_comb.empty() || !ppair_comb.empty()) {
                        long pindx = wfn_.calc_pindex(hpair_comb[0], ppair_comb[0]);
                        // std::cout << "Pair index: " << pindx << std::endl;
                        // std::cout << "hpair_comb: " << hpair_comb[0] << " " << hpair_comb[1] << std::endl;
                        // std::cout << "ppair_comb: " << ppair_comb[0] << " " << ppair_comb[1] << std::endl;
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
            // std::cout << "Handling single excitations" << std::endl;
            // Filter holes and particles to exclude used ones
            std::vector<std::size_t> remaining_holes, remaining_particles;

            // Exclude used holes
            std::copy_if(holes.begin(), holes.end(), std::back_inserter(remaining_holes),
                        [&](std::size_t h) { return std::find(used_holes.begin(), used_holes.end(), h) == used_holes.end(); });

            // Exclude used particles
            std::copy_if(particles.begin(), particles.end(), std::back_inserter(remaining_particles),
                        [&](std::size_t p) { return std::find(used_parts.begin(), used_parts.end(), p) == used_parts.end(); });

            // std::sort(remaining_holes.begin(), remaining_holes.end());
            // std::sort(remaining_particles.begin(), remaining_particles.end());
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
                        if (single_inds[0] == -1) {
                            single_inds.clear();
                        }
                        single_inds.push_back(sindx);
                        // std::cout << "Single index: " << sindx << std::endl;
                        // std::cout << "h: " << hsingle_comb[0] <<  ", p: " << psingle_comb[0] << std::endl;
                        // used_holes.push_back(hsingle_comb[0]);
                        // used_parts.push_back(psingle_comb[0]);
                    }
                }
            }
        }
        // std::cout << "Generated single indices" << std::endl;
        // for (const auto& sid : single_inds) {
        //     std::cout << sid << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "Generated pair indices" << std::endl;
        // for (const auto& pid : pair_inds) {
        //     std::cout << pid << " ";
        // }
    
    }
}

void AP1roGeneralizedSenoObjective::init_overlap(const NonSingletCI &wfn_)
{
    // default_value = std::numeric_limits<double>::quiet_NaN();
    // std::cout << "Inside init_overlap" << std::endl;
    // Initialize your class-specific variables here
    // init_Overlap objective for the AP1roGSDspin_sen-o 
    // std::cout << "wfn_.nocc_up: " << wfn_.nocc_up << "wfn_.nvir_up" << wfn_.nvir_up << std::endl;
    // std::cout << "wfn_.nocc: " << wfn_.nocc << "wfn_.nvir" << wfn_.nvir << std::endl;
    long nocc_up = wfn_.nocc / 2; 
    long nbasis = wfn_.nbasis / 2;
    // long nvir_up = nbasis - nocc_up;

    nparam = nocc_up * (nbasis - nocc_up); //paired-doubles
    // std::cout << "nparam (doubles): " << nparam << std::endl;
    nparam += wfn_.nocc * (wfn_.nbasis - wfn_.nocc); // beta singles
    // std::cout << "nparam (doubles + S_alpha + S_beta): " << nparam << std::endl;

    

    ovlp.resize(nconn);
    d_ovlp.resize(nconn * nparam);
    det_exc_param_indx.resize(nconn);
    // std::cout << "Size of d_ovlp: " << d_ovlp.size() << std::endl;

    std::size_t nword = (ulong)wfn_.nword;
    long nb = wfn_.nbasis;
    long nocc = wfn_.nocc;

    for (std::size_t idet = 0; idet != nconn; ++idet)
    {
        AlignedVector<ulong> rdet(nword);
        wfn_.fill_hartreefock_det(nb, nocc, &rdet[0]);
        const ulong *det = wfn_.det_ptr(idet);
        ensure_struct_size(det_exc_param_indx, idet+1);
        // std::cout << "Size of det_exc_param_indx: " << det_exc_param_indx.size() << std::endl;
        // std::cout << "\n---------> Det: " ;
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
        for (std::size_t iword = 0; iword != nword; ++iword)
        {
            word = rdet[iword] ^ det[iword]; //str for excitation
            hword = word & rdet[iword]; //str for hole
            pword = word & det[iword]; //str for particle
            // std::cout << "\nword: " << word << std::endl;
            // std::cout << "hword: " << hword << std::endl;
            // std::cout << "pword: " << pword << std::endl;
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
        // std::cout << "nexc: " << nexc << std::endl;
        // std::cout << "Ensured struct size" << std::endl;
        DetExcParamIndx exc_info;
        exc_info.det.resize(nword);
        exc_info.pair_inds.resize(1);
        exc_info.single_inds.resize(1);
        exc_info.pair_inds[0] = -1;
        exc_info.single_inds[0] = -1;
        // std::cout << "Assigned first elem as -1 to both pair_inds and single_inds" << std::endl;
        std::memcpy(&exc_info.det[0], &det[0], sizeof(ulong) * nword);
        // std::cout << "\nCopied det" << std::endl;
        if (!are_same) generate_excitations(holes, particles, nexc, exc_info.pair_inds, exc_info.single_inds, nbasis, wfn_);
        // std::cout << "Generated excitations" << std::endl;
        // std::cout << "size of det_exc_param_indx: " << det_exc_param_indx.size() << std::endl;
        std::sort(exc_info.pair_inds.begin(), exc_info.pair_inds.end());
        std::sort(exc_info.single_inds.begin(), exc_info.single_inds.end());
        // if (idet == 41) {
            // std::cout << "Det: ";
            // for (std::size_t k = 0; k < nword; ++k) {
            //     std::cout << det[k] << " ";
            // }
            // std::cout << std::endl;
            // std::cout << "exc_info.pair_inds: ";
            // for (const auto& pid : exc_info.pair_inds) {
            //     std::cout << pid << " ";
            // }
            // std::cout << std::endl;
            // std::cout << "exc_info.single_inds: ";
            // for (const auto& sid : exc_info.single_inds) {
            //     std::cout << sid << " ";
            // }
            // std::cout << std::endl;
        // }
        det_exc_param_indx[idet] = exc_info;
    }
}


bool AP1roGeneralizedSenoObjective::permanent_calculation(const std::vector<long>& excitation_inds, const double* x, double& permanent) {
    // Ryser's Algorithm
    std::size_t n = static_cast<std::size_t>(std::sqrt(excitation_inds.size()));
    if (n == 0) {permanent = 1.0; return true;}
    if (n == 1) {
        permanent = x[excitation_inds[0]]; 
        // std::cout << "I'm here, perm: " << permanent << std::endl;
        return true;}
    permanent = 0.0;
    std::size_t subset_count = 1UL << n; // 2^n subsets
    // std::cout << "\npermanent: " << permanent << std::endl;
    for (std::size_t subset = 0; subset < subset_count; ++subset) {
        double rowsumprod = 1.0;

        for (std::size_t  i = 0; i < n; ++i) {
            double rowsum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                // std::cout << "\ni: " << i << ", j: " << j << ", rowsum: " << rowsum << std::endl;
                if (subset & (1UL << j)) {
                    rowsum += x[excitation_inds[i * n + j]];
                    // std::cout << "x[" << excitation_inds[i * n + j] << "]: " << x[excitation_inds[i * n + j]] << std::endl;
                }
                // std::cout << "updated rowsum: " << rowsum << std::endl;
            }
            if (std::isnan(rowsum) || std::isinf(rowsum)) {
                std::cerr << "Error: rowsum is invalid (NaN or Inf) at subset " << subset << std::endl;
                return false;    
            }
            rowsumprod *= rowsum;
            // std::cout << "rowsumprod: " << rowsumprod << std::endl;
        }
        if (std::isnan(rowsumprod) || std::isinf(rowsumprod)) {
            std::cerr << "Error: rowsumprod is invalid (NaN or Inf) at subset " << subset << std::endl;
            return false;
        }

        // multiply by the parity of the subset
        // std::cout << "parity: " << ((__builtin_popcount(subset) & 1) << 1) << std::endl;
        permanent += rowsumprod * (1 - ((__builtin_popcount(subset) & 1) << 1));
        // std::cout << "permanent: " << permanent << std::endl;
    }
    // If n (matrix size) is odd, multiply by -1
    permanent *= ((n % 2 == 1) ? -1 : 1);
    // std::cout << "permanent: " << permanent << std::endl;
    
    if (std::isnan(permanent) || std::isinf(permanent)) {
        std::cerr << "Error: permanent is invalid (NaN or Inf)" << std::endl;
        return false;
    }
    return true;
}

double AP1roGeneralizedSenoObjective::compute_derivative(const std::vector<long> excitation_inds, 
    const double* x, std::size_t iparam) {
    // std::cout << "\nComputing derivative" << std::endl;

    double reduced_permanent = 0.0;
    // std::cout << "excitation_inds: ";
    // for (const auto& eid : excitation_inds) {
    //     std::cout << eid << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "iparam: " << iparam << std::endl;
    // Check if excitation_inds is empty, first element is -1
    if (excitation_inds[0] == -1) {return 0.0;} 
   

    // check if iparam is within excitation_inds
    auto it = std::find(excitation_inds.begin(), excitation_inds.end(), iparam);

    // If iparam is not in excitation_inds, return 0
    if (it == excitation_inds.end()) {return 0.0;}
  
    if (excitation_inds.size() == 1 && excitation_inds[0] == iparam) {
        reduced_permanent = 1.0;
        return reduced_permanent;
    }
    // Modify x such that it excludes the parameter at index iparam
    std::vector<double> modified_x(x, x + nparam);
    modified_x[iparam] = 1.0;
    

    // std::cout << "Modified x: ";
    //     for (const auto& rid : modified_x) {
    //         std::cout << rid << " ";
    //     }
    // std::cout << std::endl;
    permanent_calculation(excitation_inds, modified_x.data(), reduced_permanent);

    if (std::isnan(reduced_permanent) || std::isinf(reduced_permanent)) {
        std::cerr << "Error: reduced_permanent is invalid (NaN or Inf)" << std::endl;
        return 0.0;
    }
    // std::cout << "\nreduced_permanent: " << reduced_permanent << std::endl;
    return reduced_permanent;
}



void AP1roGeneralizedSenoObjective::overlap(std::size_t ndet, const double *x, double *y) {
    // std::cout << "\nInside overlap" << std::endl;
    // std::cout << "ndet: "   << ndet << std::endl;
    p_permanent.resize(nconn);
    s_permanent.resize(nconn);
    // std::cout << "Input params: ";
    // for (std::size_t i = 0; i < nparam; ++i) {
    //     std::cout << x[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Ovlp: " ;
    // for (std::size_t i = 0; i < ndet; ++i) {
    //     std::cout << y[i] << " ";
    // }
    // std::cout << std::endl;
    for (std::size_t idet = 0; idet != ndet; ++idet) {
        // std::cout << "Size of det_exc_param_indx: " << det_exc_param_indx.size() << std::endl;
    
        if (idet < det_exc_param_indx.size()) {
            // std::cout << "\n\nidet: " <<  idet << " found in storage" << std::endl;

            // Access the excitation parameter indices
            const DetExcParamIndx& exc_info = det_exc_param_indx[idet];
            double pair_permanent = 1.0;
            double single_permanent = 1.0;

            if (exc_info.pair_inds[0] != -1) {
                if (!permanent_calculation(exc_info.pair_inds, x, pair_permanent)) {
                    std::cerr << "Error calculating pair_permanent for idet" << idet << std::endl;
                    // pair_permanent = 0.0; // Default to 0 or another appropriate fallback
                }
            }

            if (exc_info.single_inds[0] != -1) {
                if (!permanent_calculation(exc_info.single_inds, x, single_permanent)) {
                    std::cerr << "Error calculating single_permanent for idet " << idet << std::endl;
                    // single_permanent = 0.0; // Default value on error
                }
            }

            p_permanent[idet] = pair_permanent;
            s_permanent[idet] = single_permanent;

            // std::cout << "exc_info.pair_inds, x[idx]: ";
            // for (const auto& pid : exc_info.pair_inds) {
            //     std::cout << pid << ", " << x[pid] << " ";
            // }
            // std::cout << "\nexc_info.single_inds: ";
            // for (const auto& sid : exc_info.single_inds) {
            //     std::cout << sid << " ";
            // }
            // std::cout << "\npair_permanent: " << pair_permanent << std::endl;
            // std::cout << "single_permanent: " << single_permanent << std::endl;

            if (y != nullptr && idet < ndet) {
                y[idet] = pair_permanent * single_permanent;
            } else {
                std::cerr << "y is nullptr or idet is out of bounds" << std::endl;
            }
            // std::cout << "y[" << idet << "]: " << y[idet] << std::endl;
        } else {
            std::cout << "idet" <<  idet << " not found in storage" << std::endl;
            y[idet] = 0.0;
            s_permanent[idet] = 0.0;
            p_permanent[idet] = 0.0;
        }
    }
}


// void AP1roGeneralizedSenoObjective::d_overlap(const NonSingletCI &wfn_, const size_t ndet, const double *x, double *y){
void AP1roGeneralizedSenoObjective::d_overlap(const size_t ndet, const double *x, double *y){
    // std::cout << "\n------>Computing d_overlap" << std::endl;
    // std::cout << "Size of s_permanent: " << s_permanent.size() << std::endl;
    // std::cout << "Size of p_permanent: " << p_permanent.size() << std::endl;
    // std::cout << "ndet: " << ndet << std::endl;
    for (std::size_t idet = 0; idet != ndet; ++idet)   {

        // Ensure we have the excitation parameters for this determinant
        if (idet < det_exc_param_indx.size()) {
            const DetExcParamIndx exc_info = det_exc_param_indx[idet];
            double pair_permanent = 1.0;
            double single_permanent = 1.0;
            if (idet < s_permanent.size() && idet < p_permanent.size()) {
                // std::cout << "\n\nidet: " <<  idet << " found in storage" << std::endl;
                pair_permanent = p_permanent[idet];
                single_permanent = s_permanent[idet];
            }
            else {
                // std::cout << "\nidet: " << idet << " not found in storage" << std::endl;
                if (exc_info.pair_inds[0] != -1) {
                    if (!permanent_calculation(exc_info.pair_inds, x, pair_permanent)) {
                        std::cerr << "Error calculating pair_permanent for idet" << idet << std::endl;
                        // pair_permanent = 0.0; // Default to 0 or another appropriate fallback
                    }
                }

                if (exc_info.single_inds[0] != -1) {
                    if (!permanent_calculation(exc_info.single_inds, x, single_permanent)) {
                        std::cerr << "Error calculating single_permanent for idet " << idet << std::endl;
                        // single_permanent = 0.0; // Default value on error
                    }
                }
                
            }

            // std::cout << "\npair_permanent: " << pair_permanent << std::endl;
            // std::cout << "single_permanent: " << single_permanent << std::endl;

            for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
                std::cout << "computing deriv of idet: " << idet << " wrt iparam: " << iparam << std::endl;
                // std::cout << "nparam: " << nparam << std::endl;
                // std::cout << "Size(pair_inds): " << exc_info.pair_inds.size() << std::endl;
                // std::cout << "Size(single_inds): " << exc_info.single_inds.size() << std::endl;
                // std::cout << "size of y array: " << &y.size() << std::endl;
                double dpair = 0.0;
                double dsingle = 0.0;
                std::cout << "exc_info.single_inds: ";
                // for (const auto& sid : exc_info.single_inds) {
                //     std::cout << sid << " ";
                // }
                // std::cout << std::endl;
                // std::cout << "exc_info.pair_inds: ";
                // for (const auto& pid : exc_info.pair_inds) {
                //     std::cout << pid << " ";
                // }
                // std::cout << std::endl;

                if (exc_info.single_inds[0] != -1) {
                    dsingle = compute_derivative(exc_info.single_inds, x, iparam);
                    // std::cout << "dsingle: " << dsingle << std::endl;
                }
                // std::cout << "dsingle: " << dsingle << std::endl;
                
                if (exc_info.pair_inds[0] != -1) {
                    dpair = compute_derivative(exc_info.pair_inds, x, iparam);
                    // std::cout << "dpair: " << dpair << std::endl;
                }
                // std::cout << "dpair: " << dpair << std::endl;
                
                
                // std::cout << "dpair: " << dpair << std::endl;
                // std::cout << "dsingle: " << dsingle <<  "\n" ;
                // std::cout << "deriv index:" << idet * nparam + iparam << std::endl;
                // std::cout <<  "final deriv: " << dpair * single_permanent + dsingle * pair_permanent << " wrt iparam: " << iparam << std::endl;

                y[ndet * iparam + idet] = dpair * single_permanent + dsingle * pair_permanent;
            }
        }
        else {
            // std::cout << "Determinant " << idet << " not found in det_map" << std::endl;
            // Set all derivatives to zero if determinant is not found
            for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
                y[ndet * iparam + idet] = 0.0;
            }
        }
    }
}


} // namespace pyci
