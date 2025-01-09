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
// #include <cmath>
#include <iostream>

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
    AlignedVector<long> occs_up, occs_dn, virs_up, virs_dn, temp_occs;
    for (int i : holes) {
        (i < nbasis ? occs_up : occs_dn).push_back(i);
    }
    for (int a : particles) {
        (a < nbasis ? virs_up : virs_dn).push_back(a);
    }

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
                        long sindx = wfn_.calc_sindex(hsingle_comb[0], psingle_comb[0]);
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

    ovlp.resize(nconn);
    d_ovlp.resize(nconn * nparam);
    det_exc_param_indx.resize(nconn);

    std::size_t nword = (ulong)wfn_.nword;
    long nb = wfn_.nbasis;
    long nocc = wfn_.nocc;

    for (std::size_t idet = 0; idet != nconn; ++idet)
    {
        AlignedVector<ulong> rdet(nword);
        wfn_.fill_hartreefock_det(nb, nocc, &rdet[0]);
        const ulong *det = wfn_.det_ptr(idet);
        ensure_struct_size(det_exc_param_indx, idet+1);

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

        DetExcParamIndx exc_info;
        exc_info.det.resize(nword);
        exc_info.pair_inds.resize(1);
        exc_info.single_inds.resize(1);
        exc_info.pair_inds[0] = -1;
        exc_info.single_inds[0] = -1;
        std::memcpy(&exc_info.det[0], &det[0], sizeof(ulong) * nword);
        if (!are_same) generate_excitations(holes, particles, nexc, exc_info.pair_inds, exc_info.single_inds, nbasis, wfn_);
        std::sort(exc_info.pair_inds.begin(), exc_info.pair_inds.end());
        std::sort(exc_info.single_inds.begin(), exc_info.single_inds.end());

        det_exc_param_indx[idet] = exc_info;
    }
}


bool AP1roGeneralizedSenoObjective::permanent_calculation(const std::vector<long>& excitation_inds, const double* x, double& permanent) {
    // Ryser's Algorithm
    std::size_t n = static_cast<std::size_t>(std::sqrt(excitation_inds.size()));
    if (n == 0) {permanent = 1.0; return true;}
    if (n == 1) {
        permanent = x[excitation_inds[0]]; 
        return true;}
    permanent = 0.0;
    std::size_t subset_count = 1UL << n; // 2^n subsets
    for (std::size_t subset = 0; subset < subset_count; ++subset) {
        double rowsumprod = 1.0;

        for (std::size_t  i = 0; i < n; ++i) {
            double rowsum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                if (subset & (1UL << j)) {
                    rowsum += x[excitation_inds[i * n + j]];
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



void AP1roGeneralizedSenoObjective::overlap(std::size_t ndet, const double *x, double *y) {
    p_permanent.resize(nconn);
    s_permanent.resize(nconn);

    for (std::size_t idet = 0; idet != ndet; ++idet) {
    
        if (idet < det_exc_param_indx.size()) {
            // Access the excitation parameter indices
            const DetExcParamIndx& exc_info = det_exc_param_indx[idet];
            double pair_permanent = 1.0;
            double single_permanent = 1.0;

            if (exc_info.pair_inds[0] != -1) {
                if (!permanent_calculation(exc_info.pair_inds, x, pair_permanent)) {
                    std::cerr << "Error calculating pair_permanent for idet" << idet << std::endl;
                }
            }

            if (exc_info.single_inds[0] != -1) {
                if (!permanent_calculation(exc_info.single_inds, x, single_permanent)) {
                    std::cerr << "Error calculating single_permanent for idet " << idet << std::endl;
                }
            }

            p_permanent[idet] = pair_permanent;
            s_permanent[idet] = single_permanent;

            if (y != nullptr && idet < ndet) {
                y[idet] = pair_permanent * single_permanent;
            } else {
                std::cerr << "y is nullptr or idet is out of bounds" << std::endl;
            }
        } else {
            std::cout << "idet" <<  idet << " not found in storage" << std::endl;
            y[idet] = 0.0;
            s_permanent[idet] = 0.0;
            p_permanent[idet] = 0.0;
        }
    }
}


void AP1roGeneralizedSenoObjective::d_overlap(const size_t ndet, const double *x, double *y){

    for (std::size_t idet = 0; idet != ndet; ++idet)   {
        // Ensure we have the excitation parameters for this determinant
        if (idet < det_exc_param_indx.size()) {
            const DetExcParamIndx exc_info = det_exc_param_indx[idet];
            double pair_permanent = 1.0;
            double single_permanent = 1.0;
            if (idet < s_permanent.size() && idet < p_permanent.size()) {
                pair_permanent = p_permanent[idet];
                single_permanent = s_permanent[idet];
            }
            // else {
            //     if (exc_info.pair_inds[0] != -1) {
            //         if (!permanent_calculation(exc_info.pair_inds, x, pair_permanent)) {
            //             std::cerr << "Error calculating pair_permanent for idet" << idet << std::endl;
            //         }
            //     }

            //     if (exc_info.single_inds[0] != -1) {
            //         if (!permanent_calculation(exc_info.single_inds, x, single_permanent)) {
            //             std::cerr << "Error calculating single_permanent for idet " << idet << std::endl;
            //         }
            //     }
                
            // }
            for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
                double dpair = 0.0;
                double dsingle = 0.0;
                if (exc_info.single_inds[0] != -1) {
                    dsingle = compute_derivative(exc_info.single_inds, x, iparam);
                }
                
                if (exc_info.pair_inds[0] != -1) {
                    dpair = compute_derivative(exc_info.pair_inds, x, iparam);
                }
                // std::cout << "dpair: " << dpair << ", dsingle: " << dsingle << std::endl;
                // std::cout << "single_permanent: " << single_permanent << ", pair_permanent: " << pair_permanent << std::endl;
                // std::cout << "idet: " << idet << " deriv: " << dpair * single_permanent + dsingle * pair_permanent << std::endl;
                y[ndet * iparam + idet] = dpair * single_permanent + dsingle * pair_permanent;
            }
        }
        else {
            for (std::size_t iparam = 0; iparam < nparam; ++iparam) {
                y[ndet * iparam + idet] = 0.0;
            }
        }
    }
}


} // namespace pyci
