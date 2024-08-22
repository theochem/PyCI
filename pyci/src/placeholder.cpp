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

// See apig.cpp for reference

PlaceholderObjective::PlaceholderObjective(const SparseOp &op_, const FullCIWfn &wfn_,
                                 const std::size_t n_detcons_,
                                 const long *idx_detcons_,
                                 const double *val_detcons_,
                                 const std::size_t n_paramcons_,
                                 const long *idx_paramcons_,
                                 const double *val_paramcons_)
: Objective<FullCIWfn>::Objective(op_, wfn_, n_detcons_, idx_detcons_, val_detcons_, n_paramcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_);
}

PlaceholderObjective::PlaceholderObjective(const SparseOp &op_, const FullCIWfn &wfn_,
                                 const pybind11::object idx_detcons_,
                                 const pybind11::object val_detcons_,
                                 const pybind11::object idx_paramcons_,
                                 const pybind11::object val_paramcons_)
: Objective<FullCIWfn>::Objective(op_, wfn_, idx_detcons_, val_detcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_);
}

PlaceholderObjective::PlaceholderObjective(const PlaceholderObjective &obj)
: Objective<FullCIWfn>::Objective(obj)
{
    return;
}

PlaceholderObjective::PlaceholderObjective(PlaceholderObjective &&obj) noexcept
: Objective<FullCIWfn>::Objective(obj){
    return;
}

void PlaceholderObjective::init_overlap(const FullCIWfn &wfn_)
{
    // Initialize your class-specific variables here
    return;
}

void PlaceholderObjective::overlap(const size_t ndet, const double *x, double *y)
{
    // x == parameters p_j
    // y == overlap vector σ_i
    return;
}

void PlaceholderObjective::d_overlap(const size_t ndet, const double *x, double *y)
{
    // x == parameters p_j
    // y == unwrapped overlap objective ∂σ_i/∂p_j
    return;
}

} // namespace pyci
