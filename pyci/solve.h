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

#pragma once

#include <vector>

#include <pyci/common.h>
#include <pyci/doci.h>
#include <pyci/fullci.h>


namespace pyci {


struct SparseOp
{
    public:

    int_t nrow, ncol;
    std::vector<double> data;
    std::vector<int_t> indices;
    std::vector<int_t> indptr;

    inline int_t rows() const { return nrow; }
    inline int_t cols() const { return ncol; }

    SparseOp();

    void perform_op(const double *, double *) const;
    void solve(const double *, const int_t, const int_t, const int_t, const double, double *, double *) const;

    void init_doci(const DOCIWfn &, const double *, const double *, const double *, const int_t);
    void init_fullci(const FullCIWfn &, const double *, const double *, const int_t);
    void init_gen(const DOCIWfn &, const double *, const double *, const int_t);

    private:

    void init_doci_run_thread(const DOCIWfn &, const double *, const double *, const double *, const int_t, const int_t);
    void init_fullci_run_thread(const FullCIWfn &, const double *, const double *, const int_t, const int_t);
    void init_gen_run_thread(const DOCIWfn &, const double *, const double *, const int_t, const int_t);
    void init_condense_thread(SparseOp &);
};


} // namespace pyci
