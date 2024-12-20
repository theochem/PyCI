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

AP1roGObjective::AP1roGObjective(const SparseOp &op_, const DOCIWfn &wfn_,
                                 const std::size_t n_detcons_,
                                 const long *idx_detcons_,
                                 const double *val_detcons_,
                                 const std::size_t n_paramcons_,
                                 const long *idx_paramcons_,
                                 const double *val_paramcons_)
: Objective<DOCIWfn>::Objective(op_, wfn_, n_detcons_, idx_detcons_, val_detcons_, n_paramcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_);
}

AP1roGObjective::AP1roGObjective(const SparseOp &op_, const DOCIWfn &wfn_,
                                 const pybind11::object idx_detcons_,
                                 const pybind11::object val_detcons_,
                                 const pybind11::object idx_paramcons_,
                                 const pybind11::object val_paramcons_)
: Objective<DOCIWfn>::Objective(op_, wfn_, idx_detcons_, val_detcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_);
}

AP1roGObjective::AP1roGObjective(const AP1roGObjective &obj)
: Objective<DOCIWfn>::Objective(obj), nrow(obj.nrow), ncol(obj.ncol),
  nexc_list(obj.nexc_list), hole_list(obj.hole_list), part_list(obj.part_list)
{
    return;
}

AP1roGObjective::AP1roGObjective(AP1roGObjective &&obj) noexcept
: Objective<DOCIWfn>::Objective(obj), nrow(std::exchange(obj.nrow, 0)), ncol(std::exchange(obj.ncol, 0)),
  nexc_list(std::move(obj.nexc_list)), hole_list(std::move(obj.hole_list)), part_list(std::move(obj.part_list))
{
    return;
}

void AP1roGObjective::init_overlap(const DOCIWfn &wfn_)
{
    nparam = wfn_.nocc_up * (wfn_.nbasis - wfn_.nocc_up);
    nrow = wfn_.nocc_up;
    ncol = wfn_.nbasis - wfn_.nocc_up;

    ovlp.resize(nconn);
    d_ovlp.resize(nconn * nparam);

    nexc_list.resize(nconn);
    hole_list.resize(wfn_.nocc_up * nconn);
    part_list.resize(wfn_.nocc_up * nconn);

    std::size_t nword = (ulong)wfn_.nword;
    for (std::size_t idet = 0; idet != nconn; ++idet) {
        std::vector<ulong> rdet(wfn_.nword);
        fill_hartreefock_det(wfn_.nocc_up, &rdet[0]);
        const ulong *det = wfn_.det_ptr(idet);
        ulong word, hword, pword;
        std::size_t h, p, nexc = 0;
        
        for (std::size_t iword = 0; iword != nword; ++iword) {
            word = rdet[iword] ^ det[iword];
            hword = word & rdet[iword];
            pword = word & det[iword];

            while (hword) {
                h = Ctz(hword);
                p = Ctz(pword);
                hole_list[idet * wfn_.nocc_up + nexc] = h + iword * Size<ulong>();
                part_list[idet * wfn_.nocc_up + nexc] = p + iword * Size<ulong>() - wfn_.nocc_up;
                
                hword &= ~(1UL << h);
                pword &= ~(1UL << p);
                ++nexc;         
            }
        }
        nexc_list[idet] = nexc;
    }
}

void AP1roGObjective::overlap(const size_t ndet, const double *x, double *y) {

    std::size_t m, i, j, k, c;
    std::size_t *hlist, *plist;
    double rowsum, rowsumprod, out;

    for (std::size_t idet = 0; idet != ndet; ++idet) {

        m = nexc_list[idet];
        if (m == 0) {
            y[idet] = 1;
            continue;
        }

        hlist = &hole_list[idet * nrow];
        plist = &part_list[idet * nrow];

        out = 0;

        /* Iterate over c = pow(2, m) submatrices (equal to (1 << m)) submatrices. */
        c = 1UL << m;

        /* Loop over columns of submatrix; compute product of row sums. */
        for (k = 0; k < c; ++k) {
            rowsumprod = 1.0;
            for (i = 0; i < m; ++i) {

                /* Loop over rows of submatrix; compute row sum. */
                rowsum = 0.0;
                for (j = 0; j < m; ++j) {

                    /* Add element to row sum if the row index is in the characteristic *
                     * vector of the submatrix, which is the binary vector given by k.  */
                    std::size_t xpq = k & (1UL << j);
                    std::cout << "k &(1UL << j): "  << xpq << std::endl;
                    if (k & (1UL << j)) {
                        rowsum += x[ncol * hlist[i] + plist[j]];
                    }
                }

                /* Update product of row sums. */
                rowsumprod *= rowsum;
            }

            /* Add term multiplied by the parity of the characteristic vector. */
            out += rowsumprod * (1 - ((__builtin_popcountll(k) & 1) << 1));
        }

        /* Return answer with the correct sign (times -1 for odd m). */
        y[idet] = out * ((m % 2 == 1) ? -1 : 1);
    }
}

void AP1roGObjective::d_overlap(const size_t ndet, const double *x, double *y) {

    std::size_t m, n, i, j, k, c;
    std::size_t *hlist, *plist;
    double rowsum, rowsumprod, out;

    for (std::size_t idet = 0; idet != ndet; ++idet) {

        for (std::size_t iparam = 0; iparam != nparam; ++iparam) {

            hlist = &hole_list[idet * nrow];
            plist = &part_list[idet * nrow];

            m = nexc_list[idet];
            if (m == 0) {
                y[ndet * iparam + idet] = 0;
                continue;
            }

            std::vector<std::size_t> rows;
            std::vector<std::size_t> cols;
            for (i = 0; i < m; ++i) {
                if (hlist[i] != iparam / ncol) {
                    rows.push_back(hlist[i]);
                }
                if (plist[i] != iparam % ncol) {
                    cols.push_back(plist[i]);
                }
            }
            m = rows.size();
            n = cols.size();
            if (m == 0 && n == 0) {
                y[ndet * iparam + idet] = 1;
                continue;
            } else if (m == nexc_list[idet] || n == nexc_list[idet] || m != n) {
                y[ndet * iparam + idet] = 0;
                continue;
            }

            out = 0;

            /* Iterate over c = pow(2, m) submatrices (equal to (1 << m)) submatrices. */
            c = 1UL << m;

            /* Loop over columns of submatrix; compute product of row sums. */
            for (k = 0; k < c; ++k) {
                rowsumprod = 1.0;
                for (i = 0; i < m; ++i) {

                    /* Loop over rows of submatrix; compute row sum. */
                    rowsum = 0.0;
                    for (j = 0; j < m; ++j) {

                        /* Add element to row sum if the row index is in the characteristic *
                         * vector of the submatrix, which is the binary vector given by k.  */
                        if (k & (1UL << j)) {
                            rowsum += x[ncol * rows[i] + cols[j]];
                        }
                    }

                    /* Update product of row sums. */
                    rowsumprod *= rowsum;
                }

                /* Add term multiplied by the parity of the characteristic vector. */
                out += rowsumprod * (1 - ((__builtin_popcountll(k) & 1) << 1));
            }

            /* Return answer with the correct sign (times -1 for odd m). */
            y[ndet * iparam + idet] = out * ((m % 2 == 1) ? -1 : 1);
        }
    }
}

} // namespace pyci
