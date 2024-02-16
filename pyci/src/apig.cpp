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

APIGObjective::APIGObjective(const SparseOp &op_, const DOCIWfn &wfn_,
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

APIGObjective::APIGObjective(const SparseOp &op_, const DOCIWfn &wfn_,
                                 const pybind11::object idx_detcons_,
                                 const pybind11::object val_detcons_,
                                 const pybind11::object idx_paramcons_,
                                 const pybind11::object val_paramcons_)
: Objective<DOCIWfn>::Objective(op_, wfn_, idx_detcons_, val_detcons_, idx_paramcons_, val_paramcons_)
{
    init_overlap(wfn_);
}

APIGObjective::APIGObjective(const APIGObjective &obj)
: Objective<DOCIWfn>::Objective(obj), nrow(obj.nrow), ncol(obj.ncol), part_list(obj.part_list)
{
    return;
}

APIGObjective::APIGObjective(APIGObjective &&obj) noexcept
: Objective<DOCIWfn>::Objective(obj), nrow(std::exchange(obj.nrow, 0)), ncol(std::exchange(obj.ncol, 0)),
  part_list(std::move(obj.part_list))
{
    return;
}

void APIGObjective::init_overlap(const DOCIWfn &wfn_)
{
    nparam = wfn_.nocc_up * wfn_.nbasis;
    nrow = wfn_.nocc_up;
    ncol = wfn_.nbasis;

    ovlp.resize(nconn);
    d_ovlp.resize(nconn * nparam);

    part_list.resize(wfn_.nocc_up * nconn);

    std::size_t nword = (std::size_t)wfn_.nword;
    for (std::size_t idet = 0; idet != nconn; ++idet) {
        const ulong *det = wfn_.det_ptr(idet);
        ulong word;
        std::size_t p, nexc = 0;
        for (std::size_t iword = 0; iword != nword; ++iword) {
            word = det[iword];
            while (word) {
                p = Ctz(word);
                part_list[idet * wfn_.nocc_up + nexc++] = p + iword * Size<ulong>();
                word &= ~(1UL << p);
            }
        }
    }
}

void APIGObjective::overlap(const size_t ndet, const double *x, double *y) {

    std::size_t i, j, k, c;
    std::size_t *plist;
    double rowsum, rowsumprod, out;

    for (std::size_t idet = 0; idet != ndet; ++idet) {

        plist = &part_list[idet * nrow];

        out = 0;

        /* Iterate over c = pow(2, m) submatrices (equal to (1 << m)) submatrices. */
        c = 1UL << nrow;

        /* Loop over columns of submatrix; compute product of row sums. */
        for (k = 0; k < c; ++k) {
            rowsumprod = 1.0;
            for (i = 0; i < nrow; ++i) {

                /* Loop over rows of submatrix; compute row sum. */
                rowsum = 0.0;
                for (j = 0; j < nrow; ++j) {

                    /* Add element to row sum if the row index is in the characteristic *
                     * vector of the submatrix, which is the binary vector given by k.  */
                    if (k & (1UL << j)) {
                        // rowsum += x[nrow * plist[i] + j];
                        rowsum += x[nrow * plist[i] + j];
                    }
                }

                /* Update product of row sums. */
                rowsumprod *= rowsum;
            }

            /* Add term multiplied by the parity of the characteristic vector. */
            out += rowsumprod * (1 - ((__builtin_popcountll(k) & 1) << 1));
        }

        /* Return answer with the correct sign (times -1 for odd m). */
        y[idet] = out * ((nrow % 2 == 1) ? -1 : 1);
    }
}

void APIGObjective::d_overlap(const size_t ndet, const double *x, double *y) {

    std::size_t m, n, i, j, k, c;
    std::size_t *plist;
    double rowsum, rowsumprod, out;

    for (std::size_t idet = 0; idet != ndet; ++idet) {

        for (std::size_t iparam = 0; iparam != nparam; ++iparam) {

            plist = &part_list[idet * nrow];

            std::vector<std::size_t> rows;
            std::vector<std::size_t> cols;
            for (i = 0; i < nrow; ++i) {
                if (i != iparam % nrow) {
                    rows.push_back(i);
                }
                if (plist[i] != iparam / nrow) {
                    cols.push_back(plist[i]);
                }
            }
            m = rows.size();
            n = cols.size();
            if (m == 0 && n == 0) {
                y[ndet * iparam + idet] = 1;
                continue;
            } else if (m == nrow || n == nrow || m != n) {
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
                            rowsum += x[nrow * cols[i] + rows[j]];
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
