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

#include <parallel_hashmap/phmap.h>

#include <pyci/common.h>


namespace pyci {


struct FullCIWfn
{
    public:

    typedef hashmap<int_t, int_t> hashmap_type;

    int_t nword, nword2, nbasis, nocc_up, nocc_dn, nvir_up, nvir_dn, ndet, maxdet_up, maxdet_dn;

    std::vector<uint_t> dets;
    hashmap_type dict;

    FullCIWfn();
    FullCIWfn(const int_t, const int_t, const int_t);
    FullCIWfn(const FullCIWfn &);
    FullCIWfn(const char *);
    FullCIWfn(const int_t, const int_t, const int_t, const int_t, const uint_t *);
    FullCIWfn(const int_t, const int_t, const int_t, const int_t, const int_t *);
    ~FullCIWfn();

    void init(const int_t, const int_t, const int_t);
    void from_fullciwfn(const FullCIWfn &);
    void from_file(const char *);
    void from_det_array(const int_t, const int_t, const int_t, const int_t, const uint_t *);
    void from_occs_array(const int_t, const int_t, const int_t, const int_t, const int_t *);

    void to_file(const char *) const;
    void to_occs_array(const int_t, const int_t, int_t *) const;

    int_t index_det(const uint_t *) const;
    int_t index_det_from_rank(const int_t) const;
    void copy_det(const int_t, uint_t *) const;
    const uint_t * det_ptr(const int_t) const;

    int_t add_det(const uint_t *);
    int_t add_det_with_rank(const uint_t *, const int_t);
    int_t add_det_from_occs(const int_t *);
    void add_all_dets();
    void add_excited_dets(const uint_t *, const int_t, const int_t);

    void reserve(const int_t);
    void squeeze();

    double compute_overlap(const double *, const FullCIWfn &, const double *) const;

    void compute_rdms(const double *, double *, double *) const;

    int_t run_hci(const double *, const double *, const double *, const double);

    private:

    void run_hci_run_thread(const FullCIWfn &, const double *, const double *, const double *,
                            const double, const int_t, const int_t);
    void run_hci_condense_thread(FullCIWfn &);
};


} // namespace pyci
