# This file is part of PyCI.
#
# PyCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# PyCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCI. If not, see <http://www.gnu.org/licenses/>.

r"""PyCI seniority CI module."""

from itertools import combinations

import numpy as np

from . import pyci


__all__ = [
    "add_seniorities",
]


def add_seniorities(wfn, *seniorities):
    r"""
    Add determinants of the specified seniority/ies to the wave function.

    Parameters
    ----------
    wfn : pyci.fullci_wfn
        FullCI wave function.
    seniorities : Sequence[int]
        List of seniorities of determinants to add.

    """
    # Check wave function
    if not isinstance(wfn, pyci.fullci_wfn):
        raise TypeError(f"invalid `wfn` type `{type(wfn)}`; must be `pyci.fullci_wfn`")

    # Check specified seniorities
    smin = wfn.nocc_up - wfn.nocc_dn
    smax = bin(
        ((1 << wfn.nocc_up) - 1) ^ ((1 << wfn.nocc_dn) - 1) ^ ((1 << wfn.nbasis) - 1)
    ).count("1")
    if any(s < smin or s > smax or s % 2 != smin % 2 for s in seniorities):
        raise ValueError(f"invalid seniority number in `seniorities = {seniorities}`")

    # Make seniority-zero occupation vectors
    sz_wfn = pyci.doci_wfn(wfn.nbasis, wfn.nocc_up, wfn.nocc_up)
    sz_wfn.add_all_dets()
    occ_up_array = sz_wfn.to_occ_array()
    del sz_wfn

    # Make working arrays
    brange = np.arange(wfn.nbasis, dtype=pyci.c_long)
    occs = np.empty((2, wfn.nocc_up), dtype=pyci.c_long)

    # Add determinants of specified seniorities
    for s in seniorities:
        if not s:
            # Seniority-zero
            for occs_up in occ_up_array:
                occs[0, :] = occs_up
                occs[1, :] = occs_up
                wfn.add_occs(occs)
        else:
            # Seniority-nonzero
            pairs = (wfn.nocc - s) // 2
            if pairs == wfn.nocc_dn:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    for occs_dn in combinations(occs_up, wfn.nocc_dn):
                        occs[1, : wfn.nocc_dn] = occs_dn
                        wfn.add_occs(occs)
            elif not pairs:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    virs_up = np.setdiff1d(brange, occs_up, assume_unique=True)
                    for occs_dn in combinations(virs_up, wfn.nocc_dn):
                        occs[1, : wfn.nocc_dn] = occs_dn
                        wfn.add_occs(occs)
            else:
                for occs_up in occ_up_array:
                    occs[0, :] = occs_up
                    virs_up = np.setdiff1d(brange, occs_up, assume_unique=True)
                    for occs_i_dn in combinations(occs_up, pairs):
                        occs[1, :pairs] = occs_i_dn
                        for occs_a_dn in combinations(virs_up, wfn.nocc_dn - pairs):
                            occs[1, pairs : wfn.nocc_dn] = occs_a_dn
                            wfn.add_occs(occs)
