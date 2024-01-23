r"""
FanCI AP1roG module.

"""

from itertools import permutations, combinations
from typing import Any, Union

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    "pCCDS",
]


class pCCDS(FanCI):
    r"""
    CC Singles and Pairs FanCI class.

    """

    def __init__(
        self,
        ham: pyci.hamiltonian,
        nocc_up: int,
        nocc_dn: int,
        nproj: int = None,
        wfn: pyci.fullci_wfn = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        nocc : int
            Number of occupied orbitals.
        nproj : int, optional
            Number of determinants in projection ("P") space.
        wfn : pyci.doci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        if not isinstance(ham, pyci.hamiltonian):
            raise TypeError(f"Invalid `ham` type `{type(ham)}`; must be `pyci.hamiltonian`")

        # Compute number of parameters (c_kl + energy)
        # FIXME: Only works for nocc_a = nocc_b
        nocc = nocc_up + nocc_dn
        nparam = nocc_up * (ham.nbasis - nocc_up) + nocc * (2*ham.nbasis - nocc) + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn (P space == single pair excitations)
        # Must be AP1roG + single excitations
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc_up, nocc_dn)
            wfn.add_excited_dets(1)
            wfn = pyci.fullci_wfn(wfn)
            pyci.add_excitations(wfn, 1)
        elif not isinstance(wfn, pyci.fullci_wfn):
            raise TypeError(f"Invalid `wfn` type `{type(wfn)}`; must be `pyci.fullci_wfn`")
        elif wfn.nocc_up != nocc_up or wfn.nocc_dn != nocc_dn:
            raise ValueError(f"wfn.nocc_{{up,dn}} does not match `nocc_{{up,dn}}={nocc_up,nocc_dn}` parameter")

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

        # Assign reference occupations
        ref_occs_up = np.arange(nocc_up, dtype=pyci.c_long)
        ref_occs_dn = np.arange(nocc_dn, dtype=pyci.c_long)

        # Save sub-class -specific attributes
        self._ref_occs = [ref_occs_up, ref_occs_dn]

        # Use set differences to get hole/particle indices
        # The are three pairs of list for each occupation vector:
        # alpha hole-particle excitation: `hlist_up` and `plist_up`
        # beta hole-particle excitation: `hlist_dn` and `plist_dn`
        # hole-particle pair excitation: `hlist` and `plist`
        hlist_up, plist_up, hlist_dn, plist_dn, hlist, plist = _get_hole_particle_indexes(self._wfn, self._ref_occs, self._sspace)
        # Change from spatial orbital to spin-orbital notation the elements in hlist_{up,dn} and
        # plist_{up,dn}. This gives hlist_ab and plist_ab.
        ab_lists = [self._sspace, hlist_up, plist_up, hlist_dn, plist_dn]
        hlist_ab, plist_ab = _make_alpha_plus_beta_strings(self._wfn, *ab_lists)
        # Make a power set of the elements in hlist, plist for each occupation vector: comb_hlist
        # and comb_plist.
        comb_hlist, comb_plist = _make_pairexc_powerset(self._sspace, hlist, plist)

        # Save sub-class -specific attributes
        self._sspace_data = [(hlist_ab, plist_ab), (comb_hlist, comb_plist)]
        self._pspace_data = [(hlist_ab[:nproj], plist_ab[:nproj]), (comb_hlist[:nproj], comb_plist[:nproj])]

    def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap array.

        """
        # Check if we can use our pre-computed {p,s}space_data
        if isinstance(occs_array, np.ndarray):
            hlist_up, plist_up, hlist_dn, plist_dn, hlist, plist = _get_hole_particle_indexes(self._wfn, self._ref_occs, occs_array)
            ab_lists = [occs_array, hlist_up, plist_up, hlist_dn, plist_dn]
            hlist_ab, plist_ab = _make_alpha_plus_beta_strings(self._wfn, *ab_lists)
            comb_hlist, comb_plist = _make_pairexc_powerset(occs_array, hlist, plist)
        elif occs_array == "P":
            occs_array = self._pspace
            hlist_ab, plist_ab = self._pspace_data[0]
            comb_hlist, comb_plist = self._pspace_data[1]
        elif occs_array == "S":
            occs_array = self._sspace
            hlist_ab, plist_ab = self._sspace_data[0]
            comb_hlist, comb_plist = self._sspace_data[1]
        else:
            raise ValueError("invalid `occs_array` argument")

        # Reshape parameter array to particles times virtuals matrix
        t_ii = x[:self._wfn.nocc_up * self._wfn.nvir_up].reshape(self._wfn.nocc_up, self._wfn.nvir_up)
        t_i = x[self._wfn.nocc_up * self._wfn.nvir_up:].reshape(self._wfn.nocc, self._wfn.nvir)

        # Compute overlaps of occupation vectors
        y = np.zeros(occs_array.shape[0], dtype=pyci.c_double)

        # # Determine which spin-orbitals correspond to pair-excitations and which to singles: Change
        # # the elements in {holes,parts} from spatial to spin-orbital notation and compare them with
        # # the ones in the singles excitation description of the occs {holes,parts}_ab. The diference
        # # gives the singles component of the excitation.
        # for i, (occs, holes, parts) in enumerate(zip(occs_array, hlist, plist)):
        #     if holes.size > parts.size:
        #     # if holes.size != parts.size:
        #         continue
        #     elif holes.size < parts.size:
        #         max_pairs = min(holes.size, parts.size)
        #         holes = holes[:max_pairs]
        #         parts = parts[:max_pairs]
        #     holes_ab, parts_ab = _get_singles_component(self._wfn, holes, parts, hlist_ab[i], plist_ab[i])
        #     y[i] += permanent(t_ii[holes, :][:, parts])*permanent(t_i[holes_ab, :][:, parts_ab])

        # 1- Compute sinlges permanent:
        # Overlap is equal to one for the reference determinant
        for i, (occs, holes, parts) in enumerate(zip(occs_array, hlist_ab, plist_ab)):
            # y[i] = permanent(t_i[holes, :][:, parts]) if np.array(holes).size else 1
            y[i] = permanent(t_i[holes, :][:, parts])
        
        # 2- Add pure pairs permanent (Only for seniority zero occs)
        # and combinations of sinlges * pair excitations
        for i, (occs, holes_dict, parts_dict) in enumerate(zip(occs_array, comb_hlist, comb_plist)):
            for exc_n in holes_dict.keys():
                holes_c = holes_dict[exc_n] # list of tuples, combinations of exc_n holes
                parts_c = parts_dict[exc_n] # list of tuples, combinations of exc_n particles
                for holes in holes_c:
                    for parts in parts_c:
                        # Determine which spin-orbitals correspond to pair-excitations and which to
                        # singles: Change the elements in {holes,parts}_c from spatial to
                        # spin-orbital notation and compare them with the ones in the singles
                        # excitation description of the occs {holes,parts}_ab. The diference gives
                        # the singles component of the excitation.
                        temp = [h + self._wfn.nocc_up for h in holes] #  holes_dn
                        temp = np.concatenate((holes, temp), axis=0) # spin-orbs of hole pairs
                        holes_ab = np.setdiff1d(hlist_ab[i], temp, assume_unique=True) # singles holes
                        temp = [p + self._wfn.nvir_up for p in parts] # parts_dn
                        temp = np.concatenate((parts, temp), axis=0) # spin-orbs of particle pairs
                        parts_ab = np.setdiff1d(plist_ab[i], temp, assume_unique=True) # singles particles

                        y[i] += permanent(t_ii[holes, :][:, parts])*permanent(t_i[holes_ab, :][:, parts_ab])

        return y

    def compute_overlap_deriv(
        self, x: np.ndarray, occs_array: Union[np.ndarray, str]
    ) -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative array.

        """
        raise NotImplementedError("Overlap derivative for pCCDS not supported.")


def permanent(matrix: np.ndarray) -> float:
    r"""
    Compute the permanent of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    result : matrix.dtype
        Permanent of the matrix.

    """
    rows = np.arange(matrix.shape[0])
    return sum(np.prod(matrix[rows, cols]) for cols in permutations(rows))


def _make_alpha_plus_beta_strings(wfn, occsarray, hlistup, plistup, hlistdn, plistdn):
    # Form alphas + betas holes/particles lists for each occupation vector
    # (`hlist_ab`/`plist_ab`)
    singles_lists = zip(occsarray, hlistup, plistup, hlistdn, plistdn)
    hlist_ab = []
    plist_ab = []
    for i, (occs, holes_up, parts_up, holes_dn, parts_dn) in enumerate(singles_lists):
        if holes_up.size:
            # Alpha and beta occupations were excited or
            # there is at least an alpha occupation being excited
            if not holes_dn.size:
                hlist_ab.append(holes_up)
                plist_ab.append(parts_up)
            else:
                holes_dn = [h + wfn.nocc_up if h.size else h for h in holes_dn]
                parts_dn = [p + wfn.nvir_up if p.size else p for p in parts_dn]
                hlist_ab.append(np.concatenate((holes_up, holes_dn), axis=0))
                plist_ab.append(np.concatenate((parts_up, parts_dn), axis=0))
        else:
            # Only one beta occupation was excited or the occupation vector is the reference one
            hlist_ab.append([h + wfn.nocc_up if h.size else h for h in holes_dn])
            plist_ab.append([p + wfn.nvir_up if p.size else p for p in parts_dn])
    return hlist_ab, plist_ab


def _get_hole_particle_indexes(wfn, ref_occs, occsarray):
    nocc_up, nocc_dn = wfn.nocc_up, wfn.nocc_dn
    hlist_up = [np.setdiff1d(ref_occs[0], occs[0], assume_unique=1) for occs in occsarray]
    plist_up = [np.setdiff1d(occs[0], ref_occs[0], assume_unique=1) - nocc_up for occs in occsarray]
    hlist_dn = [np.setdiff1d(ref_occs[1], occs[1], assume_unique=1) for occs in occsarray]
    plist_dn = [np.setdiff1d(occs[1], ref_occs[1], assume_unique=1) - nocc_dn for occs in occsarray]
    hlist = [np.intersect1d(holes_up, holes_dn, assume_unique=1) for holes_up, holes_dn in zip(hlist_up, hlist_dn)]
    plist = [np.intersect1d(parts_up, parts_dn, assume_unique=1) for parts_up, parts_dn in zip(plist_up, plist_dn)]
    return hlist_up, plist_up, hlist_dn, plist_dn, hlist, plist


def _get_singles_component(wfn, holes, parts, holesab, partsab):
    """[summary]

    Parameters
    ----------
    wfn : [type]
        PyCI wavefunction
    holes : list
        Hole pairs indexes
    parts : list
        Particle pairs indexes
    holesab : list
        Alpha and beta holes indexes
    partsab : list
        Alpha and beta particles indexes

    Returns
    -------
    list
        Component of holesab and partsab described as single excitations.
    """
    temp = [h + wfn.nocc_up for h in holes] #  holes_dn
    temp = np.concatenate((holes, temp), axis=0) # spin-orbs of hole pairs
    hs = np.setdiff1d(holesab, temp, assume_unique=True).tolist() # singles holes
    temp = [p + wfn.nvir_up for p in parts] # parts_dn
    temp = np.concatenate((parts, temp), axis=0) # spin-orbs of particle pairs
    ps = np.setdiff1d(partsab, temp, assume_unique=True).tolist()
    return hs, ps


def _make_pairexc_powerset(occsarray, hlist, plist):
    # Make all pair excitations sets (`comb_hlist --> comb_plist`) for each
    # occupation vector. Does not include the empty set.
    pairs_lists = zip(occsarray, hlist, plist)
    comb_hlist = []
    comb_plist = []
    for (occs, holes, parts) in pairs_lists:
        hole_c = {}
        part_c = {}
        for y in range(len(holes)):
            hole_c[y+1] = list(combinations(holes, y+1))
            part_c[y+1] = list(combinations(parts, y+1))
        comb_hlist.append(hole_c)
        comb_plist.append(part_c)
    return comb_hlist, comb_plist
