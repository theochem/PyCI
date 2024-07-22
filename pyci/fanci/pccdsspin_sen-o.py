r"""
FanCI AP1roG module.
Pratiksha's modifications in previous pccds.py implementation (by Gabriela) to accomodate only Spin_sen-o in AP1roG.

"""

from itertools import permutations, combinations
from typing import Any, Union

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    "AP1roGSDSpin",
]


class AP1roGSDSpin(FanCI):
    r"""
    Pair coupled cluster doubles + singles_sen-o (or "AP1roGSDspin_sen-o") FanCI class.

    .. math::

        \left| \Psi_\text{pCCD+S} \right> =
        \prod_{i=1}^{N/2} \left(
            1 + \sum_{a=N/2+1}^K t_{i\bar{i};a\bar{a}} \hat{\tau}^{a\bar{a}}_{i\bar{i}}
        \right)
        \prod_{i=1}^{N/2} \left(
            1 + \sum_{a=N/2+1}^K t_{i;a\bar{a}} \hat{\tau}^{a}_{i} \hat{n}_{\bar{i}}
        \right)
        \left| \psi_0 \right> \,.

    This is AP1roG supplemented with single excitations that preserve the spin number, i.e.,
    no alpha to beta excitation is allowed, or vice-versa. Furthermore, the single excitations
    added are those that break an occupied pair of spin orbitals, increasing the seniority number.
    This determines the `sen-o` part of the name (from every occupied pair only one element of the
    pair is excited) [pCCDS1]_.

    .. [pCCDS1] Gaikwad, Pratiksha B., et al. "Coupled Cluster-Inspired Geminal Wavefunctions."
                *arXiv preprint* arXiv:2310.01764 (2023).


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

        # Every occupation vector is described in terms of the occupied/virtual indexes of the excitation
        # that generates it from the reference, labeled as holes/particle indexes.
        # The alpha and beta commponents of the occupation are stored separatelly, and used to determine
        # which elements correspond to paired exitations; which are also stored in separate lists.
        # In total three pairs of list are generated:
        # alpha hole/particle excitation indexes: `hlist_up` and `plist_up`
        # beta hole-particle excitation indexes: `hlist_dn` and `plist_dn`
        # hole-particle pair excitation indexes: `hlist` and `plist`
        hlist_up, plist_up, hlist_dn, plist_dn, hlist, plist = _get_hole_particle_indexes(self._wfn, self._ref_occs, self._sspace)
        # Change from spatial orbital to spin-orbital notation the elements in hlist_{up,dn} and
        # plist_{up,dn}. This gives hlist_ab and plist_ab.
        ab_lists = [self._sspace, hlist_up, plist_up, hlist_dn, plist_dn]
        hlist_ab, plist_ab = _make_alpha_plus_beta_strings(self._wfn, *ab_lists)
        # # Make a power set of the elements in hlist, plist for each occupation vector: comb_hlist
        # # and comb_plist.
        # comb_hlist, comb_plist = _make_pairexc_powerset(self._sspace, hlist, plist)

        # Save sub-class -specific attributes
        self._sspace_data = [(hlist_ab, plist_ab), (hlist, plist)]
        self._pspace_data = [(hlist_ab[:nproj], plist_ab[:nproj]), (hlist[:nproj], plist[:nproj])]

    def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
        r"""
        Compute the pCCDS overlap vector.

        The occupation vector is described in terms of the occupied/virtual indexes of the excitation
        that generates it from the reference. The indexes are further identified as corresponding to
        excitations of pairs or single spin-orbitals. The overlap is computed as the product of the
        permanents of the matrices formed by the pair and single excitations indexes.

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
        elif occs_array == "P":
            occs_array = self._pspace
            hlist_ab, plist_ab = self._pspace_data[0]
            hlist, plist = self._pspace_data[1]
        elif occs_array == "S":
            occs_array = self._sspace
            hlist_ab, plist_ab = self._sspace_data[0]
            hlist, plist = self._sspace_data[1]
        else:
            raise ValueError("invalid `occs_array` argument")

        # Reshape parameter array into two matrices, one for the pair excitations (t_ii) and another for
        # the single ones (t_i). Both have dimensions of number of (pair) particles times (pair) virtuals.
        t_ii = x[:self._wfn.nocc_up * self._wfn.nvir_up].reshape(self._wfn.nocc_up, self._wfn.nvir_up)
        t_i = x[self._wfn.nocc_up * self._wfn.nvir_up:].reshape(self._wfn.nocc, self._wfn.nvir)

        # Compute overlaps of occupation vectors
        y = np.zeros(occs_array.shape[0], dtype=pyci.c_double)

        # For each occupation vector determine which spin-orbital indexes correspond to pair-excitations
        # and which to single excitations from the reference.
        # To do so map the {holes,parts} indexes of the pair excitations from spatial to spin-orbital
        # notation and contrast them with the ones in the singles excitation description of the occs
        # {holes,parts}_ab. The diference gives the singles component of the excitation.
        for i, (occs, holes, parts) in enumerate(zip(occs_array, hlist, plist)):
            if holes.size > parts.size: # Occupation vector outside pCCSDSpin_sen-o space; e.g. Phi^ab_ii-bar
                continue
            elif holes.size < parts.size:
                max_pairs = min(holes.size, parts.size)
                holes = holes[:max_pairs]
                parts = parts[:max_pairs]

            # Make all pair excitations sets including the empty set.
            # For the empty pair, the holes/particles excitation is expressed fully as single excitations
            # from the reference, and the permanent of the singles is added to the overlap of the occupation
            # vector.
            for pair_exc_order in range(len(holes)+1):
                holes_comb = list(combinations(holes, pair_exc_order))
                parts_comb = list(combinations(parts, pair_exc_order))
                for _holes in holes_comb:
                    for _parts in parts_comb:
                        holes_ab, parts_ab = _get_singles_component(self._wfn, _holes, _parts, hlist_ab[i], plist_ab[i])

                        
                        # For AP1roGSDSpin_sen-o wavefunction, EACH singles excitation considered with respect to the reference state should satisfy two conditions.
                        ## 1. Whether EACH singles is obtained by breaking doubly occupied orbital pair which wasn't broken before. 
                              # For example: t_{i\bar{i}}^{a\bar{i}}*t_{i\bar{i}}^{i\bar{a}} is not a valid term since both singles are breaking the same occupied orbital pair (i\bar{i}).
                        ## 2. EACH singles should preserve the spin. \alpha -> \beta and viceversa is not allowed. This sets the off diagonal blocks of t_i^{a} to zero. 


                        ## Checking condition #1. 
                        # Check if holes_ab has spin orbital indices belonging to same doubly occupied spatial orbital. 
                        # If there are any such pairs then move on to next holes_ab, parts_ab 
                        olp  = permanent(t_ii[_holes, :][:, _parts])
                        if _check_paired_singles(holes_ab, self.nocc_up):
                            permant_S = 0
                            y[i] += (olp * permant_S)
                            continue
                        ## Checking condition #2.
                        elif len(holes_ab):
                            permant_S, valid_terms, invalid_terms = _permanent_singles(t_i, holes_ab, parts_ab)
                            olp *= permant_S
                        y[i] += olp


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


def _is_valid_singles(term: list, nocc: int, nvir: int) -> bool:
    r"""
    Check if given indices belong to spin-preserving singles.

    Parameters
    ----------
    term : list
        List of indices (hole, particle) contibuting to product 
        of singles term in permanent.

    nocc : int
        Total number of occupied spin orbitals. 

    nvir : int
        Total number of virtual spin orbitals. 
 
    NOTE
    ----
        p and q will help to check given singles belong to which quadrant of 
        singles parameter array. 
                 __                                __
                |                  |                 |
                | \alpha -> \alpha | \alpha -> \beta |
                |------------------|-----------------|
        t_i^a = | \beta -> \alpha  | \beta -> \beta  |
                |__                |              _ _|
 
    Returns:
    True/False : boolean
        If the given term is spin-preserving then return True otherwise False.
                
    """
    for (h, p) in term:
        # Check if the singles belong to the off-diagonal quadrant.
        # In the given product term, even if one singles is non-preserving
        # then that product term is invalid and won't contribute to the permament calculation.
        if (h >= nocc// 2 and p < nvir // 2) or (h < nocc// 2 and p >= nvir // 2):
            return False
        return True


def _permanent_singles(matrix: np.ndarray, hab: list, pab: list) -> (float, list, list):
    r"""
    Prepare list of 'valid' (containing all spin-preserving singles) product terms 
    of singles contributing to permanent calculation.

    Parameters
    ----------
    matrix : np.ndarray
        Parameter array corresponding to singles.
        Array of size (wfn.nocc_up + wfn.nocc_dn) x (wfn.nvir_up + wfn.nvir_dn)

    hab : list 
        List of holes of singles component obtained from the 
        _get_singles_component function.   
   
    pab : list 
        List of particles of singles component obtained from the 
        _get_singles_component function.   
    
    Returns
    -------
    permant : float
        Permanent of all possible terms obtained from 'valid' singles.
 
    invalid_terms : list
        List of indices (hole, particle) of singles contributing to the
        invalid (non-spin-preserving) product term.
        Example: 
            indices of occ_up spin orbitals = [0,1]
            indices of occ_dn spin orbitals = [2,3]
            indices of vir_up spin orbitals = [0,1,2,3]
            indices of vir_dn spin orbitals = [4,5,6,7]
            product term contributing to permanent = t_{2}^{3}t_{1}^{5}
            ==> return list[(2,3),(1,5)]
    valid_terms : list
        List of indices (hole, particle) of singles contributing to the
        valid (spin-preserving) product term.
        In the above example: 
            product term contributing to permanent = t_{2}^{5}t_{1}^{3}
            ==> return list[(2,5),(1,3)]
    """
    # Extract the submatrix of singles parameters corresponding to given indices
    submatrix = matrix[hab, :][:, pab]

    rows = np.arange(len(hab))
    p, q = matrix.shape
    permant = 0

    valid_terms = []
    invalid_terms = []

    # Generate all permutations of the rows
    for cols in permutations(rows):
        term = [(hab[r], pab[cols[r]]) for r in rows]
        
        if _is_valid_singles(term, p, q):
            permant += np.prod([matrix[b, c] for b, c in term])
            valid_terms.append(term)
        else:
            invalid_terms.append(term)
    
    return permant, valid_terms, invalid_terms



def _check_paired_singles(lst: list, occ_up: int) -> bool:
    r"""
    Check if the given (holes_ab) list of spin-orbital indices used to include 
    faux-singles (t_{iibar}^{aabar}) generates more than one singles exciting from
    same doubly-occupied orbital pair. 

    Parameters
    ----------
    lst : list
        List of all indices of holes contributing to faux-singles.
        This is the holes_ab list is obtained from _get_singles_component function.
     
    Returns
    -------
    True/False : boolean
        If singles exciting from same doubly-occupied orbital pairs are detected,
        then returns true.
    """
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if abs(lst[i] - lst[j]) == occ_up:
                return True
    return False


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
            # Alpha and beta occupations were excited or there is at least an alpha occupation being excited
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
    # Get the list of alpha and beta hole/particle indexes for each occupation vector:
    # `hlist_{up, dn}` and `plist_{up, dn}`.
    # Example:
    # reference determinant: Phi_0
    # Slater det. from 2 alpha electrons excited: a^+ b^+ j i Phi_0 = Phi^ab_ij
    # diff(Phi_0, Phi^ab_ij) --> ij : holes up indexes
    # diff(Phi^ab_ij, Phi_0) --> ab : particles up indexes
    nocc_up, nocc_dn = wfn.nocc_up, wfn.nocc_dn
    hlist_up = [np.setdiff1d(ref_occs[0], occs[0], assume_unique=1) for occs in occsarray]
    plist_up = [np.setdiff1d(occs[0], ref_occs[0], assume_unique=1) - nocc_up for occs in occsarray]
    hlist_dn = [np.setdiff1d(ref_occs[1], occs[1], assume_unique=1) for occs in occsarray]
    plist_dn = [np.setdiff1d(occs[1], ref_occs[1], assume_unique=1) - nocc_dn for occs in occsarray]

    # The list `hlist` (`plist`) stores the indexes that compose a double excitations where an alpha-beta pair
    # is removed (added), for every occ vector in occsarray.
    # In pCCSD there can be double excitations from Phi_0 which conserve the seniority (add and remove a pair)
    # and those which don't (remove a pair and add a broken pair, or viceversa).
    # The four simple cases are:
    # a) double excitation from Phi_0 removing a pair and adding another giving Phi^aa-bar_ii-bar. The
    # corresponding occupation vector contributes one hole, `i`, and one particle index `a` to each list.
    # b) double excitation from Phi_0 removing a pair and adding a broken pair, giving Phi^ab-bar_ii-bar.
    # This results in the occ vector contributing the index `i` to `hlist` and an empty element to `plist`.
    # c) double excitation from Phi_0 removing a broken pair and adding a pair, giving Phi^aa-bar_ij-bar.
    # For this occ vector, `plist` is filled, adding the index `a`, and `hlist` gets an empty element.
    # d) The occ vector corresponds to a single, or two single excitations from the reference (no pairs).
    # Then both `hlist` and `plist` get an empty elemet.
    hlist = [np.intersect1d(holes_up, holes_dn, assume_unique=1) for holes_up, holes_dn in zip(hlist_up, hlist_dn)]
    plist = [np.intersect1d(parts_up, parts_dn, assume_unique=1) for parts_up, parts_dn in zip(plist_up, plist_dn)]
    return hlist_up, plist_up, hlist_dn, plist_dn, hlist, plist


def _get_singles_component(wfn, holes, parts, holesab, partsab):
    """
    Determine the indexes of the spin-orbitals associated with single excitations as part
    of an n-th order excitation from a reference determinant.
    Use the difference between the alpha and beta holes (particles) indexes and the holes (particles)
    pair indexes of the occupation vector.


    Parameters
    ----------
    wfn : PyCI wavefunction
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
    # Make all pair excitations sets (`comb_hlist --> comb_plist`) for each occupation vector.
    # Does not include the empty set.
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
