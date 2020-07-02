# cython : language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
#
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


cdef class wavefunction:
    r"""
    Wave function class.

    """
    pass


cdef class one_spin_wfn(wavefunction):
    r"""
    One-spin wave function class.

    """
    cdef OneSpinWfn _obj

    @classmethod
    def from_file(cls, str filename not None):
        r"""
        Return a one_spin_wfn instance by loading a ONESPIN file.

        Parameters
        ----------
        filename : str
            ONESPIN file from which to load determinants.

        Returns
        -------
        wfn : one_spin_wfn
            One-spin wave function object.

        """
        cdef one_spin_wfn wfn = cls(2, 1)
        wfn._obj.from_file(filename.encode())
        return wfn

    @classmethod
    def from_det_array(cls, int_t nbasis, int_t nocc, uint_t[:, ::1] det_array not None):
        r"""
        Return a one_spin_wfn instance from an array of determinant bitstrings.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.
        det_array : np.ndarray(c_uint(n, nword))
            Array of determinants.

        Returns
        -------
        wfn : one_spin_wfn
            One-spin wave function object.

        """
        if det_array.shape[1] != nword_det(nbasis):
            raise IndexError('nbasis, nocc given do not match up with det_array dimensions')
        cdef one_spin_wfn wfn = cls(nbasis, nocc)
        wfn._obj.from_det_array(nbasis, nocc, det_array.shape[0], <uint_t *>(&det_array[0, 0]))
        return wfn

    @classmethod
    def from_occs_array(cls, int_t nbasis, int_t nocc, int_t[:, ::1] occs_array not None):
        r"""
        Return a single_spin_wfn instance from an array of occupied indices.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.
        occs_array : np.ndarray(c_int(n, nocc))
            Array of occupied indices.

        Returns
        -------
        wfn : doci_wfn
            One-spin wave function object.

        """
        cdef one_spin_wfn wfn = cls(nbasis, nocc)
        if occs_array.shape[1] != wfn._obj.nocc:
            raise IndexError('nbasis, nocc given do not match up with occs_array dimensions')
        wfn._obj.from_occs_array(nbasis, nocc, occs_array.shape[0], <int_t *>(&occs_array[0, 0]))
        return wfn

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._obj.nbasis

    def __init__(self, int_t nbasis, int_t nocc):
        r"""
        Initialize a one_spin_wfn instance.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.

        """
        if nbasis <= nocc or nocc < 1:
            raise ValueError('failed check: nbasis > nocc > 0')
        self._obj.init(nbasis, nocc)

    def __copy__(self):
        r"""
        Copy a one_spin_wfn instance.

        Returns
        -------
        wfn : one_spin_wfn
            One-spin wave function object.

        """
        cdef one_spin_wfn wfn = self.__class__(2, 1)
        wfn._obj.from_onespinwfn(self._obj)
        return wfn

    def copy(self):
        r"""
        Copy a one_spin_wfn instance.

        Returns
        -------
        wfn : one_spin_wfn
            One-spin wave function object.

        """
        return self.__copy__()

    def to_file(self, str filename not None):
        r"""
        Write a doci_wfn instance to a DOCI file.

        Parameters
        ----------
        filename : str
            Name of DOCI file to write.

        """
        self._obj.to_file(filename.encode())

    def __len__(self):
        r"""
        Return the number of determinants in the wave function.

        Returns
        -------
        ndet : int
            Number of determinants in the wave function.

        """
        return self._obj.ndet

    def __getitem__(self, int_t index):
        r""""
        Return the specified determinant from the wave function.

        Parameters
        ----------
        index : int
            Index of determinant to return.

        Returns
        -------
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        if index < 0 or index >= self._obj.ndet:
            raise IndexError('index out of range')
        cdef np.ndarray det_array = np.empty(self._obj.nword, dtype=c_uint)
        cdef uint_t[::1] det = det_array
        self._obj.copy_det(index, <uint_t *>(&det[0]))
        return det_array

    def to_det_array(self, int_t start=-1, int_t end=-1):
        r"""
        Convert the determinant bitstrings to an array of words (bitstrings).

        Parameters
        ----------
        start : int, optional
            Works as in python built-in range function.
        end : int, optional
            Works as in python built-in range function.

        Returns
        -------
        det_array : np.ndarray(c_uint(n, nword))
            Array of words (bitstrings).

        """
        # parse arguments (similar to python range())
        if start == -1:
            start = 0
            if end == -1:
                end = self._obj.ndet
        elif end == -1:
            end = start
            start = 0
        # check ranges
        if self._obj.ndet == 0 or start < 0 or end < start or self._obj.ndet < end:
            raise IndexError('\'start\', \'stop\' parameters out of range')
        # copy det array
        cdef const uint_t *det_ptr = self._obj.det_ptr(start)
        cdef np.ndarray det_array = np.array(<uint_t[:(end - start), :self._obj.nword]>det_ptr)
        return det_array

    def to_occs_array(self, int_t start=-1, int_t end=-1):
        r"""
        Convert the determinant bitstrings to an array of integers (occupied indices).

        Parameters
        ----------
        start : int, optional
            Works as in python built-in range function.
        end : int, optional
            Works as in python built-in range function.

        Returns
        -------
        occs_array : np.ndarray(c_int(n, nocc))
            Array of occupied indices.

        """
        # parse arguments (similar to python range())
        if start == -1:
            start = 0
            if end == -1:
                end = self._obj.ndet
        elif end == -1:
            end = start
            start = 0
        # check ranges
        if self._obj.ndet == 0 or start < 0 or end < start or self._obj.ndet < end:
            raise IndexError('\'start\', \'stop\' parameters out of range')
        # compute occs array
        cdef np.ndarray occs_array = np.zeros((end - start, self._obj.nocc), dtype=c_int)
        cdef int_t[:, ::1] occs = occs_array
        self._obj.to_occs_array(start, end, <int_t *>(&occs[0, 0]))
        return occs_array

    def index_det(self, uint_t[::1] det not None):
        r"""
        Return the index of the determinant in the wave function, or -1 if not found.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        index : int
            Index of determinant, or -1 if not found.

        """
        return self._obj.index_det(<uint_t *>(&det[0]))

    def add_det(self, uint_t[::1] det not None):
        r"""
        Add a determinant to the wavefunction.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        result : int
            Index of determinant, or -1 if addition fails.

        """
        return self._obj.add_det(<uint_t *>(&det[0]))

    def add_occs(self, int_t[::1] occs not None):
        r"""
        Add a determinant to the wavefunction from an array of occupied indices.

        Parameters
        ----------
        occs : np.ndarray(c_int(nocc))
            Indices of occupied electron pairs in determinant.

        Returns
        -------
        result : int
            Index of determinant, or -1 if addition fails.

        """
        return self._obj.add_det_from_occs(<int_t *>(&occs[0]))

    def add_hartreefock_det(self):
        r"""
        Add the Hartree-Fock determinant to the wave function.

        """
        self.add_occs(np.arange(self._obj.nocc, dtype=c_int))

    def add_all_dets(self):
        r"""
        Add all determinants to the wave function.

        """
        self._obj.add_all_dets()

    def add_excited_dets(self, *exc, uint_t[::1] det=None):
        r"""
        Add pair-excited determinants from a reference determinant to the wave function.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword)), default=hartreefock_determinant
            Reference determinant. If not provided, the Hartree-Fock determinant is used.
        exc : ints
            Excitation levels to add. Zero corresponds to no excitation.

        """
        # check excitation levels
        cdef int_t emax = min(self._obj.nocc, self._obj.nvir)
        cdef int_t[::1] excv = np.array(list(set(exc)), dtype=c_int)
        cdef int_t nexc = excv.shape[0], i, e
        for i in range(nexc):
            e = excv[i]
            if e < 0 or e > emax:
                raise ValueError('invalid excitation order e < 0 or e > min(nocc, nvir)')
        # default determinant is hartree-fock determinant
        if det is None:
            det = self.occs_to_det(np.arange(self._obj.nocc, dtype=c_int))
        # add determinants
        for i in range(nexc):
            self._obj.add_excited_dets(&det[0], excv[i])

    def reserve(self, int_t n):
        r"""
        Reserve space in memory for :math:`n` elements in the wave function instance.

        Parameters
        ----------
        n : int
            Number of elements for which to reserve space.

        """
        self._obj.reserve(n)

    def squeeze(self):
        r"""
        Free up any unused memory reserved by the wave function instance.

        This can help reduce memory usage if many determinants are individually added.

        """
        self._obj.squeeze()

    def truncated(self, int_t start=-1, int_t end=-1):
        r"""
        Return a truncated version of the wave function instance.

        Parameters
        ----------
        start : int, optional
            Works as in python built-in range function.
        end : int, optional
            Works as in python built-in range function.

        Returns
        -------
        wfn : one_spin_wfn
            Truncated wave function.

        """
        return self.__class__.from_det_array(
                self._obj.nbasis, self._obj.nocc, self.to_det_array(start, end),
                )

    def occs_to_det(self, int_t[::1] occs not None):
        r"""
        Convert an array of occupied indices to a determinant.

        Parameters
        ----------
        occs : np.ndarray(c_int(nocc))
            Indices of occupied electron pairs in determinant.

        Returns
        -------
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        cdef np.ndarray det_array = np.zeros(self._obj.nword, dtype=c_uint)
        cdef uint_t[::1] det = det_array
        fill_det(self._obj.nocc, <int_t *>(&occs[0]), <uint_t *>(&det[0]))
        return det_array

    def det_to_occs(self, uint_t[::1] det not None):
        r"""
        Convert a determinant to an array of occupied indices.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        occs : np.ndarray(c_int(nocc))
            Indices of occupied electron pairs in determinant.

        """
        cdef np.ndarray occs_array = np.empty(self._obj.nocc, dtype=c_int)
        cdef int_t[::1] occs = occs_array
        fill_occs(self._obj.nword, <uint_t *>(&det[0]), <int_t *>(&occs[0]))
        return occs_array

    def det_to_virs(self, uint_t[::1] det not None):
        r"""
        Convert a determinant to an array of virtual indices.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        virs : np.ndarray(c_int(nvir))
            Indices without occupied electron pairs in determinant.

        """
        cdef np.ndarray virs_array = np.empty(self._obj.nvir, dtype=c_int)
        cdef int_t[::1] virs = virs_array
        fill_virs(self._obj.nword, self._obj.nbasis, <uint_t *>(&det[0]), <int_t *>(&virs[0]))
        return virs_array

    def excite_det(self, int_t i, int_t a, uint_t[::1] det not None):
        r"""
        Return the excitation of a determinant from pair index :math:`i` to pair index :math:`a`.

        Parameters
        ----------
        i : int
            Electron pair "hole" index.
        a : int
            Electron pair "particle" index.
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        newdet : np.ndarray(c_uint(nword))
            Excited determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[::1] newdet = newdet_array
        excite_det(i, a, <uint_t *>(&newdet[0]))
        return newdet_array

    def setbit_det(self, int_t i, uint_t[::1] det not None):
        r"""
        Return the determinant with bit :math:`i` set.

        Parameters
        ----------
        i : int
            Bit to set.
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        newdet : np.ndarray(c_uint(nword))
            New determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[::1] newdet = newdet_array
        setbit_det(i, <uint_t *>(&newdet[0]))
        return newdet_array

    def clearbit_det(self, int_t i, uint_t[::1] det not None):
        r"""
        Return the determinant with bit :math:`i` cleared.

        Parameters
        ----------
        i : int
            Bit to clear.
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        newdet : np.ndarray(c_uint(nword))
            New determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[::1] newdet = newdet_array
        clearbit_det(i, <uint_t *>(&newdet[0]))
        return newdet_array

    def excite_det_inplace(self, int_t i, int_t a, uint_t[::1] det not None):
        r"""
        Excite a determinant from pair index :math:`i` to pair index :math:`a` in-place.

        Parameters
        ----------
        i : int
            Electron pair "hole" index.
        a : int
            Electron pair "particle" index.
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        excite_det(i, a, <uint_t *>(&det[0]))

    def setbit_det_inplace(self, int_t i, uint_t[::1] det not None):
        r"""
        Set a bit in a determinant in-place.

        Parameters
        ----------
        i : int
            Bit to set.
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        setbit_det(i, <uint_t *>(&det[0]))

    def clearbit_det_inplace(self, int_t i, uint_t[::1] det not None):
        r"""
        Clear a bit in a determinant in-place.

        Parameters
        ----------
        i : int
            Bit to clear.
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        clearbit_det(i, <uint_t *>(&det[0]))

    def popcnt_det(self, uint_t[::1] det not None):
        r"""
        Count the set bits in a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        popcnt : int
            Number of set bits.

        """
        return popcnt_det(self._obj.nword, <uint_t *>(&det[0]))

    def ctz_det(self, uint_t[::1] det not None):
        r"""
        Count the number of trailing zeros in a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        ctz : int
            Number of trailing zeros.

        """
        return ctz_det(self._obj.nword, <uint_t *>(&det[0]))

    def rank_det(self, uint_t[::1] det not None):
        r"""
        Compute the rank of a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        rank : int
            Rank value.

        """
        return self._obj.rank_det(<uint_t *>(&det[0]))

    def new_det(self):
        r"""
        Return a new determinant with all bits set to zero.

        Returns
        -------
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        return np.zeros(self._obj.nword, dtype=c_uint)

    def phase_single_det(self, uint_t[::1] det not None, int_t i, int_t a):
        r"""
        Compute the phase factor of a reference determinant with a singly-excited determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.
        i : int
            Electron "hole" index.
        a : int
            Electron "particle" index.

        Returns
        -------
        phase : (+1 | -1)
            Phase factor.

        """
        return phase_single_det(self._obj.nword, i, a, <uint_t *>(&det[0]))

    def phase_double_det(self, uint_t[::1] det not None, int_t i, int_t j, int_t a, int_t b):
        r"""
        Compute the phase factor of a reference determinant with a doubly-excited determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.
        i : int
            First electron "hole" index.
        j : int
            Second electron "hole" index.
        a : int
            First electron "particle" index.
        b : int
            Second electron "particle" index.

        Returns
        -------
        phase : (+1 | -1)
            Phase factor.

        """
        return phase_double_det(self._obj.nword, i, j, a, b, <uint_t *>(&det[0]))

    def compute_overlap(self, double[::1] coeffs not None, one_spin_wfn wfn not None,
        double[::1] wfn_coeffs not None):
        r"""
        Compute the overlap of this wave function with another wave function.

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            This wave function's coefficient vector.
        wfn : one_spin_wfn
            Wave function with which to compute overlap.
        wfn_coeffs : np.ndarray(c_double(len(wfn)))
            The other wave function's coefficient vector.

        Returns
        -------
        olp : float
            Overlap.

        """
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of self, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        if wfn._obj.ndet != wfn_coeffs.size:
            raise ValueError('dimensions of wfn, wfn_coeffs do not match')
        return self._obj.compute_overlap(&coeffs[0], wfn._obj, &wfn_coeffs[0])


cdef class two_spin_wfn(wavefunction):
    r"""
    Two-spin wave function class.

    """
    cdef TwoSpinWfn _obj

    @classmethod
    def from_file(cls, str filename not None):
        r"""
        Return a two_spin_wfn instance by loading a TWOSPIN file.

        Parameters
        ----------
        filename : str
            TWOSPIN file from which to load determinants.

        Returns
        -------
        wfn : two_spin_wfn
            FullCI wave function object.

        """
        cdef two_spin_wfn wfn = cls(2, 1, 1)
        wfn._obj.from_file(filename.encode())
        return wfn

    @classmethod
    def from_det_array(cls, int_t nbasis, int_t nocc_up, int_t nocc_dn, uint_t[:, :, ::1] det_array not None):
        r"""
        Return a two_spin_wfn instance from an array of determinant bitstrings.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of occupied spin-up indices.
        nocc_dn : int
            Number of occupied spin-down indices.
        det_array : np.ndarray(c_uint(n, 2, nword))
            Array of determinants.

        Returns
        -------
        wfn : two_spin_wfn
            Two-spin wave function object.

        """
        if det_array.shape[1] != 2 or det_array.shape[2] != nword_det(nbasis):
            raise IndexError('nbasis, nocc_{up,dn} given do not match up with det_array dimensions')
        cdef two_spin_wfn wfn = cls(nbasis, nocc_up, nocc_dn)
        wfn._obj.from_det_array(nbasis, nocc_up, nocc_dn, det_array.shape[0], <uint_t *>(&det_array[0, 0, 0]))
        return wfn

    @classmethod
    def from_occs_array(cls, int_t nbasis, int_t nocc_up, int_t nocc_dn, int_t[:, :, ::1] occs_array not None):
        r"""
        Return a two_spin_wfn instance from an array of occupied indices.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of occupied spin-up indices.
        nocc_dn : int
            Number of occupied spin-down indices.
        occs_array : np.ndarray(c_uint(n, 2, nocc_up))
            Array of occupied indices.

        Returns
        -------
        wfn : two_spin_wfn
            Two-spin wave function object.

        """
        if occs_array.shape[1] != 2 or occs_array.shape[2] != nocc_up:
            raise IndexError('nbasis, nocc_{up,dn} given do not match up with det_array dimensions')
        cdef two_spin_wfn wfn = cls(nbasis, nocc_up, nocc_dn)
        wfn._obj.from_occs_array(nbasis, nocc_up, nocc_dn, occs_array.shape[0], <int_t *>(&occs_array[0, 0, 0]))
        return wfn

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._obj.nbasis

    def __init__(self, int_t nbasis, int_t nocc_up, int_t nocc_dn):
        r"""
        Initialize a two_spin_wfn instance.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of spin-up occupied indices.
        nocc_dn : int
            Number of spin-down occupied indices.

        """
        if (nbasis < nocc_up or nbasis <= nocc_dn or nocc_up <= 0 or nocc_dn < 0 or nocc_up < nocc_dn):
            raise ValueError('failed check: nbasis >= nocc_up > 0, nbasis > nocc_dn >= 0, nocc_up >= nocc_dn')
        self._obj.init(nbasis, nocc_up, nocc_dn)

    def __len__(self):
        r"""
        Return the number of determinants in the wave function.

        Returns
        -------
        ndet : int
            Number of determinants in the wave function.

        """
        return self._obj.ndet

    def __getitem__(self, int_t index):
        r""""
        Return the specified determinant from the wave function.

        Parameters
        ----------
        index : int
            Index of determinant to return.

        Returns
        -------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        """
        if index < 0 or index >= self._obj.ndet:
            raise IndexError('index out of range')
        cdef np.ndarray det_array = np.empty((2, self._obj.nword), dtype=c_uint)
        cdef uint_t[:, ::1] det = det_array
        self._obj.copy_det(index, <uint_t *>(&det[0, 0]))
        return det_array

    def __copy__(self):
        r"""
        Copy a two_spin_wfn instance.

        Returns
        -------
        wfn : two_spin_wfn
            Two-spin wave function object.

        """
        cdef two_spin_wfn wfn = self.__class__(2, 1, 1)
        wfn._obj.from_twospinwfn(self._obj)
        return wfn

    def copy(self):
        r"""
        Copy a two_spin_wfn instance.

        Returns
        -------
        wfn : two_spin_wfn
            Two-spin wave function object.

        """
        return self.__copy__()

    def to_file(self, str filename not None):
        r"""
        Write a two_spin_wfn instance to a TWOSPIN file.

        Parameters
        ----------
        filename : str
            Name of TWOSPIN file to write.

        """
        self._obj.to_file(filename.encode())

    def to_det_array(self, int_t start=-1, int_t end=-1):
        r"""
        Convert the determinant bitstrings to an array of words (bitstrings).

        Parameters
        ----------
        start : int, optional
            Works as in python built-in range function.
        end : int, optional
            Works as in python built-in range function.

        Returns
        -------
        det_array : np.ndarray(c_uint(n, 2, nword))
            Array of words (bitstrings).

        """
        # parse arguments (similar to python range())
        if start == -1:
            start = 0
            if end == -1:
                end = self._obj.ndet
        elif end == -1:
            end = start
            start = 0
        # check ranges
        if self._obj.ndet == 0 or start < 0 or end < start or self._obj.ndet < end:
            raise IndexError('\'start\', \'stop\' parameters out of range')
        # copy det array
        cdef const uint_t *det_ptr = self._obj.det_ptr(start)
        cdef np.ndarray det_array = np.array(<uint_t[:(end - start), :2, :self._obj.nword]>det_ptr)
        return det_array

    def to_occs_array(self, int_t start=-1, int_t end=-1):
        r"""
        Convert the determinant bitstrings to an array of occupied indices (integers).

        Parameters
        ----------
        start : int, optional
            Works as in python built-in range function.
        end : int, optional
            Works as in python built-in range function.

        Returns
        -------
        occs_array : np.ndarray(c_uint(n, 2, nocc_up))
            Array of occupied indices (integers).

        """
        # parse arguments (similar to python range())
        if start == -1:
            start = 0
            if end == -1:
                end = self._obj.ndet
        elif end == -1:
            end = start
            start = 0
        # check ranges
        if self._obj.ndet == 0 or start < 0 or end < start or self._obj.ndet < end:
            raise IndexError('\'start\', \'stop\' parameters out of range')
        # compute occs array
        cdef np.ndarray occs_array = np.zeros((end - start, 2, self._obj.nocc_up), dtype=c_int)
        cdef int_t[:, :, ::1] occs = occs_array
        self._obj.to_occs_array(start, end, <int_t *>(&occs[0, 0, 0]))
        return occs_array

    def index_det(self, uint_t[:, ::1] det not None):
        r"""
        Return the index of the determinant in the wave function, or -1 if not found.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        Returns
        -------
        index : int
            Index of determinant, or -1 if not found.

        """
        return self._obj.index_det(<uint_t *>(&det[0, 0]))

    def add_det(self, uint_t[:, ::1] det not None):
        r"""
        Add a determinant to the wavefunction.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        Returns
        -------
        result : int
            Index of determinant, or -1 if addition fails.

        """
        return self._obj.add_det(<uint_t *>(&det[0, 0]))

    def add_occs(self, int_t[:, ::1] occs not None):
        r"""
        Add a determinant to the wavefunction from an array of occupied indices.

        Parameters
        ----------
        occs : np.ndarray(c_int(2, nocc_up))
            Indices of occupied spin-up and spin-down electrons in determinant.

        Returns
        -------
        result : int
            Index of determinant, or -1 if addition fails.

        """
        return self._obj.add_det_from_occs(<int_t *>(&occs[0, 0]))

    def add_hartreefock_det(self):
        r"""
        Add the Hartree-Fock determinant to the wave function.

        """
        cdef np.ndarray occs = np.zeros((2, self._obj.nocc_up), dtype=c_int)
        occs[0, :self._obj.nocc_up] = np.arange(self._obj.nocc_up, dtype=c_int)
        occs[1, :self._obj.nocc_dn] = np.arange(self._obj.nocc_dn, dtype=c_int)
        self.add_occs(occs)

    def add_all_dets(self):
        r"""
        Add all determinants to the wave function.

        """
        self._obj.add_all_dets()

    def add_excited_dets(self, *exc, uint_t[:, ::1] det=None):
        r"""
        Add excited determinants from a reference determinant to the wave function.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword)), default=hartreefock_determinant
            Reference determinant. If not provided, the Hartree-Fock determinant is used.
        exc : ints
            Excitation levels to add. Zero corresponds to no excitation.

        """
        # check excitation levels
        cdef int_t nocc = self._obj.nocc_up + self._obj.nocc_dn
        cdef int_t nvir = self._obj.nvir_up + self._obj.nvir_dn
        cdef int_t emax = min(nocc, nvir)
        cdef int_t emax_up = min(self._obj.nocc_up, self._obj.nvir_up)
        cdef int_t emax_dn = min(self._obj.nocc_dn, self._obj.nvir_dn)
        cdef int_t[::1] excv = np.array(list(set(exc)), dtype=c_int)
        cdef int_t nexc = excv.shape[0], i, e, a, b
        for i in range(nexc):
            e = excv[i]
            if e < 0 or e > emax:
                raise ValueError('invalid excitation order e < 0 or e > min(nocc, nvir)')
        # default determinant is hartree-fock determinant
        cdef np.ndarray occs
        if det is None:
            occs = np.zeros((2, self._obj.nocc_up), dtype=c_int)
            occs[0, :self._obj.nocc_up] = np.arange(self._obj.nocc_up, dtype=c_int)
            occs[1, :self._obj.nocc_dn] = np.arange(self._obj.nocc_dn, dtype=c_int)
            det = self.occs_to_det(occs)
        # add determinants
        for i in range(nexc):
            e = excv[i]
            a = min(e, self._obj.nocc_up, self._obj.nvir_up)
            b = e - a
            while (a >= 0) and (b <= emax_dn):
                self._obj.add_excited_dets(&det[0, 0], a, b)
                a -= 1
                b += 1

    def reserve(self, int_t n):
        r"""
        Reserve space in memory for :math:`n` elements in the wave function instance.

        Parameters
        ----------
        n : int
            Number of elements for which to reserve space.

        """
        self._obj.reserve(n)

    def squeeze(self):
        r"""
        Free up any unused memory reserved by the wave function instance.

        This can help reduce memory usage if many determinants are individually added.

        """
        self._obj.squeeze()

    def truncated(self, int_t start=-1, int_t end=-1):
        r"""
        Return a truncated version of the wave function instance.

        Parameters
        ----------
        start : int, optional
            Works as in python built-in range function.
        end : int, optional
            Works as in python built-in range function.

        Returns
        -------
        wfn : two_spin_wfn
            Truncated wave function.

        """
        return self.__class__.from_det_array(
                self._obj.nbasis, self._obj.nocc_up, self._obj.nocc_dn, self.to_det_array(start, end),
                )

    def occs_to_det(self, int_t[:, ::1] occs not None):
        r"""
        Convert an array of occupied indices to a determinant.

        Parameters
        ----------
        occs : np.ndarray(c_int(2, nocc_up))
            Indices of occupied spin-up and spin-down electrons in determinant.

        Returns
        -------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        """
        cdef np.ndarray det_array = np.zeros((2, self._obj.nword), dtype=c_uint)
        cdef uint_t[:, ::1] det = det_array
        fill_det(self._obj.nocc_up, <int_t *>(&occs[0, 0]), <uint_t *>(&det[0, 0]))
        fill_det(self._obj.nocc_dn, <int_t *>(&occs[1, 0]), <uint_t *>(&det[1, 0]))
        return det_array

    def det_to_occs(self, uint_t[:, ::1] det not None):
        r"""
        Convert a determinant to an array of occupied indices.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        Returns
        -------
        occs : np.ndarray(c_int(2, nocc_up))
            Indices of occupied spin-up and spin-down electrons in determinant.

        """
        cdef np.ndarray occs_array = np.zeros((2, self._obj.nocc_up), dtype=c_int)
        cdef int_t[:, ::1] occs = occs_array
        fill_occs(self._obj.nword, <uint_t *>(&det[0, 0]), <int_t *>(&occs[0, 0]))
        fill_occs(self._obj.nword, <uint_t *>(&det[1, 0]), <int_t *>(&occs[1, 0]))
        return occs_array

    def det_to_virs(self, uint_t[:, ::1] det not None):
        r"""
        Convert a determinant to an array of virtual indices.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        Returns
        -------
        virs : np.ndarray(c_int(2, nvir_dn))
            Spin-up and spin-down indices without occupied electrons in determinant.

        """
        cdef np.ndarray virs_array = np.zeros((2, self._obj.nvir_dn), dtype=c_int)
        cdef int_t[:, ::1] virs = virs_array
        fill_virs(self._obj.nword, self._obj.nbasis, <uint_t *>(&det[0, 0]), <int_t *>(&virs[0, 0]))
        fill_virs(self._obj.nword, self._obj.nbasis, <uint_t *>(&det[1, 0]), <int_t *>(&virs[1, 0]))
        return virs_array

    def excite_det(self, int_t i, int_t a, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Return the excitation of a determinant from index :math:`i` to index :math:`a`.

        Parameters
        ----------
        i : int
            Electron "hole" index.
        a : int
            Electron "particle" index.
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        Returns
        -------
        newdet : np.ndarray(c_uint(2, nword))
            Excited determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[:, ::1] newdet = newdet_array
        excite_det(i, a, <uint_t *>(&newdet[<int_t>spin, 0]))
        return newdet_array

    def setbit_det(self, int_t i, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Return the determinant with bit :math:`i` set.

        Parameters
        ----------
        i : int
            Bit to set.
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        Returns
        -------
        newdet : np.ndarray(c_uint(2, nword))
            New determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[:, ::1] newdet = newdet_array
        setbit_det(i, <uint_t *>(&newdet[<int_t>spin, 0]))
        return newdet_array

    def clearbit_det(self, int_t i, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Return the determinant with bit :math:`i` cleared.

        Parameters
        ----------
        i : int
            Bit to clear.
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        Returns
        -------
        newdet : np.ndarray(c_uint(2, nword))
            New determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[:, ::1] newdet = newdet_array
        clearbit_det(i, <uint_t *>(&newdet[<int_t>spin, 0]))
        return newdet_array

    def excite_det_inplace(self, int_t i, int_t a, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Excite a determinant from index :math:`i` to index :math:`a` in-place.

        Parameters
        ----------
        i : int
            Electron "hole" index.
        a : int
            Electron "particle" index.
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        """
        excite_det(i, a, <uint_t *>(&det[<int_t>spin, 0]))

    def setbit_det_inplace(self, int_t i, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Set a bit in a determinant in-place.

        Parameters
        ----------
        i : int
            Bit to set.
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        """
        setbit_det(i, <uint_t *>(&det[<int_t>spin, 0]))

    def clearbit_det_inplace(self, int_t i, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Clear a bit in a determinant in-place.

        Parameters
        ----------
        i : int
            Bit to clear.
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        """
        clearbit_det(i, <uint_t *>(&det[<int_t>spin, 0]))

    def popcnt_det(self, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Count the set bits in a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        Returns
        -------
        popcnt : int
            Number of set bits.

        """
        return popcnt_det(self._obj.nword, <uint_t *>(&det[<int_t>spin, 0]))

    def ctz_det(self, uint_t[:, ::1] det not None, SpinLabel spin):
        r"""
        Count the number of trailing zeros in a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        Returns
        -------
        ctz : int
            Number of trailing zeros.

        """
        return ctz_det(self._obj.nword, <uint_t *>(&det[<int_t>spin, 0]))

    def phase_single_det(self, uint_t[:, ::1] det not None, int_t i, int_t a, SpinLabel spin):
        r"""
        Compute the phase factor of a reference determinant with a singly-excited determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        i : int
            Electron "hole" index.
        a : int
            Electron "particle" index.
        spin : (SPIN_UP | SPIN_DN)
            The spin upon which to perform the operation.

        Returns
        -------
        phase : (+1 | -1)
            Phase factor.

        """
        return phase_single_det(self._obj.nword, i, a, <uint_t *>(&det[<int_t>spin, 0]))

    def phase_double_det(self, uint_t[:, ::1] det not None, int_t i, int_t j, int_t a, int_t b,
        SpinLabel spin1, SpinLabel spin2):
        r"""
        Compute the phase factor of a reference determinant with a doubly-excited determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.
        i : int
            First electron "hole" index.
        j : int
            Second electron "hole" index.
        a : int
            First electron "particle" index.
        b : int
            Second electron "particle" index.
        spin1 : (SPIN_UP | SPIN_DN)
            Spin of first excitation.
        spin2 : (SPIN_UP | SPIN_DN)
            Spin of second excitation.

        Returns
        -------
        phase : (+1 | -1)
            Phase factor.

        """
        if spin1 == spin2:
            return phase_double_det(self._obj.nword, i, j, a, b, <uint_t *>(&det[<int_t>spin1, 0]))
        else:
            return phase_single_det(self._obj.nword, i, a, <uint_t *>(&det[<int_t>spin1, 0])) \
                    * phase_single_det(self._obj.nword, j, b, <uint_t *>(&det[<int_t>spin2, 0]))

    def rank_det(self, uint_t[:, ::1] det not None):
        r"""
        Compute the rank of a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        Returns
        -------
        rank : int
            Rank value.

        """
        return self._obj.rank_det(<uint_t *>(&det[0, 0]))

    def new_det(self):
        r"""
        Return a new determinant with all bits set to zero.

        Returns
        -------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        """
        return np.zeros((2, self._obj.nword), dtype=c_uint)

    def compute_overlap(self, double[::1] coeffs not None, fullci_wfn wfn not None,
        double[::1] wfn_coeffs not None):
        r"""
        Compute the overlap of this wave function with another wave function.

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            This wave function's coefficient vector.
        wfn : fullci_wfn
            Wave function with which to compute overlap.
        wfn_coeffs : np.ndarray(c_double(len(wfn)))
            This wave function's coefficient vector.

        Returns
        -------
        olp : float
            Overlap.

        """
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of self, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        if wfn._obj.ndet != wfn_coeffs.size:
            raise ValueError('dimensions of wfn, wfn_coeffs do not match')
        return self._obj.compute_overlap(&coeffs[0], wfn._obj, &wfn_coeffs[0])


cdef class doci_wfn(one_spin_wfn):
    r"""
    DOCI wave function class.

    Attributes
    ----------
    nbasis : int
        Number of orbital basis functions.
    nocc : int
        Number of occupied indices.
    nocc_up : int
        Number of occupied spin-up indices.
    nocc_dn : int
        Number of occupied spin-down indices.
    nvir : int
        Number of virtual indices.
    nvir_up : int
        Number of virtual spin-up indices.
    nvir_dn : int
        Number of virtual spin-down indices.

    """

    @property
    def nocc(self):
        r"""
        Number of occupied indices.

        """
        return self._obj.nocc * 2

    @property
    def nocc_up(self):
        r"""
        Number of spin-up occupied indices.

        """
        return self._obj.nocc

    @property
    def nocc_dn(self):
        r"""
        Number of spin-down occupied indices.

        """
        return self._obj.nocc

    @property
    def nvir(self):
        r"""
        Number of virtual indices.

        """
        return self._obj.nvir * 2

    @property
    def nvir_up(self):
        r"""
        Number of spin-up virtual indices.

        """
        return self._obj.nvir

    @property
    def nvir_dn(self):
        r"""
        Number of spin-down virtual indices.

        """
        return self._obj.nvir

    def to_fullci_wfn(self):
        r"""
        Convert a DOCI wave function to a FullCI wave function.

        Returns
        -------
        wfn : fullci_wfn
            FullCI wave funtion object.

        """
        cdef fullci_wfn wfn = fullci_wfn(2, 1, 1)
        wfn._obj.from_onespinwfn(self._obj)
        return wfn

    def to_genci_wfn(self):
        r"""
        Convert a DOCI wave function to a Generalized CI wave function.

        Returns
        -------
        wfn : genci_wfn
            Generalized CI wave funtion object.

        """
        return self.to_fullci_wfn().to_genci_wfn()

    def compute_rdms(self, double[::1] coeffs not None, str mode='d'):
        r"""
        Compute the 1- and 2- particle reduced density matrices (RDMs) of a wave function.

        .. math::

            d_{pq} = \left<p|q\right>

        .. math::

            D_{pqrs} = \left<pq|rs\right>

        This method returns two nbasis-by-nbasis matrices by default, which include the
        unique seniority-zero and seniority-two terms from the full 2-RDMs:

        .. math::

            D_0 = \left<pp|qq\right>

        .. math::

            D_2 = \left<pq|pq\right>

        The diagonal elements of :math:`D_0` are equal to the 1-RDM elements :math:`\left<p|p\right>`.

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        mode : ('d' | 'r' | 'g'), default='d'
            Whether to return the :math:`D_0` and :math:`D2` matrices, or the restricted or
            generalized 1- and 2- particle RDMs.

        Returns
        -------
        rdm1 : np.ndarray(c_double(...))
            :math:`D_0` matrix or 1-particle reduced density matrix, :math:`d`.
        rdm2 : np.ndarray(c_double(...))
            :math:`D_2` matrix or 2-particle reduced density matrix, :math:`D`.

        """
        cdef tuple result = None
        mode = mode.lower()[0]
        # Check params
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif mode not in 'drg':
            raise ValueError('mode must be one of \'d\', \'r\', \'g\'')
        # Compute DOCI matrices
        cdef np.ndarray d0_array = np.zeros((self._obj.nbasis, self._obj.nbasis), dtype=c_double)
        cdef np.ndarray d2_array = np.zeros((self._obj.nbasis, self._obj.nbasis), dtype=c_double)
        cdef double[:, ::1] d0 = d0_array
        cdef double[:, ::1] d2 = d2_array
        self._obj.compute_rdms_doci(<double *>(&coeffs[0]), <double *>(&d0[0, 0]), <double *>(&d2[0, 0]))
        # DOCI matrices
        if mode == 'd':
            result = d0_array, d2_array
        # Restricted RDMs
        elif mode == 'r':
            result = self.generate_restricted_rdms(d0, d2)
        # Generalized RDMs
        elif mode == 'g':
            result = self.generate_generalized_rdms(d0, d2)
        # Invalid mode
        else:
            raise ValueError('mode must be one of \'d\', \'r\', \'g\'')
        return result

    def compute_enpt2(self, hamiltonian ham not None, double[::1] coeffs not None,
        double energy, double eps=1.0e-6):
        r"""
        Compute the second-order Epstein-Nesbet perturbation theory correction to the energy.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        energy : float
            Variational energy.
        eps : float, default=1.0e-6
            Threshold value for which determinants to include.

        Returns
        -------
        enpt2_energy : float
           ENPT2-corrected energy.

        """
        cdef double result = np.nan
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Restricted DOCI
        elif isinstance(ham, restricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            result = self._obj.compute_enpt2_doci(
                    <double *>(&ham._one_mo[0, 0]),
                    <double *>(&ham._two_mo[0, 0, 0, 0]),
                    <double *>(&coeffs[0]),
                    energy - ham._ecore,
                    eps,
                    ) + energy
        # Unrestricted DOCI
        elif isinstance(ham, unrestricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Generalized DOCI
        elif isinstance(ham, generalized_ham):
            if self._obj.nbasis * 2 != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Invalid Hamiltonian
        else:
            raise TypeError('invalid ham type')
        return result

    def run_hci(self, hamiltonian ham not None, double[::1] coeffs not None, double eps):
        r"""
        Run an iteration of heat-bath CI.

        Adds all determinants connected to determinants currently in the wave function,
        if they satisfy the criteria
        :math:`|\left<f|H|d\right> c_d| > \epsilon` for :math:`f = P^\dagger_i P_a d`.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        eps : float
            Threshold value for which determinants to include.

        Returns
        -------
        n : int
            Number of determinants added to wave function.

        """
        cdef int_t n = 0
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif ham._h is None:
            raise AttributeError('seniority-zero integrals were not computed')
        # Restricted DOCI
        elif isinstance(ham, restricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            n = self._obj.run_hci_doci(<double *>(&ham._v[0, 0]), <double *>(&coeffs[0]), eps)
        # Unrestricted DOCI
        elif isinstance(ham, unrestricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Generalized DOCI
        elif isinstance(ham, generalized_ham):
            if self._obj.nbasis * 2 != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Invalid Hamiltonian
        else:
            raise TypeError('invalid ham type')
        return n

    @staticmethod
    def generate_restricted_rdms(double[:, ::1] d0 not None, double[:, ::1] d2 not None):
        r"""
        Generate restricted one- and two- particle RDMs from the :math:`D_0` and :math:`D_2` matrices.

        Parameters
        ----------
        d0 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_0` matrix.
        d2 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_2` matrix.

        Returns
        -------
        rdm1 : np.ndarray(c_double(nbasis, nbasis))
            Restricted 1-particle RDM.
        rdm2 : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Restricted 2-particle RDM.

        """
        # TODO: test this!
        if not (d0.shape[0] == d0.shape[1] == d2.shape[0] == d2.shape[1]):
            raise ValueError('dimensions of d0, d2 do not match')
        cdef int_t nbasis = d0.shape[0], p, q
        cdef np.ndarray rdm1_array = np.zeros((nbasis, nbasis), dtype=c_double)
        cdef np.ndarray rdm2_array = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=c_double)
        cdef double[:, ::1] rdm1 = rdm1_array
        cdef double[:, :, :, ::1] rdm2 = rdm2_array
        for p in range(nbasis):
            rdm1[p, p] += d0[p, p]
            for q in range(nbasis):
                rdm2[p, p, q, q] += d0[p, q]
                rdm2[p, q, p, q] += d2[p, q]
        rdm1_array *= 2
        rdm2_array -= np.transpose(rdm2_array, axes=(1, 0, 2, 3))
        rdm2_array -= np.transpose(rdm2_array, axes=(0, 1, 3, 2))
        return rdm1_array, rdm2_array

    @staticmethod
    def generate_generalized_rdms(double[:, ::1] d0 not None, double[:, ::1] d2 not None):
        r"""
        Generate generalized one- and two- particle RDMs from the :math:`D_0` and :math:`D_2` matrices.

        Parameters
        ----------
        d0 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_0` matrix.
        d2 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_2` matrix.

        Returns
        -------
        rdm1 : np.ndarray(c_double(2 * nbasis, 2 * nbasis))
            Generalized 1-particle RDM.
        rdm2 : np.ndarray(c_double(2 * nbasis, 2 * nbasis, 2 * nbasis, 2 * nbasis))
            Generalized 2-particle RDM.

        """
        if not (d0.shape[0] == d0.shape[1] == d2.shape[0] == d2.shape[1]):
            raise ValueError('dimensions of d0, d2 do not match')
        cdef int_t nbasis = d0.shape[0]
        cdef int_t nspin = nbasis * 2, p, q
        cdef np.ndarray rdm1_array = np.zeros((nspin, nspin), dtype=c_double)
        cdef np.ndarray rdm2_array = np.zeros((nspin, nspin, nspin, nspin), dtype=c_double)
        cdef double[:, :] rdm1_a = rdm1_array[:nbasis, :nbasis]
        cdef double[:, :] rdm1_b = rdm1_array[nbasis:, nbasis:]
        cdef double[:, :, :, :] rdm2_abab = rdm2_array[:nbasis, nbasis:, :nbasis, nbasis:]
        cdef double[:, :, :, :] rdm2_baba = rdm2_array[nbasis:, :nbasis, nbasis:, :nbasis]
        cdef double[:, :, :, :] rdm2_aaaa = rdm2_array[:nbasis, :nbasis, :nbasis, :nbasis]
        cdef double[:, :, :, :] rdm2_bbbb = rdm2_array[nbasis:, nbasis:, nbasis:, nbasis:]
        for p in range(nbasis):
            rdm1_a[p, p] += d0[p, p]
            rdm1_b[p, p] += d0[p, p]
            for q in range(nbasis):
                rdm2_abab[p, p, q, q] += d0[p, q]
                rdm2_baba[p, p, q, q] += d0[p, q]
                rdm2_aaaa[p, q, p, q] += d2[p, q]
                rdm2_bbbb[p, q, p, q] += d2[p, q]
                rdm2_abab[p, q, p, q] += d2[p, q]
                rdm2_baba[p, q, p, q] += d2[p, q]
        rdm2_array -= np.transpose(rdm2_array, axes=(1, 0, 2, 3))
        rdm2_array -= np.transpose(rdm2_array, axes=(0, 1, 3, 2))
        rdm2_array *= 0.5
        return rdm1_array, rdm2_array


cdef class fullci_wfn(two_spin_wfn):
    r"""
    FullCI wave function class.

    Attributes
    ----------
    nbasis : int
        Number of orbital basis functions.
    nocc : int
        Number of occupied indices.
    nocc_up : int
        Number of occupied spin-up indices.
    nocc_dn : int
        Number of occupied spin-down indices.
    nvir : int
        Number of virtual indices.
    nvir_up : int
        Number of virtual spin-up indices.
    nvir_dn : int
        Number of virtual spin-down indices.

    """

    @property
    def nocc(self):
        r"""
        Number of occupied indices.

        """
        return self._obj.nocc_up + self._obj.nocc_dn

    @property
    def nocc_up(self):
        r"""
        Number of spin-up occupied indices.

        """
        return self._obj.nocc_up

    @property
    def nocc_dn(self):
        r"""
        Number of spin-down occupied indices.

        """
        return self._obj.nocc_dn

    @property
    def nvir(self):
        r"""
        Number of virtual indices.

        """
        return self._obj.nvir_up + self._obj.nvir_dn

    @property
    def nvir_up(self):
        r"""
        Number of spin-up virtual indices.

        """
        return self._obj.nvir_up

    @property
    def nvir_dn(self):
        r"""
        Number of spin-down virtual indices.

        """
        return self._obj.nvir_dn

    def to_genci_wfn(self):
        r"""
        Convert a FullCI wave function to a Generalized CI wave function.

        Returns
        -------
        wfn : genci_wfn
            Generalized CI wave funtion object.

        """
        cdef genci_wfn wfn = genci_wfn(2, 1)
        wfn._obj.from_twospinwfn(self._obj)
        return wfn

    def compute_rdms(self, double[::1] coeffs not None, str mode='r'):
        r"""
        Compute the 1- and 2- particle reduced density matrices (RDMs) of a wave function.

        .. math::

            d_{pq} = \left<p|q\right>

        .. math::

            D_{pqrs} = \left<pq|rs\right>

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        mode : ('r' | 'g'), default='r'
            Whether to return the restricted or generalized RDMs.

        Returns
        -------
        rdm1 : np.ndarray(c_double(:, :))
            1-particle reduced density matrix, :math:`d`.
        rdm2 : np.ndarray(c_double(:, :, :, :))
            2-particle reduced density matrix, :math:`D`.

        """
        mode = mode.lower()[0]
        cdef tuple result = None
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif mode not in 'rg':
            raise ValueError('mode must be one of \'r\', \'g\'')
        # Compute restricted RDMs
        cdef tuple nbasis2 = (self._obj.nbasis, self._obj.nbasis)
        cdef tuple nbasis4 = (self._obj.nbasis, self._obj.nbasis, self._obj.nbasis, self._obj.nbasis)
        cdef np.ndarray rdm1_array = np.zeros(nbasis2, dtype=c_double)
        cdef np.ndarray rdm2_array = np.zeros(nbasis4, dtype=c_double)
        cdef double[:, ::1] rdm1 = rdm1_array
        cdef double[:, :, :, ::1] rdm2 = rdm2_array
        self._obj.compute_rdms_fullci(<double *>(&coeffs[0]), <double *>(&rdm1[0, 0]), <double *>(&rdm2[0, 0, 0, 0]))
        # Restricted RDMs
        if mode == 'r':
            result = rdm1_array, rdm2_array
        # Generalized RDMs
        elif mode == 'g':
            result = fullci_wfn.generate_generalized_rdms(rdm1, rdm2)
        # Invalid mode
        else:
            raise ValueError('mode must be one of \'r\', \'g\'')
        return result

    def compute_enpt2(self, hamiltonian ham not None, double[::1] coeffs not None,
        double energy, double eps=1.0e-6):
        r"""
        Compute the second-order Epstein-Nesbet perturbation theory correction to the energy.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        energy : float
            Variational energy.
        eps : float, default=1.0e-6
            Threshold value for which determinants to include.

        Returns
        -------
        enpt2_energy : float
           ENPT2-corrected energy.

        """
        cdef double result = np.nan
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Restricted FullCI
        elif isinstance(ham, restricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            result = self._obj.compute_enpt2_fullci(
                    <double *>(&ham._one_mo[0, 0]),
                    <double *>(&ham._two_mo[0, 0, 0, 0]),
                    <double *>(&coeffs[0]),
                    energy - ham._ecore,
                    eps,
                    ) + energy
        # Unrestricted FullCI
        elif isinstance(ham, unrestricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Generalized FullCI
        elif isinstance(ham, generalized_ham):
            if self._obj.nbasis * 2 != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Invalid Hamiltonian
        else:
            raise TypeError('invalid ham type')
        return result

    def run_hci(self, hamiltonian ham not None, double[::1] coeffs not None, double eps):
        r"""
        Run an iteration of heat-bath CI.

        Adds all determinants connected to determinants currently in the wave function,
        if they satisfy the criteria
        :math:`|\left<f|H|d\right> c_d| > \epsilon` for :math:`f = a^\dagger_i a_a d` or
        :math:`f = a^\dagger_i a^\dagger_j a_b a_a d`.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        eps : float
            Threshold value for which determinants to include.

        Returns
        -------
        n : int
            Number of determinants added to wave function.

        """
        cdef int_t n = 0
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif ham._h is None:
            raise AttributeError('seniority-zero integrals were not computed')
        # Restricted FullCI
        elif isinstance(ham, restricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            n = self._obj.run_hci_fullci(
                    <double *>(&ham._one_mo[0, 0]),
                    <double *>(&ham._two_mo[0, 0, 0, 0]),
                    <double *>(&coeffs[0]),
                    eps,
                    )
        # Unrestricted FullCI
        elif isinstance(ham, unrestricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        # Generalized FullCI
        elif isinstance(ham, generalized_ham):
            if self._obj.nbasis * 2 != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            raise NotImplementedError
        return n

    @staticmethod
    def generate_generalized_rdms(double[:, ::1] rdm1 not None, double[:, :, :, ::1] rdm2 not None):
        r"""
        Generate generalized one- and two- particle RDMs from the restricted RDMs.

        Parameters
        ----------
        rdm1 : np.ndarray(c_double(:, :))
            Restricted 1-particle RDM.
        rdm2 : np.ndarray(c_double(:, :, :, :))
            Restricted 2-particle RDM.

        Returns
        -------
        rdm1 : np.ndarray(c_double(2 * nbasis, 2 * nbasis))
            Generalized 1-particle RDM.
        rdm2 : np.ndarray(c_double(2 * nbasis, 2 * nbasis, 2 * nbasis, 2 * nbasis))
            Generalized 2-particle RDM.

        """
        raise NotImplementedError


cdef class genci_wfn(one_spin_wfn):
    r"""
    Generalized CI wave function class.

    Attributes
    ----------
    nbasis : int
        Number of orbital basis functions.
    nocc : int
        Number of occupied indices.
    nocc_up : int
        Number of occupied spin-up indices.
    nocc_dn : int
        Number of occupied spin-down indices.
    nvir : int
        Number of virtual indices.
    nvir_up : int
        Number of virtual spin-up indices.
    nvir_dn : int
        Number of virtual spin-down indices.

    """

    @property
    def nocc(self):
        r"""
        Number of occupied indices.

        """
        return self._obj.nocc

    @property
    def nocc_up(self):
        r"""
        Number of spin-up occupied indices.

        """
        return self._obj.nocc

    @property
    def nocc_dn(self):
        r"""
        Number of spin-down occupied indices.

        """
        return 0

    @property
    def nvir(self):
        r"""
        Number of virtual indices.

        """
        return self._obj.nvir

    @property
    def nvir_up(self):
        r"""
        Number of spin-up virtual indices.

        """
        return self._obj.nvir

    @property
    def nvir_dn(self):
        r"""
        Number of spin-down virtual indices.

        """
        return 0

    def compute_rdms(self, double[::1] coeffs not None, str mode='g'):
        r"""
        Compute the 1- and 2- particle reduced density matrices (RDMs) of a wave function.

        .. math::

            d_{pq} = \left<p|q\right>

        .. math::

            D_{pqrs} = \left<pq|rs\right>

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.

        Returns
        -------
        rdm1 : np.ndarray(c_double(nbasis, nbasis))
            Generalized 1-particle reduced density matrix, :math:`d`.
        rdm2 : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Generalized 2-particle reduced density matrix, :math:`D`.

        """
        mode = mode.lower()[0]
        # Check parameters
        if mode != 'g':
            raise ValueError('mode must be \'g\'')
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        # Compute generalized RDMs
        cdef tuple nbasis2 = (self._obj.nbasis, self._obj.nbasis)
        cdef tuple nbasis4 = (self._obj.nbasis, self._obj.nbasis, self._obj.nbasis, self._obj.nbasis)
        cdef np.ndarray rdm1_array = np.zeros(nbasis2, dtype=c_double)
        cdef np.ndarray rdm2_array = np.zeros(nbasis4, dtype=c_double)
        cdef double[:, ::1] rdm1 = rdm1_array
        cdef double[:, :, :, ::1] rdm2 = rdm2_array
        self._obj.compute_rdms_genci(<double *>(&coeffs[0]), <double *>(&rdm1[0, 0]), <double *>(&rdm2[0, 0, 0, 0]))
        return rdm1_array, rdm2_array

    def compute_enpt2(self, generalized_ham ham not None, double[::1] coeffs not None,
        double energy, double eps=1.0e-6):
        r"""
        Compute the second-order Epstein-Nesbet perturbation theory correction to the energy.

        Parameters
        ----------
        ham : generalized_ham
            Hamiltonian object.
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        energy : float
            Variational energy.
        eps : float, default=1.0e-6
            Threshold value for which determinants to include.

        Returns
        -------
        enpt2_energy : float
           ENPT2-corrected energy.

        """
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif self._obj.nbasis != ham._nbasis:
            raise ValueError('dimensions of wfn, ham do not match')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Generalized CI
        return self._obj.compute_enpt2_genci(
                <double *>(&ham._one_mo[0, 0]),
                <double *>(&ham._two_mo[0, 0, 0, 0]),
                <double *>(&coeffs[0]),
                energy - ham._ecore,
                eps,
                ) + energy

    def run_hci(self, generalized_ham ham not None, double[::1] coeffs not None, double eps):
        r"""
        Run an iteration of heat-bath CI.

        Adds all determinants connected to determinants currently in the wave function,
        if they satisfy the criteria
        :math:`|\left<f|H|d\right> c_d| > \epsilon` for :math:`f = a^\dagger_i a_a d` or
        :math:`f = a^\dagger_i a^\dagger_j a_b a_a d`.

        Parameters
        ----------
        ham : generalized_ham
            Hamiltonian object.
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.
        eps : float
            Threshold value for which determinants to include.

        Returns
        -------
        n : int
            Number of determinants added to wave function.

        """
        # Check parameters
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif self._obj.nbasis != ham._nbasis:
            raise ValueError('dimensions of wfn, ham do not match')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Generalized CI
        return self._obj.run_hci_genci(
                <double *>(&ham._one_mo[0, 0]),
                <double *>(&ham._two_mo[0, 0, 0, 0]),
                <double *>(&coeffs[0]),
                eps,
                )


# vim: set ft=pyrex:
