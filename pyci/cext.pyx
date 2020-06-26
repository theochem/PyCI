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

r"""
PyCI C extension module.

"""

from libc.stdint cimport int64_t, uint64_t
from libcpp.vector cimport vector

cimport numpy as np

import numpy as np

from scipy.sparse import csr_matrix

from pyci.common cimport binomial, fill_det, fill_occs, fill_virs
from pyci.common cimport excite_det, setbit_det, clearbit_det, popcnt_det, ctz_det
from pyci.common cimport phase_single_det, phase_double_det, rank_det
from pyci.doci cimport DOCIWfn
from pyci.fullci cimport FullCIWfn
from pyci.solve cimport SparseOp
from pyci import fcidump


__all__ = [
    'SpinLabel',
    'SPIN_UP',
    'SPIN_DN',
    '_get_version',
    'hamiltonian',
    'doci_wfn',
    'fullci_wfn',
    'sparse_op',
    ]


ctypedef int64_t int_t
ctypedef uint64_t uint_t


cdef np.dtype c_int = np.dtype(np.int64)
cdef np.dtype c_uint = np.dtype(np.uint64)
cdef np.dtype c_double = np.dtype(np.double)


cpdef enum SpinLabel:
    SPIN_UP = 0
    SPIN_DN = 1


def _get_version():
    r"""
    Return the version number string from the C extension.

    Returns
    -------
    version : str
        Version number string.

    """
    return PYCI_VERSION


cdef class hamiltonian:
    r"""
    Hamiltonian class.

    .. math::

        H = \sum_{pq}{t_{pq} a^\dagger_p a_q} + \sum_{pqrs}{g_{pqrs} a^\dagger_p a^\dagger_q a_s a_r}

    .. math::

        H = \sum_{p}{h_p N_p} + \sum_{p \neq q}{v_{pq} P^\dagger_p P_q} + \sum_{pq}{w_{pq} N_p N_q}

    where

    .. math::

        h_{p} = \left<p|T|p\right> = t_{pp}

    .. math::

        v_{pq} = \left<pp|V|qq\right> = g_{ppqq}

    .. math::

        w_{pq} = 2 \left<pq|V|pq\right> - \left<pq|V|qp\right> = 2 * g_{pqpq} - g_{pqqp}

    Attributes
    ----------
    nbasis : int
        Number of orbital basis functions.
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.
    h : np.ndarray(c_double(nbasis))
        Seniority-zero one-electron integrals.
    v : np.ndarray(c_double(nbasis, nbasis))
        Seniority-zero two-electron integrals.
    w : np.ndarray(c_double(nbasis, nbasis))
        Seniority-two two-electron integrals.

    """
    cdef int_t _nbasis
    cdef double _ecore
    cdef double[:, ::1] _one_mo
    cdef double[:, :, :, ::1] _two_mo
    cdef double[::1] _h
    cdef double[:, ::1] _v
    cdef double[:, ::1] _w

    @staticmethod
    def from_file(object filename not None, bint keep_mo=True, bint doci=True):
        r"""
        Return a Hamiltonian instance by loading an FCIDUMP file.

        Parameters
        ----------
        filename : str
            FCIDUMP file from which to load integrals.
        keep_mo : bool, default=True
            Whether to keep the full MO arrays.
        doci : bool, default=True
            Whether to compute the seniority-zero integral arrays.

        Returns
        -------
        ham : hamiltonian
            Hamiltonian object.

        """
        return hamiltonian(*fcidump.read(filename)[:3], keep_mo=keep_mo, doci=doci)

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._nbasis

    @property
    def ecore(self):
        r"""
        Constant/"zero-electron" integral.

        """
        return self._ecore

    @property
    def one_mo(self):
        r"""
        Full one-electron integral array.

        """
        if self._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        return np.asarray(self._one_mo)

    @property
    def two_mo(self):
        r"""
        Full two-electron integral array.

        """
        if self._two_mo is None:
            raise AttributeError('full integral arrays were not saved')
        return np.asarray(self._two_mo)

    @property
    def h(self):
        r"""
        Seniority-zero one-electron integrals.

        """
        if self._h is None:
            raise AttributeError('seniority-zero integrals were not computed')
        return np.asarray(self._h)

    @property
    def v(self):
        r"""
        Seniority-zero two-electron integrals.

        """
        if self._v is None:
            raise AttributeError('seniority-zero integrals were not computed')
        return np.asarray(self._v)

    @property
    def w(self):
        r"""
        Seniority-two two-electron integrals.

        """
        if self._w is None:
            raise AttributeError('seniority-zero integrals were not computed')
        return np.asarray(self._w)

    def __init__(self, double ecore, double[:, ::1] one_mo not None,
        double[:, :, :, ::1] two_mo not None, bint keep_mo=True, bint doci=True):
        """
        Initialize a Hamiltonian instance.

        Parameters
        ----------
        ecore : float
            Constant/"zero-electron" integral.
        one_mo : np.ndarray(c_double(nbasis, nbasis))
            Full one-electron integral array.
        two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Full two-electron integral array.
        keep_mo : bool, default=True
            Whether to keep the full MO arrays.
        doci : bool, default=True
            Whether to compute the seniority-zero integral arrays.

        """
        if not (one_mo.shape[0] == one_mo.shape[1] == two_mo.shape[0] == \
                two_mo.shape[1] == two_mo.shape[2] == two_mo.shape[3]):
            raise ValueError('(one_mo, two_mo) shapes are incompatible')
        self._nbasis = one_mo.shape[0]
        self._ecore = ecore
        if keep_mo:
            self._one_mo = one_mo
            self._two_mo = two_mo
        else:
            self._one_mo = None
            self._two_mo = None
        if doci:
            self._h = np.copy(np.diagonal(one_mo))
            self._v = np.copy(np.diagonal(np.diagonal(two_mo)))
            self._w = np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 2, 3, 1)))) * 2 \
                    - np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 3, 2, 1))))
        else:
            self._h = None
            self._v = None
            self._w = None

    def to_file(self, object filename not None, int_t nelec=0, int_t ms2=0):
        r"""
        Write a Hamiltonian instance to an FCIDUMP file.

        Parameters
        ----------
        filename : str
            Name of FCIDUMP file to write.
        nelec : int, default=0
            Electron number to write to FCIDUMP file.
        ms2 : int, default=0
            Spin number to write to FCIDUMP file.

        """
        fcidump.write(filename, self._ecore, self.one_mo, self.two_mo, nelec=nelec, ms2=ms2)

    def reduced_v(self, int_t nocc):
        r"""
        Generate the reduced seniority-zero molecular orbital integrals.

        Returns the matrix

        .. math::

            K^0_{ab} = \left(\frac{2}{N_\text{elec} - 1}\right) h_{ab} \delta_{ab} + v_{ab}.

        Parameters
        ----------
        nocc : int
            Number of occupied indices.

        Returns
        -------
        v : np.ndarray(c_double(nbasis, nbasis))
            Reduced seniority-zero molecular orbital integrals.

        """
        cdef np.ndarray v_array = np.diag(self.h)
        v_array *= 2.0 / (2.0 * nocc - 1.0)
        v_array += self._v
        return v_array

    def reduced_w(self, int_t nocc):
        r"""
        Generate the reduced seniority-two molecular orbital integrals.

        Returns the matrix

        .. math::

            K^2_{ab} = \left(\frac{2}{N_\text{elec} - 1}\right) \left(h_{aa} + h_{bb}\right) + w_{ab}.

        Parameters
        ----------
        nocc : int
            Number of occupied indices.

        Returns
        -------
        w : np.ndarray(c_double(nbasis, nbasis))
            Reduced seniority-two molecular orbital integrals.

        """
        cdef int_t i, j
        cdef np.ndarray w_array = np.empty_like(self.v)
        cdef double[:, :] w = w_array
        for i in range(self._nbasis):
            for j in range(self._nbasis):
                w[i, j] = self._h[i] + self._h[j]
        w_array *= 2.0 / (2.0 * nocc - 1.0)
        w_array += self._w
        return w_array

    def _doci_elem_diag(self, int_t[::1] occs not None):
        r"""
        Compute Hamiltonian element :math:`\left<d|H|d\right>` for determinant :math:`d`.

        Parameters
        ----------
        occs : np.ndarray(c_int(nocc))
            Indices of occupied electron pairs in determinant.

        Returns
        -------
        elem : float
            Hamiltonian element.

        """
        if self._h is None:
            raise AttributeError('seniority-zero integrals were not computed')
        cdef int_t nocc = occs.shape[0], i, j, k
        cdef double elem1 = 0.0, elem2 = 0.0
        for i in range(nocc):
            j = occs[i]
            elem1 += self._v[j, j]
            elem2 += self._h[j]
            for k in range(i):
                elem2 += self._w[j, occs[k]]
        return elem1 + elem2 * 2

    def _fullci_elem_diag(self, int_t[::1] occs_up not None, int_t[::1] occs_dn not None):
        r"""
        Compute Hamiltonian element :math:`\left<d|H|d\right>` for determinant :math:`d`.

        Parameters
        ----------
        occs_up : np.ndarray(c_int(nocc_up))
            Indices of occupied spin-up electrons in determinant.
        occs_dn : np.ndarray(c_int(nocc_dn))
            Indices of occupied spin-down electrons in determinant.

        Returns
        -------
        elem : float
            Hamiltonian element.

        """
        if self._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        cdef int_t nocc_up = occs_up.shape[0], nocc_dn = occs_dn.shape[0], i, j, k, l
        cdef double elem = 0.0
        for i in range(nocc_up):
            j = occs_up[i]
            elem += self._one_mo[j, j]
            for k in range(i):
                l = occs_up[i]
                elem += self._two_mo[j, l, j, l] - self._two_mo[j, l, l, j]
            for k in range(nocc_dn):
                l = occs_dn[k]
                elem += self._two_mo[j, l, j, l]
        for i in range(nocc_dn):
            j = occs_dn[i]
            elem += self._one_mo[j, j]
            for k in range(i):
                l = occs_dn[i]
                elem += self._two_mo[j, l, j, l] - self._two_mo[j, l, l, j]
        return elem


cdef class doci_wfn:
    r"""
    DOCI wave function class.

    Attributes
    ----------
    nbasis : int
        Number of orbital basis functions.
    nocc : int
        Number of occupied indices.
    nvir : int
        Number of virtual indices.

    """
    cdef DOCIWfn _obj

    @staticmethod
    def from_file(object filename not None):
        r"""
        Return a doci_wfn instance by loading a DOCI file.

        Parameters
        ----------
        filename : str
            DOCI file from which to load determinants.

        Returns
        -------
        wfn : doci_wfn
            DOCI wave function object.

        """
        cdef doci_wfn wfn = doci_wfn(2, 1)
        wfn._obj.from_file(filename.encode())
        return wfn

    @staticmethod
    def from_det_array(int_t nbasis, int_t nocc, uint_t[:, ::1] det_array not None):
        r"""
        Return a doci_wfn instance from an array of determinant bitstrings.

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
        wfn : doci_wfn
            DOCI wave function object.

        """
        cdef doci_wfn wfn = doci_wfn(nbasis, nocc)
        if det_array.ndim != 2 or det_array.shape[1] != wfn._obj.nword:
            raise IndexError('nbasis, nocc given do not match up with det_array dimensions')
        wfn._obj.from_det_array(nbasis, nocc, det_array.shape[0], <uint_t *>(&det_array[0, 0]))
        return wfn

    @staticmethod
    def from_occs_array(int_t nbasis, int_t nocc, int_t[:, ::1] occs_array not None):
        r"""
        Return a doci_wfn instance from an array of occupied indices.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.
        occs_array : np.ndarray(c_int(n, nword))
            Array of occupied indices.

        Returns
        -------
        wfn : doci_wfn
            DOCI wave function object.

        """
        cdef doci_wfn wfn = doci_wfn(nbasis, nocc)
        if occs_array.ndim != 2 or occs_array.shape[1] != wfn._obj.nocc:
            raise IndexError('nbasis, nocc given do not match up with occs_array dimensions')
        wfn._obj.from_occs_array(nbasis, nocc, occs_array.shape[0], <int_t *>(&occs_array[0, 0]))
        return wfn

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._obj.nbasis

    @property
    def nocc(self):
        r"""
        Number of occupied indices.

        """
        return self._obj.nocc

    @property
    def nvir(self):
        r"""
        Number of virtual indices.

        """
        return self._obj.nvir

    def __init__(self, int_t nbasis, int_t nocc):
        r"""
        Initialize a doci_wfn instance.

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
        cdef uint_t[:] det = det_array
        self._obj.copy_det(index, <uint_t *>(&det[0]))
        return det_array

    def __copy__(self):
        r"""
        Copy a doci_wfn instance.

        Returns
        -------
        wfn : doci_wfn
            DOCI wave function object.

        """
        cdef doci_wfn wfn = doci_wfn(2, 1)
        wfn._obj.from_dociwfn(self._obj)
        return wfn

    def to_file(self, object filename not None):
        r"""
        Write a doci_wfn instance to a DOCI file.

        Parameters
        ----------
        filename : str
            Name of DOCI file to write.

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
        cdef uint_t *det_ptr = &self._obj.dets[start * self._obj.nword]
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

    def copy(self):
        r"""
        Copy a doci_wfn instance.

        Returns
        -------
        wfn : doci_wfn
            DOCI wave function object.

        """
        return self.__copy__()

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
        cdef int_t emax = min(self._obj.nocc, self._obj.nvir), ndet = 0, i, e, nexc
        cdef int_t[:] excv = np.array(list(set(exc)), dtype=c_int)
        nexc = excv.shape[0]
        for i in range(nexc):
            e = excv[i]
            if e < 0 or e > emax:
                raise ValueError('invalid excitation order e < 0 or e > min(nocc, nvir)')
            ndet += binomial(self._obj.nocc, e) * binomial(self._obj.nvir, e)
        # default determinant is hartree-fock determinant
        if det is None:
            det = self.occs_to_det(np.arange(self._obj.nocc, dtype=c_int))
        # reserve space for determinants
        self._obj.reserve(ndet)
        # add determinants
        for i in range(nexc):
            self._obj.add_excited_dets(&det[0], excv[i])

    def reserve(self, int_t n):
        r"""
        Reserve space in memory for :math:`n` elements in the doci_wfn instance.

        Parameters
        ----------
        n : int
            Number of elements for which to reserve space.

        """
        self._obj.reserve(n)

    def squeeze(self):
        r"""
        Free up any unused memory reserved by the doci_wfn instance.

        This can help reduce memory usage if many determinants are individually added.

        """
        self._obj.squeeze()

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
        cdef uint_t[:] det = det_array
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
        cdef int_t[:] occs = occs_array
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
        cdef int_t[:] virs = virs_array
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
        cdef uint_t[:] newdet = newdet_array
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
        cdef uint_t[:] newdet = newdet_array
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
        cdef uint_t[:] newdet = newdet_array
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
        return rank_det(self._obj.nbasis, self._obj.nocc, <uint_t *>(&det[0]))

    def new_det(self):
        r"""
        Return a new determinant with all bits set to zero.

        Returns
        -------
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        return np.zeros(self._obj.nword, dtype=c_uint)

    def compute_rdms(self, double[::1] coeffs not None):
        r"""
        Compute the 2-particle reduced density matrix (RDM) of a wave function.

        This method returns two nbasis-by-nbasis matrices, which include the unique
        seniority-zero and seniority-two terms from the full 2-RDMs:

        .. math::

            D_0 = \left<pp|qq\right>

        .. math::

            D_2 = \left<pq|pq\right>

        The diagonal elements of :math:`D_0` are equal to the 1-RDM elements :math:`\left<p|p\right>`.

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            Coefficient vector.

        Returns
        -------
        d0 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_0` matrix.
        d2 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_2` matrix.

        """
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        cdef np.ndarray d0_array = np.zeros((self._obj.nbasis, self._obj.nbasis), dtype=c_double)
        cdef np.ndarray d2_array = np.zeros((self._obj.nbasis, self._obj.nbasis), dtype=c_double)
        cdef double[:, :] d0 = d0_array
        cdef double[:, :] d2 = d2_array
        self._obj.compute_rdms(<double *>(&coeffs[0]), <double *>(&d0[0, 0]), <double *>(&d2[0, 0]))
        return d0_array, d2_array

    def run_hci(self, hamiltonian ham not None, double[::1] coeffs not None, double eps):
        r"""
        Run an iteration of seniority-zero heat-bath CI.

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
        if self._obj.ndet != coeffs.shape[0]:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif self._obj.nbasis != ham._nbasis:
            raise ValueError('dimensions of wfn, ham do not match')
        return self._obj.run_hci(<double *>(&ham._v[0, 0]), <double *>(&coeffs[0]), eps)

    @staticmethod
    def generate_rdms(double[:, ::1] d0 not None, double[:, ::1] d2 not None):
        r"""
        Generate full one- and two- particle RDMs from the :math:`D_0` and :math:`D_2` matrices.

        Parameters
        ----------
        d0 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_0` matrix.
        d2 : np.ndarray(c_double(nbasis, nbasis))
            :math:`D_2` matrix.

        Returns
        -------
        rdm1 : np.ndarray(c_double(2 * nbasis, 2 * nbasis))
            One-particle RDM.
        rdm2 : np.ndarray(c_double(2 * nbasis, 2 * nbasis, 2 * nbasis, 2 * nbasis))
            Two-particle RDM.

        """
        if not (d0.shape[0] == d0.shape[1] == d2.shape[0] == d2.shape[1]):
            raise ValueError('dimensions of d0, d2 do not match')
        cdef int_t nbasis = d0.shape[0]
        cdef int_t nspin = nbasis * 2
        cdef int_t p, q
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


cdef class fullci_wfn:
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
    cdef FullCIWfn _obj

    @staticmethod
    def from_file(object filename not None):
        r"""
        Return a fullci_wfn instance by loading a FullCI file.

        Parameters
        ----------
        filename : str
            FullCI file from which to load determinants.

        Returns
        -------
        wfn : fullci_wfn
            FullCI wave function object.

        """
        cdef fullci_wfn wfn = fullci_wfn(2, 1, 1)
        wfn._obj.from_file(filename.encode())
        return wfn

    @staticmethod
    def from_det_array(int_t nbasis, int_t nocc_up, int_t nocc_dn, uint_t[:, :, ::1] det_array not None):
        r"""
        Return a fullci_wfn instance from an array of determinant bitstrings.

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
        wfn : fullci_wfn
            FullCI wave function object.

        """
        cdef fullci_wfn wfn = fullci_wfn(nbasis, nocc_up, nocc_dn)
        if det_array.ndim != 3 or det_array.shape[1] != 2 or det_array.shape[2] != wfn._obj.nword:
            raise IndexError('nbasis, nocc_{up,dn} given do not match up with det_array dimensions')
        wfn._obj.from_det_array(nbasis, nocc_up, nocc_dn, det_array.shape[0],
                                <uint_t *>(&det_array[0, 0, 0]))
        return wfn

    @staticmethod
    def from_occs_array(int_t nbasis, int_t nocc_up, int_t nocc_dn, int_t[:, :, ::1] occs_array not None):
        r"""
        Return a fullci_wfn instance from an array of occupied indices.

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
        wfn : fullci_wfn
            FullCI wave function object.

        """
        cdef fullci_wfn wfn = fullci_wfn(nbasis, nocc_up, nocc_dn)
        if occs_array.ndim != 3 or occs_array.shape[1] != 2 or occs_array.shape[2] != nocc_up:
            raise IndexError('nbasis, nocc_{up,dn} given do not match up with det_array dimensions')
        wfn._obj.from_occs_array(nbasis, nocc_up, nocc_dn, occs_array.shape[0],
                                 <int_t *>(&occs_array[0, 0, 0]))
        return wfn

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._obj.nbasis

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

    def __init__(self, int_t nbasis, int_t nocc_up, int_t nocc_dn):
        r"""
        Initialize a doci_wfn instance.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of spin-up occupied indices.
        nocc_dn : int
            Number of spin-down occupied indices.

        """
        if (nbasis < nocc_up or nbasis <= nocc_dn or \
            nocc_up <= 0 or nocc_dn < 0 or nocc_up < nocc_dn):
            raise ValueError(
                'failed check: nbasis >= nocc_up > 0, nbasis > nocc_dn >= 0, nocc_up >= nocc_dn'
                )
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
        cdef uint_t[:, :] det = det_array
        self._obj.copy_det(index, <uint_t *>(&det[0, 0]))
        return det_array

    def __copy__(self):
        r"""
        Copy a fullci_wfn instance.

        Returns
        -------
        wfn : fullci_wfn
            FullCI wave function object.

        """
        cdef fullci_wfn wfn = fullci_wfn(2, 1, 1)
        wfn._obj.from_fullciwfn(self._obj)
        return wfn

    def to_file(self, object filename not None):
        r"""
        Write a fullci_wfn instance to a FullCI file.

        Parameters
        ----------
        filename : str
            Name of FullCI file to write.

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
        cdef uint_t *det_ptr = &self._obj.dets[start * self._obj.nword2]
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

    def copy(self):
        r"""
        Copy a fullci_wfn instance.

        Returns
        -------
        wfn : fullci_wfn
            FullCI wave function object.

        """
        return self.__copy__()

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
        cdef int_t ndet = 0, i, nexc, e, a, b
        cdef int_t[:] excv = np.array(list(set(exc)), dtype=c_int)
        nexc = excv.shape[0]
        for i in range(nexc):
            e = excv[i]
            if e < 0 or e > emax:
                raise ValueError('invalid excitation order e < 0 or e > min(nocc, nvir)')
            a = min(e, self._obj.nocc_up, self._obj.nvir_up)
            b = e - a
            while (a >= 0) and (b <= emax_dn):
                ndet += binomial(self._obj.nocc_up, a) * binomial(self._obj.nvir_up, a) \
                      * binomial(self._obj.nocc_dn, b) * binomial(self._obj.nvir_dn, b)
                a -= 1
                b += 1
        # default determinant is hartree-fock determinant
        cdef np.ndarray occs
        if det is None:
            occs = np.zeros((2, self._obj.nocc_up), dtype=c_int)
            occs[0, :self._obj.nocc_up] = np.arange(self._obj.nocc_up, dtype=c_int)
            occs[1, :self._obj.nocc_dn] = np.arange(self._obj.nocc_dn, dtype=c_int)
            det = self.occs_to_det(occs)
        # reserve space for determinants
        self._obj.reserve(ndet)
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
        Reserve space in memory for :math:`n` elements in the doci_wfn instance.

        Parameters
        ----------
        n : int
            Number of elements for which to reserve space.

        """
        self._obj.reserve(n)

    def squeeze(self):
        r"""
        Free up any unused memory reserved by the doci_wfn instance.

        This can help reduce memory usage if many determinants are individually added.

        """
        self._obj.squeeze()

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
        cdef uint_t[:, :] det = det_array
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
        cdef int_t[:, :] occs = occs_array
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
        cdef int_t[:, :] virs = virs_array
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
            Which spin upon which to perform the oepration.

        Returns
        -------
        newdet : np.ndarray(c_uint(2, nword))
            Excited determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[:, :] newdet = newdet_array
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
            Which spin upon which to perform the oepration.

        Returns
        -------
        newdet : np.ndarray(c_uint(2, nword))
            New determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[:, :] newdet = newdet_array
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
            Which spin upon which to perform the oepration.

        Returns
        -------
        newdet : np.ndarray(c_uint(2, nword))
            New determinant.

        """
        cdef np.ndarray newdet_array = np.copy(det)
        cdef uint_t[:, :] newdet = newdet_array
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
            Which spin upon which to perform the oepration.

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
            Which spin upon which to perform the oepration.

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
            Which spin upon which to perform the oepration.

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
            Which spin upon which to perform the oepration.

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
            Which spin upon which to perform the oepration.

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
            Which spin upon which to perform the oepration.

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
            Electron "hole" index.
        j : int
            Electron "hole" index.
        a : int
            Electron "particle" index.
        b : int
            Electron "particle" index.
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
            return phase_single_det(self._obj.nword, i, a, <uint_t *>(&det[<int_t>spin1, 0])) * \
                   phase_single_det(self._obj.nword, j, b, <uint_t *>(&det[<int_t>spin2, 0]))

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
        return rank_det(self._obj.nbasis, self._obj.nocc_up, <uint_t *>(&det[0, 0])) \
             * self._obj.maxdet_dn \
             + rank_det(self._obj.nbasis, self._obj.nocc_dn, <uint_t *>(&det[1, 0]))

    def new_det(self):
        r"""
        Return a new determinant with all bits set to zero.

        Returns
        -------
        det : np.ndarray(c_uint(2, nword))
            Determinant.

        """
        return np.zeros((2, self._obj.nword), dtype=c_uint)


cdef class gen_wfn(doci_wfn):
    r"""
    Generalized wave function class.

    """

    @staticmethod
    def from_file(object filename not None):
        r"""
        Return a gen_wfn instance by loading a DOCI file.

        Parameters
        ----------
        filename : str
            DOCI file from which to load determinants.

        Returns
        -------
        wfn : gen_wfn
            Generalized wave function object.

        """
        cdef gen_wfn wfn = gen_wfn(2, 1)
        wfn._obj.from_file(filename.encode())
        return wfn

    @staticmethod
    def from_det_array(int_t nbasis, int_t nocc, uint_t[:, ::1] det_array not None):
        r"""
        Return a gen_wfn instance from an array of determinant bitstrings.

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
        wfn : gen_wfn
            Generalized wave function object.

        """
        cdef gen_wfn wfn = gen_wfn(nbasis, nocc)
        if det_array.ndim != 2 or det_array.shape[1] != wfn._obj.nword:
            raise IndexError('nbasis, nocc given do not match up with det_array dimensions')
        wfn._obj.from_det_array(nbasis, nocc, det_array.shape[0], <uint_t *>(&det_array[0, 0]))
        return wfn

    @staticmethod
    def from_occs_array(int_t nbasis, int_t nocc, int_t[:, ::1] occs_array not None):
        r"""
        Return a gen_wfn instance from an array of occupied indices.

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.
        occs_array : np.ndarray(c_int(n, nword))
            Array of occupied indices.

        Returns
        -------
        wfn : gen_wfn
            Generalized wave function object.

        """
        cdef gen_wfn wfn = gen_wfn(nbasis, nocc)
        if occs_array.ndim != 2 or occs_array.shape[1] != wfn._obj.nocc:
            raise IndexError('nbasis, nocc given do not match up with occs_array dimensions')
        wfn._obj.from_occs_array(nbasis, nocc, occs_array.shape[0], <int_t *>(&occs_array[0, 0]))
        return wfn

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
            Electron "hole" index.
        j : int
            Electron "hole" index.
        a : int
            Electron "particle" index.
        b : int
            Electron "particle" index.

        Returns
        -------
        phase : (+1 | -1)
            Phase factor.

        """
        return phase_double_det(self._obj.nword, i, j, a, b, <uint_t *>(&det[0]))


cdef class sparse_op:
    r"""
    Sparse matrix operator class.

    Attributes
    ----------
    shape : (int, int)
        Shape of matrix operator.
    ecore : float
        Constant/"zero-electron" integral.

    """
    cdef SparseOp _obj
    cdef tuple _shape
    cdef double _ecore
    cdef double _ref_elem

    @property
    def shape(self):
        r"""
        Return the shape of the sparse matrix operator.

        """
        return self._shape

    @property
    def ecore(self):
        r"""
        Return the constant/"zero-electron" integral.

        """
        return self._ecore

    def __init__(self, hamiltonian ham not None, object wfn not None, int_t nrow=-1):
        r"""
        Initialize a sparse matrix operator instance.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        wfn : (doci_wfn | fullci_wfn)
            Wave function object.
        nrow : int, optional
            Number of rows (<= number of determinants in wavefunction). Default is square matrix.

        """
        # check inputs
        if ham.nbasis != wfn.nbasis:
            raise ValueError('dimension of ham, wfn do not match')
        elif len(wfn) == 0:
            raise ValueError('wfn must contain at least one determinant')
        # intialize object
        cdef double[:] h
        cdef double[:, :] v, w
        cdef double[:, :, :, :] x
        cdef int_t[:, :] occs
        if isinstance(wfn, doci_wfn):
            h = ham.h
            v = ham.v
            w = ham.w
            self._obj.init(
                (<doci_wfn>wfn)._obj,
                <double *>(&h[0]),
                <double *>(&v[0, 0]),
                <double *>(&w[0, 0]),
                nrow)
            self._ref_elem = ham._doci_elem_diag(wfn.det_to_occs(wfn[0]))
        elif isinstance(wfn, fullci_wfn):
            w = ham.one_mo
            x = ham.two_mo
            self._obj.init(
                (<fullci_wfn>wfn)._obj,
                <double *>(&w[0, 0]),
                <double *>(&x[0, 0, 0, 0]),
                nrow,
                )
            occs = wfn.det_to_occs(wfn[0])
            self._ref_elem = ham._fullci_elem_diag(occs[0, :wfn.nocc_up], occs[1, :wfn.nocc_dn])
        else:
            raise TypeError('invalid wfn type')
        self._shape = <object>(self._obj.nrow), <object>(self._obj.ncol)
        self._ecore = ham.ecore

    def to_csr_matrix(self):
        r"""
        Convert the sparse matrix operator to a scipy.sparse.csr_matrix instance.

        Returns
        -------
        mat : scipy.sparse.csr_matrix
            CSR matrix instance.

        """
        cdef double *data_ptr = &self._obj.data[0]
        cdef int_t *indices_ptr = &self._obj.indices[0]
        cdef int_t *indptr_ptr = &self._obj.indptr[0]
        cdef np.ndarray data = np.asarray(<double[:self._obj.data.size()]>data_ptr)
        cdef np.ndarray indices = np.asarray(<int_t[:self._obj.indices.size()]>indices_ptr)
        cdef np.ndarray indptr = np.asarray(<int_t[:self._obj.indptr.size()]>indptr_ptr)
        return csr_matrix((data, indices, indptr), shape=self._shape, copy=True)

    def dot(self, double[::1] x not None, double[::1] out=None):
        r"""
        Compute the dot product of the sparse operator :math:`A` with a vector :math:`x`.

        Parameters
        ----------
        x : np.ndarray(c_double(n))
            Operand vector.
        out : np.ndarray(c_double(n)), optional
            Output parameter, as in NumPy (e.g., numpy.dot).

        Returns
        -------
        y : np.ndarray(c_double(n))
           Result vector.

        """
        cdef np.ndarray y
        if x.size != self._obj.ncol:
            raise ValueError('Dimensions of operator and \'x\' do not match')
        # set y and out variables
        if out is None:
            y = np.empty(self._obj.nrow, dtype=c_double)
            out = y
        else:
            if out.size != self._obj.nrow:
                raise ValueError('Dimensions of operator and \'out\' do not match')
            y = np.asarray(out)
        # return result of operation
        self._obj.perform_op(&x[0], &out[0])
        return y

    def solve(self, int_t n=1, int_t ncv=-1, double[::1] c0=None, int_t maxiter=-1, double tol=1.0e-6):
        r"""
        Solve the CI problem for the energy/energies and coefficient vector(s).

        Parameters
        ----------
        n : int, default=1
            Number of lowest-energy solutions for which to solve.
        ncv : int, default=max(n + 1, min(20, nrow))
            Number of Lanczos vectors to use for eigensolver.
            More is generally faster and more reliably convergent.
        c0 : np.ndarray(c_double(nrow)), optional
            Initial guess for lowest-energy coefficient vector.
            If not provided, the default is [1, 0, 0, ..., 0, 0].
        maxiter : int, default=1000*n
            Maximum number of iterations for eigensolver to run.
        tol : float, default=1.0e-6
            Convergence tolerance for eigensolver.

        Returns
        -------
        evals : np.ndarray(c_double(n))
            Energies.
        evecs : np.ndarray(c_double(n, nrow))
            Coefficient vectors.

        """
        if self._shape[0] != self._shape[1]:
            raise ValueError('cannot solve for a rectangular operator')
        cdef int_t ndet = self._shape[0]
        # set default number of lanczos vectors n < ncv <= len(c0)
        if ncv == -1:
            ncv = max(n + 1, min(20, ndet))
        # default initial guess c = [1, 0, ..., 0]
        if c0 is None:
            c0 = np.zeros(ndet, dtype=c_double)
            c0[0] = 1.0
        elif ndet != c0.shape[0]:
            raise ValueError('dimension of sparse_op, c0 do not match')
        # default maxiter = 1000 * n
        if maxiter == -1:
            maxiter = 1000 * n
        # solve eigenproblem
        cdef np.ndarray evals_array = np.empty(n, dtype=c_double)
        cdef np.ndarray evecs_array = np.empty((n, ndet), dtype=c_double)
        cdef double[:] evals = evals_array
        cdef double[:, :] evecs = evecs_array
        self._obj.solve(<double *>(&c0[0]), n, ncv, maxiter, tol,
                        <double *>(&evals[0]), <double *>(&evecs[0, 0]))
        evals_array += self._ecore
        return evals_array, evecs_array
