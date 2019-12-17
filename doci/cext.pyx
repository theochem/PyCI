# cython : language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
#
# This file is part of DOCI.
#
# DOCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# DOCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with DOCI. If not, see <http://www.gnu.org/licenses/>.

r"""
DOCI C extension module.

"""

cimport numpy as np
import numpy as np

from doci.cext cimport int_t, uint_t, DOCIWfn, dociham, dociwfn
from doci.cext cimport solve_sparse, solve_direct, compute_rdms_, compute_energy_, run_hci_
from doci.cext cimport binomial, fill_det, fill_occs, fill_virs
from doci.cext cimport excite_det, setbit_det, clearbit_det, popcnt_det, ctz_det, hash_det
from doci.fcidump import read as read_fcidump, write as write_fcidump


__all__ = [
    'get_version',
    'comb',
    'dociham',
    'dociwfn',
    'solve_ci',
    'compute_rdms',
    'compute_energy',
    'run_hci',
    'generate_rdms',
    ]


cdef np.dtype c_int = np.dtype(np.int64)

cdef np.dtype c_uint = np.dtype(np.uint64)

cdef np.dtype c_double = np.dtype(np.double)


def get_version():
    r"""
    Return the version number string from the C extension.

    Returns
    -------
    version : str
        Version number string.

    """
    return DOCI_VERSION


def comb(int_t n, int_t k):
    r"""
    Compute the binomial coefficient :math:`{n}\choose{k}`.

    Parameters
    ----------
    n : int
        :math:`n`.
    k : int
        :math:`k`.

    Returns
    -------
    comb : int
        :math:`{n}\choose{k}`.

    """
    if n < 0 or k < 0:
        raise ValueError('n and k must be non-negative integers')
    return binomial(n, k)


cdef class dociham:
    r"""
    DOCI Hamiltonian class.

    .. math::

        H = \sum_{p}{ h_p N_p } + \sum_{p \neq q}{ v_{pq} P^\dagger_p P_q } + \sum_{pq}{ w_{pq} N_p N_q }

    where

    .. math::

        h_{p} = \left<p|T|p\right>

    .. math::

        v_{pq} = \left<pp|V|qq\right>

    .. math::

        w_{pq} = 2 \left<pq|V|pq\right> - \left<pq|V|qp\right>

    Attributes
    ----------
    nbasis : int
        Number of orbital basis functions.
    ecore : float
        Constant/"zero-electron" integral.
    h : np.ndarray(c_double(nbasis))
        Seniority-zero one-electron integrals.
    v : np.ndarray(c_double(nbasis, nbasis))
        Seniority-zero two-electron integrals.
    w : np.ndarray(c_double(nbasis, nbasis))
        Seniority-two two-electron integrals.
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.

    """

    @classmethod
    def from_mo_arrays(cls, double ecore, double[:, ::1] one_mo not None, double[:, :, :, ::1] two_mo not None,
        bint keep_mo=True):
        r"""
        Return a dociham instance from input molecular orbital arrays.

        Parameters
        ----------
        ecore : float
            Constant/"zero-electron" integral.
        one_mo : np.ndarray(c_double(nbasis, nbasis))
            Full one-electron integral array.
        two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Full two-electron integral array.
        keep_mo : bool, default=True
            Whether to keep full MO arrays in memory.

        Returns
        -------
        ham : dociham
            DOCI Hamiltonian object.

        """
        if not (one_mo.shape[0] == one_mo.shape[1] == two_mo.shape[0] == \
                two_mo.shape[1] == two_mo.shape[2] == two_mo.shape[3]):
            raise ValueError('(one_mo, two_mo) shapes are incompatible')
        cdef double[:] h = np.copy(np.diagonal(one_mo))
        cdef double[:, :] v = np.copy(np.diagonal(np.diagonal(two_mo)))
        cdef double[:, :] w = np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 2, 3, 1)))) * 2
        cdef np.ndarray w_array = np.asarray(w)
        w_array -= np.diagonal(np.diagonal(np.transpose(two_mo, axes=(0, 3, 2, 1))))
        if not keep_mo:
            one_mo = None
            two_mo = None
        return cls(ecore, h, v, w, one_mo=one_mo, two_mo=two_mo)

    @classmethod
    def from_file(cls, object filename not None, bint keep_mo=True):
        r"""
        Return a dociham instance by loading an FCIDUMP file.

        Parameters
        ----------
        filename : str
            FCIDUMP file from which to load integrals.
        keep_mo : bool, default=True
            Whether to keep full MO arrays in memory.

        Returns
        -------
        ham : dociham
            DOCI Hamiltonian object.

        """
        return cls.from_mo_arrays(*read_fcidump(filename)[:3], keep_mo=keep_mo)

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
    def h(self):
        r"""
        Seniority-zero one-electron integrals.

        """
        return np.asarray(self._h)

    @property
    def v(self):
        r"""
        Seniority-zero two-electron integrals.

        """
        return np.asarray(self._v)

    @property
    def w(self):
        r"""
        Seniority-two two-electron integrals.

        """
        return np.asarray(self._w)

    @property
    def one_mo(self):
        r"""
        Full one-electron integral array.

        """
        if self._one_mo is None:
            raise AttributeError('one_mo was not passed to dociham instance')
        return np.asarray(self._one_mo)

    @property
    def two_mo(self):
        r"""
        Full two-electron integral array.

        """
        if self._two_mo is None:
            raise AttributeError('two_mo was not passed to dociham instance')
        return np.asarray(self._two_mo)

    def __init__(self, double ecore, double[::1] h not None, double[:, ::1] v not None, double[:, ::1] w not None,
        double[:, ::1] one_mo=None, double[:, :, :, ::1] two_mo=None):
        """
        Initialize a dociham instance.

        Parameters
        ----------
        ecore : float
            Constant/"zero-electron" integral.
        h : np.ndarray(c_double(nbasis))
            Seniority-zero one-electron integrals.
        v : np.ndarray(c_double(nbasis, nbasis))
            Seniority-zero two-electron integrals.
        w : np.ndarray(c_double(nbasis, nbasis))
            Seniority-two two-electron integrals.
        one_mo : np.ndarray(c_double(nbasis, nbasis)), optional
            Full one-electron integral array.
        two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis)), optional
            Full two-electron integral array.

        """
        if not (h.shape[0] == v.shape[0] == v.shape[1] == w.shape[0] == w.shape[1]):
            raise ValueError('(h, v, w) shapes are incompatible')
        self._nbasis = h.shape[0]
        self._ecore = ecore
        self._h = h
        self._v = v
        self._w = w
        self._one_mo = one_mo
        self._two_mo = two_mo

    def to_file(self, object filename not None, int_t nelec=0, int_t ms2=0):
        r"""
        Write a dociham instance to an FCIDUMP file.

        Parameters
        ----------
        filename : str
            Name of FCIDUMP file to write.
        nelec : int, default=0
            Electron number to write to FCIDUMP file.
        ms2 : int, default=0
            Spin number to write to FCIDUMP file.

        """
        write_fcidump(filename, self._ecore, self._one_mo, self._two_mo, nelec=nelec, ms2=ms2)

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
        cdef np.ndarray v_array = np.diag(self._h)
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
        cdef np.ndarray w_array = np.empty_like(self._v)
        cdef double[:, :] w = w_array
        for i in range(self._nbasis):
            for j in range(self._nbasis):
                w[i, j] = self._h[i] + self._h[j]
        w_array *= 2.0 / (2.0 * nocc - 1.0)
        w_array += self._w
        return w_array

    def elem_diag(self, int_t[::1] occs not None):
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
        cdef int_t nocc = occs.shape[0], i, j, k
        cdef double elem1 = 0.0, elem2 = 0.0
        for i in range(nocc):
            j = occs[i]
            elem1 += self._v[j, j]
            elem2 += self._h[j]
            for k in range(i):
                elem2 += self._w[j, occs[k]]
        return elem1 + elem2 * 2

    def elem_double(self, int_t i, int_t a):
        r"""
        Compute Hamiltonian element :math:`\left<f|H|d\right>` for determinants :math:`d` and :math:`f = P^\dagger_a P_i d`.

        Parameters
        ----------
        i : int
            Electron pair "hole" index.
        a : int
            Electron pair "particle" index.

        Returns
        -------
        elem : float
            Hamiltonian element.

        """
        return self._v[i, a]


cdef class dociwfn:
    r"""
    DOCI wave function class.

    Attributes
    ----------
    nword : int
        Number of words (unsigned 64-bit ints) per determinant.
    nbasis : int
        Number of orbital basis functions.
    nocc : int
        Number of occupied indices.
    nvir : int
        Number of virtual indices.

    """

    @classmethod
    def from_file(cls, object filename not None):
        r"""
        Return a dociwfn instance by loading a DOCI file.

        Parameters
        ----------
        filename : str
            DOCI file from which to load determinants.

        Returns
        -------
        wfn : dociwfn
            DOCI wave function object.

        """
        cdef dociwfn wfn = cls(2, 1)
        wfn._obj.from_file(filename.encode())
        return wfn

    @property
    def nword(self):
        r"""
        Number of words (unsigned 64-bit ints) per determinant.

        """
        return self._obj.nword

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
        Initialize a dociwfn instance.

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

    def __iter__(self):
        r"""
        Return an iterator over all determinants in the wave function.

        Yields
        -------
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        cdef int_t ndet = self._obj.ndet, i
        cdef np.ndarray det_array
        cdef uint_t[:] det
        for i in range(ndet):
            det_array = np.empty(self._obj.nword, dtype=c_uint)
            det = det_array
            self._obj.copy_det(i, <uint_t *>(&det[0]))
            yield det_array

    def to_file(self, object filename not None):
        r"""
        Write a dociwfn instance to a DOCI file.

        Parameters
        ----------
        filename : str
            Name of DOCI file to write.

        """
        self._obj.to_file(filename.encode())

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

    def add_det_from_occs(self, int_t[::1] occs not None):
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
        self.add_det_from_occs(np.arange(self._obj.nocc, dtype=c_int))

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
            det = self.det_from_occs(np.arange(self._obj.nocc, dtype=c_int))
        # reserve space for determinants
        self._obj.reserve(ndet)
        # add determinants
        for i in range(nexc):
            self._obj.add_excited_dets(&det[0], excv[i])

    def reserve(self, int_t n):
        r"""
        Reserve space in memory for :math:`n` elements in the dociwfn instance.

        Parameters
        ----------
        n : int
            Number of elements for which to reserve space.

        """
        self._obj.reserve(n)

    def squeeze(self):
        r"""
        Free up any unused memory reserved by the dociwfn instance.

        This can help reduce memory usage if many determinants are individually added.

        """
        self._obj.squeeze()

    def det_from_occs(self, int_t[::1] occs not None):
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

    def occs_from_det(self, uint_t[::1] det not None):
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

    def virs_from_det(self, uint_t[::1] det not None):
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

    def hash_det(self, uint_t[::1] det not None):
        r"""
        Compute the hash of a determinant.

        Parameters
        ----------
        det : np.ndarray(c_uint(nword))
            Determinant.

        Returns
        -------
        hash : int
            Hash value.

        """
        return hash_det(self._obj.nbasis, self._obj.nocc, <uint_t *>(&det[0]))

    def new_det(self):
        r"""
        Return a new determinant with all bits set to zero.

        Returns
        -------
        det : np.ndarray(c_uint(nword))
            Determinant.

        """
        return np.zeros(self._obj.nword, dtype=c_uint)


def solve_ci(dociham ham not None, dociwfn wfn not None, int_t n=1, int_t ncv=20, double[::1] c0=None,
    int_t maxiter=-1, double tol=1.0e-6, object mode='sparse'):
    r"""
    Solve the CI problem for the energy/energies and coefficient vector(s).

    Parameters
    ----------
    ham : dociham
        DOCI Hamiltonian object.
    wfn : dociwfn
        DOCI wave function object.
    n : int, default=1
        Number of lowest-energy solutions for which to solve.
    ncv : int, default=20
        Number of Lanczos vectors to use for eigensolver.
        More is generally faster and more reliably convergent.
    c0 : np.ndarray(c_double(len(wfn))), optional
        Initial guess for lowest-energy coefficient vector.
        If not provided, the default is [1, 0, 0, ..., 0, 0].
    maxiter : int, default=1000*n
        Maximum number of iterations for eigensolver to run.
    tol : float, default=1.0e-6
        Convergence tolerance for eigensolver.
    mode : ('sparse' | 'direct'), default='sparse'
        Whether to use the sparse matrix or direct CI eigensolver.
        'direct' mode is much slower, but uses much less memory than 'sparse' mode.

    Returns
    -------
    evals : np.ndarray(c_double(n))
        Energies.
    evecs : np.ndarray(c_double(n, len(wfn)))
        Coefficient vectors.

    """
    # check inputs
    if ham._nbasis != wfn._obj.nbasis:
        raise ValueError('dimension of ham, wfn do not match')
    elif wfn._obj.ndet == 0:
        raise ValueError('wfn must contain at least one determinant')
    # handle ndet = 1 case
    if wfn._obj.ndet == 1:
        return (np.full(1, ham.elem_diag(wfn.occs_from_det(wfn[0])) + ham._ecore, dtype=c_double),
                np.ones((1, 1), dtype=c_double))
    # set number of lanczos vectors n < ncv <= len(c0)
    ncv = max(n + 1, min(ncv, wfn._obj.ndet))
    # default initial guess c = [1, 0, ..., 0]
    if c0 is None:
        c0 = np.zeros(wfn._obj.ndet, dtype=c_double)
        c0[0] = 1.0
    elif wfn._obj.ndet != c0.shape[0]:
        raise ValueError('dimension of wfn, c0 do not match')
    # default maxiter = 1000 * n
    if maxiter == -1:
        maxiter = 1000 * n
    # solve eigenproblem
    cdef np.ndarray evals_array = np.empty(n, dtype=c_double)
    cdef np.ndarray evecs_array = np.empty((n, wfn._obj.ndet), dtype=c_double)
    cdef double[:] evals = evals_array
    cdef double[:, :] evecs = evecs_array
    if mode == 'sparse':
        solve_sparse(wfn._obj, <double *>(&ham._h[0]), <double *>(&ham._v[0, 0]), <double *>(&ham._w[0, 0]),
                     <double *>(&c0[0]), n, ncv, maxiter, tol, <double *>(&evals[0]), <double *>(&evecs[0, 0]))
    elif mode == 'direct':
        solve_direct(wfn._obj, <double *>(&ham._h[0]), <double *>(&ham._v[0, 0]), <double *>(&ham._w[0, 0]),
                     <double *>(&c0[0]), n, ncv, maxiter, tol, <double *>(&evals[0]), <double *>(&evecs[0, 0]))
    else:
        raise ValueError('\'mode\' option must be either \'sparse\' or \'direct\'')
    evals_array += ham._ecore
    return evals_array, evecs_array


def compute_rdms(dociwfn wfn not None, double[::1] coeffs not None):
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
    wfn : dociwfn
        DOCI wave function object.
    coeffs : np.ndarray(c_double(len(wfn)))
        Coefficient vector.

    Returns
    -------
    d0 : np.ndarray(c_double(wfn.nbasis, wfn.nbasis))
        :math:`D_0` matrix.
    d2 : np.ndarray(c_double(wfn.nbasis, wfn.nbasis))
        :math:`D_2` matrix.

    """
    if wfn._obj.ndet != coeffs.shape[0]:
        raise ValueError('dimensions of wfn, coeffs do not match')
    elif wfn._obj.ndet == 0:
        raise ValueError('wfn must contain at least one determinant')
    cdef np.ndarray d0_array = np.zeros((wfn._obj.nbasis, wfn._obj.nbasis), dtype=c_double)
    cdef np.ndarray d2_array = np.zeros((wfn._obj.nbasis, wfn._obj.nbasis), dtype=c_double)
    cdef double[:, :] d0 = d0_array
    cdef double[:, :] d2 = d2_array
    compute_rdms_(wfn._obj, <double *>(&coeffs[0]), <double *>(&d0[0, 0]), <double *>(&d2[0, 0]))
    return d0_array, d2_array


def compute_energy(dociham ham not None, dociwfn wfn not None, double[::1] coeffs not None):
    r"""
    Compute the energy of a wave function from the Hamiltonian and coefficients.

    Parameters
    ----------
    ham : dociham
        DOCI Hamiltonian object.
    wfn : dociwfn
        DOCI wave function object.
    coeffs : np.ndarray(c_double(len(wfn)))
        Coefficient vector.

    Returns
    -------
    energy : float
        Energy.

    """
    if wfn._obj.ndet != coeffs.shape[0]:
        raise ValueError('dimensions of wfn, coeffs do not match')
    elif wfn._obj.ndet == 0:
        raise ValueError('wfn must contain at least one determinant')
    elif wfn._obj.nbasis != ham._nbasis:
        raise ValueError('dimensions of wfn, ham do not match')
    return compute_energy_(wfn._obj, <double *>(&ham._h[0]), <double *>(&ham._v[0, 0]),
                           <double *>(&ham._w[0, 0]), <double *>(&coeffs[0])) + ham._ecore


def run_hci(dociham ham not None, dociwfn wfn not None, double[::1] coeffs not None, double eps):
    r"""
    Run an iteration of seniority-zero heat-bath CI.

    Adds all determinants connected to determinants currently in the wave function,
    if they satisfy the criteria
    :math:`|\left<f|H|d\right> c_d| > \epsilon` for :math:`f = P^\dagger_i P_a d`.

    Parameters
    ----------
    ham : dociham
        DOCI Hamiltonian object.
    wfn : dociwfn
        DOCI wave function object.
    coeffs : np.ndarray(c_double(len(wfn)))
        Coefficient vector.
    eps : float
        Threshold value for which determinants to include.

    Returns
    -------
    n : int
        Number of determinants added to wave function.

    """
    if wfn._obj.ndet != coeffs.shape[0]:
        raise ValueError('dimensions of wfn, coeffs do not match')
    elif wfn._obj.ndet == 0:
        raise ValueError('wfn must contain at least one determinant')
    elif wfn._obj.nbasis != ham._nbasis:
        raise ValueError('dimensions of wfn, ham do not match')
    return run_hci_(wfn._obj, <double *>(&ham._v[0, 0]), <double *>(&coeffs[0]), eps)


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
