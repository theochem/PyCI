# cython: language_level=3, wraparound=False, binding=False
# cython: initializedcheck=False, nonecheck=False, boundscheck=False
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

cimport numpy as np
import numpy as np

from pyci cimport *


__all__ = [
        '__version__',
        'c_int',
        'c_uint',
        'c_double',
        'SpinLabel',
        'SPIN_UP',
        'SPIN_DN',
        'hamiltonian',
        'restricted_ham',
        'unrestricted_ham',
        'generalized_ham',
        'wavefunction',
        'one_spin_wfn',
        'two_spin_wfn',
        'doci_wfn',
        'fullci_wfn',
        'genci_wfn',
        'sparse_op',
        ]


cdef np.dtype int_dtype = np.dtype(np.int64)


cdef np.dtype uint_dtype = np.dtype(np.uint64)


cdef np.dtype double_dtype = np.dtype(np.double)


__version__ = PYCI_VERSION
r"""
PyCI version string.

"""


c_int = np.dtype(np.int64)
r"""
C extension signed int dtype.

"""


c_uint = np.dtype(np.uint64)
r"""
C extension unsigned int dtype.

"""


c_double = np.dtype(np.double)
r"""
C extension floating-point dtype.

"""


cpdef enum SpinLabel:
    SPIN_UP = 0
    SPIN_DN = 1
r"""
Spin label enumeration.

"""


def read_fcidump(str filename not None):
    r"""
    Read an FCIDUMP file.

    Parameters
    ----------
    filename : str
        Name of FCIDUMP file to read.

    Returns
    -------
    ecore : float
        Constant/"zero-electron" integral.
    one_mo : np.ndarray(c_double(nbasis, nbasis))
        Full one-electron integral array.
    two_mo : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
        Full two-electron integral array.

    """
    cdef object f
    cdef list fields
    cdef str line, field
    cdef double[:, ::1] one_mo
    cdef double[:, :, :, ::1] two_mo
    cdef double ecore = 0.0, val
    cdef int_t nbasis, i, j, k, l
    with open(filename, 'r', encoding='utf-8') as f:
        # check header
        line = next(f)
        if not line.startswith(' &FCI NORB='):
            raise IOError('Error in FCIDUMP file header')
        # read nbasis from header
        nbasis = int(line[11:line.find(',')])
        # skip rest of header
        for line in f:
            field = line.split()[0]
            if field == '&END' or field == '/END' or field == '/':
                break
        # read integrals
        one_mo = np.zeros((nbasis, nbasis), dtype=double_dtype)
        two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=double_dtype)
        for line in f:
            fields = line.split()
            if len(fields) != 5:
                raise IOError('Expecting 5 fields on each data line in FCIDUMP')
            val = float(fields[0])
            if fields[3] != '0':
                i = int(fields[1]) - 1
                j = int(fields[2]) - 1
                k = int(fields[3]) - 1
                l = int(fields[4]) - 1
                two_mo[i, k, j, l] = val
                two_mo[k, i, l, j] = val
                two_mo[j, k, i, l] = val
                two_mo[i, l, j, k] = val
                two_mo[j, l, i, k] = val
                two_mo[l, j, k, i] = val
                two_mo[k, j, l, i] = val
                two_mo[l, i, k, j] = val
            elif fields[1] != '0':
                i = int(fields[1]) - 1
                j = int(fields[2]) - 1
                one_mo[i, j] = val
                one_mo[j, i] = val
            else:
                ecore = val
    return ecore, one_mo, two_mo


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

    def __init__(self, *pyargs, bint keep_mo=True, bint doci=True):
        """
        Initialize a Hamiltonian instance.

        Parameters
        ----------
        filename : str
            FCIDUMP file to read.
        keep_mo : bool, default=True
            Whether to keep the full MO arrays.
        doci : bool, default=True
            Whether to compute the seniority-zero integral arrays.

        or

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
        cdef tuple args = pyargs
        cdef double ecore
        cdef double[:, ::1] one_mo
        cdef double[:, :, :, ::1] two_mo
        if len(args) == 1 and hasattr(args[0], 'encode'):
            args = read_fcidump(args[0])
        elif len(args) != 3:
            raise TypeError('args must be a filename or (ecore, one_mo, two_mo)')
        ecore = args[0]
        one_mo = args[1]
        two_mo = args[2]
        if not (one_mo.shape[0] == one_mo.shape[1] == two_mo.shape[0]
                == two_mo.shape[1] == two_mo.shape[2] == two_mo.shape[3]):
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

    def to_file(self, str filename not None, int_t nelec=0, int_t ms2=0, double tol=1.0e-18):
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
        tol : float, default=1.0e-18
            Write elements with magnitude larger than this value.

        """
        cdef object f
        cdef int_t nbasis = self.one_mo.shape[0], i, j, k, l
        cdef double val
        with open(filename, 'w', encoding='utf-8') as f:
            # write header
            f.write(f' &FCI NORB={nbasis},NELEC={nelec},MS2={ms2},\n')
            f.write(f'  ORBSYM={"1," * nbasis}\n  ISYM=1\n &END\n')
            # write two-electron integrals
            for i in range(nbasis):
                for j in range(i + 1):
                    for k in range(nbasis):
                        for l in range(k + 1):
                            if (i * (i + 1)) // 2 + j >= (k * (k + 1)) // 2 + l:
                                val = self._two_mo[i, k, j, l]
                                if abs(val) > tol:
                                    f.write(f'{val:23.16E} {i + 1:4d} {j + 1:4d} {k + 1:4d} {l + 1:4d}\n')
            # write one-electron integrals
            for i in range(nbasis):
                for j in range(i + 1):
                    val = self._one_mo[i, j]
                    if abs(val) > tol:
                        f.write(f'{val:23.16E} {i + 1:4d} {j + 1:4d}    0    0\n')
            # write zero-energy integrals
            f.write(f'{self._ecore if abs(self._ecore) > tol else 0:23.16E}    0    0    0    0\n')

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
        cdef double[:, ::1] w = w_array
        for i in range(self._nbasis):
            for j in range(self._nbasis):
                w[i, j] = self._h[i] + self._h[j]
        w_array *= 2.0 / (2.0 * nocc - 1.0)
        w_array += self._w
        return w_array


cdef class restricted_ham(hamiltonian):
    r"""
    Restricted Hamiltonian class.

    """
    pass


cdef class unrestricted_ham(hamiltonian):
    r"""
    Unrestricted Hamiltonian object.

    """

    @property
    def one_mo(self):
        r"""
        Full one-electron integral array.

        """
        if self._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        raise NotImplementedError

    @property
    def two_mo(self):
        r"""
        Full two-electron integral array.

        """
        if self._two_mo is None:
            raise AttributeError('full integral arrays were not saved')
        raise NotImplementedError

    @property
    def h(self):
        r"""
        Seniority-zero one-electron integrals.

        """
        if self._h is None:
            raise AttributeError('seniority-zero integrals were not computed')
        raise NotImplementedError

    @property
    def v(self):
        r"""
        Seniority-zero two-electron integrals.

        """
        if self._v is None:
            raise AttributeError('seniority-zero integrals were not computed')
        raise NotImplementedError

    @property
    def w(self):
        r"""
        Seniority-two two-electron integrals.

        """
        if self._w is None:
            raise AttributeError('seniority-zero integrals were not computed')
        raise NotImplementedError

    def __init__(self, *pyargs, bint keep_mo=True, bint doci=True):
        """
        Initialize a Hamiltonian instance.

        Parameters
        ----------
        filename : str
            FCIDUMP file to read.
        keep_mo : bool, default=True
            Whether to keep the full MO arrays.
        doci : bool, default=True
            Whether to compute the seniority-zero integral arrays.

        or

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
        raise NotImplementedError

    def to_file(self, str filename not None, int_t nelec=0, int_t ms2=0, double tol=1.0e-18):
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
        tol : float, default=1.0e-18
            Write elements with magnitude larger than this value.

        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError


cdef class generalized_ham(hamiltonian):
    r"""
    Generalized Hamiltonian class.

    """
    pass


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

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._obj.nbasis

    def __init__(self, *pyargs):
        r"""
        Initialize a one_spin_wfn instance.

        Parameters
        ----------
        filename : str
            Name of ONESPIN file to read.

        or

        Parameters
        ----------
        wfn : one_spin_wfn
            Wave function from which to initialize.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.
        array : (np.ndarray(c_int(n, nocc)) | np.ndarray(c_uint(n, nword)))
            Array of occupied indices or determinant bitstrings.

        """
        cdef tuple args = pyargs
        cdef one_spin_wfn wfn
        cdef np.ndarray array
        cdef int_t[:, ::1] iarray
        cdef uint_t[:, ::1] uarray
        cdef int_t case = len(args)
        if case == 1:
            if isinstance(args[0], one_spin_wfn):
                wfn = args[0]
                self._obj.from_onespinwfn(wfn._obj);
                return
            elif hasattr(args[0], 'encode'):
                self._obj.from_file(args[0].encode())
                return
        elif case == 2:
            self._obj.init(args[0], args[1])
            return
        elif case == 3:
            if isinstance(args[2], np.ndarray):
                array = args[2]
                if array.dtype == int_dtype:
                    iarray = array
                    if iarray.shape[1] != args[1]:
                        raise ValueError('invalid array shape')
                    self._obj.from_occs_array(args[0], args[1], iarray.shape[0], <int_t *>(&iarray[0, 0]))
                elif array.dtype == uint_dtype:
                    uarray = array
                    if uarray.shape[1] != nword_det(args[0]):
                        raise ValueError('invalid array shape')
                    self._obj.from_det_array(args[0], args[1], uarray.shape[0], <uint_t *>(&uarray[0, 0]))
                else:
                    raise TypeError('invalid array dtype')
                return
        raise TypeError('invalid arguments to __init__')

    def __copy__(self):
        r"""
        Copy a one_spin_wfn instance.

        Returns
        -------
        wfn : one_spin_wfn
            One-spin wave function object.

        """
        return self.__class__(self)

    def copy(self):
        r"""
        Copy a one_spin_wfn instance.

        Returns
        -------
        wfn : one_spin_wfn
            One-spin wave function object.

        """
        return self.__class__(self)

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
        cdef np.ndarray det_array = np.empty(self._obj.nword, dtype=uint_dtype)
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
        cdef np.ndarray occs_array = np.zeros((end - start, self._obj.nocc), dtype=int_dtype)
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
        self.add_occs(np.arange(self._obj.nocc, dtype=int_dtype))

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
        cdef int_t[::1] excv = np.array(sorted(set(exc)), dtype=int_dtype)
        cdef int_t nexc = excv.size, i, e
        if excv[0] < 0 or excv[nexc - 1] > emax:
            raise ValueError('invalid excitation order e < 0 or e > min(nocc, nvir)')
        # default determinant is hartree-fock determinant
        if det is None:
            det = self.occs_to_det(np.arange(self._obj.nocc, dtype=int_dtype))
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
        return self.__class__(self._obj.nbasis, self._obj.nocc, self.to_det_array(start, end))

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
        cdef np.ndarray det_array = np.zeros(self._obj.nword, dtype=uint_dtype)
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
        cdef np.ndarray occs_array = np.empty(self._obj.nocc, dtype=int_dtype)
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
        cdef np.ndarray virs_array = np.empty(self._obj.nvir, dtype=int_dtype)
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
        return np.zeros(self._obj.nword, dtype=uint_dtype)

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

    @property
    def nbasis(self):
        r"""
        Number of orbital basis functions.

        """
        return self._obj.nbasis

    def __init__(self, *pyargs):
        r"""
        Initialize a two_spin_wfn instance.

        Parameters
        ----------
        filename : str
            Name of TWOSPIN file to read.

        or

        Parameters
        ----------
        wfn : two_spin_wfn
            Wave function from which to initialize.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of spin-up occupied indices.
        nocc_dn : int
            Number of spin-down occupied indices.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of spin-up occupied indices.
        nocc_dn : int
            Number of spin-down occupied indices.
        array : (np.ndarray(c_int(n, nocc)) | np.ndarray(c_uint(n, nword)))
            Array of occupied indices or determinant bitstrings.

        """
        cdef tuple args = pyargs
        cdef two_spin_wfn wfn
        cdef np.ndarray array
        cdef int_t[:, :, ::1] iarray
        cdef uint_t[:, :, ::1] uarray
        cdef int_t case = len(args)
        if case == 1:
            if isinstance(args[0], two_spin_wfn):
                wfn = args[0]
                self._obj.from_twospinwfn(wfn._obj);
                return
            elif hasattr(args[0], 'encode'):
                self._obj.from_file(args[0].encode())
                return
        elif case == 3:
            self._obj.init(args[0], args[1], args[2])
            return
        elif case == 4:
            if isinstance(args[3], np.ndarray):
                array = args[3]
                if array.dtype == int_dtype:
                    iarray = array
                    if iarray.shape[1] != 2 or iarray.shape[2] != args[1]:
                        raise ValueError('invalid array shape')
                    self._obj.from_occs_array(
                            args[0], args[1], args[2],
                            iarray.shape[0], <int_t *>(&iarray[0, 0, 0]),
                            )
                elif array.dtype == uint_dtype:
                    uarray = array
                    if uarray.shape[1] != 2 or uarray.shape[2] != nword_det(args[0]):
                        raise ValueError('invalid array shape')
                    self._obj.from_det_array(args[0], args[1], args[2],
                            uarray.shape[0], <uint_t *>(&uarray[0, 0, 0]),
                            )
                else:
                    raise TypeError('invalid array dtype')
                return
        raise TypeError('invalid arguments to __init__')

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
        cdef np.ndarray det_array = np.empty((2, self._obj.nword), dtype=uint_dtype)
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
        return self.__class__(self)

    def copy(self):
        r"""
        Copy a two_spin_wfn instance.

        Returns
        -------
        wfn : two_spin_wfn
            Two-spin wave function object.

        """
        return self.__class__(self)

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
        cdef np.ndarray occs_array = np.zeros((end - start, 2, self._obj.nocc_up), dtype=int_dtype)
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
        cdef np.ndarray occs = np.zeros((2, self._obj.nocc_up), dtype=int_dtype)
        occs[0, :self._obj.nocc_up] = np.arange(self._obj.nocc_up, dtype=int_dtype)
        occs[1, :self._obj.nocc_dn] = np.arange(self._obj.nocc_dn, dtype=int_dtype)
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
        cdef int_t[::1] excv = np.array(sorted(set(exc)), dtype=int_dtype)
        cdef int_t nexc = excv.size, i, e, a, b
        if excv[0] < 0 or excv[nexc - 1] > emax:
            raise ValueError('invalid excitation order e < 0 or e > min(nocc, nvir)')
        # default determinant is hartree-fock determinant
        cdef np.ndarray occs
        if det is None:
            occs = np.zeros((2, self._obj.nocc_up), dtype=int_dtype)
            occs[0, :self._obj.nocc_up] = np.arange(self._obj.nocc_up, dtype=int_dtype)
            occs[1, :self._obj.nocc_dn] = np.arange(self._obj.nocc_dn, dtype=int_dtype)
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
        return self.__class__(
            self._obj.nbasis, self._obj.nocc_up, self._obj.nocc_dn,
            self.to_det_array(start, end),
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
        cdef np.ndarray det_array = np.zeros((2, self._obj.nword), dtype=uint_dtype)
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
        cdef np.ndarray occs_array = np.zeros((2, self._obj.nocc_up), dtype=int_dtype)
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
        cdef np.ndarray virs_array = np.zeros((2, self._obj.nvir_dn), dtype=int_dtype)
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
        return np.zeros((2, self._obj.nword), dtype=uint_dtype)

    def compute_overlap(self, double[::1] coeffs not None, two_spin_wfn wfn not None,
        double[::1] wfn_coeffs not None):
        r"""
        Compute the overlap of this wave function with another wave function.

        Parameters
        ----------
        coeffs : np.ndarray(c_double(ndet))
            This wave function's coefficient vector.
        wfn : two_spin_wfn
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

    def compute_rdms(self, double[::1] coeffs not None):
        r"""
        Compute the 1- and 2- particle reduced density matrices (RDMs) of a wave function.

        .. math::

            d_{pq} = \left<p|q\right>

        .. math::

            D_{pqrs} = \left<pq|rs\right>

        This method returns two nbasis-by-nbasis matrices, which include the
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

        Returns
        -------
        d0 : np.ndarray(c_double(...))
            :math:`D_0` matrix.
        d2 : np.ndarray(c_double(...))
            :math:`D_2` matrix.

        """
        # Check params
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        # Compute DOCI matrices
        cdef np.ndarray d0_array = np.zeros((self._obj.nbasis, self._obj.nbasis), dtype=double_dtype)
        cdef np.ndarray d2_array = np.zeros((self._obj.nbasis, self._obj.nbasis), dtype=double_dtype)
        cdef double[:, ::1] d0 = d0_array
        cdef double[:, ::1] d2 = d2_array
        self._obj.compute_rdms_doci(<double *>(&coeffs[0]), <double *>(&d0[0, 0]), <double *>(&d2[0, 0]))
        return d0_array, d2_array

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
        cdef double result = energy
        # Check parameters
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Restricted DOCI
        elif isinstance(ham, restricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            result += self._obj.compute_enpt2_doci(
                    <double *>(&ham._one_mo[0, 0]), <double *>(&ham._two_mo[0, 0, 0, 0]),
                    <double *>(&coeffs[0]), energy - ham._ecore, eps,
                    )
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
        if self._obj.ndet != coeffs.size:
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
    def make_rdms(double[:, ::1] d0 not None, double[:, ::1] d2 not None):
        r"""
        Generate generalized 1- and 2- particle RDMs from the :math:`D_0` and :math:`D_2` matrices.

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
        cdef np.ndarray rdm1_array = np.zeros((nspin, nspin), dtype=double_dtype)
        cdef np.ndarray rdm2_array = np.zeros((nspin, nspin, nspin, nspin), dtype=double_dtype)
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

    def __init__(self, *pyargs):
        r"""
        Initialize a fullci_wfn instance.

        Parameters
        ----------
        filename : str
            Name of TWOSPIN file to read.

        or

        Parameters
        ----------
        wfn : (doci_wfn | fullci_wfn)
            Wave function from which to initialize.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of spin-up occupied indices.
        nocc_dn : int
            Number of spin-down occupied indices.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc_up : int
            Number of spin-up occupied indices.
        nocc_dn : int
            Number of spin-down occupied indices.
        array : (np.ndarray(c_int(n, nocc)) | np.ndarray(c_uint(n, nword)))
            Array of occupied indices or determinant bitstrings.

        """
        cdef tuple args = pyargs
        cdef doci_wfn wfn
        if len(args) == 1 and isinstance(args[0], doci_wfn):
            wfn = args[0]
            self._obj.from_onespinwfn(wfn._obj)
            return
        two_spin_wfn.__init__(self, *args)

    def compute_rdms(self, double[::1] coeffs not None):
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
        aa : np.ndarray(c_double(nbasis, nbasis))
            Up-up 1-RDM spin-block.
        bb : np.ndarray(c_double(nbasis, nbasis))
            Down-down 1-RDM spin-block.
        aaaa : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Up-up-up-up 2-particle RDM spin-block.
        bbbb : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Down-down-down-down 2-particle RDM spin-block.
        abab : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Up-down-up-down 2-particle RDM spin-block.

        """
        # Check parameters
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        # Compute restricted RDMs
        cdef tuple nbasis2 = (self._obj.nbasis, self._obj.nbasis)
        cdef tuple nbasis4 = (self._obj.nbasis, self._obj.nbasis, self._obj.nbasis, self._obj.nbasis)
        cdef np.ndarray aa_array = np.zeros(nbasis2, dtype=double_dtype)
        cdef np.ndarray bb_array = np.zeros(nbasis2, dtype=double_dtype)
        cdef np.ndarray aaaa_array = np.zeros(nbasis4, dtype=double_dtype)
        cdef np.ndarray bbbb_array = np.zeros(nbasis4, dtype=double_dtype)
        cdef np.ndarray abab_array = np.zeros(nbasis4, dtype=double_dtype)
        cdef double[:, ::1] aa = aa_array
        cdef double[:, ::1] bb = bb_array
        cdef double[:, :, :, ::1] aaaa = aaaa_array
        cdef double[:, :, :, ::1] bbbb = bbbb_array
        cdef double[:, :, :, ::1] abab = abab_array
        self._obj.compute_rdms_fullci(
                <double *>(&coeffs[0]), <double *>(&aa[0, 0]), <double *>(&bb[0, 0]),
                <double *>(&aaaa[0, 0, 0, 0]), <double *>(&bbbb[0, 0, 0, 0]),
                <double *>(&abab[0, 0, 0, 0]),
                )
        return aa_array, bb_array, aaaa_array, bbbb_array, abab_array

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
        cdef double result = energy
        # Check parameters
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Restricted FullCI
        elif isinstance(ham, restricted_ham):
            if self._obj.nbasis != ham._nbasis:
                raise ValueError('dimensions of wfn, ham do not match')
            result += self._obj.compute_enpt2_fullci(
                    <double *>(&ham._one_mo[0, 0]), <double *>(&ham._two_mo[0, 0, 0, 0]),
                    <double *>(&coeffs[0]), energy - ham._ecore, eps,
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
        if self._obj.ndet != coeffs.size:
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
                    <double *>(&ham._one_mo[0, 0]), <double *>(&ham._two_mo[0, 0, 0, 0]),
                    <double *>(&coeffs[0]), eps,
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
    def make_rdms(double[:, ::1] aa not None, double[:, ::1] bb not None,
            double[:, :, :, ::1] aaaa not None, double[:, :, :, ::1] bbbb not None,
            double[:, :, :, ::1] abab not None):
        r"""
        Generate generalized 1- and 2- particle RDMs from the RDM spin-blocks.

        Parameters
        ----------
        aa : np.ndarray(c_double(nbasis, nbasis))
            Up-up 1-RDM spin-block.
        bb : np.ndarray(c_double(nbasis, nbasis))
            Down-down 1-RDM spin-block.
        aaaa : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Up-up-up-up 2-particle RDM spin-block.
        bbbb : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Down-down-down-down 2-particle RDM spin-block.
        abab : np.ndarray(c_double(nbasis, nbasis, nbasis, nbasis))
            Up-down-up-down 2-particle RDM spin-block.

        Returns
        -------
        rdm1 : np.ndarray(c_double(2 * nbasis, 2 * nbasis))
            Generalized 1-particle RDM.
        rdm2 : np.ndarray(c_double(2 * nbasis, 2 * nbasis, 2 * nbasis, 2 * nbasis))
            Generalized 2-particle RDM.

        """
        if not (aa.shape[0] == aa.shape[1] == bb.shape[0] == bb.shape[1]
                == aaaa.shape[0] == aaaa.shape[1] == aaaa.shape[2] == aaaa.shape[3]
                == bbbb.shape[0] == bbbb.shape[1] == bbbb.shape[2] == bbbb.shape[3]
                == abab.shape[0] == abab.shape[1] == abab.shape[2] == abab.shape[3]):
            raise ValueError('dimensions of d0, d2 do not match')
        cdef int_t nbasis = aa.shape[0]
        cdef int_t nspin = nbasis * 2
        cdef np.ndarray rdm1 = np.zeros((nspin, nspin), dtype=double_dtype)
        cdef np.ndarray rdm2 = np.zeros((nspin, nspin, nspin, nspin), dtype=double_dtype)
        rdm1[:nbasis, :nbasis] = aa
        rdm1[nbasis:, nbasis:] = bb
        rdm2[:nbasis, :nbasis, :nbasis, :nbasis] = aaaa
        rdm2[nbasis:, nbasis:, nbasis:, nbasis:] = bbbb
        rdm2[:nbasis, nbasis:, :nbasis, nbasis:] = abab  #abab
        rdm2[nbasis:, :nbasis, nbasis:, :nbasis] = abab  #baba
        rdm2[:nbasis, nbasis:, nbasis:, :nbasis] -= abab #abba
        rdm2[nbasis:, :nbasis, :nbasis, nbasis:] -= abab #baab
        return rdm1, rdm2


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

    def __init__(self, *pyargs):
        r"""
        Initialize a genci_wfn instance.

        Parameters
        ----------
        filename : str
            Name of ONESPIN file to read.

        or

        Parameters
        ----------
        wfn : (doci_wfn | fullci_wfn | genci_wfn)
            Wave function from which to initialize.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.

        or

        Parameters
        ----------
        nbasis : int
            Number of orbital basis functions.
        nocc : int
            Number of occupied indices.
        array : (np.ndarray(c_int(n, nocc)) | np.ndarray(c_uint(n, nword)))
            Array of occupied indices or determinant bitstrings.

        """
        cdef tuple args = pyargs
        cdef fullci_wfn wfn
        if len(args) == 1:
            if isinstance(args[0], doci_wfn):
                wfn = fullci_wfn(args[0])
                self._obj.from_twospinwfn(wfn._obj)
                return
            elif isinstance(args[0], fullci_wfn):
                wfn = args[0]
                self._obj.from_twospinwfn(wfn._obj)
                return
        two_spin_wfn.__init__(self, *args)

    def compute_rdms(self, double[::1] coeffs not None):
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
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        # Compute generalized RDMs
        cdef tuple nbasis2 = (self._obj.nbasis, self._obj.nbasis)
        cdef tuple nbasis4 = (self._obj.nbasis, self._obj.nbasis, self._obj.nbasis, self._obj.nbasis)
        cdef np.ndarray rdm1_array = np.zeros(nbasis2, dtype=double_dtype)
        cdef np.ndarray rdm2_array = np.zeros(nbasis4, dtype=double_dtype)
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
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif self._obj.nbasis != ham._nbasis:
            raise ValueError('dimensions of wfn, ham do not match')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Generalized CI
        return energy + self._obj.compute_enpt2_genci(
                <double *>(&ham._one_mo[0, 0]), <double *>(&ham._two_mo[0, 0, 0, 0]),
                <double *>(&coeffs[0]), energy - ham._ecore, eps,
                )

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
        if self._obj.ndet != coeffs.size:
            raise ValueError('dimensions of wfn, coeffs do not match')
        elif self._obj.ndet == 0:
            raise ValueError('wfn must contain at least one determinant')
        elif self._obj.nbasis != ham._nbasis:
            raise ValueError('dimensions of wfn, ham do not match')
        elif ham._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        # Generalized CI
        return self._obj.run_hci_genci(
                <double *>(&ham._one_mo[0, 0]), <double *>(&ham._two_mo[0, 0, 0, 0]),
                <double *>(&coeffs[0]), eps,
                )


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
    cdef double _ecore

    @property
    def size(self):
        r"""
        The number of non-zero elements in the sparse matrix operator.

        """
        return self._obj.size

    @property
    def shape(self):
        r"""
        The shape of the sparse matrix operator.

        """
        return self._obj.nrow, self._obj.ncol

    @property
    def ecore(self):
        r"""
        Constant/"zero-electron" integral.

        """
        return self._ecore

    def __init__(self, hamiltonian ham not None, wavefunction wfn not None, int_t nrow=-1):
        r"""
        Initialize a sparse matrix operator instance.

        Parameters
        ----------
        ham : hamiltonian
            Hamiltonian object.
        wfn : (doci_wfn | fullci_wfn | genci_wfn)
            Wave function object.
        nrow : int, optional
            Number of rows (<= number of determinants in wavefunction). Default is square matrix.

        """
        # Check parameters
        if len(wfn) == 0:
            raise ValueError('wfn must contain at least one determinant')
        # DOCI wave function
        elif isinstance(wfn, doci_wfn):
            if ham._h is None:
                raise AttributeError('seniority-zero integrals were not computed')
            # Restricted DOCI operator
            elif isinstance(ham, restricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                self._obj.init_doci(
                        (<doci_wfn>wfn)._obj,
                        <double *>(&ham._h[0]),
                        <double *>(&ham._v[0, 0]),
                        <double *>(&ham._w[0, 0]),
                        nrow,
                        )
            # Unrestricted DOCI operator
            elif isinstance(ham, unrestricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            # Generalized DOCI operator
            elif isinstance(ham, generalized_ham):
                if wfn.nbasis * 2 != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            else:
                raise ValueError('invalid ham type')
        # FullCI wave function
        elif isinstance(wfn, fullci_wfn):
            if ham._one_mo is None:
                raise AttributeError('full integral arrays were not saved')
            # Restricted FullCI operator
            elif isinstance(ham, restricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                self._obj.init_fullci(
                        (<fullci_wfn>wfn)._obj,
                        <double *>(&ham._one_mo[0, 0]),
                        <double *>(&ham._two_mo[0, 0, 0, 0]),
                        nrow,
                        )
            # Unrestricted FullCI operator
            elif isinstance(ham, unrestricted_ham):
                if wfn.nbasis != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            # Generalized FullCI operator
            elif isinstance(ham, generalized_ham):
                if wfn.nbasis * 2 != ham.nbasis:
                    raise ValueError('dimension of ham, wfn do not match')
                raise NotImplementedError
            else:
                raise ValueError('invalid ham type')
        # Generalized CI wave function
        elif isinstance(wfn, genci_wfn):
            if not isinstance(ham, generalized_ham):
                raise ValueError('invalid ham type')
            elif ham._one_mo is None:
                raise AttributeError('full integral arrays were not saved')
            elif wfn.nbasis != ham.nbasis:
                raise ValueError('dimension of ham, wfn do not match')
            self._obj.init_genci((<genci_wfn>wfn)._obj, <double *>(&ham._one_mo[0, 0]),
                    <double *>(&ham._two_mo[0, 0, 0, 0]), nrow)
        else:
            raise TypeError('wfn type must be one of \'doci_wfn\', \'fullci_wfn\', \'genci_wfn\'')
        self._ecore = ham.ecore

    def __call__(self, double[::1] x not None, double[::1] out=None):
        r"""
        Compute the result of the sparse operator :math:`A` applied to vector :math:`x`.

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
            y = np.empty(self._obj.nrow, dtype=double_dtype)
            out = y
        else:
            if out.size != self._obj.nrow:
                raise ValueError('Dimensions of operator and \'out\' do not match')
            y = np.asarray(out)
        # return result of operation
        self._obj.perform_op(&x[0], &out[0])
        return y

    def to_csr_matrix(self):
        r"""
        Convert the sparse matrix operator to CSR matrix data in NumPy arrays.

        Returns
        -------
        csr_mat : tuple(np.ndarray, np.ndarray, np.ndarray)
            CSR matrix data. Can be passed to `scipy.sparse.csr_matrix`.

        """
        cdef np.ndarray data = np.copy(<double[:self._obj.size]>self._obj.data_ptr(0))
        cdef np.ndarray indices = np.copy(<int_t[:self._obj.size]>self._obj.indices_ptr(0))
        cdef np.ndarray indptr = np.copy(<int_t[:self._obj.nrow + 1]>self._obj.indptr_ptr(0))
        return data, indices, indptr

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
        if self._obj.nrow != self._obj.ncol:
            raise ValueError('cannot solve for a rectangular operator')
        # set default number of lanczos vectors n < ncv <= len(c0)
        if ncv == -1:
            ncv = max(n + 1, min(20, self._obj.nrow))
        # default initial guess c = [1, 0, ..., 0]
        if c0 is None:
            c0 = np.zeros(self._obj.nrow, dtype=double_dtype)
            c0[0] = 1.0
        elif self._obj.nrow != c0.size:
            raise ValueError('dimension of sparse_op, c0 do not match')
        # default maxiter = 1000 * n
        if maxiter == -1:
            maxiter = 1000 * n
        # solve eigenproblem
        cdef np.ndarray evals_array = np.empty(n, dtype=double_dtype)
        cdef np.ndarray evecs_array = np.empty((n, self._obj.nrow), dtype=double_dtype)
        cdef double[::1] evals = evals_array
        cdef double[:, ::1] evecs = evecs_array
        self._obj.solve(<double *>(&c0[0]), n, ncv, maxiter, tol,
                <double *>(&evals[0]), <double *>(&evecs[0, 0]))
        evals_array += self._ecore
        return evals_array, evecs_array
