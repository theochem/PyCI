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

    @classmethod
    def from_file(cls, str filename not None, bint keep_mo=True, bint doci=True):
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
        cdef str line, key, val, field
        cdef list fields
        cdef dict header_info
        cdef int_t nbasis, nelec, ms2, i, j, k, l
        cdef double[:, ::1] one_mo
        cdef double[:, :, :, ::1] two_mo
        cdef double ecore = 0.0, fval
        with open(filename, 'r', encoding='utf-8') as f:
            # check header
            line = next(f)
            if not line.startswith(' &FCI NORB='):
                raise IOError('Error in FCIDUMP file header')
            # read info from header
            fields = line[5:].split(',')
            header_info = dict()
            for field in fields:
                if field.count('=') == 1:
                    key, val = field.split('=')
                    header_info[key.strip()] = val.strip()
            nbasis = int(header_info['NORB'])
            nelec = int(header_info.get('NELEC', '0'))
            ms2 = int(header_info.get('MS2', '0'))
            # skip rest of header
            for line in f:
                field = line.split()[0]
                if field == '&END' or field == '/END' or field == '/':
                    break
            # read integrals
            one_mo = np.zeros((nbasis, nbasis), dtype=np.double)
            two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.double)
            for line in f:
                fields = line.split()
                if len(fields) != 5:
                    raise IOError('Expecting 5 fields on each data line in FCIDUMP')
                fval = float(fields[0].strip())
                if fields[3] != '0':
                    i = int(fields[1].strip()) - 1
                    j = int(fields[2].strip()) - 1
                    k = int(fields[3].strip()) - 1
                    l = int(fields[4].strip()) - 1
                    two_mo[i, k, j, l] = fval
                    two_mo[k, i, l, j] = fval
                    two_mo[j, k, i, l] = fval
                    two_mo[i, l, j, k] = fval
                    two_mo[j, l, i, k] = fval
                    two_mo[l, j, k, i] = fval
                    two_mo[k, j, l, i] = fval
                    two_mo[l, i, k, j] = fval
                elif fields[1] != '0':
                    i = int(fields[1].strip()) - 1
                    j = int(fields[2].strip()) - 1
                    one_mo[i, j] = fval
                    one_mo[j, i] = fval
                else:
                    ecore = fval
        return cls(ecore, one_mo, two_mo, keep_mo=keep_mo, doci=doci)

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
        if not (one_mo.shape[0] == one_mo.shape[1] == two_mo.shape[0] \
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


cdef class generalized_ham(hamiltonian):
    r"""
    Generalized Hamiltonian class.

    """
    pass


cdef class unrestricted_ham(hamiltonian):
    r"""
    Unrestricted Hamiltonian object.

    """

    @classmethod
    def from_file(cls, str filename not None, bint keep_mo=True, bint doci=True):
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
        raise NotImplementedError

    @property
    def one_mo(self):
        r"""
        Full one-electron integral array.

        """
        if self._one_mo is None:
            raise AttributeError('full integral arrays were not saved')
        return np.asarray(self._one_mo).reshape(2, self._nbasis, self._nbasis, self._nbasis, self._nbasis)

    @property
    def two_mo(self):
        r"""
        Full two-electron integral array.

        """
        if self._two_mo is None:
            raise AttributeError('full integral arrays were not saved')
        return np.asarray(self._two_mo).reshape(2, self._nbasis, self._nbasis, self._nbasis, self._nbasis)

    @property
    def h(self):
        r"""
        Seniority-zero one-electron integrals.

        """
        if self._h is None:
            raise AttributeError('seniority-zero integrals were not computed')
        return np.asarray(self._h).reshape(2, self._nbasis)

    @property
    def v(self):
        r"""
        Seniority-zero two-electron integrals.

        """
        if self._v is None:
            raise AttributeError('seniority-zero integrals were not computed')
        return np.asarray(self._v).reshape(2, self._nbasis, self._nbasis)

    @property
    def w(self):
        r"""
        Seniority-two two-electron integrals.

        """
        if self._w is None:
            raise AttributeError('seniority-zero integrals were not computed')
        return np.asarray(self._w).reshape(2, self._nbasis, self._nbasis)

    def __init__(self, double ecore, double[:, :, ::1] one_mo not None,
            double[:, :, :, :, ::1] two_mo not None, bint keep_mo=True, bint doci=True):
        """
        Initialize a Hamiltonian instance.

        Parameters
        ----------
        ecore : float
            Constant/"zero-electron" integral.
        one_mo : np.ndarray(c_double(2, nbasis, nbasis))
            Full one-electron integral array.
        two_mo : np.ndarray(c_double(2, nbasis, nbasis, nbasis, nbasis))
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


# vim: set ft=pyrex:
