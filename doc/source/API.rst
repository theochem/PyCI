..
    : This file is part of PyCI.
    :
    : PyCI is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : PyCI is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with PyCI. If not, see <http://www.gnu.org/licenses/>.

PyCI API Documentation
======================

Numerical Types and Constants
-----------------------------

.. autodata:: pyci.c_int

.. autodata:: pyci.c_uint

.. autodata:: pyci.c_double

.. autodata:: pyci.SpinLabel

.. autodata:: pyci.SPIN_UP

.. autodata:: pyci.SPIN_DN

Hamiltonians
------------

.. autoclass:: pyci.hamiltonian
    :members:

.. autoclass:: pyci.restricted_ham
    :members:

.. autoclass:: pyci.unrestricted_ham
    :members:

.. autoclass:: pyci.generalized_ham
    :members:

Wave functions
--------------

.. autoclass:: pyci.wavefunction
    :members:

.. autoclass:: pyci.one_spin_wfn
    :members:

.. autoclass:: pyci.two_spin_wfn
    :members:

.. autoclass:: pyci.doci_wfn
    :members:

.. autoclass:: pyci.fullci_wfn
    :members:

.. autoclass:: pyci.genci_wfn
    :members:

Sparse matrix solver
--------------------

.. autoclass:: pyci.sparse_op
    :members:
