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

NumPy dtypes
------------

.. autodata:: pyci.c_int

.. autodata:: pyci.c_uint

.. autodata:: pyci.c_double

Constants
---------

.. autodata:: pyci.fullci.SPIN_UP

.. autodata:: pyci.fullci.SPIN_DN

DOCI Hamiltonian
----------------

.. autoclass:: pyci.doci.ham

.. autoclass:: pyci.cext.doci_ham
    :members:

DOCI wave function
------------------

.. autoclass:: pyci.doci.wfn

.. autoclass:: pyci.cext.doci_wfn
    :members:

DOCI routines
-------------

.. autofunction:: pyci.doci.solve_ci

.. autofunction:: pyci.doci.compute_rdms

.. autofunction:: pyci.doci.generate_rdms

.. autofunction:: pyci.doci.compute_energy

.. autofunction:: pyci.doci.run_hci

FullCI Hamiltonian
------------------

.. autoclass:: pyci.fullci.ham

.. autoclass:: pyci.cext.fullci_ham
    :members:

FullCI wave function
--------------------

.. autoclass:: pyci.fullci.wfn

.. autoclass:: pyci.cext.fullci_wfn
    :members:

FullCI routines
---------------

WIP.
