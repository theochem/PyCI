..
    : This file is part of DOCI.
    :
    : DOCI is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : DOCI is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with DOCI. If not, see <http://www.gnu.org/licenses/>.

DOCI API Documentation
======================

NumPy dtypes
------------

.. autodata:: doci.c_int

.. autodata:: doci.c_uint

.. autodata:: doci.c_double

DOCI Hamiltonian
----------------

.. autoclass:: doci.dociham
    :members:

DOCI wave function
------------------

.. autoclass:: doci.dociwfn
    :members:

DOCI solver
-----------

.. autofunction:: doci.solve_ci

Reduced density matrices
------------------------

.. autofunction:: doci.compute_rdms

.. autofunction:: doci.generate_rdms

Compute energy
--------------

.. autofunction:: doci.compute_energy

Heat-bath CI
------------

.. autofunction:: doci.run_hci
