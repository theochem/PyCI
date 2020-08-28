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

API Documentation
#################

C++ extension
=============

.. autodata:: pyci.__version__

Numerical Types
---------------

.. autodata:: pyci.c_long

.. autodata:: pyci.c_ulong

.. autodata:: pyci.c_double

Threading
---------

The number of threads to use is decided as follows, in decreasing priority:

- The number of threads specified by a ``nthread=`` keyword to a PyCI function or method.
- The number of threads specified by calling ``pyci.set_num_threads`` at run time.
- The number of threads specified by environment variable ``PYCI_NUM_THREADS`` at run time.
- The number of threads specified by macro variable ``PYCI_NUM_THREADS_DEFAULT`` at compile time.

The sparse matrix operator matrix-vector product and eigenproblem methods will also use OpenMP for
multi-threading if PyCI was compiled with OpenMP support, i.e., with the ``-fopenmp`` C++ flag.
The threading for these methods can be controlled as usual for OpenMP, or simply by compiling PyCI
without OpenMP support.

.. autofunction:: pyci.get_num_threads

.. autofunction:: pyci.set_num_threads

Functions
=========

Selected CI routines
--------------------

.. autofunction:: pyci.add_excitations

.. autofunction:: pyci.add_seniorities

.. autofunction:: pyci.add_gkci

.. autofunction:: pyci.add_hci

CI Solver
---------

.. autofunction:: pyci.solve

Post-CI routines
----------------

Integral transformations
------------------------

.. autofunction:: pyci.make_senzero_integrals

.. autofunction:: pyci.reduce_senzero_integrals

.. autofunction:: pyci.make_rdms

FCIDUMP
-------

.. autofunction:: pyci.read_fcidump
.. autofunction:: pyci.write_fcidump

Classes
=======

Hamiltonian
-----------

.. autoclass:: pyci.hamiltonian
    :members:

Base wave function classes
--------------------------

.. autoclass:: pyci.wavefunction
    :members:

.. autoclass:: pyci.one_spin_wfn
    :members:

.. autoclass:: pyci.two_spin_wfn
    :members:

DOCI wave function
------------------

.. autoclass:: pyci.doci_wfn
    :members:

FullCI wave function
--------------------

.. autoclass:: pyci.fullci_wfn
    :members:

Generalized CI wave function
----------------------------

.. autoclass:: pyci.genci_wfn
    :members:

Sparse matrix operator
----------------------

.. autoclass:: pyci.sparse_op
    :members:
