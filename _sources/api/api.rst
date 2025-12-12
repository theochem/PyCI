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

Constants
=========

.. autodata:: pyci.__version__

Numerical Types
---------------

.. autodata:: pyci.c_long

.. autodata:: pyci.c_ulong

.. autodata:: pyci.c_double

Classes
=======

Second-quantized operator class
-------------------------------

.. autoclass:: pyci.secondquant_op
    :members:

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

Functions
=========

Threading
---------

The number of threads to use is decided as follows, in decreasing priority:

- The value of a keyword argument ``nthread=`` to a particular PyCI function or method call
- The value of ``n`` from the most recent invokation of ``pyci.set_num_threads(n)``
- The value of environment variable ``PYCI_NUM_THREADS`` when Python is started
- The number of threads supported by the hardware (``std::thread::hardware_concurrency()``)

.. autofunction:: pyci.get_num_threads

.. autofunction:: pyci.set_num_threads

Selected CI routines
--------------------

.. autofunction:: pyci.add_hci

.. autofunction:: pyci.add_excitations

.. autofunction:: pyci.add_seniorities

.. autofunction:: pyci.add_gkci


Post-CI routines
----------------

.. autofunction:: pyci.compute_rdms

.. autofunction:: pyci.spinize_rdms

.. autofunction:: pyci.compute_enpt2

.. autofunction:: pyci.compute_overlap

Integral transformations
------------------------

.. autofunction:: pyci.make_senzero_integrals

.. autofunction:: pyci.reduce_senzero_integrals

FanCI classes
=============

.. autoclass:: pyci.fanci.AP1roG
    :members:

.. autoclass:: pyci.fanci.APIG
    :members:

.. autoclass:: pyci.fanci.DetRatio
    :members:

.. autoclass:: pyci.fanci.pCCDS
    :members:
