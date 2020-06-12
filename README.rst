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

|Python 2.7| |Python 3.8|

PyCI
====

PyCI is a flexible *ab-initio* quantum chemistry library for Configuration
Interaction consisting of a C++ library and a Python package.

PyCI is distributed under the GNU General Public License version 3 (GPLv3+).

See http://www.gnu.org/licenses/ for more information.

Dependencies
------------

The following programs/libraries are required to run PyCI:

-  Python_ (≥3.x or ≥2.7)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.17)

The following programs/libraries are required to build PyCI:

-  GCC_ (≥4.8) or `Clang/LLVM`_ (≥3.3) C++ compiler
-  Python_ (≥3.x or ≥2.7, including system headers)
-  NumPy_ (≥1.13)
-  Cython_ (≥0.24)
-  Nosetests_ (optional: to run tests)

The following programs/libraries are required to build the PyCI documentation:

-  Sphinx_
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

The following header-only libraries are included as git submodules:

-  `Parallel Hashmap`__
-  Eigen_
-  Spectra_

__ Parallel-Hashmap_

Installation
------------

Basic Compilation and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to download PyCI and its submodules via git:

.. code:: shell

    git clone --recursive https://github.com/msricher/pyci.git

Then, run the following to build and install PyCI:

.. code:: shell

    python setup.py install --user

Compiling on OSX
~~~~~~~~~~~~~~~~

Since the default Xcode Clang compiler for OSX does not support OpenMP, the C and C++ compilers must
be specified by using the ``CC`` and ``CXX`` environment variables:

.. code:: shell

    CC=/path/to/your/gcc/or/clang CXX=/path/to/your/g++/or/clang++ python setup.py install --user

Examples
--------

Full seniority-zero CI
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from pyci import doci

    # Load Hamiltonian from FCIDUMP file
    ham = doci.ham.from_file('Ne.FCIDUMP')

    # Generate full wave function with 5 electron pairs
    wfn = doci.wfn(ham.nbasis, 5)
    wfn.add_all_dets()

    # Solve CI problem and compute RDMs
    op = doci.sparse_op(ham, wfn)
    evals, evecs = op.solve(n=1)
    rdm0, rdm2 = doci.compute_rdms(wfn, evecs[0])

CI pairs singles and doubles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Generate CI pairs singles and doubles wave function
    wfn = doci.wfn(ham.nbasis, 5)
    wfn.add_excited_dets(0, 1, 2)

    # Solve CI problem and compute RDMs
    op = doci.sparse_op(ham, wfn)
    evals, evecs = op.solve(n=1)
    rdm0, rdm2 = doci.compute_rdms(wfn, evecs[0])

Seniority-zero Heat-Bath CI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Generate Hartree-Fock wave function
    wfn = doci.wfn(ham.nbasis, 5)
    wfn.add_hartreefock_det()

    # Solve initial CI problem
    op = doci.sparse_op(ham, wfn)
    evals, evecs = op.solve(n=1)

    # Run HCI iterations at epsilon=1.0e-5 until no more determinants are added
    epsilon = 1.0e-5
    dets_added = 1
    while dets_added:
        dets_added = doci.run_hci(ham, wfn, evecs[0], epsilon)
        op = doci.sparse_op(ham, wfn)
        evals, evecs = op.solve(n=1)

    # Compute RDMs
    rdm0, rdm2 = doci.compute_rdms(wfn, evecs[0])

Excited states
~~~~~~~~~~~~~~

.. code:: python

    # Solve CI problem for three lowest-energy states
    op = doci.sparse_op(ham, wfn)
    evals, evecs = op.solve(n=3)

    # Compute RDMs of ground state and first two excited states
    e0_rdm0, e0_rdm2 = doci.compute_rdms(wfn, evecs[0])
    e1_rdm0, e1_rdm2 = doci.compute_rdms(wfn, evecs[1])
    e2_rdm0, e2_rdm2 = doci.compute_rdms(wfn, evecs[2])

.. _Python:             http://python.org/
.. _NumPy:              http://numpy.org/
.. _SciPy:              http://docs.scipy.org/doc/scipy/reference/
.. _Cython:             http://cython.org/
.. _GCC:                http://gcc.gnu.org/
.. _`Clang/LLVM`:       http://clang.llvm.org/
.. _Nosetests:          http://nose.readthedocs.io/
.. _Sphinx:             http://sphinx-doc.org/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/
.. _Parallel-Hashmap:   http://github.com/greg7mdp/parallel-hashmap/
.. _Eigen:              http://eigen.tuxfamily.org/
.. _Spectra:            http://spectralib.org/

.. |Python 2.7| image:: http://img.shields.io/badge/python-2.7-blue.svg
   :target: https://docs.python.org/2.7/

.. |Python 3.8| image:: http://img.shields.io/badge/python-3.8-blue.svg
   :target: https://docs.python.org/3.8/
