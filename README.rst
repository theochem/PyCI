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

|Python 3.8| |Travis|

PyCI
====

PyCI_ is a flexible *ab-initio* quantum chemistry library for Configuration Interaction.

PyCI is distributed under the GNU General Public License version 3 (GPLv3).

See http://www.gnu.org/licenses/ for more information.

Dependencies
------------

The following programs/libraries are required to run PyCI:

-  Python_ (≥3.6)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)
-  Pytest_ (optional: to run tests)

The following programs/libraries are required to build PyCI:

-  Make_
-  Git_
-  GCC_ (≥4.8) or `Clang/LLVM`_ (≥3.3) C++ compiler
-  Python_ (≥3.6, including C headers)
-  NumPy_ (≥1.13, including C headers)

Some header-only C++ libraries downloaded automatically:

-  Eigen_
-  `Parallel Hashmap`__
-  Pybind11_
-  Spectra_
-  SpookyHash_

__ Parallel-Hashmap_

The following programs/libraries are required to build the PyCI documentation:

-  Sphinx_
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

Installation
------------

Basic Compilation and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to download PyCI via git:

.. code:: shell

    git clone https://github.com/msricher/pyci.git && cd pyci

Then, run the following to build and install PyCI:

.. code:: shell

    make
    python3 -m pip install -e .

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to install the Read the Docs Sphinx theme via pip:

.. code:: shell

    python3 -m pip install --user sphinx-rtd-theme

Then, after installing PyCI, run the following to build the HTML documentation:

.. code:: shell

    cd doc && make html

.. _`Clang/LLVM`:       http://clang.llvm.org/
.. _Eigen:              http://eigen.tuxfamily.org/
.. _GCC:                http://gcc.gnu.org/
.. _Make:               http://gnu.org/software/make/
.. _Git:                http://git-scm.com/
.. _NumPy:              http://numpy.org/
.. _Parallel-Hashmap:   http://github.com/greg7mdp/parallel-hashmap/
.. _PyCI:               http://github.com/msricher/PyCI/
.. _Pybind11:           http://pybind11.readthedocs.io/en/stable/
.. _Pytest:             http://docs.pytest.org/en/latest/
.. _Python:             http://python.org/
.. _SciPy:              http://docs.scipy.org/doc/scipy/reference/
.. _Spectra:            http://spectralib.org/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/
.. _Sphinx:             http://sphinx-doc.org/
.. _SpookyHash:         http://www.burtleburtle.net/bob/hash/spooky.html

.. |Python 3.8| image:: http://img.shields.io/badge/python-3.8-blue.svg
   :target: https://docs.python.org/3.8/
.. |Travis| image:: http://travis-ci.com/msricher/PyCI.svg?token=cXv5xZ8ji4xAnkUvpsev&branch=master
   :target: http://travis-ci.com/msricher/PyCI/
