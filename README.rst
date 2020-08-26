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

|Python 3.8|

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
-  GCC_ (≥4.8) or `Clang/LLVM_` (≥3.3) C++ compiler
-  Python_ (≥3.6, including C headers)
-  NumPy_ (≥1.13, including C headers)

Some header-only C++ libraries are included with this git repository:

-  Pybind11_ (git submodule)
-  `Parallel Hashmap`__ (git submodule)
-  SpookyHash_ (files included directly)

__ Parallel-Hashmap_

The following programs/libraries are required to build the PyCI documentation:

-  Sphinx_
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

Installation
------------

Basic Compilation and Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to download PyCI and its submodules via git:

.. code:: shell

    git clone --recursive https://github.com/msricher/pyci.git && cd pyci

Then, run the following to build and install PyCI:

.. code:: shell

    make
    python setup.py install --user

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Run the following in your shell to install the Read the Docs Sphinx theme via pip:

.. code:: shell

    pip install sphinx-rtd-theme --user

Then, after installing PyCI, run the following to build the HTML documentation:

.. code:: shell

    cd doc && make html

.. _GCC:                http://gcc.gnu.org/
.. _Make:               http://gnu.org/software/make/
.. _NumPy:              http://numpy.org/
.. _Parallel-Hashmap:   http://github.com/greg7mdp/parallel-hashmap/
.. _PyCI:               http://github.com/msricher/PyCI/
.. _Pybind11:           http://pybind11.readthedocs.io/en/stable/
.. _Pytest:             http://docs.pytest.org/en/latest/
.. _Python:             http://python.org/
.. _SciPy:              http://docs.scipy.org/doc/scipy/reference/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/
.. _Sphinx:             http://sphinx-doc.org/
.. _SpookyHash:         http://www.burtleburtle.net/bob/hash/spooky.html
.. _Clang/LLVM:         http://clang.llvm.org/

.. |Python 3.8| image:: http://img.shields.io/badge/python-3.8-blue.svg
   :target: https://docs.python.org/3.8/
