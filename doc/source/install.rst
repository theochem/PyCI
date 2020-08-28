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

Installation
############

Dependencies
============

The following programs/libraries are required to run PyCI:

-  Python_ (≥3.6)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)
-  Pytest_ (optional: to run tests)

The following programs/libraries are required to build PyCI:

-  Make_
-  GCC_ (≥4.8) or `Clang/LLVM`_ (≥3.3) C++ compiler
-  Python_ (≥3.6, including C headers)
-  NumPy_ (≥1.13, including C headers)

Some header-only C++ libraries are included with this git repository as git submodules:

-  Eigen_
-  `Parallel Hashmap`__
-  Pybind11_
-  Spectra_

__ Parallel-Hashmap_

The programs required to get started can be installed with your operating system's package manager.

E.g., for Debian- or Ubuntu- based Linux systems:

.. code:: shell

    sudo apt-get install make gcc python3-pip python3-sphinx

The required Python packages can be installed with Pip:

.. code:: shell

    python3 -m pip install numpy scipy pytest

Download
========

Run the following in your shell to download PyCI and its submodules via git:

.. code:: shell

    git clone --recursive https://github.com/msricher/pyci.git && cd pyci

Build
=====

Run the following to build PyCI:

.. code:: shell

    PYTHON=python3 make

Install
=======

Run the following to intall PyCI:

.. code:: shell

    python3 -m pip install .

Test
====

Run the following to test PyCI:

.. code:: shell

    python3 -m pytest pyci

Build Documentation
===================

The following programs/libraries are required to build the PyCI documentation:

-  Sphinx_
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

Run the following in your shell to install the Read the Docs Sphinx theme via pip:

.. code:: shell

    python3 -m pip install sphinx-rtd-theme

Then, after building PyCI, run the following to build the HTML documentation:

.. code:: shell

    cd doc && make html

.. _`Clang/LLVM`:       http://clang.llvm.org/
.. _Eigen:              http://eigen.tuxfamily.org/
.. _GCC:                http://gcc.gnu.org/
.. _Make:               http://gnu.org/software/make/
.. _NumPy:              http://numpy.org/
.. _Parallel-Hashmap:   http://github.com/greg7mdp/parallel-hashmap/
.. _Pybind11:           http://pybind11.readthedocs.io/en/stable/
.. _Pytest:             http://docs.pytest.org/en/latest/
.. _Python:             http://python.org/
.. _SciPy:              http://docs.scipy.org/doc/scipy/reference/
.. _Spectra:            http://spectralib.org/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/
.. _Sphinx:             http://sphinx-doc.org/
