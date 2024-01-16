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

|Python 3|

PyCI
====

PyCI_ is a flexible quantum chemistry Configuration Interaction library for Python 3.

PyCI is distributed under the GNU General Public License version 3 (GPLv3).

See http://www.gnu.org/licenses/ for more information.

Installation
------------

Dependencies
~~~~~~~~~~~~

The following programs/libraries are required to run PyCI:

-  Python_ (≥3.6)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)

The following programs/libraries are required to build PyCI:

-  Make_
-  Git_
-  GCC_ (≥4.8) or `Clang/LLVM`_ (≥3.3) C++ compiler
-  Python_ (≥3.6, including C headers)
-  NumPy_ (≥1.13, including C headers)
-  Pytest_ (optional: to run tests)
-  Pycodestyle_ (optional: to run tests)
-  Pydocstyle_ (optional: to run tests)
-  Sphinx_ (optional: to build the documentation)
-  `Read the Docs Sphinx Theme`__ (optional: to build the documentation)

__ Sphinx-RTD-Theme_

Some header-only C++ libraries are downloaded automatically:

-  Eigen_
-  `Parallel Hashmap`__
-  Pybind11_
-  Spectra_
-  CLHash_

__ Parallel-Hashmap_

Install dependencies
~~~~~~~~~~~~~~~~~~~~

The programs required to build and run PyCI can be installed with your operating system's package
manager.

E.g., for Debian- or Ubuntu- based Linux systems:

.. code:: shell

    sudo apt-get install make git gcc python3 python3-devel python3-pip python3-sphinx

The required Python packages can then be installed with pip:

.. code:: shell

    python3 -m pip install numpy scipy pytest pycodestyle pydocstyle

Download PyCI
~~~~~~~~~~~~~

Run the following in your shell to download PyCI via git:

.. code:: shell

    git clone https://github.com/msricher/pyci.git && cd pyci

Install PyCI
~~~~~~~~~~~~

Run the following to build and install PyCI:

.. code:: shell

    make
    python3 -m pip install .

Run the following to test PyCI:

.. code:: shell

    python3 -m pytest -v ./pyci

Build documentation
~~~~~~~~~~~~~~~~~~~

Run the following in your shell to install the Read the Docs Sphinx theme via pip:

.. code:: shell

    python3 -m pip install sphinx-rtd-theme

Then, after building PyCI, run the following to build the HTML API documentation:

.. code:: shell

    cd doc && make html

Citing PyCI
-----------

See the CONTRIBUTORS file.

.. _Eigen:              http://eigen.tuxfamily.org/
.. _GCC:                http://gcc.gnu.org/
.. _Git:                http://git-scm.com/
.. _Make:               http://gnu.org/software/make/
.. _NumPy:              http://numpy.org/
.. _Parallel-Hashmap:   http://github.com/greg7mdp/parallel-hashmap/
.. _PyCI:               http://github.com/msricher/PyCI/
.. _Pybind11:           http://pybind11.readthedocs.io/en/stable/
.. _Pycodestyle:        http://pycodestyle.pycqa.org/en/latest/
.. _Pydocstyle:         http://www.pydocstyle.org/en/latest/
.. _Pytest:             http://docs.pytest.org/en/latest/
.. _Python:             http://python.org/
.. _SciPy:              http://docs.scipy.org/doc/scipy/reference/
.. _Spectra:            http://spectralib.org/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/
.. _Sphinx:             http://sphinx-doc.org/
.. _CLHash:             https://github.com/lemire/clhash/
.. _`Clang/LLVM`:       http://clang.llvm.org/

.. |Python 3| image:: http://img.shields.io/badge/python-3-blue.svg
   :target: https://docs.python.org/3.8/
