# This file is part of PyCI.
#
# PyCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# PyCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCI. If not, see <http://www.gnu.org/licenses/>.

r"""
PyCI setup script.

Run `python setup.py --help` for help.

"""

from os import environ
from shutil import which

from setuptools import Extension, setup

import numpy

# Uncomment this to use a specific non-negative integer seed for the SpookyHash algorithm.
# PYCI_SPOOKYHASH_SEED = 0

name = "pyci"


version = "0.5.0"


license = "GPLv3"


author = "Michael Richer"


author_email = "richerm@mcmaster.ca"


url = "https://github.com/msricher/PyCI"


description = "A flexible ab-initio quantum chemistry library for Configuration Interaction."


long_description = open("README.rst", "r", encoding="utf-8").read()


classifiers = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Science/Engineering :: Molecular Science",
]


install_requires = [
    "numpy>=1.13",
]


extras_require = {
    "test": ["pytest"],
    "doc": ["sphinx", "sphinx_rtd_theme"],
}


packages = [
    "pyci",
    "pyci.test",
]


package_data = {
    "pyci": ["include/*.h", "src/*.cpp"],
    "pyci.test": ["data/*.fcidump", "data/*.npy", "data/*.npz"],
}


sources = [
    "pyci/src/pyci.cpp",
]


include_dirs = [
    numpy.get_include(),
    "lib/parallel-hashmap",
    "lib/pybind11/include",
    "pyci/include",
]


extra_compile_args = [
    "-Wall",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    f"-DPYCI_VERSION={version}",
    "-fvisibility=hidden",
    "-fopenmp",
]


extra_link_args = [
    "-fopenmp",
]


cext = {
    "name": "pyci.pyci",
    "language": "c++",
    "sources": sources,
    "include_dirs": include_dirs,
    "extra_compile_args": extra_compile_args,
    "extra_link_args": extra_link_args,
}


if __name__ == "__main__":

    if "CXX" not in environ and which("clang++") is not None:
        environ.update(CC="clang", CXX="clang++")

    try:
        extra_compile_args.append(f"-DPYCI_SPOOKYHASH_SEED={hex(abs(PYCI_SPOOKYHASH_SEED))}U")
    except NameError:
        pass

    pyci_extension = Extension(**cext)

    ext_modules = [
        pyci_extension,
    ]

    setup(
        name=name,
        version=version,
        license=license,
        author=author,
        author_email=author_email,
        url=url,
        description=description,
        long_description=long_description,
        classifiers=classifiers,
        install_requires=install_requires,
        extras_require=extras_require,
        packages=packages,
        package_data=package_data,
        include_package_data=True,
        ext_modules=ext_modules,
    )
