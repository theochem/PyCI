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

name = "pyci"


version = "0.6.1"


license = "GPLv3"


author = "Michelle Richer"


author_email = "richerm@mcmaster.ca"


url = "https://github.com/msricher/PyCI"


description = "A flexible quantum chemistry CI library for Python 3."


long_description = open("README.rst", "r", encoding="utf-8").read()


classifiers = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Science/Engineering :: Molecular Science",
]


install_requires = [
    "numpy>=1.13",
    "scipy>=1.0",
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
    "pyci": ["pyci.so", "include/*.h", "src/*.cpp"],
    "pyci.test": ["data/*.fcidump", "data/*.npy", "data/*.npz"],
}


if __name__ == "__main__":

    from setuptools import setup

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
    )
