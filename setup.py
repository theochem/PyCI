# This file is part of DOCI.
#
# DOCI is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# DOCI is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with DOCI. If not, see <http://www.gnu.org/licenses/>.

r"""
DOCI setup script.

Run `python setup.py --help` for help.

"""

from io import open
from os import path, rename
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

from setuptools import setup

import numpy


name = 'doci'

version = '0.1.0'

license = 'GPLv3'

author = 'Michael Richer'

author_email = 'richerm@mcmaster.ca'

url = 'https://github.com/msricher/doci'

description = 'A flexible seniority-zero configuraion interaction libary.'

long_description = open('README.rst', 'r', encoding='utf-8').read()


classifiers = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Science/Engineering :: Molecular Science',
    ]


install_requires = [
    'numpy>=1.13',
    ]


extras_require = {
    'build': ['cython'],
    'test': ['nose'],
    'doc': ['sphinx', 'sphinx_rtd_theme'],
    }


packages = [
    'doci',
    'doci.test',
    ]


package_data = {
    'doci': ['doci.h', 'doci.cpp', 'cext.pxd', 'cext.pyx', 'cext.cpp'],
    'doci.test': ['data/*.fcidump', 'data/*.npz'],
    }


include_dirs = [
    'parallel-hashmap',
    'eigen',
    'spectra/include',
    numpy.get_include(),
    ]


compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-Wall',
    '-fopenmp',
    '-O3',
    ]


cext = {
    'name': 'doci.cext',
    'language': 'c++',
    'sources': ['doci/cext.cpp', 'doci/doci.cpp'],
    'include_dirs': include_dirs,
    'extra_compile_args': compile_args,
    'extra_link_args': compile_args,
    }


try:
    from Cython.Distutils import build_ext, Extension
    cext['sources'] = ['doci/cext.pyx', 'doci/doci.cpp']
    cext['cython_compile_time_env'] = dict(DOCI_VERSION=str(version))
except ImportError:
    from setuptools.command.build_ext import build_ext
    from setuptools import Extension


class BuildExtCommand(build_ext):
    r"""
    Custom "build_ext" command that downloads C++ header libraries prior to building extensions.

    """
    header_libraries = {
        'parallel-hashmap':
            ('parallel-hashmap-master',
             'https://github.com/greg7mdp/parallel-hashmap/archive/master.zip'),
        'eigen':
            ('eigen-master',
             'https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.zip'),
        'spectra':
            ('spectra-master',
             'https://github.com/yixuan/spectra/archive/master.zip'),
        }

    def run(self):
        r"""
        Download header libraries and build extensions.

        """
        self.announce('checking for C++ header libraries', level=2)
        for lib, (dirname, url) in self.header_libraries.items():
            if path.exists(lib):
                self.announce('found {0:s}'.format(lib), level=2)
            else:
                self.announce('downloading {0:s}'.format(lib), level=2)
                ZipFile(urlretrieve(url, NamedTemporaryFile().name)[0], 'r').extractall()
                rename(dirname, lib)
        build_ext.run(self)


if __name__ == '__main__':

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
        ext_modules=[Extension(**cext)],
        cmdclass={'build_ext': BuildExtCommand},
        )
