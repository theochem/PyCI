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
DOCI test module.

"""

from os import path


__all__ = [
    'datafile',
    ]


DATAPATH = path.join(path.abspath(path.dirname(__file__)), 'data/')


def datafile(name):
    r"""
    Return the full path of a DOCI test data file.

    Parameters
    ----------
    name : str
        Name of data file.

    Returns
    -------
    filename : str
        Path to file.

    """
    return path.abspath(path.join(DATAPATH, name))
