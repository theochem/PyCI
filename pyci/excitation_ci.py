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

r"""PyCI excitation CI module."""

__all__ = [
    "add_excitations",
]


def add_excitations(wfn, *excitations, ref=None):
    r"""
    Add excited determinants to a wave function.

    Parameters
    ----------
    wfn : pyci.wavefunction
        Wave function.
    excitations : Sequence[int]
        List of excitation levels of determinants to add.
    ref : numpy.ndarray, optional
        Reference determinant by which to determine excitation levels.
        Default is the Hartree-Fock determinant.

    """
    print("Inside excitation_ci/add_excitations")
    for e in excitations:
        print(f"Adding {e}-excited determinants...")
        wfn.add_excited_dets(e, ref=ref)
