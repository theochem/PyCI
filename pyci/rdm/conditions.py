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

r"""PyCI RDM Conditions module."""

import numpy as np

__all__ = [
    "find_closest_sdp",
]

def find_closest_sdp(dm, constraint, alpha):
    r"""
    Projection onto a semidefinite constraint.

    Parameters
    ----------
    dm : np.ndarray
        Density matrix.
    constraint : function
        Positive semidefinite constraint, linear mapping.
    alpha : float
        Value of the correct trace.

    """
    #symmetrize if necessary
    L = constraint(dm) + constraint(dm).conj().T
    #find eigendecomposition
    vals, vecs = np.linalg.eig(L)
    #calculate the shift, sigma0
    sigma0 = calculate_shift(vals)
    
    #calculate the closest semidefinite positive matrix with correct trace
    L_closest = vals @ np.diag(vecs - sigma0) @ vecs.conj().T

    # return the reconstructed density matrix
    return constraint(L_closest).conj().T


def calculate_shift(eigenvalues, alpha):
    r"""
    Calculate the shift for density matrix for it to have the correct trace. 
    This step shifts the spectrum and eliminates the negative eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of not shifted matrix.
    alpha : float
        Value of the coprrect trace.
        
    """
    pass

