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

r"""PyCI RDM Constraints module."""

import numpy as np

from scipy.optimize import root  


__all__ = [
    "find_closest_sdp",
    "calc_P",
    "calc_Q",
    "calc_G",
    "calc_T1",
    "calc_T2",
    "calc_T2_prime",
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
    constrained = constraint(dm)
    L = constrained + constrained.conj().T
    #find eigendecomposition
    vals, vecs = np.linalg.eigh(L)
    #calculate the shift, sigma0
    sigma0 = calculate_shift(vals, alpha)
    
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
    #sample code, to be confirmed
    trace = lambda sigma0: np.sum(np.heaviside(eigenvalues - sigma0, 0.5)*(eigenvalues - sigma0))  
    constraint = lambda x: trace(x) - alpha  
    res = root(constraint, 0) 
    return res.x

def calc_P():
    pass

def calc_Q():
    pass

def calc_G():
    pass

def calc_T1():
    pass

def calc_T2():
    pass

def calc_T2_prime():
    pass

