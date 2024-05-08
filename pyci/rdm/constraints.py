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

def calc_G(gamma, N, conjugate=False):
    """
    Calculating G tensor

    Parameters
    ----------
    gamma: np.ndarray
        1DM tensor
    N: int
        number of electrons in the system
    conjugate: bool
        conjugate or regular condition

    Returns
    -------
    np.ndarray

    Notes
    -----
    G is defined as:

    .. math::
        \mathcal{G}_1(\Gamma)_{\alpha \beta ; \gamma \delta}=\delta_{\beta \delta} \rho_{\alpha \gamma}-\Gamma_{\alpha \delta ; \gamma \beta}
        \mathcal{G}^{\prime}(\Gamma)_{\alpha \beta ; \gamma \delta}=\delta_{\beta \delta} \rho_{\alpha \gamma}-\Gamma_{\alpha \delta ; \gamma \beta}-\rho_{\alpha \beta} \rho_{\gamma \delta}
    """
    eye = np.eye(gamma.shape[0])
    a_bar = np.einsum('abgb -> ag', gamma)
    rho = 1/(N - 1) * a_bar
    if not conjugate:
        return np.einsum('bd, ag -> abgd', eye, rho) - np.einsum('adgb -> abgd', gamma)
    term_1 = 1/(N-1) *\
          (np.einsum('bd, ag -> abgd', eye, a_bar) - np.einsum('ad, bg -> abgd', eye, a_bar) -\
           np.einsum('bg, ad -> abgd', eye, a_bar) + np.einsum('ag, bd -> abgd', eye, a_bar)
    )
    term_2 = -np.einsum('adgb -> abgd', gamma) + np.einsum('bdga -> abgd', gamma) +\
              np.einsum('agdb -> abgd', gamma) - np.einsum('bgda -> abgd', gamma)
    return term_1 + term_2

def calc_T1():
     """
    Calculating T1 tensor

    Parameters
    ----------
    gamma: np.ndarray
        1DM tensor
    N: int
        number of electrons in the system
    conjugate: bool
        conjugate or regular condition

    Returns
    -------
    np.ndarray

    Notes
    -----
    G is defined as:

    .. math::
        \mathcal{G}_1(\Gamma)_{\alpha \beta ; \gamma \delta}=\delta_{\beta \delta} \rho_{\alpha \gamma}-\Gamma_{\alpha \delta ; \gamma \beta}
        \mathcal{G}^{\prime}(\Gamma)_{\alpha \beta ; \gamma \delta}=\delta_{\beta \delta} \rho_{\alpha \gamma}-\Gamma_{\alpha \delta ; \gamma \beta}-\rho_{\alpha \beta} \rho_{\gamma \delta}
    """

def calc_T2(gamma, N, conjugate=False):
    """
    Calculating T2 tensor

    Parameters
    ----------
    gamma: np.ndarray
        1DM tensor
    N: int
        number of electrons in the system
    conjugate: bool
        conjugate or regular condition

    Returns
    -------
    np.ndarray

    Notes
    -----
    T2 is defined as:

    .. math::
        \begin{aligned}
        \mathcal{T}_2(\Gamma)_{\alpha \beta \gamma ; \delta \epsilon \zeta}= & \left(\delta_{\alpha \delta} \delta_{\beta \epsilon}-\delta_{\alpha \epsilon} \delta_{\beta \delta}\right) \rho_{\gamma \zeta}+\delta_{\gamma \zeta} \Gamma_{\alpha \beta ; \delta \epsilon} \\
        & -\delta_{\alpha \delta} \Gamma_{\gamma \epsilon ; \zeta \beta}+\delta_{\beta \delta} \Gamma_{\gamma \epsilon ; \zeta \alpha}+\delta_{\alpha \epsilon} \Gamma_{\gamma \delta ; \zeta \beta}-\delta_{\beta \epsilon} \Gamma_{\gamma \delta ; \zeta \alpha}
        \end{aligned}

        \begin{aligned}
        \mathcal{T}_2^{\dagger}(A)_{\alpha \beta ; \gamma \delta}= & \frac{1}{2(N-1)}\left[\delta_{\beta \delta} \tilde{\tilde{A}}_{\alpha \gamma}-\delta_{\alpha \delta} \tilde{\tilde{A}}_{\beta \gamma}-\delta_{\beta \gamma} \tilde{\tilde{A}}_{\alpha \delta}+\delta_{\alpha \gamma} \tilde{\tilde{A}}_{\beta \delta}\right] \\
        & +\bar{A}_{\alpha \beta ; \gamma \delta}-\left[\tilde{A}_{\delta \alpha ; \beta \gamma}-\tilde{A}_{\delta \beta ; \alpha \gamma}-\tilde{A}_{\gamma \alpha ; \beta \delta}+\tilde{A}_{\gamma \beta ; \alpha \delta}\right]
\       end{aligned}
    """
    eye = np.eye(gamma.shape[0])
    if not conjugate:
        rho = 1/(N-1) * np.einsum('abgb -> ag', gamma)
        term_1 = np.einsum('ad, be, gz -> abgdez', eye, eye, rho) -\
                 np.einsum('ae, bd, gz -> abgdez', eye, eye, rho)
        term_2 = np.einsum('gz, abde -> abgdez', eye, gamma)
        term_3 = np.einsum('ad, gezb -> abgdez', eye, gamma)
        term_4 = np.einsum('bd, geza -> abgdez', eye, gamma)
        term_5 = np.einsum('ae, gdzb -> abgdez', eye, gamma)
        term_6 = np.einsum('be, gdza -> abgdez', eye, gamma)
        return term_1 + term_2 - term_3 + term_4 + term_5 - term_6
    a_dtilda = np.einsum('lkalkg -> ag', gamma)
    a_tilda = np.einsum('lablgd -> abgd', gamma)
    a_bar = np.einsum('ablgdl -> abgd', gamma)

    term_1 = np.einsum('bd, ag -> abgd', eye, a_dtilda)
    term_2 = np.einsum('ad, bg -> abgd', eye, a_dtilda)
    term_3 = np.einsum('bg, ad -> abgd', eye, a_dtilda)
    term_4 = np.einsum('ag, bd -> abgd', eye, a_dtilda)    
    # term_5 = a_bar
    term_6 = np.einsum('dabg -> abgd', a_tilda)
    term_7 = np.einsum('dbag -> abgd', a_tilda)
    term_8 = np.einsum('gabd -> abgd', a_tilda)
    term_9 = np.einsum('gbad -> abgd', a_tilda)
    return 0.5/(N-1) * (term_1 - term_2 - term_3 + term_4) +\
        a_bar - (term_6 - term_7 - term_8 + term_9)
    eye = np.eye(gamma.shape[0])
    rho = 1/(N-1) * np.einsum('abgb -> ag', gamma)
    if not conjugate:
        term_1 = np.einsum('ad, be, gz -> abgdez', eye, eye, rho) -\
                 np.einsum('ae, bd, gz -> abgdez', eye, eye, rho)
        term_2 = np.einsum('gz, abde -> abgdez', eye, gamma)
        term_3 = np.einsum('ad, gezb -> abgdez', eye, gamma)
        term_4 = np.einsum('bd, geza -> abgdez', eye, gamma)
        term_5 = np.einsum('ae, gdzb -> abgdez', eye, gamma)
        term_6 = np.einsum('be, gdza -> abgdez', eye, gamma)
        return term_1 + term_2 - term_3 + term_4 + term_5 - term_6
    a_dtilda = np.einsum('lkalkg -> ag', gamma)
    a_tilda = np.einsum('lablgd -> abgd', gamma)
    a_bar = np.einsum('ablgdl -> abgd', gamma)

    term_1 = np.einsum('bd, ag -> abgd', eye, a_dtilda)
    term_2 = np.einsum('ad, bg -> abgd', eye, a_dtilda)
    term_3 = np.einsum('bg, ad -> abgd', eye, a_dtilda)
    term_4 = np.einsum('ag, bd -> abgd', eye, a_dtilda)    
    # term_5 = a_bar
    term_6 = np.einsum('dabg -> abgd', a_tilda)
    term_7 = np.einsum('dbag -> abgd', a_tilda)
    term_8 = np.einsum('gabd -> abgd', a_tilda)
    term_9 = np.einsum('gbad -> abgd', a_tilda)
    return 0.5/(N-1) * (term_1 - term_2 - term_3 + term_4) +\
        a_bar - (term_6 - term_7 - term_8 + term_9)


def calc_T2_prime():
    pass

