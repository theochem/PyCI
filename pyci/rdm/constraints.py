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
from pyci.rdm.tools import flat_tensor


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

def calc_P(gamma, N, conjugate=False):
    """
    Calculating P tensor

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
    
    """
    return gamma

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

def calc_T1(gamma, N, conjugate):
    """
    Calculating T1 tensor

    Parameters
    ----------
    gamma: np.ndarray
        1DM or 2DM tensor
    N: int
        number of electrons in the system
    conjugate: bool
        conjugate or regular condition

    Returns
    -------
    np.ndarray

    Notes
    -----
    T1 is defined as:

    .. math::
        \begin{aligned}
            \mathcal{T}_1^{\dagger}(\Gamma)_{\alpha \beta ; \gamma \delta}=&\frac{2N}{N(N-1)}(\delta_{\alpha \gamma}\delta_{\beta \delta} - \delta_{\alpha \delta}\delta_{\beta \gamma}) \rm{Tr} A +\bar{A}_{\alpha\beta;\gamma\delta} \\
            &-\frac{1}{2(N-1)} [\delta_{\beta \delta} \bar{\bar{A}}_{\alpha \gamma} - \delta_{\alpha \delta}\bar{\bar{A}}_{\beta \gamma} - \delta_{\beta \gamma}\bar{\bar{A}}_{\alpha \delta} + \delta_{\alpha \gamma}\bar{\bar{A}}_{\beta \delta}]
        \end{aligned}

        \begin{aligned}
            \mathcal{T}_1^(\Gamma)_{\alpha \beta \gamma; \delta \epsilon \zeta}=&   \delta_{\gamma\zeta}\delta_{\beta\epsilon}\delta_{\alpha\delta} - \delta_{\gamma\epsilon}\delta_{\alpha\delta}\delta_{\beta\zeta} 
                                                                                  + \delta_{\alpha\zeta}\delta_{\gamma\epsilon}\delta_{\beta\delta} + \delta_{\gamma\zeta}\delta_{\alpha\epsilon}\delta_{\beta\delta}
                                                                                  + \delta_{\beta\zeta}\delta_{\alpha\epsilon}\delta_{\gamma\delta} - \delta_{\alpha\zeta}\delta_{\beta\epsilon}\delta_{\gamma\delta} \\
                                                                                & -(\delta_{\gamma\zeta}\delta_{\beta\epsilon} - \delta_{\beta\zeta}\delta_{gamma\epsilon})\rho_{\alpha\delta}
                                                                                  +(\delta_{\gamma]zeta}\delta_{\aplha\epsilon} - \delta_{\alpha\zeta}\delta_{\gamma\epsilon})\rho_{\beta\delta}
                                                                                  -(\delta_{\beta\zeta}\delta_{\alpha\epsilon} - \delta_{\alpha\zeta}\delta_{\beta\epsilon})\rho_{\gamma\delta} \\
                                                                                & +(\delta_{\gamma\zeta}\delta_{\beta\delta} - \delta_{\beta\zeta}\delta_{\gamma\delta})\rho_{\alpha\epsilon}
                                                                                  -(\delta_{\gamma\zeta}\delta_{\alpha\delta} - \delta_{\alpha\zeta}\delta_{\gamma\delta})\rho_{\epsilon\beta}
                                                                                  +(\delta_{\beta\zeta}\delta_{\alpha\delta} - \delta_{alpha\zeta}\delta_{\beta\delta})\rho_{gamma\epsilon} \\
                                                                                & -(\delta_{\beta\delta}\delta_{\gamma\epsilon} - \delta_{\beta\epsilon}\delta_{\gamma\delta})\rho_{alpha\zeta} 
                                                                                  +(\delta_{\gamma\epsilon}\delta_{\alpha\delta} - \delta_{\alpha\epsilon}\delta_{\gamma\delta})\rho_{\beta\zeta}
                                                                                  -(\delta_{\beta\epsilon}\delta_{\alpha\delta} - \delta_{\alpha\epsilon}\delta_{\beta\delta})\rho_{\gamma\zeta} \\
                                                                                & + \delta_{\gamma\zeta}\Gamma_{\alpha\beta;\delta\epsilon} - \delta_{\beta\zeta}\Gamma_{\alpha\gamma;\delta\epsilon}
                                                                                  + \delta_{\alpha\zeta}\Gamma_{\beta\gamma;\delta\epsilon} - \delta_{\gamma\epsilon}\Gamma_{\alpha\beta;\delta\zeta}
                                                                                  + \delta_{\beta\epsilon}\Gamma_{\alpha\gemma;\delta\zeta} - \delta_{\alpha\epsilon}\Gamma_{\beta\gamma;\delta]zeta} \\
                                                                                & + \delta_{\gamma\delta}\Gamma_{\alpha\beta;\epsilon\zeta} - \delta_{\beta\delta}\Gamma_{\alpha\gamma;\epsilon\zeta}
                                                                                  + \delta_{\alpha\delta}\Gamma_{\beta\gamma;\epsilon\zeta} .                                                        
        \end{aligned}
        

    """
    eye = np.eye(gamma.shape[0])

    if not conjugate:
        rho = 1 / (N-1) * np.einsum('abgb -> ag', gamma)
        term_1 = np.einsum('gz, be, ad -> abgdez', eye, eye, eye) + \
                 np.einsum('ge, ad, bz -> abgdez', eye, eye, eye) + \
                 np.einsum('az, ge, bd -> abgdez', eye, eye, eye) + \
                 np.einsum('gz, ae, bd -> abgdez', eye, eye, eye) + \
                 np.einsum('az, be, gd -> abgdez', eye, eye, eye)
        term_2 = - np.einsum('gz, be, ad -> abgdez', eye, eye, rho) + \
                   np.einsum('bz, ge, ad -> abgdez', eye, eye, rho) + \
                   np.einsum('gz, ae, bd -> abgdez', eye, eye, rho) - \
                   np.einsum('az, ge, bd -> abgdez', eye, eye, rho) - \
                   np.einsum('bz, ae, gd -> abgdez', eye, eye, rho) + \
                   np.einsum('az, be, gd -> abgdez', eye, eye, rho)
        term_3 = np.einsum('gz, bd, ae -> abgdez', eye, eye, rho) - \
                 np.einsum('bz, gd, ae -> abgdez', eye, eye, rho) - \
                 np.einsum('gz, ad, eb -> abgdez', eye, eye, rho) + \
                 np.einsum('az, gd, eb -> abgdez', eye, eye, rho) + \
                 np.einsum('bz, ad, ge -> abgdez', eye, eye, rho) - \
                 np.einsum('az, bd, ge -> abgdez', eye, eye, rho)
        term_4 = - np.einsum('bd, ge, az -> abgdez', eye, eye, rho) + \
                   np.einsum('be, gd, az -> abgdez', eye, eye, rho) + \
                   np.einsum('ge, ad, bz -> abgdez', eye, eye, rho) - \
                   np.einsum('ae, gd, bz -> abgdez', eye, eye, rho) - \
                   np.einsum('be, ad, gz -> abgdez', eye, eye, rho) + \
                   np.einsum('ae, bd, gz -> abgdez', eye, eye, rho)
        term_5 = np.einsum('gz, abde -> abgdez', eye, gamma) - np.einsum('bz, agde -> abgdez', eye, gamma) + \
                 np.einsum('az, bgde -> abgdez', eye, gamma) - np.einsum('ge, abdz -> abgdez', eye, gamma) + \
                 np.einsum('be, agdz -> abgdez', eye, gamma) - np.einsum('ae, bgdz -> abgdez', eye, gamma) + \
                 np.einsum('gd, abez -> abgdez', eye, gamma) - np.einsum('bd, agez -> abgdez', eye, gamma) + \
                 np.einsum('ad, bgez -> abgdez', eye, gamma)
        return term_1 + term_2 + term_3 + term_4 + term_5

    else:
        tr_gamma = np.einsum('aaaaaa', gamma)
        gamma_abgd = np.einsum('ablgdl -> abgd', gamma)
        term_1 = 2 / (N*N - N) *\
                (np.einsum('ag, bd -> abgd', eye, eye) - np.einsum('ad, bg -> abgd', eye, eye)) * tr_gamma + gamma_abgd        
        
        gamma_ag = np.einsum('abgb -> ag', gamma_abgd)
        gamma_bg = np.einsum('abag -> bg', gamma_abgd)
        gamma_ad = np.einsum('agdg -> ad', gamma_abgd)
        gamma_bd = np.einsum('abda -> bd', gamma_abgd)
        
        term_2 = - 2 / (2*N - 2)*\
                (np.einsum('bd, ag -> abgd', eye, gamma_ag) - np.einsum('ad, bg -> abgd', eye, gamma_bg) -\
                 np.einsum('bg, ad -> abgd', eye, gamma_ad) + np.einsum('ag, bd -> abgd', eye, gamma_bd))

        return term_1 + term_2


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


def calc_T2_prime(gamma, N, conjugate=False):
    """
    Calculating T2' tensor

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
    T2' is defined as:

    .. math::
        \begin{aligned}
        \mathcal{T}'_{2}(\Gamma)
        = \left( \begin{matrix}
            \mathcal{T}_{2}(\Gamma)_{\alpha \beta \gamma; \delta \epsilon \zeta} & (\Gamma_\omega)_{\alpha \beta \gamma; \nu} \\
            (\Gamma_\omega)_{\mu; \delta \epsilon \zeta} & (\Gamma_\rho)_{\mu \nu}
        \end{matrix} \right)
        \end{aligned}

        \begin{aligned}
        \mathcal{T}'^{\dagger}_{2}(\Gamma)_{\alpha \beta; \gamma \delta} 
        = &
            \mathcal{T}^{\dagger}_{2}(A_{\mathcal{T}}) 
            + (\Gamma_{\omega})_{\alpha \beta \delta; \gamma} 
            + (\Gamma_{\omega})_{\gamma \delta \beta; \alpha} 
        \\
        &
            - (\Gamma_{\omega})_{\alpha \beta \gamma; \delta} 
            - (\Gamma_{\omega})_{\gamma \delta \alpha; \beta} 
        \\
        &
            + \frac{1}{N-1} 
            \left( 
                \delta_{\beta \delta} (\Gamma_{\rho})_{\gamma \alpha} 
                - \delta_{\alpha \delta} (\Gamma_{\rho})_{\gamma \beta} 
                - \delta_{\beta \gamma} (\Gamma_{\rho})_{\delta \alpha} 
                + \delta_{\alpha \gamma} (\Gamma_{\rho})_{\delta \beta} 
            \right)
    \end{aligned}
    """
    omega = np.einsum('abgd -> abdg', gamma)
    rho = 1/(N-1) * np.einsum('abgb -> ag', gamma)

    if not conjugate:
        t2 = calc_T2(gamma, N, False)
        n = t2.shape[0]
        return np.block([  
            [flat_tensor(t2, (n**3, n**3)), flat_tensor(omega, (n**3, n))],  
            [flat_tensor(omega, (n, n**3)), rho]  
        ])
    else:
        t2 = calc_T2(gamma, N, False)
        t2_d = calc_T2(t2, N, True)
        eye = np.eye(N)

        term1 = t2_d
        term2 = np.einsum('abdg -> abgd', omega)
        term3 = np.einsum('gdba -> abgd', omega)
        term4 = omega
        term5 = np.einsum('gdab -> abgd', omega)
        term6 = np.einsum('bd, ga -> abgd', eye, rho)
        term7 = np.einsum('ad, gb -> abgd', eye, rho)
        term8 = np.einsum('bg, da -> abgd', eye, rho)
        term9 = np.einsum('ag, db -> abgd', eye, rho)

        return term1 + term2 + term3 - term4 - term5 + 1 / (N - 1) * (term6 - term7 - term8 + term9)
