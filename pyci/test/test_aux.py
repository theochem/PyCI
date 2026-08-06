import numpy as np
import numpy.testing as npt
from pyscf import fci
from iodata import load_one

from scipy.special import comb

import pyci
from pyci.test import datafile


def extract_seniority_zero_sector_sf(h1,h2):
        """
        Extract the seniority-zero sector of the one and two body electron integrals.

        Parameters
        ----------
        one_ints : list[np.ndarray]
            One-body term in spatial basis in physics notation
        two_ints : [np.ndarray]
            Two-body term in spatial basis in physics notation

        Returns
        -------
        S1aa,S1bb,S2aaaa,S2bbbb,S2abab : tuple(np.ndarray)
            one and two-body electron integrals for the seniority zero sector in sptial basis
            and physics notation
        """

        n=h1.shape[0]
        one_ints_diag=np.diag(h1)
        S1=np.diag(one_ints_diag)
        S2=np.zeros(h2.shape)
        for p in range(n):
            S2[p][p][p][p]=h2[p][p][p][p]
            for q in range(n):
                if p!=q:
                    S2[p][q][q][p]=h2[p][q][q][p] 
                    S2[p][q][p][q]=h2[p][q][p][q]
                    S2[p][p][q][q]=h2[p][p][q][q]
                else:
                    continue
        return S1,S2

ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format("BH3")))
occs = (4, 4)
wfn = pyci.doci_wfn(ham.nbasis, *occs)
wfn.add_all_dets()
op = pyci.sparse_op(ham, wfn)
es, cs = op.solve(n=1, tol=1.0e-6)
d0, d2, d3, d4, d5, d6, d7 = pyci.compute_rdms_1234(wfn, cs[0])
rdm1,rdm2,rdm3, _ = pyci.spin_free_rdms(d0, d2, d3, d4, d5, d6, d7, flag= '3RDM')
rdm1_new, rdm2_new, rdm3_new = pyci.spin_free_rdms_123_new_version(d0, d2, d3, d4, flag = '3RDM')
data = load_one("/workspaces/Sen-0ic-MRCC/pyci_repo/pyci/test/data/BH3.fcidump", fmt="fcidump")
h1 = data.one_ints["core_mo"]
h2 = data.two_ints["two_mo"]
nelec = occs
norb = h1.shape[0]
S1, S2 = extract_seniority_zero_sector_sf(h1,h2)
e_seniority_zero, ci_seniority_zero = fci.direct_spin0.kernel(S1, np.einsum('ijkl->ikjl',S2), norb, nelec, nroots=1,
                                            max_space=12, max_cycle=50)

rdm1_spin0,rdm2_spin0,rdm3_spin0,rdm4_spin0=fci.rdm.make_dm1234('FCI4pdm_kern_spin0',ci_seniority_zero,ci_seniority_zero,norb,nelec)
rdm1_spin0, rdm2_spin0, rdm3_spin0,rdm4_spin0 = fci.rdm.reorder_dm1234(rdm1_spin0, rdm2_spin0, rdm3_spin0, rdm4_spin0, inplace=True)
rdm2_spin0=np.einsum('pqrs-> prqs',rdm2_spin0)
rdm3_spin0=np.einsum('pqrstu-> prtqsu',rdm3_spin0)
rdm4_spin0=np.einsum('pqrstuvw-> prtvqsuw',rdm4_spin0)
print(np.allclose(rdm1, rdm1_new))
print(np.allclose(rdm2, rdm2_new))
print(np.allclose(rdm3, rdm3_new))
