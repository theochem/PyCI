import numpy as np
import numpy.testing as npt

from scipy.special import comb

import pyci
from pyci.test import datafile


ham = pyci.secondquant_op(datafile("{0:s}.fcidump".format("h6_sto_3g")))
occs = (3, 3)
wfn = pyci.doci_wfn(ham.nbasis, *occs)
wfn.add_all_dets()
op = pyci.sparse_op(ham, wfn)
es, cs = op.solve(n=1, tol=1.0e-6)
d0, d2, d3, d4, d5, d6, d7 = pyci.compute_rdms_1234(wfn, cs[0])
rdm1,rdm2,rdm3, _ = pyci.spin_free_rdms(d0, d2, d3, d4, d5, d6, d7, flag= '3RDM')
rdm1_new, rdm2_new, rdm3_new = pyci.spin_free_rdms_123_new_version(d0, d2, d3, d4)
print(np.allclose(rdm1, rdm1_new))
print(np.allclose(rdm2, rdm2_new))
print(np.allclose(rdm3, rdm3_new))
print(np.linalg.norm(rdm3_new-rdm3))
print(rdm3_new[0][1][2][0][1][2]-rdm3[0][1][2][0][1][2])
