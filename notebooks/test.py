import numpy as np
import pyci
import numpy as np
import iodata
from iodata import load_one
import pyscf
from pyscf import scf, fci
import time

data = load_one("/workspaces/PyCI/pyci/test/data/h6_sto_3g.fcidump", fmt="fcidump")
h2=data.two_ints['two_mo']
h1=data.one_ints['core_mo']

n=h1.shape[0]
h1_diag=np.diag(h1)
s1=np.diag(h1_diag)
s2=np.zeros(h2.shape)
for p in range(n):
    s2[p][p][p][p]=h2[p][p][p][p]
    for q in range(n):
        if p!=q:
          s2[p][q][q][p]=h2[p][q][q][p] 
          s2[p][q][p][q]=h2[p][q][p][q]
          s2[p][p][q][q]=h2[p][p][q][q]
        else:
          continue

occs = (3,3)    
nbasis = 6

ecore = 0.0                   
ham = pyci.hamiltonian(ecore, s1, s2)
wfn3 = pyci.doci_wfn(ham.nbasis, *occs)
wfn3.add_all_dets()
op = pyci.sparse_op(ham, wfn3)
e_vals, e_vecs3 = op.solve(n=1, tol=1.0e-9)
e_vals,wfn3
d1, d2, d3, d4, d5, d6, d7 = pyci.compute_rdms_1234(wfn3, e_vecs3[0])


rdm1_pyci,rdm2_pyci,rdm3_pyci,rdm4_pyci=pyci.spinize_rdms_1234(d1, d2, d3, d4, d5, d6, d7)
print(np.einsum('ijklijkl ->',rdm4_pyci))
abbbabbb=rdm4_pyci[:n,n:,n:,n:,:n,n:,n:,n:]
baaabaaa=rdm4_pyci[n:,:n,:n,:n,n:,:n,:n,:n]
aaabaaab=rdm4_pyci[:n,:n,:n,n:,:n,:n,:n,n:]
bbbabbba=rdm4_pyci[n:,n:,n:,:n,n:,n:,n:,:n]
abababab=rdm4_pyci[:n,n:,:n,n:,:n,n:,:n,n:]
babababa=rdm4_pyci[n:,:n,n:,:n,n:,:n,n:,:n]
aaaaaaaa = rdm4_pyci[:n,:n,:n,:n,:n,:n,:n,:n]
bbbbbbbb = rdm4_pyci[n:,n:,n:,n:,n:,n:,n:,n:]
print(np.isclose(np.einsum('ijklijkl -> ', rdm4_pyci),(occs[0] * 2)*(occs[0] * 2 - 1) * (occs[0] * 2 - 2) * (occs[0] * 2 - 3) ,rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', abbbabbb),(occs[0]) * (occs[1]) * (occs[1] - 1) * (occs[1] - 2) ,rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', baaabaaa),(occs[1]) * (occs[0]) * (occs[0] - 1) * (occs[0] - 2), rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', aaabaaab),(occs[1]) * (occs[0]) * (occs[0] - 1) * (occs[0] - 2) , rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', bbbabbba),(occs[0]) * (occs[1]) * (occs[1] - 1) * (occs[1] - 2) , rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', abababab),(occs[0]) * (occs[1]) * (occs[0] - 1) * (occs[1] - 1) , rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', babababa),(occs[0]) * (occs[1]) * (occs[0] - 1) * (occs[1] - 1) , rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', aaaaaaaa),(occs[0])*(occs[0] - 1) * (occs[0] - 2) * (occs[0] - 3) , rtol=0, atol=1.0e-2))
print(np.isclose(np.einsum('ijklijkl -> ', bbbbbbbb),(occs[1])*(occs[1]- 1) * (occs[1] - 2) * (occs[1] - 3), rtol=0, atol=1.0e-2))



#Defining d3 blocks
d3_aabaab_block=rdm3_pyci[:n, :n, n:, :n, :n, n:]
d3_bbabba_block=rdm3_pyci[n:, n:, :n, n:, n:, :n]
d3_abbabb_block=rdm3_pyci[:n, n:, n:, :n, n:, n:]
d3_baabaa_block=rdm3_pyci[n:, :n, :n, n:, :n, :n]
d3_aaaaaa_block=rdm3_pyci[:n, :n, :n, :n, :n, :n]
d3_bbbbbb_block=rdm3_pyci[n:, n:, n:, n:, n:, n:]

#Defining d2 blocks

d2_abab_block=rdm2_pyci[:n, n:, :n, n:]
d2_baba_block=rdm2_pyci[n:, :n, n:, :n]
d2_aaaa_block=rdm2_pyci[:n, :n, :n, :n]
d2_bbbb_block=rdm2_pyci[n:, n:, n:, n:]

# # Testing aaabaaab/bbbabbba blocks
#With the 3rdms
fac=1/(occs[0]-2)
print(np.allclose(np.einsum('pijkplmn ->ijklmn',aaabaaab)*fac,d3_aabaab_block))
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',aaabaaab)*fac,d3_aabaab_block))
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',aaabaaab)*fac,d3_aabaab_block))
fac=1/(occs[1])
print(np.allclose(np.einsum('ijkplmnp ->ijklmn',aaabaaab)*fac,d3_aaaaaa_block))
fac=1/(occs[1]-2)
print(np.allclose(np.einsum('pijkplmn ->ijklmn',bbbabbba)*fac,d3_bbabba_block))
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',bbbabbba)*fac,d3_bbabba_block))
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',bbbabbba)*fac,d3_bbabba_block))
fac=1/(occs[1])
print(np.allclose(np.einsum('ijkplmnp ->ijklmn',bbbabbba)*fac,d3_bbbbbb_block))

#With the 2rdms
fac=1/((occs[0]-2)*(occs[0]-1))
print(np.allclose(np.einsum('pqijpqkl ->ijkl',aaabaaab)*fac,d2_abab_block))
print(np.allclose(np.einsum('piqjpkql ->ijkl',aaabaaab)*fac,d2_abab_block))
print(np.allclose(np.einsum('ipqjkpql ->ijkl',aaabaaab)*fac,d2_abab_block))
fac=1/((occs[1]-2)*(occs[1]-1))
print(np.allclose(np.einsum('pqijpqkl ->ijkl',bbbabbba)*fac,d2_baba_block))
print(np.allclose(np.einsum('piqjpkql ->ijkl',bbbabbba)*fac,d2_baba_block))
print(np.allclose(np.einsum('ipqjkpql ->ijkl',bbbabbba)*fac,d2_baba_block))
fac=1/((occs[1])*(occs[0]-2))
print(np.allclose(np.einsum('ijpqklpq ->ijkl',aaabaaab)*fac,d2_aaaa_block))
fac=1/((occs[0])*(occs[1]-2))
print(np.allclose(np.einsum('ijpqklpq ->ijkl',bbbabbba)*fac,d2_bbbb_block))

# # Testing abababab/babababa blocks

#With the 3rdms
fac=1/(occs[1]-1)
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',abababab)*fac,d3_aabaab_block))
fac=1/(occs[0]-1)
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',abababab)*fac,d3_abbabb_block))
fac=1/(occs[0]-1)
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',babababa)*fac,d3_bbabba_block))
fac=1/(occs[1]-1)
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',babababa)*fac,d3_baabaa_block))

#With the 2rdms

fac=1/((occs[0]-1) * (occs[1]-1))
print(np.allclose(np.einsum('pqijpqkl ->ijkl',abababab)*fac, d2_abab_block))
print(np.allclose(np.einsum('ipqjkpql ->ijkl',abababab)*fac, d2_abab_block))
print(np.allclose(np.einsum('ijpqklpq ->ijkl',abababab)*fac, d2_abab_block))
print(np.allclose(np.einsum('pqijpqkl ->ijkl',babababa)*fac, d2_baba_block))
print(np.allclose(np.einsum('ipqjkpql ->ijkl',babababa)*fac, d2_baba_block))
print(np.allclose(np.einsum('ijpqklpq ->ijkl',babababa)*fac, d2_baba_block))
fac=1/((occs[0]) * (occs[0]-1))
print(np.allclose(np.einsum('piqjpkql ->ijkl',abababab)*fac,d2_bbbb_block))
fac=1/((occs[0]-1) * (occs[1]-1))
print(np.allclose(np.einsum('pijqpklq ->ijkl',abababab)*fac,d2_baba_block))
fac=1/((occs[1]) * (occs[1]-1))
print(np.allclose(np.einsum('ipjqkplq ->ijkl',abababab)*fac,d2_aaaa_block))

fac=1/((occs[1]) * (occs[1]-1))
print(np.allclose(np.einsum('piqjpkql ->ijkl',babababa)*fac,d2_aaaa_block))
fac=1/((occs[1]-1) * (occs[0]-1))
print(np.allclose(np.einsum('pijqpklq ->ijkl',babababa)*fac,d2_abab_block))
fac=1/((occs[0]) * (occs[0]-1))
print(np.allclose(np.einsum('ipjqkplq ->ijkl',babababa)*fac,d2_bbbb_block))

# # Testing abbbabbb/baaabaaa blocks

#With the 3rdms
fac=1 / (occs[1]-2)
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',abbbabbb)*fac, d3_abbabb_block))
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',abbbabbb)*fac, d3_abbabb_block))
print(np.allclose(np.einsum('ijkplmnp ->ijklmn',abbbabbb)*fac, d3_abbabb_block))
fac=1 / (occs[0])
print(np.allclose(np.einsum('pijkplmn ->ijklmn',abbbabbb)*fac, d3_bbbbbb_block))

fac=1 / (occs[0]-2)
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',baaabaaa)*fac, d3_baabaa_block))
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',baaabaaa)*fac, d3_baabaa_block))
print(np.allclose(np.einsum('ijkplmnp ->ijklmn',baaabaaa)*fac, d3_baabaa_block))
fac=1 / (occs[1])
print(np.allclose(np.einsum('pijkplmn ->ijklmn',baaabaaa)*fac, d3_aaaaaa_block))


#With the 2rdms

fac=1 / ((occs[1] - 2) * (occs[0]))
print(np.allclose(np.einsum('pqijpqkl ->ijkl',abbbabbb)*fac, d2_bbbb_block))
print(np.allclose(np.einsum('piqjpkql ->ijkl',abbbabbb)*fac, d2_bbbb_block))
print(np.allclose(np.einsum('pijqpklq ->ijkl',abbbabbb)*fac, d2_bbbb_block))

fac=1 / ((occs[1] - 2) * (occs[1] - 1))
print(np.allclose(np.einsum('ipqjkpql ->ijkl',abbbabbb)*fac, d2_abab_block))
print(np.allclose(np.einsum('ipjqkplq ->ijkl',abbbabbb)*fac, d2_abab_block))
print(np.allclose(np.einsum('ijpqklpq ->ijkl',abbbabbb)*fac, d2_abab_block))

fac=1 / ((occs[0] - 2) * (occs[1]))
print(np.allclose(np.einsum('pqijpqkl ->ijkl',baaabaaa)*fac, d2_aaaa_block))
print(np.allclose(np.einsum('piqjpkql ->ijkl',baaabaaa)*fac, d2_aaaa_block))
print(np.allclose(np.einsum('pijqpklq ->ijkl',baaabaaa)*fac, d2_aaaa_block))

fac=1 / ((occs[0] - 2) * (occs[0] - 1))
print(np.allclose(np.einsum('ipqjkpql ->ijkl',baaabaaa)*fac, d2_baba_block))
print(np.allclose(np.einsum('ipjqkplq ->ijkl',baaabaaa)*fac, d2_baba_block))
print(np.allclose(np.einsum('ijpqklpq ->ijkl',baaabaaa)*fac, d2_baba_block))

'''
# # Testing All-alpha/beta blocks
#With the 3rdms
fac=1/(occs[0]-3)
print(np.allclose(np.einsum('pijkplmn ->ijklmn',aaaaaaaa)*fac,d3_aaaaaa_block))
print(np.allclose(np.einsum('pijkplmn ->ijklmn',aaaaaaaa)*fac,d3_aaaaaa_block))
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',aaaaaaaa)*fac,d3_aaaaaa_block))
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',aaaaaaaa)*fac,d3_aaaaaa_block))
print(np.allclose(np.einsum('ijkplmnp ->ijklmn',aaaaaaaa)*fac,d3_aaaaaa_block))
fac=1/(occs[1]-3)
print(np.allclose(np.einsum('pijkplmn ->ijklmn',bbbbbbbb)*fac,d3_bbbbbb_block))
print(np.allclose(np.einsum('pijkplmn ->ijklmn',bbbbbbbb)*fac,d3_bbbbbb_block))
print(np.allclose(np.einsum('ipjklpmn ->ijklmn',bbbbbbbb)*fac,d3_bbbbbb_block))
print(np.allclose(np.einsum('ijpklmpn ->ijklmn',bbbbbbbb)*fac,d3_bbbbbb_block))
print(np.allclose(np.einsum('ijkplmnp ->ijklmn',bbbbbbbb)*fac,d3_bbbbbb_block))

#With the 2rdms
fac=1/((occs[0]-3)*(occs[0]-2))
print(np.allclose(np.einsum('pqijpqkl ->ijklmn',aaaaaaaa)*fac,d2_aaaa_block))
print(np.allclose(np.einsum('piqjpkql ->ijklmn',aaaaaaaa)*fac,d2_aaaa_block))
print(np.allclose(np.einsum('ipqjkpql ->ijklmn',aaaaaaaa)*fac,d2_aaaa_block))
print(np.allclose(np.einsum('ipjqkplq ->ijklmn',aaaaaaaa)*fac,d2_aaaa_block))
print(np.allclose(np.einsum('pijqpklq ->ijklmn',aaaaaaaa)*fac,d2_aaaa_block))
print(np.allclose(np.einsum('ijpqklpq ->ijklmn',aaaaaaaa)*fac,d2_aaaa_block))

fac=1/((occs[1]-3)*(occs[1]-2))
print(np.allclose(np.einsum('pqijpqkl ->ijklmn',bbbbbbbb)*fac,d2_bbbb_block))
print(np.allclose(np.einsum('piqjpkql ->ijklmn',bbbbbbbb)*fac,d2_bbbb_block))
print(np.allclose(np.einsum('ipqjkpql ->ijklmn',bbbbbbbb)*fac,d2_bbbb_block))
print(np.allclose(np.einsum('ipjqkplq ->ijklmn',bbbbbbbb)*fac,d2_bbbb_block))
print(np.allclose(np.einsum('pijqpklq ->ijklmn',bbbbbbbb)*fac,d2_bbbb_block))
print(np.allclose(np.einsum('ijpqklpq ->ijklmn',bbbbbbbb)*fac,d2_bbbb_block))
'''
#aaaaaaa normalized to (occs[0])*(occs[0]-1)*(occs[0]-2)*(occs[1]-3)
#abababab normalized to (occs[0])*(occs[1])*(occs[0]-1)*(occs[1]-1)
#abbbabbb normalized to (occs[0])*(occs[1])*(occs[1]-1)*(occs[1]-2)
#aaabaaab normalized to (occs[1])*(occs[0])*(occs[0]-1)*(occs[0]-2)
#rdm4 must be normalized to (wfn.nocc_up * 2)*(wfn.nocc_up * 2 - 1) * (wfn.nocc_up * 2 - 2)* (wfn.nocc_up * 2 - 3)


