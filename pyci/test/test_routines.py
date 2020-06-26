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

from nose.tools import assert_raises

import numpy as np
import numpy.testing as npt

from scipy.special import comb

import pyci
from pyci.test import datafile


class TestRoutines:

    CASES = [
        ('h2o_ccpvdz', pyci.doci_wfn,   (5,),   -75.634588422),
        ('be_ccpvdz',  pyci.doci_wfn,   (2,),   -14.600556994),
        ('li2_ccpvdz', pyci.doci_wfn,   (3,),   -14.878455349),
        ('he_ccpvqz',  pyci.doci_wfn,   (1,),    -2.886809116),
        ('he_ccpvqz',  pyci.fullci_wfn, (1, 1),  -2.886809116),
        ('be_ccpvdz',  pyci.fullci_wfn, (2, 2), -14.600556994),
        ]

    def test_solve_sparse(self):
        for filename, wfn_type, occs, energy in self.CASES:
            yield self.run_solve_sparse, filename, wfn_type, occs, energy

    def test_sparse_rectangular(self):
        for filename, wfn_type, occs, energy in self.CASES:
            yield self.run_sparse_rectangular, filename, wfn_type, occs, energy

    def test_compute_rdms(self):
        for filename, wfn_type, occs, energy in self.CASES:
            yield self.run_compute_rdms, filename, wfn_type, occs, energy

    def test_run_hci(self):
        for filename, wfn_type, occs, energy in self.CASES[:3]:
            yield self.run_run_hci, filename, wfn_type, occs, energy

    def run_solve_sparse(self, filename, wfn_type, occs, energy):
        ham = pyci.hamiltonian.from_file(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_all_dets()
        op = pyci.sparse_op(ham, wfn)
        es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)

    def run_sparse_rectangular(self, filename, wfn_type, occs, energy):
        ham = pyci.hamiltonian.from_file(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_all_dets()
        nrow = len(wfn) - 10
        op = pyci.sparse_op(ham, wfn, nrow)
        assert op.shape == (nrow, len(wfn))
        y = op.dot(np.ones(op.shape[1], dtype=pyci.c_double))
        assert y.ndim == 1
        assert y.shape[0] == op.shape[0]

    def run_compute_rdms(self, filename, wfn_type, occs, energy):
        if wfn_type is pyci.fullci_wfn:
            raise AssertionError('not implemented')
        ham = pyci.hamiltonian.from_file(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_all_dets()
        op = pyci.sparse_op(ham, wfn)
        es, cs = op.solve(n=1, ncv=30, tol=1.0e-6)
        if isinstance(wfn, pyci.doci_wfn):
            d0, d2 = wfn.compute_rdms(cs[0])
            npt.assert_allclose(np.trace(d0), wfn.nocc_up, rtol=0, atol=1.0e-9)
            npt.assert_allclose(np.sum(d2), wfn.nocc_up * (wfn.nocc_up - 1), rtol=0, atol=1.0e-9)
            k0 = ham.reduced_v(wfn.nocc_up)
            k2 = ham.reduced_w(wfn.nocc_up)
            energy = ham.ecore
            energy += np.einsum('ij,ij', k0, d0)
            energy += np.einsum('ij,ij', k2, d2)
            npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)
            rdm1, rdm2 = wfn.generate_rdms(d0, d2)
        elif isinstance(wfn, pyci.fullci_wfn):
            rdm1, rdm2 = wfn.compute_rdms(cs[0])
        else:
            raise ValueError('wfn_type must be doci_wfn or fullci_wfn')
        with np.load(datafile('{0:s}_spinres.npz'.format(filename))) as f:
            one_mo = f['one_mo']
            two_mo = f['two_mo']
        energy = ham.ecore
        energy += np.einsum('ij,ij', one_mo, rdm1)
        energy += 0.25 * np.einsum('ijkl,ijkl', two_mo, rdm2)
        npt.assert_allclose(energy, es[0], rtol=0.0, atol=1.0e-9)

    def run_run_hci(self, filename, wfn_type, occs, energy):
        if wfn_type is pyci.fullci_wfn:
            raise AssertionError('not implemented')
        ham = pyci.hamiltonian.from_file(datafile('{0:s}.fcidump'.format(filename)))
        wfn = wfn_type(ham.nbasis, *occs)
        wfn.add_hartreefock_det()
        es, cs = pyci.sparse_op(ham, wfn).solve(n=1, tol=1.0e-6)
        dets_added = 1
        niter = 0
        while dets_added:
            dets_added = wfn.run_hci(ham, cs[0], eps=1.0e-5)
            es, cs = pyci.sparse_op(ham, wfn).solve(n=1, tol=1.0e-6)
            niter += 1
        assert niter > 1
        assert len(wfn) < np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-6)
        dets_added = 1
        while dets_added:
            dets_added = wfn.run_hci(ham, cs[0], eps=0.0)
            op = pyci.sparse_op(ham, wfn)
            es, cs = op.solve(n=1, tol=1.0e-6)
        assert len(wfn) == np.prod([comb(wfn.nbasis, occ, exact=True) for occ in occs])
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)

