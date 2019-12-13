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

from __future__ import absolute_import, unicode_literals

from nose.tools import assert_raises

import numpy as np
import numpy.testing as npt

from doci import comb, dociham, dociwfn, solve_ci, compute_rdms, compute_energy, run_hci
from doci.test import unicode_str, datafile


class TestRoutines:

    CASES = {
        'he_ccpvqz': (1, -2.886809115915473),
        'be_ccpvdz': (2, -14.60055699423718),
        'li2_ccpvdz': (3, -14.878455348858425),
        'h2o_ccpvdz': (5, -75.63458842226694),
        }

    def test_solve_ci_sparse(self):
        for filename in self.CASES.keys():
            yield self.run_solve_ci_sparse, filename

    def test_solve_ci_direct(self):
        for filename in self.CASES.keys():
            # skip slow test
            if filename == 'h2o_ccpvdz':
                continue
            yield self.run_solve_ci_direct, filename

    def test_compute_rdms(self):
        for filename in self.CASES.keys():
            yield self.run_compute_rdms, filename

    def test_compute_energy(self):
        for filename in self.CASES.keys():
            yield self.run_compute_energy, filename

    def test_run_hci(self):
        # prepare problem
        nocc, energy = self.CASES['h2o_ccpvdz']
        ham = dociham.from_file(unicode_str(datafile('h2o_ccpvdz.fcidump')))
        wfn = dociwfn(ham.nbasis, nocc)
        wfn.reserve(comb(wfn.nbasis, wfn.nocc))
        wfn.add_hartreefock_det()
        es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='sparse')
        dets_added = 1
        niter = 0
        while dets_added:
            dets_added = run_hci(ham, wfn, cs[0], eps=1.0e-5)
            es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='sparse')
            niter += 1
        assert niter > 1
        assert len(wfn) < comb(wfn.nbasis, wfn.nocc)
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-6)
        dets_added = 1
        while dets_added:
            dets_added = run_hci(ham, wfn, cs[0], eps=0.0)
            es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='sparse')
        assert len(wfn) == comb(wfn.nbasis, wfn.nocc)
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)

    def run_solve_ci_sparse(self, filename):
        # prepare problem
        nocc, energy = self.CASES[filename]
        ham = dociham.from_file(unicode_str(datafile('{0:s}.fcidump'.format(filename))))
        wfn = dociwfn(ham.nbasis, nocc)
        wfn.add_all_dets()
        # test solve_ci sparse
        es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='sparse')
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)

    def run_solve_ci_direct(self, filename):
        # prepare problem
        nocc, energy = self.CASES[filename]
        ham = dociham.from_file(unicode_str(datafile('{0:s}.fcidump'.format(filename))))
        wfn = dociwfn(ham.nbasis, nocc)
        wfn.add_all_dets()
        # test solve_ci direct
        es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='direct')
        npt.assert_allclose(es[0], energy, rtol=0.0, atol=1.0e-9)

    def run_compute_rdms(self, filename):
        # prepare problem
        nocc, energy = self.CASES[filename]
        ham = dociham.from_file(unicode_str(datafile('{0:s}.fcidump'.format(filename))))
        wfn = dociwfn(ham.nbasis, nocc)
        wfn.add_all_dets()
        es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='sparse')
        # test compute_rdms
        d0, d2 = compute_rdms(wfn, cs[0])
        npt.assert_allclose(np.trace(d0), wfn.nocc, rtol=0, atol=1.0e-9)
        npt.assert_allclose(np.sum(d2), wfn.nocc * (wfn.nocc - 1), rtol=0, atol=1.0e-9)

    def run_compute_energy(self, filename):
        # prepare problem
        nocc, energy = self.CASES[filename]
        ham = dociham.from_file(unicode_str(datafile('{0:s}.fcidump'.format(filename))))
        wfn = dociwfn(ham.nbasis, nocc)
        wfn.add_all_dets()
        es, cs = solve_ci(ham, wfn, n=1, ncv=30, tol=1.0e-6, mode='sparse')
        # test compute_energy
        npt.assert_allclose(compute_energy(ham, wfn, cs[0]), energy, rtol=0.0, atol=1.0e-9)
