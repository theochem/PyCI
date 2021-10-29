/* This file is part of PyCI.
 *
 * PyCI is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * PyCI is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PyCI. If not, see <http://www.gnu.org/licenses/>. */

#include <pyci.h>

namespace pyci {

namespace {

template<class WfnType>
double compute_overlap_thread(const WfnType &wfn1, const WfnType &wfn2, const double *coeffs1,
                              const double *coeffs2, const long start, const long end) {
    double olp = 0.0;
    long j;
    for (long i = start; i < end; ++i) {
        j = wfn2.index_det(wfn1.det_ptr(i));
        if (j != -1)
            olp += coeffs1[i] * coeffs2[j];
    }
    return olp;
}

} // namespace

template<class WfnType>
double compute_overlap(const WfnType &wfn1, const WfnType &wfn2, const double *coeffs1,
                       const double *coeffs2) {
    if (wfn1.ndet > wfn2.ndet)
        return compute_overlap<WfnType>(wfn2, wfn1, coeffs2, coeffs1);
    long nthread = get_num_threads(), start, end;
    long chunksize = wfn1.ndet / nthread + ((wfn1.ndet % nthread) ? 1 : 0);
    Vector<std::future<double>> v_threads;
    v_threads.reserve(nthread);
    for (long i = 0; i < nthread; ++i) {
        start = i * chunksize;
        end = (start + chunksize < wfn1.ndet) ? start + chunksize : wfn1.ndet;
        v_threads.push_back(std::async(std::launch::async, &compute_overlap_thread<WfnType>,
                                       std::ref(wfn1), std::ref(wfn2), coeffs1, coeffs2, start,
                                       end));
    }
    double olp = 0.0;
    for (auto &future : v_threads)
        olp += future.get();
    return olp;
}

template double compute_overlap<OneSpinWfn>(const OneSpinWfn &, const OneSpinWfn &, const double *,
                                            const double *);

template double compute_overlap<TwoSpinWfn>(const TwoSpinWfn &, const TwoSpinWfn &, const double *,
                                            const double *);

template<class WfnType>
double py_compute_overlap(const WfnType &wfn1, const WfnType &wfn2, const Array<double> coeffs1,
                          const Array<double> coeffs2) {
    return compute_overlap<WfnType>(wfn1, wfn2,
                                    reinterpret_cast<const double *>(coeffs1.request().ptr),
                                    reinterpret_cast<const double *>(coeffs2.request().ptr));
}

template double py_compute_overlap<OneSpinWfn>(const OneSpinWfn &, const OneSpinWfn &,
                                               const Array<double>, const Array<double>);

template double py_compute_overlap<TwoSpinWfn>(const TwoSpinWfn &, const TwoSpinWfn &,
                                               const Array<double>, const Array<double>);

} // namespace pyci
