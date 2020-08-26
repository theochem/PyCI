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

Wfn::Wfn(const Wfn &wfn)
    : nbasis(wfn.nbasis), nocc(wfn.nocc), nocc_up(wfn.nocc_up), nocc_dn(wfn.nocc_dn),
      nvir(wfn.nvir), nvir_up(wfn.nvir_up), nvir_dn(wfn.nvir_dn), ndet(wfn.ndet), nword(wfn.nword),
      nword2(wfn.nword2), maxrank_up(wfn.maxrank_up), maxrank_dn(wfn.maxrank_dn), dets(wfn.dets),
      dict(wfn.dict) {
}

Wfn::Wfn(Wfn &&wfn) noexcept
    : nbasis(std::exchange(wfn.nbasis, 0)), nocc(std::exchange(wfn.nocc, 0)),
      nocc_up(std::exchange(wfn.nocc_up, 0)), nocc_dn(std::exchange(wfn.nocc_dn, 0)),
      nvir(std::exchange(wfn.nvir, 0)), nvir_up(std::exchange(wfn.nvir_up, 0)),
      nvir_dn(std::exchange(wfn.nvir_dn, 0)), ndet(std::exchange(wfn.ndet, 0)),
      nword(std::exchange(wfn.nword, 0)), nword2(std::exchange(wfn.nword2, 0)),
      maxrank_up(std::exchange(wfn.maxrank_up, 0)), maxrank_dn(std::exchange(wfn.maxrank_dn, 0)),
      dets(std::move(wfn.dets)), dict(std::move(wfn.dict)) {
}

Wfn::Wfn(const long nb, const long nu, const long nd) {
    init(nb, nu, nd);
}

long Wfn::length(void) const {
    return ndet;
}

void Wfn::squeeze(void) {
    dets.shrink_to_fit();
}

Wfn::Wfn(void){};

void Wfn::init(const long nb, const long nu, const long nd) {
    if (nd < 0)
        throw std::domain_error("nocc_dn is < 0");
    else if (nu < nd)
        throw std::domain_error("nocc_up is < nocc_dn");
    else if (nb < nu)
        throw std::domain_error("nbasis is < nocc_up");
    nbasis = nb;
    nocc = nu + nd;
    nocc_up = nu;
    nocc_dn = nd;
    nvir = nb * 2 - nu - nd;
    nvir_up = nb - nu;
    nvir_dn = nb - nd;
    ndet = 0;
    nword = nword_det(nb);
    nword2 = nword * 2;
    maxrank_up = binomial_cutoff(nb, nu);
    maxrank_dn = binomial_cutoff(nb, nd);
}

} // namespace pyci
