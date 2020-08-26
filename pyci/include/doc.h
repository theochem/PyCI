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

#pragma once

#include <string>

namespace pyci {

const std::string module_doc = R"""(
PyCI C++ extension module.
)""";

const std::string hamiltonian_doc = R"""(
Hamiltonian class.

For arbitrary-seniority systems:

.. math::

    H = \sum_{pq}{t_{pq} a^\dagger_p a_q} + \sum_{pqrs}{g_{pqrs} a^\dagger_p a^\dagger_q a_s a_r}

For seniority-zero systems:

    .. math::

    H = \sum_{p}{h_p N_p} + \sum_{p \neq q}{v_{pq} P^\dagger_p P_q} + \sum_{pq}{w_{pq} N_p N_q}

where

.. math::

    h_{p} = \left<p|T|p\right> = t_{pp}

.. math::

    v_{pq} = \left<pp|V|qq\right> = g_{ppqq}

.. math::

    w_{pq} = 2 \left<pq|V|pq\right> - \left<pq|V|qp\right> = 2 * g_{pqpq} - g_{pqqp}

)""";

const std::string wavefunction_doc = R"""(
Wavefunction base class.
)""";

const std::string onespinwfn_doc = R"""(
One-spin wavefunction class base class.
)""";

const std::string twospinwfn_doc = R"""(
Two-spin wavefunction base class.
)""";

const std::string dociwfn_doc = R"""(
DOCI wavefunction class.
)""";

const std::string fullciwfn_doc = R"""(
FullCI wave function class.
)""";

const std::string genciwfn_doc = R"""(
Generalized CI wavefunction class.
)""";

const std::string sparseop_doc = R"""(
Sparse matrix operator class.
)""";

} // namespace pyci
