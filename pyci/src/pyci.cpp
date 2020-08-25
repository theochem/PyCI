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

/*
Note: This seems to be the only way to get setuptools to compile everything together. This
decreases the size of the binary by about 50%, and the compile time is still very short.
*/

#include <pyci.h>

#include "SpookyV2.cpp"

#include "common.cpp"

#include "ham.cpp"

#include "wfn.cpp"

#include "onespinwfn.cpp"
#include "twospinwfn.cpp"

#include "dociwfn.cpp"
#include "fullciwfn.cpp"
#include "genciwfn.cpp"

#include "sparseop.cpp"

#include "enpt2.cpp"
#include "hci.cpp"
#include "overlap.cpp"
#include "rdm.cpp"

#include "binding.cpp"
