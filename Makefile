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

PYTHON ?= python
CXX ?= g++

SPOOKYHASH_SEED ?= 0xdeadbeefdeadbeefUL
NUM_THREADS_DEFAULT ?= 4

_CFLAGS += --std=c++14
_CFLAGS += -Wall
_CFLAGS += -pipe
_CFLAGS += -O3

_CFLAGS += -march=x86-64
_CFLAGS += -mtune=generic

_CFLAGS += -fPIC
_CFLAGS += -fno-plt
_CFLAGS += -fwrapv
_CFLAGS += -fvisibility=hidden

_CFLAGS += -pthread

_CFLAGS += -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
_CFLAGS += -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
_CFLAGS += -Ilib/pybind11/include
_CFLAGS += -Ilib/parallel-hashmap
_CFLAGS += -Ipyci/include

_CFLAGS += -DPYCI_VERSION=$(shell $(PYTHON) -c "from setup import version; print(version)")
_CFLAGS += -DPYCI_SPOOKYHASH_SEED=$(SPOOKYHASH_SEED)
_CFLAGS += -DPYCI_NUM_THREADS_DEFAULT=$(NUM_THREADS_DEFAULT)

_CFLAGS += $(CFLAGS)

.PHONY: all
all: pyci/pyci.so

pyci/pyci.so:
	$(CXX) $(_CFLAGS) -shared pyci/src/pyci.cpp -o pyci/pyci.so

.PHONY: clean
clean:
	rm -rf ./pyci/pyci.so ./build ./dist ./pyci.egg-info
