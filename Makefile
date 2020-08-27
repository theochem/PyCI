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
CXX ?= c++

SPOOKYHASH_SEED ?= 0xdeadbeefdeadbeefUL
NUM_THREADS_DEFAULT ?= 4

CFLAGS += --std=c++14
CFLAGS += -Wall
CFLAGS += -pipe
CFLAGS += -O3

CFLAGS += -fPIC
CFLAGS += -flto
CFLAGS += -fno-plt
CFLAGS += -fwrapv
CFLAGS += -fvisibility=hidden

CFLAGS += -pthread

CFLAGS += -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
CFLAGS += -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
CFLAGS += -Ilib/eigen
CFLAGS += -Ilib/parallel-hashmap
CFLAGS += -Ilib/pybind11/include
CFLAGS += -Ipyci/include

CFLAGS += -DPYCI_VERSION=$(shell $(PYTHON) -c "from setup import version; print(version)")
CFLAGS += -DPYCI_SPOOKYHASH_SEED=$(SPOOKYHASH_SEED)
CFLAGS += -DPYCI_NUM_THREADS_DEFAULT=$(NUM_THREADS_DEFAULT)

ifeq ($(shell uname -s),Darwin)
CFLAGS += -undefined dynamic_lookup
endif

.PHONY: all
all: pyci/pyci.so

.PHONY: clean
clean:
	rm -rf ./pyci/pyci.so ./build ./dist ./pyci.egg-info

pyci/pyci.so:
	$(CXX) $(CFLAGS) -shared pyci/src/pyci.cpp -o pyci/pyci.so