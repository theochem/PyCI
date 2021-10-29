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

PYTHON ?= python3

CXX ?= c++

CHUNKSIZE_MIN ?= 1024

SPARSEOP_RESIZE_FACTOR ?= 1.5

SPOOKYHASH_SEED ?= 0xdeadbeefdeadbeefUL

INC_DIRS := eigen spectra/include parallel-hashmap pybind11/include

CFLAGS := --std=c++14 -Wall -pipe -O3 -pthread -fPIC -flto -fno-plt -fwrapv -fvisibility=hidden

CFLAGS += -Ipyci/include
CFLAGS += -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
CFLAGS += -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
CFLAGS += $(addprefix -Ilib/,$(INC_DIRS))

CFLAGS += -DPYCI_VERSION=$(shell $(PYTHON) -c "from setup import version; print(version)")
CFLAGS += -DPYCI_CHUNKSIZE_MIN=$(CHUNKSIZE_MIN)
CFLAGS += -DPYCI_SPARSEOP_RESIZE_FACTOR=$(SPARSEOP_RESIZE_FACTOR)
CFLAGS += -DPYCI_SPOOKYHASH_SEED=$(SPOOKYHASH_SEED)

ifeq ($(shell uname -s),Darwin)
CFLAGS += -undefined dynamic_lookup
endif

SRCS := $(shell find pyci/src -name '*.cpp')
HDRS := $(shell find pyci/include -name '*.h')
LIBS := lib/eigen lib/spectra lib/parallel-hashmap lib/pybind11

.PHONY: all
all: pyci/pyci.so

.PHONY: clean
clean:
	rm -rf ./pyci/pyci.so ./build ./dist ./pyci.egg-info

.PHONY: clean_lib
clean_lib:
	rm -rf ./lib

.PHONY: clean_all
clean_all: clean clean_lib

.PHONY: test
test:
	$(PYTHON) -m pytest -sv ./pyci
	$(PYTHON) -m pycodestyle -v ./pyci
	$(PYTHON) -m pydocstyle -v ./pyci

pyci/pyci.so: $(LIBS) $(SRCS) $(HDRS)
	$(CXX) $(CFLAGS) -shared $(SRCS) -o $@

lib/eigen:
	@git clone https://gitlab.com/libeigen/eigen.git $@

lib/spectra:
	@git clone https://github.com/yixuan/spectra.git $@

lib/parallel-hashmap:
	@git clone https://github.com/greg7mdp/parallel-hashmap.git $@

lib/pybind11:
	@git clone https://github.com/pybind/pybind11.git $@
