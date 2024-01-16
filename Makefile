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


# Setup
# -----

# Set C compiler executable
CC ?= cc
export CC

# Set C++ compiler executable
CXX ?= c++
export CXX

# Set Python executable
PYTHON ?= python3

# Set C++ compile flags
CFLAGS := --std=c++14 -Wall -pipe -O3
CFLAGS += -fPIC -flto=auto -fvisibility=hidden
CFLAGS += -pthread
CFLAGS += -Ipyci/include

# Set Python include directories
CFLAGS += $(shell $(PYTHON) tools/python_include_dirs.py)

# Set external projects and their include directories
DEPS := $(addprefix deps/,eigen spectra parallel-hashmap clhash pybind11)
CFLAGS += $(addprefix -Ideps/,eigen spectra/include parallel-hashmap clhash/include pybind11/include)

# This C++ compile flag is needed in order for Macs to find system libraries
ifeq ($(shell uname -s),Darwin)
CFLAGS += -undefined dynamic_lookup
endif

# Set PyCI version number
VERSION_MAJOR := 0
VERSION_MINOR := 6
VERSION_PATCH := 1
PYCI_VERSION := $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)

# Set preprocessor directives
DEFS := -D_PYCI_VERSION='$(PYCI_VERSION)'
DEFS += -D_GIT_BRANCH='$(shell git rev-parse --abbrev-ref HEAD)'
DEFS += -D_BUILD_TIME='$(shell date -u +%F\ %T)'
DEFS += -D_COMPILER_VERSION='$(shell $(CXX) --version | head -n 1)'

# Set objects
OBJECTS := $(patsubst %.cpp,%.o,$(wildcard pyci/src/*.cpp))


# Make commands
# -------------

.PHONY: all
all: pyci/pyci.so.$(PYCI_VERSION) pyci/pyci.so.$(VERSION_MAJOR) pyci/pyci.so

.PHONY: test
test:
	$(PYTHON) -m pytest -sv ./pyci

.PHONY: clean
clean:
	rm -rf pyci/src/*.o pyci/pyci.so*

.PHONY: cleandeps
cleandeps:
	rm -rf deps


# Make targets
# ------------

compile_flags.txt:
	echo $(CFLAGS) | tr ' ' '\n' > $(@)

pyci/src/%.o: pyci/src/%.cpp pyci/include/pyci.h $(DEPS)
	$(CXX) $(CFLAGS) $(DEFS) -c $(<) -o $(@)

pyci/pyci.so.$(PYCI_VERSION): $(OBJECTS) deps/clhash/clhash.o
	$(CXX) $(CFLAGS) $(DEFS) -shared $(^) -o $(@)

pyci/pyci.so.$(VERSION_MAJOR): pyci/pyci.so.$(PYCI_VERSION)
	ln -s $(notdir $(<)) $(@)

pyci/pyci.so: pyci/pyci.so.$(PYCI_VERSION)
	ln -s $(notdir $(<)) $(@)

deps/eigen:
	@git clone https://gitlab.com/libeigen/eigen.git $(@)

deps/spectra:
	@git clone https://github.com/yixuan/spectra.git $(@)

deps/parallel-hashmap:
	@git clone https://github.com/greg7mdp/parallel-hashmap.git $(@)

deps/clhash:
	@git clone https://github.com/lemire/clhash.git $(@)

deps/clhash/clhash.o: deps/clhash
	$(MAKE) -C $(dir $(@))

deps/pybind11:
	@git clone https://github.com/pybind/pybind11.git $(@)
