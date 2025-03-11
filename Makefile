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
CFLAGS := -std=c++14 -Wall -Wextra -pipe -O3
CFLAGS += -fPIC -flto=auto -fvisibility=hidden
CFLAGS += -pthread
CFLAGS += -Ipyci/include

ifneq ($(MAKE_NATIVE),)
CFLAGS += -mavx -mavx2 -msse4.2 -march=native -mtune=native
endif

# Set Python include directories
CFLAGS += $(shell $(PYTHON) tools/python_include_dirs.py)

# Set external projects and their include directories
DEPS := $(addprefix deps/,eigen spectra parallel-hashmap pybind11 rapidhash)
CFLAGS += $(addprefix -Ideps/,eigen spectra/include parallel-hashmap pybind11/include rapidhash)

# This C++ compile flag is needed in order for Macs to find system libraries
ifeq ($(shell uname -s),Darwin)
CFLAGS += -undefined dynamic_lookup
PYCI_EXTENSION := dylib
else
PYCI_EXTENSION := so
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
ifeq ($(shell uname -s),Darwin)
DEFS += -D_USE_RAPIDHASH='1'
endif

# Set objects
OBJECTS := $(patsubst %.cpp,%.o,$(wildcard pyci/src/*.cpp))


# Make commands
# -------------

.PHONY: all
all: pyci/_pyci.$(PYCI_EXTENSION).$(PYCI_VERSION) pyci/_pyci.$(PYCI_EXTENSION).$(VERSION_MAJOR) pyci/_pyci.$(PYCI_EXTENSION)

.PHONY: test
test:
	@set -e; $(PYTHON) -m pytest -sv ./pyci

.PHONY: clean
clean:
	rm -rf pyci/src/*.o pyci/_pyci.$(PYCI_EXTENSION)*

.PHONY: cleandeps
cleandeps:
	rm -rf deps

.PHONY: compile_flags.txt
compile_flags.txt:
	echo "$(CFLAGS)" | tr ' ' '\n' > $(@)


# Make targets
# ------------

pyci/src/%.o: pyci/src/%.cpp pyci/include/pyci.h $(DEPS)
	$(CXX) $(CFLAGS) $(DEFS) -c $(<) -o $(@)

pyci/_pyci.$(PYCI_EXTENSION).$(PYCI_VERSION): $(OBJECTS)
	$(CXX) $(CFLAGS) $(DEFS) -shared $(^) -o $(@)

pyci/_pyci.$(PYCI_EXTENSION).$(VERSION_MAJOR): pyci/_pyci.$(PYCI_EXTENSION).$(PYCI_VERSION)
	ln -sf $(notdir $(<)) $(@)

pyci/_pyci.$(PYCI_EXTENSION): pyci/_pyci.$(PYCI_EXTENSION).$(PYCI_VERSION)
	ln -sf $(notdir $(<)) $(@)

deps/eigen:
	[ -d $@ ] || git clone https://gitlab.com/libeigen/eigen.git $@

deps/spectra:
	[ -d $@ ] || git clone https://github.com/yixuan/spectra.git $@

deps/parallel-hashmap:
	[ -d $@ ] || git clone https://github.com/greg7mdp/parallel-hashmap.git $@

deps/pybind11:
	[ -d $@ ] || git clone https://github.com/pybind/pybind11.git $@

deps/rapidhash:
	[ -d $@ ] || git clone https://github.com/nicoshev/rapidhash.git $@
