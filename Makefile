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

ifndef (CXX)
	CXX =g++
endif

ifndef (SPOOKYHASH_SEED)
	SPOOKYHASH_SEED =0xdeadbeefdeadbeefUL
endif

ifndef (NUM_THREADS_DEFAULT)
	NUM_THREADS_DEFAULT =4
endif

CFLAGS  =--std=c++14
CFLAGS += -Wall
CFLAGS += -O3
CFLAGS += -mtune=native
CFLAGS += -fPIC
CFLAGS += -fvisibility=hidden
CFLAGS += -pthread
CFLAGS += -DPYCI_VERSION=$(shell python -c "from setup import version; print(version)")
CFLAGS += -DPYCI_SPOOKYHASH_SEED=$(SPOOKYHASH_SEED)
CFLAGS += -DPYCI_NUM_THREADS_DEFAULT=$(NUM_THREADS_DEFAULT)

LIBS =-lm

INCLUDES  =$(shell python-config --includes)
INCLUDES += -I$(shell python -c "import numpy; print(numpy.get_include())")
INCLUDES += -Ilib/pybind11/include
INCLUDES += -Ilib/parallel-hashmap
INCLUDES += -Ipyci/include

PYTHON_SUFFIX =$(shell python-config --extension-suffix | sed 's/.so//g')

.PHONY: all
all: pyci/pyci$(PYTHON_SUFFIX).so

pyci/pyci$(PYTHON_SUFFIX).so:
	$(CXX) $(CFLAGS) $(LIBS) $(INCLUDES) -shared pyci/src/pyci.cpp -o pyci/pyci$(PYTHON_SUFFIX).so

.PHONY: clean
clean:
	rm -f ./pyci/pyci$(PYTHON_SUFFIX).so
