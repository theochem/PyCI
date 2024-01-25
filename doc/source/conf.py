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

import sphinx_rtd_theme

import pyci


project = "PyCI"


copyright = "2021, Michael Richer"


author = "Michael Richer"


release = pyci.__version__


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
]


exclude_patterns = []


html_theme = "sphinx_rtd_theme"


html_static_path = []


templates_path = []


mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js"


mathjax_config = {
    "extensions": ["tex2jax.js"],
    "jax": ["input/TeX", "output/HTML-CSS"],
}
