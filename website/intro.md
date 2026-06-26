<!-- #region -->
# PyCI

[PyCI](https://github.com/theochem/PyCI) is a free, open-source, and cross-platform Python library designed to help you effortlessly run arbitrary determinant CI.

Please use the following citation in any publication using PyCI library:

**"PyCI: A Python-scriptable library for arbitrary determinant CI"**.
*The Journal of Chemical Physics* **161**(13), 132502 (2024).
M. Richer, G. Sánchez-Díaz, M. Martínez-González, V. Chuiko, T.D. Kim, A. Tehrani, S. Wang, P.B. Gaikwad, C.E.V. de Moura, C. Masschelein, R.A. Miranda-Quintana, A. Gerolin, F. Heidar-Zadeh, and P.W. Ayers

```
@article{richerPyCI2024,
  title = {{{PyCI}}: {{A Python-scriptable}} Library for Arbitrary Determinant {{CI}}},
  shorttitle = {{{PyCI}}},
  author = {Richer, Michelle and {S{\'a}nchez-D{\'i}az}, Gabriela and {Mart{\'i}nez-Gonz{\'a}lez}, Marco and Chuiko, Valerii and Kim, Taewon David and Tehrani, Alireza and Wang, Shuoyang and Gaikwad, Pratiksha B. and {de Moura}, Carlos E. V. and Masschelein, Cassandra and {Miranda-Quintana}, Ram{\'o}n Alain and Gerolin, Augusto and {Heidar-Zadeh}, Farnaz and Ayers, Paul W.},
  year = 2024,
  month = oct,
  journal = {The Journal of Chemical Physics},
  volume = {161},
  number = {13},
  pages = {132502},
  issn = {0021-9606},
  doi = {10.1063/5.0219010},
  urldate = {2024-12-16},
  abstract = {PyCI is a free and open-source Python library for setting up and running arbitrary determinant-driven configuration interaction (CI) computations, as well as their generalizations to cases where the coefficients of the determinant are nonlinear functions of optimizable parameters. PyCI also includes functionality for computing the residual correlation energy, along with the ability to compute spin-polarized one- and two-electron (transition) reduced density matrices. PyCI was originally intended to replace the ab~initio quantum chemistry functionality in the HORTON library but emerged as a standalone research tool, primarily intended to aid in method development, while maintaining high performance so that it is suitable for practical calculations. To this end, PyCI is written in Python, adopting principles of modern software development, including comprehensive documentation, extensive testing, continuous integration/delivery protocols, and package management. Computationally intensive steps, notably operations related to generating Slater determinants and computing their expectation values, are delegated to low-level C++ code. This article marks the official release of the PyCI library, showcasing its functionality and scope.},
}
```

The PyCI source code is hosted on [GitHub](https://github.com/theochem/PyCI) and is released under the [GNU General Public License version 3 (GPLv3)](https://github.com/theochem/PyCI/blob/master/LICENSE). We welcome any contributions to the PyCI library in accordance with our [Code of Conduct](CODE_OF_CONDUCT.md); please see our [Contributing Guidelines](CONTRIBUTING.md). Please report any issues you encounter while using GBasis library on [GitHub Issues](https://github.com/theochem/PyCI/issues). For further information and inquiries please contact us at [qcdevs@gmail.com](mailto:qcdevs@gmail.com).

## Why PyCI?
PyCI is a versatile, free, and open-source Python 3 library designed for setting up and executing arbitrary determinant CI (Configuration Interaction) and FanCI (Flexible Ansatz for N-electron Configuration Interaction) computations. This library is your gateway to efficiently generate arbitrary sets of Slater determinants and construct selected CI wave functions through intuitive Python scripting.

### Key Features:
1. Flexible CI Computations: Seamlessly set up and run arbitrary determinant CI and FanCI computations with ease.
2. Python Scripting: Construct CI wave functions through Python scripting, allowing for unprecedented flexibility and control.
3. FanCI Wave Functions: Build arbitrary FanCI wave functions by implementing overlap functions and gradients into a well-defined Python interface.
4. Efficient Multi-threading: PyCI leverages a core C++ library with multi-threading capabilities for optimized (Fan)CI computations.
5. ENPT2 Energy Correction: Compute second-order Epstein-Nesbet perturbation theory (ENPT2) energy corrections effortlessly.
6. RDM Calculations: Calculate spin-polarized one- and two-electron (transition) Reduced Density Matrices (RDMs) of wave functions.

Get Started with PyCI Today
Experience the freedom to innovate and accelerate your research in molecular electronic structure theory. Join the QC-Devs community and elevate your computational chemistry endeavors.
<!-- #endregion -->