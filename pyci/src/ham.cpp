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

#include <iomanip>
#include <iostream>
#include <regex>

#include <pyci.h>

namespace pyci {

Ham::Ham(void) {
}

Ham::Ham(const Ham &ham)
    : nbasis(ham.nbasis), ecore(ham.ecore), one_mo(ham.one_mo), two_mo(ham.two_mo), h(ham.h),
      v(ham.v), w(ham.w), one_mo_array(ham.one_mo_array), two_mo_array(ham.two_mo_array),
      h_array(ham.h_array), v_array(ham.v_array), w_array(ham.w_array) {
}

Ham::Ham(Ham &&ham) noexcept
    : nbasis(std::exchange(ham.nbasis, 0)), ecore(std::exchange(ham.ecore, 0.0)),
      one_mo(std::exchange(ham.one_mo, nullptr)), two_mo(std::exchange(ham.two_mo, nullptr)),
      h(std::exchange(ham.h, nullptr)), v(std::exchange(ham.v, nullptr)),
      w(std::exchange(ham.w, nullptr)), one_mo_array(std::move(ham.one_mo_array)),
      two_mo_array(std::move(ham.two_mo_array)), h_array(std::move(ham.h_array)),
      v_array(std::move(ham.v_array)), w_array(std::move(ham.w_array)) {
}

namespace {

template<typename T>
T read_parameter(const std::string &header, const std::string &name,
                 const std::string &regex_string) {
    std::regex r(name + regex_string);
    std::smatch m;
    T parameter;
    if (std::regex_search(header, m, r)) {
        std::string parameter_string = m[1];
        if (std::is_same<T, bool>::value) {
            std::transform(parameter_string.begin(), parameter_string.end(),
                           parameter_string.begin(), ::tolower);
            std::istringstream(parameter_string) >> std::boolalpha >> parameter;
        } else {
            std::istringstream(parameter_string) >> parameter;
        }
    } else {
        throw std::invalid_argument(name + " is not found.");
    }
    return parameter;
}

template<typename T>
T read_parameter(const std::string &header, const std::string &name,
                 const std::string &regex_string, T default_value) {
    T parameter;
    try {
        parameter = read_parameter<T>(header, name, regex_string);
    } catch (const std::exception &e) {
        parameter = default_value;
    }
    return parameter;
}

} // namespace

Ham::Ham(const std::string &filename) {
    std::ifstream f(filename);
    if (f.fail())
        throw std::ios_base::failure("Failed to read the FCIDUMP file " + filename);

    /* See https://github.com/quan-tum/CDFCI/, the header reader is copied from there. */
    std::string header, line;
    while (std::getline(f, line) && line.find("&END") == std::string::npos &&
           line.find("/") == std::string::npos) {
        header += " ";
        header += line;
    }
    header += " &END";

    if (f.eof())
        throw std::ios_base::failure("FCIDUMP has the wrong header");

    const std::string int_regex = R"([ ]*=[ ]*(\d+))";
    const std::string bool_regex = R"([ ]*=[ .]*(FALSE|TRUE))";

    long norb = read_parameter<int>(header, "NORB", int_regex);
    /* long nelec = read_parameter<int>(header, "NELEC", int_regex); */
    /* long ms2 = read_parameter<int>(header, "MS2", int_regex); */
    bool uhf = read_parameter<bool>(header, "UHF", bool_regex, false);

    nbasis = norb;
    one_mo_array = Array<double>({nbasis, nbasis});
    two_mo_array = Array<double>({nbasis, nbasis, nbasis, nbasis});
    h_array = Array<double>(nbasis);
    v_array = Array<double>({nbasis, nbasis});
    w_array = Array<double>({nbasis, nbasis});
    one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
    two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
    h = reinterpret_cast<double *>(h_array.request().ptr);
    v = reinterpret_cast<double *>(v_array.request().ptr);
    w = reinterpret_cast<double *>(w_array.request().ptr);

    long n1, n2, n3;
    n1 = nbasis;
    n2 = n1 * n1;
    n3 = n2 * n1;
    ecore = 0;
    std::fill(one_mo, one_mo + n2, static_cast<double>(0.));
    std::fill(two_mo, two_mo + n3 * n1, static_cast<double>(0.));
    if (uhf) {
        throw std::runtime_error("Unrestricted FCIDUMP not implemented");
    } else {
        long i, j, k, l;
        double integral;
        while (f >> integral >> i >> j >> k >> l) {
            if (i && j && k && l) {
                --i;
                --j;
                --k;
                --l;
                two_mo[i * n3 + k * n2 + j * n1 + l] = integral;
                two_mo[k * n3 + i * n2 + l * n1 + j] = integral;
                two_mo[j * n3 + k * n2 + i * n1 + l] = integral;
                two_mo[i * n3 + l * n2 + j * n1 + k] = integral;
                two_mo[j * n3 + l * n2 + i * n1 + k] = integral;
                two_mo[l * n3 + j * n2 + k * n1 + i] = integral;
                two_mo[k * n3 + j * n2 + l * n1 + i] = integral;
                two_mo[l * n3 + i * n2 + k * n1 + j] = integral;
            } else if (i && j) {
                --i;
                --j;
                one_mo[i * n1 + j] = integral;
                one_mo[j * n1 + i] = integral;
            } else {
                ecore = integral;
            }
        }
    }
    long i, j, k = 0, l = 0;
    for (i = 0; i != n1; ++i) {
        h[k++] = one_mo[i * (n1 + 1)];
        for (j = 0; j != n1; ++j) {
            v[l] = two_mo[i * n3 + i * n2 + j * n1 + j];
            w[l++] =
                two_mo[i * n3 + j * n2 + i * n1 + j] * 2 - two_mo[i * n3 + j * n2 + j * n1 + i];
        }
    }
}

Ham::Ham(const double e, const Array<double> mo1, const Array<double> mo2)
    : nbasis(mo1.request().shape[0]), ecore(e), one_mo_array(mo1), two_mo_array(mo1),
      h_array(nbasis), v_array({nbasis, nbasis}), w_array({nbasis, nbasis}) {
    one_mo = reinterpret_cast<double *>(one_mo_array.request().ptr);
    two_mo = reinterpret_cast<double *>(two_mo_array.request().ptr);
    h = reinterpret_cast<double *>(h_array.request().ptr);
    v = reinterpret_cast<double *>(v_array.request().ptr);
    w = reinterpret_cast<double *>(w_array.request().ptr);
    long n1 = nbasis;
    long n2 = nbasis * n1;
    long n3 = nbasis * n2;
    long i, j, k = 0, l = 0;
    for (i = 0; i != n1; ++i) {
        h[k++] = one_mo[i * (n1 + 1)];
        for (j = 0; j != n1; ++j) {
            v[l] = two_mo[i * n3 + i * n2 + j * n1 + j];
            w[l++] =
                two_mo[i * n3 + j * n2 + i * n1 + j] * 2 - two_mo[i * n3 + j * n2 + j * n1 + i];
        }
    }
}

void Ham::to_file(const std::string &filename, const long nelec, const long ms2,
                  const double tol) const {
    bool uhf = false;
    long n1, n2, n3;
    n1 = nbasis;
    n2 = n1 * n1;
    n3 = n2 * n1;
    std::ofstream f(filename);
    if (f.fail())
        throw std::ios_base::failure("Failed to open the FCIDUMP file " + filename);

    f << "&FCIDUMP\nNORB=" << nbasis << ",\nNELEC=" << nelec << ",\nMS2=" << ms2
      << ",\nUHF=" << (uhf ? ".TRUE." : ".FALSE.") << ",\nORBSYM=";
    for (long i = 0; i != nbasis; ++i)
        f << "1,";
    f << "\nISYM=1,\n&END\n";
    long i, j, k, l;
    for (i = 0; i != nbasis; ++i)
        for (j = 0; j <= i; ++j)
            for (k = 0; k != nbasis; ++k)
                for (l = 0; l <= k; ++l)
                    if ((i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l)
                        f << std::setw(28) << std::setprecision(20) << std::scientific
                          << two_mo[i * n3 + k * n2 + j * n1 + l] << ' ' << i + 1 << ' ' << j + 1
                          << ' ' << k + 1 << ' ' << l + 1 << "\n";
    for (i = 0; i != nbasis; ++i)
        for (j = 0; j <= i; ++j)
            f << std::setw(28) << std::setprecision(20) << std::scientific << one_mo[i * n1 + j]
              << ' ' << i + 1 << ' ' << j + 1 << " 0 0\n";

    f << std::setw(28) << std::setprecision(20) << std::scientific << ecore << " 0 0 0 0\n";
}

} // namespace pyci
