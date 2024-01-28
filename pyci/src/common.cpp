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

#include <pyci.h>

namespace pyci {

namespace {

long gcd(long, long);

}

long g_number_threads{1L};

clhasher hasher{0x23a23cf5033c3c81UL, 0xb3816f6a2c68e530UL};

long get_num_threads(void) {
    return g_number_threads;
}

void set_num_threads(const long n) {
    g_number_threads = std::max(n, 1L);
    Eigen::setNbThreads(g_number_threads);
}

long binomial(long n, long k) {
    if (k == 0)
        return 1;
    else if (k == 1)
        return n;
    else if (k >= n)
        return (k == n);
    else if (k > n / 2)
        k = n - k;
    long binom = 1, g, nr, dr;
    for (long d = 1; d <= k; ++d) {
        if (binom >= Max<long>() / n) {
            g = gcd(n, d);
            nr = n / g;
            dr = d / g;
            g = gcd(binom, dr);
            binom = binom / g;
            dr = dr / g;
            if (binom >= Max<long>() / nr)
                return Max<long>();
            binom = binom * nr / dr;
        } else
            binom = binom * n-- / d;
    }
    return binom;
}

void fill_hartreefock_det(long nocc, ulong *det) {
    long i = 0;
    while (nocc >= Size<ulong>()) {
        det[i++] = Max<ulong>();
        nocc -= Size<ulong>();
    }
    if (nocc)
        det[i] = (1UL << nocc) - 1;
}

void fill_det(const long nocc, const long *occs, ulong *det) {
    for (long i = 0, j; i < nocc; ++i) {
        j = occs[i];
        det[j / Size<ulong>()] |= 1UL << (j % Size<ulong>());
    }
}

void fill_occs(const long nword, const ulong *det, long *occs) {
    long p, j = 0, offset = 0;
    ulong word;
    for (long i = 0; i < nword; ++i) {
        word = det[i];
        while (word) {
            p = Ctz(word);
            occs[j++] = p + offset;
            word &= ~(1UL << p);
        }
        offset += Size<ulong>();
    }
}

void fill_virs(const long nword, long nbasis, const ulong *det, long *virs) {
    long p, j = 0, offset = 0;
    ulong word, mask;
    for (long i = 0; i < nword; ++i) {
        mask = (nbasis < Size<ulong>()) ? ((1UL << nbasis) - 1) : Max<ulong>();
        word = det[i] ^ mask;
        while (word) {
            p = Ctz(word);
            virs[j++] = p + offset;
            word &= ~(1UL << p);
        }
        offset += Size<ulong>();
        nbasis -= Size<ulong>();
    }
}

void next_colex(long *indices) {
    long i = 0;
    while (indices[i + 1] - indices[i] == 1) {
        indices[i] = i;
        ++i;
    }
    ++(indices[i]);
}

long rank_colex(const long nbasis, const long nocc, const ulong *det) {
    long k = 0, binom = 1, rank = 0;
    for (long i = 0; i < nbasis; ++i) {
        if (k == nocc)
            break;
        else if (det[i / Size<ulong>()] & (1UL << (i % Size<ulong>()))) {
            ++k;
            binom = (k == i) ? 1 : binom * i / k;
            rank += binom;
        } else
            binom = (k >= i) ? 1 : binom * i / (i - k);
    }
    return rank;
}

void unrank_colex(long nbasis, const long nocc, long rank, long *occs) {
    for (long i = 0, j, k, binom; i < nocc; ++i) {
        j = nocc - i;
        binom = binomial(nbasis, j);
        if (binom <= rank) {
            for (k = 0; k < j; ++k)
                occs[k] = k;
            break;
        }
        while (binom > rank)
            binom = binomial(--nbasis, j);
        occs[j - 1] = nbasis;
        rank -= binom;
    }
}

long phase_single_det(const long nword, const long i, const long a, const ulong *det) {
    (void)nword; // nword is deliberately unused.
    long j, k, l, m, n, high, low, nperm = 0;
    ulong mask;
    if (i > a) {
        high = i;
        low = a;
    } else {
        high = a;
        low = i;
    }
    k = high / Size<ulong>();
    m = high % Size<ulong>();
    j = low / Size<ulong>();
    n = low % Size<ulong>();
    mask = det[j];
    mask &= Max<ulong>();
    if (j == k) {
        mask &= (1UL << m) - 1;
        mask &= ~(1UL << (n + 1)) + 1;
        nperm += Pop(mask);
    } else {
        mask &= ~(1UL << (n + 1)) + 1;
        nperm += Pop(mask);
        mask = det[k];
        mask &= (1UL << m) - 1;
        nperm += Pop(mask);
        for (l = j + 1; l < k; ++l) {
            mask = det[l];
            mask &= Max<ulong>();
            nperm += Pop(mask);
        }
    }
    return (nperm % 2) ? -1 : 1;
}

long phase_double_det(const long nword, const long i1, const long i2, const long a1, const long a2,
                      const ulong *det) {
    (void)nword; // nword is deliberately unused.
    long j, k, l, m, n, high, low, nperm = 0;
    ulong mask;
    // first excitation
    if (i1 > a1) {
        high = i1;
        low = a1;
    } else {
        high = a1;
        low = i1;
    }
    k = high / Size<ulong>();
    m = high % Size<ulong>();
    j = low / Size<ulong>();
    n = low % Size<ulong>();
    mask = det[j];
    mask &= Max<ulong>();
    if (j == k) {
        mask &= (1UL << m) - 1;
        mask &= ~(1UL << (n + 1)) + 1;
        nperm += Pop(mask);
    } else {
        mask &= ~(1UL << (n + 1)) + 1;
        nperm += Pop(mask);
        mask = det[k];
        mask &= (1UL << m) - 1;
        nperm += Pop(mask);
        for (l = j + 1; l < k; ++l) {
            mask = det[l];
            mask &= Max<ulong>();
            nperm += Pop(mask);
        }
    }
    // second excitation
    if (i2 > a2) {
        high = i2;
        low = a2;
    } else {
        high = a2;
        low = i2;
    }
    k = high / Size<ulong>();
    m = high % Size<ulong>();
    j = low / Size<ulong>();
    n = low % Size<ulong>();
    mask = det[j];
    mask &= Max<ulong>();
    if (j == k) {
        mask &= (1UL << m) - 1;
        mask &= ~(1UL << (n + 1)) + 1;
        nperm += Pop(mask);
    } else {
        mask &= ~(1UL << (n + 1)) + 1;
        nperm += Pop(mask);
        mask = det[k];
        mask &= (1UL << m) - 1;
        nperm += Pop(mask);
        for (l = j + 1; l < k; ++l) {
            mask = det[l];
            mask &= Max<ulong>();
            nperm += Pop(mask);
        }
    }
    // order excitations properly
    if ((i2 < a1) || (i1 > a2))
        ++nperm;
    return (nperm % 2) ? -1 : 1;
}

long popcnt_det(const long nword, const ulong *det) {
    long popcnt = 0;
    for (long i = 0; i < nword; ++i)
        popcnt += Pop(det[i]);
    return popcnt;
}

long ctz_det(const long nword, const ulong *det) {
    ulong word;
    for (long i = 0; i < nword; ++i) {
        word = det[i];
        if (word)
            return Ctz(word) + i * Size<ulong>();
    }
    return 0;
}

long nword_det(const long n) {
    return n / Size<ulong>() + ((n % Size<ulong>()) ? 1 : 0);
}

void excite_det(const long i, const long a, ulong *det) {
    det[i / Size<ulong>()] &= ~(1UL << (i % Size<ulong>()));
    det[a / Size<ulong>()] |= 1UL << (a % Size<ulong>());
}

void setbit_det(const long i, ulong *det) {
    det[i / Size<ulong>()] |= 1UL << (i % Size<ulong>());
}

void clearbit_det(const long i, ulong *det) {
    det[i / Size<ulong>()] &= ~(1UL << (i % Size<ulong>()));
}

long py_popcnt(const Array<ulong> det) {
    pybind11::buffer_info buf = det.request();
    return popcnt_det(buf.shape[0], reinterpret_cast<const ulong *>(buf.ptr));
}

long py_ctz(const Array<ulong> det) {
    pybind11::buffer_info buf = det.request();
    return ctz_det(buf.shape[0], reinterpret_cast<const ulong *>(buf.ptr));
}

namespace {

long gcd(long x, long y) {
    long t;
    if (y < x) {
        t = x;
        x = y;
        y = t;
    }
    while (y > 0) {
        t = y;
        y = x % y;
        x = t;
    }
    return x;
}

} // namespace

} // namespace pyci
