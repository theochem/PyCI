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

#pragma once

#include <cstddef>

#include <algorithm>
#include <iterator>
#include <utility>

/* see <http://stackoverflow.com/a/46370189> */

namespace std {

namespace sort_with_arg {

template<typename _Data, typename _Order>
struct value_reference_t;

template<typename _Data, typename _Order>
struct value_t {
    _Data data;
    _Order val;

    inline value_t(_Data _data, _Order _val) : data(_data), val(_val) {
    }

    inline value_t(const value_reference_t<_Data, _Order> &rhs);
};

template<typename _Data, typename _Order>
struct value_reference_t {
    _Data *pdata;
    _Order *pval;

    value_reference_t(_Data *_itData, _Order *_itVal) : pdata(_itData), pval(_itVal) {
    }

    inline value_reference_t &operator=(const value_reference_t &rhs) {
        *pdata = *rhs.pdata;
        *pval = *rhs.pval;
        return *this;
    }

    inline value_reference_t &operator=(const value_t<_Data, _Order> &rhs) {
        *pdata = rhs.data;
        *pval = rhs.val;
        return *this;
    }

    inline bool operator<(const value_reference_t &rhs) {
        return *pval < *rhs.pval;
    }
};

template<typename _Data, typename _Order>
struct value_iterator_t {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = ptrdiff_t;
    using value_type = value_t<_Data, _Order>;
    using pointer = value_t<_Data, _Order> *;
    using reference = value_reference_t<_Data, _Order>;

    _Data *itData;
    _Order *itVal;
    value_iterator_t(_Data *_itData, _Order *_itVal) : itData(_itData), itVal(_itVal) {
    }

    inline ptrdiff_t operator-(const value_iterator_t &rhs) const {
        return itVal - rhs.itVal;
    }

    inline value_iterator_t operator+(ptrdiff_t off) const {
        return value_iterator_t(itData + off, itVal + off);
    }

    inline value_iterator_t operator-(ptrdiff_t off) const {
        return value_iterator_t(itData - off, itVal - off);
    }

    inline value_iterator_t &operator++() {
        ++itData;
        ++itVal;
        return *this;
    }

    inline value_iterator_t &operator--() {
        --itData;
        --itVal;
        return *this;
    }

    inline value_iterator_t operator++(int) {
        return value_iterator_t(itData++, itVal++);
    }

    inline value_iterator_t operator--(int) {
        return value_iterator_t(itData--, itVal--);
    }

    inline value_t<_Data, _Order> operator*() const {
        return value_t<_Data, _Order>(*itData, *itVal);
    }

    inline value_reference_t<_Data, _Order> operator*() {
        return value_reference_t<_Data, _Order>(itData, itVal);
    }

    inline bool operator<(const value_iterator_t &rhs) const {
        return itVal < rhs.itVal;
    }

    inline bool operator==(const value_iterator_t &rhs) const {
        return itVal == rhs.itVal;
    }

    inline bool operator!=(const value_iterator_t &rhs) const {
        return itVal != rhs.itVal;
    }
};

template<typename _Data, typename _Order>
inline value_t<_Data, _Order>::value_t(const value_reference_t<_Data, _Order> &rhs)
    : data(*rhs.pdata), val(*rhs.pval) {
}

template<typename _Data, typename _Order>
bool operator<(const value_t<_Data, _Order> &lhs, const value_reference_t<_Data, _Order> &rhs) {
    return lhs.val < *rhs.pval;
}

template<typename _Data, typename _Order>
bool operator<(const value_reference_t<_Data, _Order> &lhs, const value_t<_Data, _Order> &rhs) {
    return *lhs.pval < rhs.val;
}

template<typename _Data, typename _Order>
void swap(value_reference_t<_Data, _Order> lhs, value_reference_t<_Data, _Order> rhs) {
    std::swap(*lhs.pdata, *rhs.pdata);
    std::swap(*lhs.pval, *rhs.pval);
}

} // namespace sort_with_arg

} // namespace std
