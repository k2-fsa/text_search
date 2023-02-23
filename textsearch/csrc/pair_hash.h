// Copied from
// https://github.com/Martinsos/edlib/blob/gen-seqs/edlib/include/pairHash.hpp

// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <functional>
// This header file provides a hashing function for std::pair of any custom
// types. Since there is no specialization of std::hash for std::pair<T1,T2>
// provided in the standard library, we can use the implementation in this
// header file.

/**
 * Takes a seed and a value of any type. A Seed can be a hash code of other
 * values or simply 0. It will generate a new hash code using the given seed and
 * value. It can be called repeatedly to incrementally create a hash value from
 * several variables. This "magical" implementation is adopted from Boost C++
 * Libraries.
 * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
 * @param [in] seed The hash code of previous variables
 * @param [in] v The value to be hashed using the given seed
 */
template <class T> inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename S, typename T> struct pair_hash {
  inline std::size_t operator()(const std::pair<S, T> &v) const {
    std::size_t seed = 0;
    hash_combine(seed, v.first);
    hash_combine(seed, v.second);
    return seed;
  }
};
