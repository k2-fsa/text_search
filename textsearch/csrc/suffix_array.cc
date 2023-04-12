/**
 * Copyright      2021 - 2023     Xiaomi Corporation (authors: Daniel Povey
 *                                                             Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "textsearch/csrc/suffix_array.h"
#include <cstdint>
#include <vector>

namespace fasttextsearch {
template <typename T> inline bool Leq(T a1, T a2, T b1, T b2) {
  // lexicographic order for pairs, used in CreateSuffixArray()
  return (a1 < b1 || a1 == b1 && a2 <= b2);
}
template <typename T> inline bool Leq(T a1, T a2, T a3, T b1, T b2, T b3) {
  // lexicographic order for triples, used in CreateSuffixArray()
  return (a1 < b1 || a1 == b1 && Leq(a2, a3, b2, b3));
}

/*
  Helper function for CreateSuffixArray().
  Stably sorts a[0..n-1] to b[0..n-1] with keys in 0..K from r;
  i.e. the values in a are interpreted as indexes into the array
  `r` and the values in `r` are used for comparison, so that
  at exit, r[b[i]] <= r[b[i+1]].
*/
template <typename T>
static void RadixPass(const T *a, T *b, const T *r, T n, T K) {
  std::vector<T> c(K + 1, 0); // counter array
  for (T i = 0; i < n; i++)
    c[r[a[i]]]++;                       // count occurrences
  for (T i = 0, sum = 0; i <= K; i++) { // exclusive prefix sums
    T t = c[i];
    c[i] = sum;
    sum += t;
  }
  for (T i = 0; i < n; i++)
    b[c[r[a[i]]]++] = a[i]; // sort
}

// See documentation in suffix_array.h, where we use different names
// for the arguments (here, we leave the names the same as in
// https://algo2.iti.kit.edu/documents/jacm05-revised.pdf.
template <typename T> void CreateSuffixArray(const T *text, T n, T K, T *SA) {
  if (n == 1) { // The paper's code didn't seem to handle n == 1 correctly.
    SA[0] = 0;
    return;
  }
  T n0 = (n + 2) / 3, n1 = (n + 1) / 3, n2 = n / 3, n02 = n0 + n2;
  std::vector<T> R(n02 + 3, 0);
  std::vector<T> SA12(n02 + 3, 0);
  std::vector<T> R0(n0, 0);
  std::vector<T> SA0(n0, 0);

  //******* Step 0: Construct sample ********
  // generate positions of mod 1 and mod 2 suffixes
  // the "+(n0-n1)" adds a dummy mod 1 suffix if n%3 == 1
  for (T i = 0, j = 0; i < n + (n0 - n1); i++) {
    if (i % 3 != 0) {
      R[j++] = i;
    }
  }
  //******* Step 1: Sort sample suffixes ********
  // lsb radix sort the mod 1 and mod 2 triples
  RadixPass(R.data(), SA12.data(), text + 2, n02, K);
  RadixPass(SA12.data(), R.data(), text + 1, n02, K);
  RadixPass(R.data(), SA12.data(), text, n02, K);

  // find lexicographic names of triples and
  // write them to correct places in R
  T name = 0, c0 = -1, c1 = -1, c2 = -1;
  for (T i = 0; i < n02; i++) {
    if (text[SA12[i]] != c0 || text[SA12[i] + 1] != c1 ||
        text[SA12[i] + 2] != c2) {
      name++;
      c0 = text[SA12[i]];
      c1 = text[SA12[i] + 1];
      c2 = text[SA12[i] + 2];
    }
    if (SA12[i] % 3 == 1) {
      R[SA12[i] / 3] = name;
    } // write to R1
    else {
      R[SA12[i] / 3 + n0] = name;
    } // write to R2
  }
  // recurse if names are not yet unique
  if (name < n02) {
    CreateSuffixArray(R.data(), n02, name, SA12.data());
    // store unique names in R using the suffix array
    for (T i = 0; i < n02; i++)
      R[SA12[i]] = i + 1;
  } else // generate the suffix array of R directly
    for (T i = 0; i < n02; i++)
      SA12[R[i] - 1] = i;
  //******* Step 2: Sort nonsample suffixes ********
  // stably sort the mod 0 suffixes from SA12 by their first character
  for (T i = 0, j = 0; i < n02; i++)
    if (SA12[i] < n0)
      R0[j++] = 3 * SA12[i];
  RadixPass(R0.data(), SA0.data(), text, n0, K);
  //******* Step 3: Merge ********
  // merge sorted SA0 suffixes and sorted SA12 suffixes
  for (T p = 0, t = n0 - n1, k = 0; k < n; k++) {
    // i is pos of current offset 12 suffix
    T i = (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2);
    T j = SA0[p]; // pos of current offset 0 suffix
    if (SA12[t] < n0
            ? // different compares for mod 1 and mod 2 suffixes
            Leq(text[i], R[SA12[t] + n0], text[j], R[j / 3])
            : Leq(text[i], text[i + 1], R[SA12[t] - n0 + 1], text[j],
                  text[j + 1], R[j / 3 + n0])) { // suffix from SA12 is smaller
      SA[k] = i;
      t++;
      if (t == n02) // done --- only SA0 suffixes left
        for (k++; p < n0; p++, k++)
          SA[k] = SA0[p];
    } else { // suffix from SA0 is smaller
      SA[k] = j;
      p++;
      if (p == n0) // done --- only SA12 suffixes left
        for (k++; t < n02; t++, k++)
          SA[k] = (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2);
    }
  }
}

template void CreateSuffixArray(const int32_t *text, int32_t n, int32_t K,
                                int32_t *SA);
} // namespace fasttextsearch
