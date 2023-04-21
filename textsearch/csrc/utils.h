// textsearch/csrc/utils.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef TEXTSEARCH_CSRC_UTILS_H_
#define TEXTSEARCH_CSRC_UTILS_H_
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fasttextsearch {

/** This function is copied/modified from
 * https://github.com/k2-fsa/k2/blob/master/k2/csrc/utils.h#L315
 *
 * @param [in] num_elems   The number of elements in the irregular array
 * @param [in] row_ids   row_ids vector of length num_elems (
 *                  row_ids[num_elems - 1] + 1 must equal num_rows). Must be
 *                  non-decreasing.
 * @param [in] num_rows   Number of rows in the irregular array, must
 *                  be greater than any element of row_ids
 * @param [out] row_splits  Row-splits vector that this function
 *                  writes to, of length num_rows + 1.  row_splits[num_rows]
 *                  will equal num_elems.
 */
void RowIdsToRowSplits(int32_t num_elems, const uint32_t *row_ids,
                       int32_t num_rows, uint32_t *row_splits);

/**
 * Assuming the suffix array was created from a text where the first
 * ``query_len`` positions represent the query text and the remaining positions
 * represent the reference text, return a list indicating, for each suffix
 * position in the query text, the ``num_close_matches`` suffix positions in the
 * reference text that immediately precede and follow it lexicographically.
 * (I think suffix position refers to the last character of a suffix).
 *
 * This is easy to do from the suffix array without computing, for example,
 * the LCP array; and it produces exactly ``num_close_matches`` matches per
 * position in the query text, which is also convenient.
 *
 * @param [in] suffix_array  A suffix array as created by
 *                `CreateSuffixArray` of dtype ``int32``.
 * @param [in] seq_len  The length of suffix_array.
 * @param [in] query_len  The length of query, must satisfy
 *               ``0 <= query_len < seq_len``.
 * @param [in] num_close_matches  The number of close_matches for each query
 *               element.
 * @param [in, out] close_matches  The array container that close_matches will
 *                    write to, its size is ``query_len * num_close_matches``,
 *                    in which positions
 *                    ``num_close_matches*i, num_close_matches*i + 1, ...``
 *                    represent the num_close_matches positions in the original
 *                    text that are within the reference portion, and which
 *                    immediately precede and follow, in the suffix array, query
 *                    position ``i``.  This means that the suffixes ending at
 *                    those positions are reverse-lexicographically close to the
 *                    suffix ending at position ``i``.  As a special case, if
 *                    one of these returned numbers would equal the EOS position
 *                    (position seq_len - 1), or if a query position is before
 *                    any reference position in the suffix aray, we output
 *                    ``seq_len - 2`` instead to avoid having to handle special
 *                    cases later on(anyway, these would not represent a close
 *                    match).
 */
void FindCloseMatches(const int32_t *suffix_array, int32_t seq_len,
                      int32_t query_len, int32_t num_close_matches,
                      int32_t *close_matches);

} // namespace fasttextsearch

#endif // TEXTSEARCH_CSRC_UTILS_H_
