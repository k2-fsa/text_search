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

/** This function is copied/modified from
 * https://github.com/k2-fsa/k2/blob/master/k2/csrc/algorithms.h
 *
 * @param [in] keep  keep array of length num_old_elems indicating whether to
 *                   keep current element.
 * @param [in] num_old_elems  The number of elements of keep vector.
 *
 * @param [out] new2old  An array mapping the new indexes to the old indexes.
 *                       Its dimension is the number of new indexes
 *                       (i.e. the number of true in keep)
 */
void GetNew2Old(const bool *keep, uint32_t num_old_elems,
                std::vector<uint32_t> *new2old);
} // namespace fasttextsearch

#endif // TEXTSEARCH_CSRC_UTILS_H_
