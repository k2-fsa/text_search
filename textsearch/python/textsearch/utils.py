from bisect import bisect_left
from typing import List, Tuple
import numpy as np
from _textsearch import row_ids_to_row_splits as _row_ids_to_row_splits


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")


def row_ids_to_row_splits(row_ids: np.ndarray) -> np.ndarray:
    """Convert row ids to row splits.

    Args:
      row_ids:
        A 1-D array of dtype np.uint32 containing row ids.
    Returns:
      Return a 1-D array of dtype np.uint32 containing row splits.
    """
    assert row_ids.ndim == 1, row_ids.ndim
    assert row_ids.dtype == np.uint32, row_ids.dtype

    row_ids = np.ascontiguousarray(row_ids)
    num_rows = row_ids[-1] + 1
    row_splits = np.empty(num_rows + 1, dtype=np.uint32)

    _row_ids_to_row_splits(row_ids, row_splits)
    return row_splits


def is_overlap(
    ranges: List[Tuple[int, int]], query: Tuple[int, int], overlap_ratio: float = 0.5
) -> bool:
    """
    Return if the given range overlaps with the existing ranges.

    Caution:
      `ranges` will be modified in this function (when returning False)

    Note: overlapping here means the length of overlapping area is greater than
    some threshold (currently, the threshold is `overlap_ratio` multiply the length
    of the shorter overlapping ranges).

    Args:
      ranges:
        The existing ranges, it is sorted in ascending order on input, and we will
        keep it sorted in this function.
      query:
        The given range.

    Return:
      Return True if having overlap otherwise False.
    """
    is_overlap = False
    index = bisect_left(ranges, query)
    if index == 0:
        if ranges:
            is_overlap = (
                query[1] - ranges[0][0]
                > min(ranges[0][1] - ranges[0][0], query[1] - query[0]) * overlap_ratio
            )
    elif index == len(ranges):
        is_overlap = (
            ranges[index - 1][1] - query[0]
            > min(ranges[index - 1][1] - ranges[index - 1][0], query[1] - query[0])
            * overlap_ratio
        )
    else:
        is_overlap = (
            ranges[index - 1][1] - query[0]
            > min(ranges[index - 1][1] - ranges[index - 1][0], query[1] - query[0])
            * overlap_ratio
        ) or (
            query[1] - ranges[index][0]
            > min(ranges[index][1] - ranges[index][0], query[1] - query[0])
            * overlap_ratio
        )

    if not is_overlap:
        ranges.insert(index, query)
    return is_overlap


def is_punctuation(c: str, eos_only: bool = False) -> bool:
    """
    Return True if the given character is a punctuation.

    Args:
      c:
        The given character.
      eos_only:
        If True the punctuations are only those indicating end of a sentence (.?! for now).
    """
    if eos_only:
        return c in ".?!"
    return c in ',.;?!():-<>-/"'
