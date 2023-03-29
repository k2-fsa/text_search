#!/usr/bin/env python3

import unittest

import numpy as np

from textsearch import row_ids_to_row_splits


class TestRowIdsToRowSplits(unittest.TestCase):
    def test_row_ids_to_row_splits(self):
        row_ids = np.array([0, 0, 0, 1, 1], dtype=np.uint32)
        row_splits = row_ids_to_row_splits(row_ids)
        expected_row_splits = np.array([0, 3, 5], dtype=np.uint32)
        np.testing.assert_equal(row_splits, expected_row_splits)
        assert row_splits.dtype == np.uint32, row_splits.dtype

    def test_row_ids_to_row_splits_case2(self):
        row_ids = np.array([1, 1, 3, 3, 5], dtype=np.uint32)
        row_splits = row_ids_to_row_splits(row_ids)
        print(row_splits)
        expected_row_splits = np.array([0, 0, 2, 2, 4, 4, 5], dtype=np.uint32)
        np.testing.assert_equal(row_splits, expected_row_splits)
        assert row_splits.dtype == np.uint32, row_splits.dtype


if __name__ == "__main__":
    unittest.main()
