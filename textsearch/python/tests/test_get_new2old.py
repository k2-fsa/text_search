#!/usr/bin/env python3

import unittest

import numpy as np

from textsearch import get_new2old


class TestGetNew2Old(unittest.TestCase):
    def test_get_new2old_basic(self):
        keep = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        new2old = get_new2old(keep)
        expected_new2old = np.array([1, 2, 5, 6], dtype=np.uint32)
        np.testing.assert_equal(new2old, expected_new2old)
        assert new2old.dtype == np.uint32, new2old.dtype

    def test_get_new2old_empty(self):
        keep = np.array([], dtype=bool)
        new2old = get_new2old(keep)
        expected_new2old = np.array([], dtype=np.uint32)
        np.testing.assert_equal(new2old, expected_new2old)
        assert new2old.dtype == np.uint32, new2old.dtype

    def test_get_new2old_drop_all(self):
        keep = np.array([0, 0, 0, 0, 0, 0, 0], dtype=bool)
        new2old = get_new2old(keep)
        expected_new2old = np.array([], dtype=np.uint32)
        np.testing.assert_equal(new2old, expected_new2old)
        assert new2old.dtype == np.uint32, new2old.dtype


if __name__ == "__main__":
    unittest.main()
