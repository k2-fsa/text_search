#!/usr/bin/env python3

import unittest

import numpy as np

from textsearch import TextSource


class TestTextSource(unittest.TestCase):
    def test_use_utf8(self):
        name = "test"
        s = "zażółć gęślą jaźń\n你好Hallo"
        source = TextSource.from_str(name=name, s=s, use_utf8=True)

        assert source.name == name, (source.name, name)

        assert isinstance(source.binary_text, np.ndarray)
        assert source.binary_text.dtype == np.uint8, source.binary_text.dtype

        expected_binary_text = s.encode("utf-8")
        assert source.binary_text.tobytes() == expected_binary_text

        assert isinstance(source.text, str), type(source.text)
        assert source.text == s, (source.text, s)

        assert source.pos is None, source.pos

    def test_without_using_utf8(self):
        name = "test"
        s = "zażółć gęślą jaźń\n你好Hallo"
        source = TextSource.from_str(name=name, s=s, use_utf8=False)

        assert source.name == name, (source.name, name)

        assert isinstance(source.binary_text, np.ndarray)
        assert source.binary_text.dtype == np.int32, source.binary_text.dtype

        expected_binary_text = np.fromiter((ord(i) for i in s), dtype=np.int32)
        np.testing.assert_equal(source.binary_text, expected_binary_text)

        assert isinstance(source.text, str), type(source.text)
        assert source.text == s, (source.text, s)

        assert isinstance(source.pos, np.ndarray), type(source.pos)
        assert source.pos.dtype == np.uint32, source.pos.dtype
        # fmt: off
        # A Polish character occupies 2 bytes in utf-8 while
        # a Chinese character occupies 3 bytes
        np.testing.assert_equal(
            source.pos,
            np.array(
                [0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 17, 19, 20, 21, 22, 24,
                 26, 27, 30, 33, 34, 35, 36, 37], dtype=np.uint32
            ),
        )
        # fmt: on


if __name__ == "__main__":
    unittest.main()
