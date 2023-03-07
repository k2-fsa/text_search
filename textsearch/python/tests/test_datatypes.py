#!/usr/bin/env python3

from tempfile import NamedTemporaryFile

import numpy as np

from textsearch import TextSource

import unittest


class TestTextSource(unittest.TestCase):
    def test(self):
        with NamedTemporaryFile(suffix=".txt", encoding="utf8", mode="w+") as f:
            s = "zażółć gęślą jaźń\n你好Hallo"
            f.write(s)
            f.flush()
            f.seek(0)

            source = TextSource.from_file(f.name)

            assert isinstance(source.name, str)
            assert source.name == f.name, (source.name, f.name)

            assert isinstance(source.binary_text, np.ndarray)
            assert np.issubdtype(source.binary_text.dtype, "S1")
            s_bytes = s.encode("utf-8")
            assert source.binary_text.tobytes() == s_bytes

            assert isinstance(source.text, str)
            assert source.text == s

            assert isinstance(source.pos, np.ndarray)
            assert np.issubdtype(source.pos.dtype, np.uint32), source.pos.dtype
            # fmt: off
            # A Polish character occupies 2 bytes in utf-8 while
            # a Chinese character occupies 3 bytes
            np.testing.assert_equal(
                source.pos,
                np.array(
                    [0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 17, 19, 20, 21, 22, 24,
                     26, 27, 30, 33, 34, 35, 36, 37]
                ),
            )
            # fmt: on


if __name__ == "__main__":
    unittest.main()
