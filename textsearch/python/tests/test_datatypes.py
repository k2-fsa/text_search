from tempfile import NamedTemporaryFile

import numpy as np

from textsearch.python.textsearch.datatypes import TextSource


def test_text_source():
    with NamedTemporaryFile(suffix=".txt", encoding="utf8", mode="w+") as f:
        f.write("zażółć gęślą jaźń\n")
        f.flush()
        f.seek(0)

        source = TextSource.from_file(f.name)

        assert isinstance(source.name, str)

        assert isinstance(source.binary_text, np.ndarray)
        assert np.issubdtype(source.binary_text.dtype, "S1")
        np.testing.assert_equal(
            source.binary_text,
            np.array(
                [
                    b"z",
                    b"a",
                    b"\xc5",
                    b"\xbc",
                    b"\xc3",
                    b"\xb3",
                    b"\xc5",
                    b"\x82",
                    b"\xc4",
                    b"\x87",
                    b" ",
                    b"g",
                    b"\xc4",
                    b"\x99",
                    b"\xc5",
                    b"\x9b",
                    b"l",
                    b"\xc4",
                    b"\x85",
                    b" ",
                    b"j",
                    b"a",
                    b"\xc5",
                    b"\xba",
                    b"\xc5",
                    b"\x84",
                    b"\n",
                ]
            ),
        )

        assert isinstance(source.text, str)
        assert source.text == "zażółć gęślą jaźń\n"

        assert isinstance(source.pos, np.ndarray)
        assert np.issubdtype(source.pos.dtype, np.int64)
        np.testing.assert_equal(
            source.pos,
            np.array(
                [0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 17, 19, 20, 21, 22, 24, 26]
            ),
        )
