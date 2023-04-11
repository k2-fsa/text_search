#!/usr/bin/env python3

import numpy as np
import textsearch


def main():
    s = "banana"
    array = np.fromstring(s, dtype=np.int8)

    assert array.max() < np.iinfo(array.dtype).max - 1
    tail = np.array([array.max() + 1, 0, 0, 0], dtype=np.int8)
    array = np.concatenate([array, tail])

    suffix_array = textsearch.create_suffix_array(array)
    print(suffix_array)


if __name__ == "__main__":
    main()
