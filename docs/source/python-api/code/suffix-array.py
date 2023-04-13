#!/usr/bin/env python3

import numpy as np
import textsearch

a = np.fromstring("banana", dtype=np.int8)
print(a)

suffix_array = textsearch.create_suffix_array(a)
print(suffix_array)

for i in suffix_array[:-1]:
    print(a[i:].tobytes().decode("utf-8") + "$")

"""
The output is:

[ 98  97 110  97 110  97]
[1 3 5 0 2 4 6]

anana$
ana$
a$
banana$
nana$
na$
"""
