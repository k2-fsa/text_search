#!/usr/bin/env python3

import numpy as np
import textsearch

q = "abcd"
t = "acdefabd"
query = np.fromstring(q, dtype=np.int8).astype(np.int32)
target = np.fromstring(t, dtype=np.int8).astype(np.int32)
distance, alignments = textsearch.levenshtein_distance(query, target)
print(distance)
print(alignments)
aligns = textsearch.get_nice_alignments(alignments, q, t)
print(aligns[0])
print("-." * 10)
print(aligns[1])

"""
The output is

1
[(0, 2, 'CICC'), (5, 7, 'CCIC')]
a b c d
| + | |
a * c d
-.-.-.-.-.-.-.-.-.-.
a b c d
| | + |
a b * d
"""
