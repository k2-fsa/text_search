#!/usr/bin/env python3

import numpy as np
import textsearch

query = "hi"
document = "howareyou"
full_text = np.fromstring(query + document, dtype=np.int8)

suffix_array = textsearch.create_suffix_array(full_text)

close_matches = textsearch.find_close_matches(
    suffix_array=suffix_array,
    query_len=len(query),
)

print("n\t\tpos\t\ttype\t\tsubstring")
print("-" * 65)
for i in range(suffix_array.size - 2):
    t = "query" if suffix_array[i] < len(query) else "document"
    sub = full_text[suffix_array[i] :].tobytes().decode("utf-8")
    print(i, suffix_array[i], t, sub, sep="\t\t")

print(close_matches)

"""
The output is:

n               pos             type            substring
-----------------------------------------------------------------
0               5               document                areyou
1               7               document                eyou
2               0               query           hihowareyou
3               2               document                howareyou
4               1               query           ihowareyou
5               9               document                ou
6               3               document                owareyou
7               6               document                reyou
8               10              document                u
9               4               document                wareyou
[[7 2]
 [2 9]]
"""
