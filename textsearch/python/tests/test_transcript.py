#!/usr/bin/env python3


import unittest

import numpy as np
from textsearch import Transcript, TextSource


class TestTranscript(unittest.TestCase):
    def test(self):
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }
        transcript = Transcript.from_dict(name="test", d=d)
        source = TextSource.from_str(name="test", s="".join(text))

        np.testing.assert_equal(transcript.binary_text, source.binary_text)

        last_pos = 0
        for t, time in zip(text, begin_times):
            num_bytes = len(t.encode("utf-8"))
            np.testing.assert_equal(
                transcript.times[last_pos : last_pos + num_bytes], time
            )
            last_pos += num_bytes


if __name__ == "__main__":
    unittest.main()
