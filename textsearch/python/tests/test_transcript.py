#!/usr/bin/env python3

import unittest

import numpy as np

from textsearch import TextSource, Transcript


class TestTranscript(unittest.TestCase):
    def test_use_utf8(self):
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }
        transcript = Transcript.from_dict(name="test", d=d, use_utf8=True)
        source = TextSource.from_str(name="test", s="".join(text), use_utf8=True)

        assert transcript.binary_text.dtype == np.uint8, transcript.binary_text.dtype
        np.testing.assert_equal(transcript.binary_text, source.binary_text)

        assert transcript.binary_text.size == transcript.times.size, (
            transcript.binary_text.size,
            transcript.time.size,
        )

        last_pos = 0
        for t, time in zip(text, begin_times):
            num_bytes = len(t.encode("utf-8"))
            np.testing.assert_equal(
                transcript.times[last_pos : last_pos + num_bytes], time
            )
            last_pos += num_bytes

    def test_without_using_utf8(self):
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }
        transcript = Transcript.from_dict(name="test", d=d, use_utf8=False)
        source = TextSource.from_str(name="test", s="".join(text), use_utf8=False)

        assert transcript.binary_text.dtype == np.int32, transcript.binary_text.dtype

        np.testing.assert_equal(transcript.binary_text, source.binary_text)

        assert transcript.binary_text.size * 4 == transcript.times.size, (
            transcript.binary_text.size * 4,
            transcript.times.size,
        )

        last_pos = 0
        for t, time in zip(text, begin_times):
            num_bytes = len(t) * 4
            np.testing.assert_equal(
                transcript.times[last_pos : last_pos + num_bytes], time
            )
            last_pos += num_bytes


if __name__ == "__main__":
    unittest.main()
