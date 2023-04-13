#!/usr/bin/env python3

import unittest

import numpy as np

from textsearch import SourcedText
from textsearch import TextSource
from textsearch import Transcript
from textsearch import append_texts
from textsearch import filter_texts
from textsearch import texts_to_sourced_texts


class TestSourcedText(unittest.TestCase):
    def test_from_text_source_utf8(self):
        """Test constructing a SourcedText from a TextSource using utf-8"""
        name = "test_text_source_utf8"
        s = "zażółć gęślą jaźń\n你好Hallo"
        text_source = TextSource.from_str(name=name, s=s, use_utf8=True)

        sourced_text = texts_to_sourced_texts([text_source])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(text_source.binary_text.size)
        )
        np.testing.assert_equal(sourced_text[0].doc,np.zeros(text_source.binary_text.size))
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is text_source

    def test_from_text_source_unicode_code_point(self):
        """Test constructing a SourcedText from a TextSource using Unicode code point"""
        name = "test_text_source_unicode_code_point"
        s = "zażółć gęślą jaźń\n你好Hallo"
        text_source = TextSource.from_str(name=name, s=s, use_utf8=False)

        sourced_text = texts_to_sourced_texts([text_source])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(text_source.binary_text.size)
        )
        np.testing.assert_equal(sourced_text[0].doc, np.zeros(text_source.binary_text.size))
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is text_source

    def test_from_transcript_utf8(self):
        """Test constructing a SourcedText from a Transcript using utf-8"""
        name = "test_from_transcript_utf8"
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }

        transcript = Transcript.from_dict(name=name, d=d, use_utf8=True)

        sourced_text = texts_to_sourced_texts([transcript])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(transcript.binary_text.size)
        )

        np.testing.assert_equal( sourced_text[0].doc, np.zeros(transcript.binary_text.size))
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is transcript

    def test_from_transcript_unicode_code_point(self):
        """Test constructing a SourcedText from a Transcript using Unicode code point"""
        name = "test_from_transcript_unicode_code_point"
        text = "▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe".split()
        begin_times = []
        for i in range(len(text)):
            begin_times.append(i * 0.25)
        d = {
            "text": text,
            "begin_times": begin_times,
        }

        transcript = Transcript.from_dict(name=name, d=d, use_utf8=False)

        sourced_text = texts_to_sourced_texts([transcript])
        assert len(sourced_text) == 1, len(sourced_text)
        assert sourced_text[0].pos.dtype == np.uint32, sourced_text[0].pos.dtype
        np.testing.assert_equal(
            sourced_text[0].pos, np.arange(transcript.binary_text.size)
        )
        np.testing.assert_equal(sourced_text[0].doc, np.zeros(transcript.binary_text.size))
        assert len(sourced_text[0].sources) == 1, len(sourced_text[0].sources)
        assert sourced_text[0].sources[0] is transcript

    def test_append_texts_text_source_utf8(self):
        name0 = "test_append_texts_text_source_utf8_0"
        s0 = "zażółć gęślą jaźń\n"
        text_source0 = TextSource.from_str(name=name0, s=s0, use_utf8=True)

        name1 = "test_append_texts_text_source_utf8_1"
        s1 = "你好Hallo"
        text_source1 = TextSource.from_str(name=name1, s=s1, use_utf8=True)

        sourced_text0 = texts_to_sourced_texts([text_source0])
        sourced_text1 = texts_to_sourced_texts([text_source1])
        sourced_text = append_texts([sourced_text0[0], sourced_text1[0]])

        # check binary_text
        expected_binary_text = np.concatenate(
            [text_source0.binary_text, text_source1.binary_text]
        )
        np.testing.assert_equal(sourced_text.binary_text, expected_binary_text), (
            sourced_text.binary_text,
            expected_binary_text,
        )
        assert (
            sourced_text.binary_text.dtype == np.uint8
        ), sourced_text.binary_text.dtype

        # check pos
        expected_pos0 = np.arange(text_source0.binary_text.size, dtype=np.uint32)
        expected_pos1 = np.arange(text_source1.binary_text.size, dtype=np.uint32)
        expected_pos = np.concatenate([expected_pos0, expected_pos1])
        np.testing.assert_equal(sourced_text.pos, expected_pos), (
            sourced_text.pos,
            expected_pos,
        )
        assert sourced_text.pos.dtype == np.uint32, sourced_text.pos.dtype

        # check doc
        expected_doc0 = np.zeros(text_source0.binary_text.size, dtype=np.uint32)
        expected_doc1 = np.ones(text_source1.binary_text.size, dtype=np.uint32)
        expected_doc = np.concatenate([expected_doc0, expected_doc1])
        np.testing.assert_equal(sourced_text.doc, expected_doc), (
            sourced_text.doc,
            expected_doc,
        )
        assert sourced_text.doc.dtype == np.uint32, sourced_text.doc.dtype

        # check sources
        assert len(sourced_text.sources) == 2, len(sourced_text.sources)
        assert sourced_text.sources[0] is text_source0
        assert sourced_text.sources[1] is text_source1

        expected_doc_splits = np.array(
            [
                0,
                text_source0.binary_text.size,
                text_source0.binary_text.size + text_source1.binary_text.size,
            ],
            dtype=np.uint32,
        )
        np.testing.assert_equal(sourced_text.doc_splits, expected_doc_splits)
        assert sourced_text.doc_splits.dtype == np.uint32, sourced_text.doc_splits.dtype

    def test_append_texts_text_source_unicode_code_point(self):
        name0 = "test_append_texts_text_source_unicode_code_point_0"
        s0 = "zażółć gęślą jaźń\n"
        text_source0 = TextSource.from_str(name=name0, s=s0, use_utf8=False)

        name1 = "test_append_texts_text_source_unicode_code_point_1"
        s1 = "你好Hallo"
        text_source1 = TextSource.from_str(name=name1, s=s1, use_utf8=False)

        sourced_text0 = texts_to_sourced_texts([text_source0])
        sourced_text1 = texts_to_sourced_texts([text_source1])
        sourced_text = append_texts([sourced_text0[0], sourced_text1[0]])

        # check binary_text
        expected_binary_text = np.concatenate(
            [text_source0.binary_text, text_source1.binary_text]
        )
        np.testing.assert_equal(sourced_text.binary_text, expected_binary_text), (
            sourced_text.binary_text,
            expected_binary_text,
        )
        assert (
            sourced_text.binary_text.dtype == np.int32
        ), sourced_text.binary_text.dtype

        # check pos
        expected_pos0 = np.arange(text_source0.binary_text.size, dtype=np.uint32)
        expected_pos1 = np.arange(text_source1.binary_text.size, dtype=np.uint32)
        expected_pos = np.concatenate([expected_pos0, expected_pos1])
        np.testing.assert_equal(sourced_text.pos, expected_pos), (
            sourced_text.pos,
            expected_pos,
        )
        assert sourced_text.pos.dtype == np.uint32, sourced_text.pos.dtype

        # check doc
        expected_doc0 = np.zeros(text_source0.binary_text.size, dtype=np.uint32)
        expected_doc1 = np.ones(text_source1.binary_text.size, dtype=np.uint32)
        expected_doc = np.concatenate([expected_doc0, expected_doc1])
        np.testing.assert_equal(sourced_text.doc, expected_doc), (
            sourced_text.doc,
            expected_doc,
        )
        assert sourced_text.doc.dtype == np.uint32, sourced_text.doc.dtype

        # check sources
        assert len(sourced_text.sources) == 2, len(sourced_text.sources)
        assert sourced_text.sources[0] is text_source0
        assert sourced_text.sources[1] is text_source1

        expected_doc_splits = np.array(
            [
                0,
                text_source0.binary_text.size,
                text_source0.binary_text.size + text_source1.binary_text.size,
            ],
            dtype=np.uint32,
        )
        np.testing.assert_equal(sourced_text.doc_splits, expected_doc_splits)
        assert sourced_text.doc_splits.dtype == np.uint32, sourced_text.doc_splits.dtype

    def test_filter_texts(self):
        name0 = "test_filter_texts_0"
        s0 = "Higher, faster, stronger, together"
        text_source0 = TextSource.from_str(name=name0, s=s0, use_utf8=False)

        sourced_text0 = texts_to_sourced_texts([text_source0])
        sourced_text = append_texts([sourced_text0[0]])

        # test keep
        keep = np.array([True if x != "," else False for x in s0], dtype=bool)
        new_sourced_text = filter_texts(sourced_text, keep=keep)

        new_s0 = "Higher faster stronger together"
        new_text_source0 = TextSource.from_str(name=name0, s=new_s0, use_utf8=False)
        np.testing.assert_equal(
            new_sourced_text.binary_text, new_text_source0.binary_text
        )

        new_pos_list = list(range(len(s0)))
        new_pos_list.remove(6)
        new_pos_list.remove(14)
        new_pos_list.remove(24)
        new_pos = np.array(new_pos_list, dtype=np.uint32)
        np.testing.assert_equal(new_sourced_text.pos, new_pos)

        # test fn
        fn = lambda x: x != ord(",")
        new_sourced_text = filter_texts(sourced_text, fn=fn)
        np.testing.assert_equal(
            new_sourced_text.binary_text, new_text_source0.binary_text
        )
        np.testing.assert_equal(new_sourced_text.pos, new_pos)


if __name__ == "__main__":
    unittest.main()
