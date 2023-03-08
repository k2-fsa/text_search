import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union

import numpy as np


@dataclass
class TextSource:
    """
    Represents the full text from a UTF-8 encoded document.
    We store the contents in memory as a numpy byte array,
    together with the start positions of each Unicode symbol contained within.
    """

    # filename or ID
    name: str

    # the text, probably as a sequence of bytes but could be utf-32 i.e. np.int32.
    # Length of the array may not be >= 2**32.
    binary_text: np.ndarray

    # Only used if text is an np.int32: the mapping from
    # UTF character position to byte position.
    pos: Optional[np.ndarray] = None

    @property
    def text(self) -> str:
        """Return Python string representation of self.binary_text decoded as UTF-8."""
        return self.binary_text.tobytes().decode("utf-8")

    @staticmethod
    def from_file(path: Union[str, Path]) -> "TextSource":
        binary_text = Path(path).read_bytes()
        # try to decode it to trigger UnicodeDecodeError when it's not valid UTF-8
        binary_text.decode("utf-8")
        pos = _find_byte_offsets_for_utf8_symbols(binary_text)
        return TextSource(
            name=str(path),
            binary_text=np.frombuffer(binary_text, dtype="S1"),
            pos=pos,
        )

    @staticmethod
    def from_str(name: str, s: str) -> "TextSource":
        binary_text = s.encode("utf-8")
        pos = _find_byte_offsets_for_utf8_symbols(binary_text)
        return TextSource(
            name=name,
            binary_text=np.frombuffer(binary_text, dtype="S1"),
            pos=pos,
        )


def _find_byte_offsets_for_utf8_symbols(binary_text: bytes) -> np.ndarray:
    """
    Args:
      binary_text:
        A 1-D array containing utf-8 encoded characters.
    Returns:
      Return a np.uint32 1-D array. Its length equals to the number of
      characters contained in binary_text before encoding. ans[i]
      contains the byte position in binary_text for the i-th character.
    """
    byte_offsets = []

    # Iterate over each byte in the binary text
    byte_index = 0
    while byte_index < len(binary_text):
        # Determine the length of the UTF-8 symbol starting at the current byte
        byte = binary_text[byte_index]
        if byte < 0x80:
            # Single-byte UTF-8 symbol
            symbol_length = 1
        elif byte < 0xE0:
            # Two-byte UTF-8 symbol
            symbol_length = 2
        elif byte < 0xF0:
            # Three-byte UTF-8 symbol
            symbol_length = 3
        else:
            # Four-byte UTF-8 symbol
            symbol_length = 4

        # Add the byte index to the list of offsets
        byte_offsets.append(byte_index)

        # Move the byte index to the start of the next symbol
        byte_index += symbol_length

    # Note: uint32 should be sufficient for 4GB of text,
    # I don't expect we'd deal with such large documents here.
    return np.asarray(byte_offsets, dtype=np.uint32)


@dataclass
class Transcript:
    """
    Type that represents automatically transcribed text.
    The time marks are at the byte level.  We can easily turn automatic
    transcripts of text into this type as long as we keep track of the time for each symbol.
    """

    # A filename or an ID.
    name: str

    # the text as a sequence of bytes or (more likely) int32 corresponding to UTF codepoints.
    # We will have to expand BPE tokens into bytes, presumably converting _ into space.
    binary_text: np.ndarray

    # Gives the time in seconds for each byte of the text.  (Should be in non-decreasing order).
    times: np.ndarray

    @property
    def text(self) -> str:
        """Return Python string representation of self.binary_text decoded as UTF-8."""
        return self.binary_text.tobytes().decode("utf-8")

    @staticmethod
    def from_dict(name: str, d: dict) -> "Transcript":
        """
        Args:
          name:
            A filename or an ID.
          d:
            A dict containing:

              - d["text"]: List[str]. Each element in the list is a token or a word.

              - d["begin_times]: List[float]. len(d["text"]) == d["begin_times"].
                d["begin_times"][i] is the begin time for d["text"][i]
        """
        assert "text" in d, list(d.keys())
        assert "begin_times" in d, list(d.keys())
        assert len(d["text"]) == len(d["begin_times"]), (
            len(d["text"]),
            len(d["begin_times"]),
            d["text"],
            d["begin_times"],
        )

        bytes_list = []
        times_list = []
        for text, begin_time in zip(d["text"], d["begin_times"]):
            b = text.encode("utf-8")

            if times_list:
                # Check that begin_time is non-decreasing.
                #
                # < here requires that it is strictly increasing.
                assert times_list[-1] < begin_time, (times_list[-1], begin_time)

            # bytes belonging to the same text have the same begin time
            times_list += [begin_time] * len(b)
            bytes_list.append(b)

        return Transcript(
            name=name,
            binary_text=np.frombuffer(b"".join(bytes_list), dtype="S1"),
            times=np.asarray(times_list, dtype=np.float32),
        )


# we'll have a global list of text sources during the program lifetime, and indexes into this list
# (probably int32) will represent the text source.
TextSources = List[Union[TextSource, Transcript]]


@dataclass
class SourcedText:
    """
    Represents a text with some meta-info that records from where in a collection of
    texts it came.

    # TODO(pzelasko): if I understand correctly this class hides whether the text came from TextSource or Transcript
    # TODO(pzelasko): test it
    """

    # A 1-d array, probably a sequence of bytes of UTF-8 encoded text.
    # Note: this text array is 'owned' here so it can be modified by the user.
    binary_text: np.ndarray

    # The text position in the original source for each element of the text. Of type uint32;
    pos: np.ndarray

    # The document index for each element of the text.
    # The np.ndarray would be of type uint32.
    doc: Union[int, np.ndarray]

    # For reference, the list of available text sources that this text might come from.
    sources: List[TextSource]

    # Optional: an array derived from `doc` which is like the row
    # splits in k2 (or in tensorflow ragged arrays), with `doc` being
    # the row-ids.
    doc_splits: Optional[np.ndarray] = None


def texts_to_sourced_texts(sources: List[TextSource]) -> List[SourcedText]:
    # TODO(pzelasko): test it
    return [
        SourcedText(
            binary_text=s.binary_text,
            # TODO(pzelasko): not sure if 'pos' here has the same meaning as in
            #  TextSource (byte offset for each UTF-8 symbol start)
            #  or as suggested by docstring in SourcedText
            #  (text position in the original source for each element of the text).
            #  For now I am assuming the latter but if that's right we should find a new name
            #  to avoid ambiguity.
            pos=np.arange(len(s.binary_text), dtype=np.uint32),
            doc=doc_idx,
            sources=[s],
            doc_splits=None,
        )
        for doc_idx, s in enumerate(sources)
    ]


def append_texts(texts: List[SourcedText]) -> SourcedText:
    # TODO(pzelasko): ensure this is right; my interpretation here is that
    #   the resulting SourcedText is similar to k2 Fsa / ragged tensors
    #   in that it's actually a batch of texts with varying sizes,
    #   and doc + doc_splits let us recover the individual items.
    # TODO(pzelasko): test it
    return SourcedText(
        binary_text=np.concatenate([t.binary_text for t in texts], axis=0),
        pos=np.concatenate([t.pos for t in texts], axis=0),
        doc=np.array(
            [
                item
                for t in texts
                for item in (t.doc if isinstance(t.doc, np.ndarray) else [t.doc])
            ]
        ),
        sources=[s for t in texts for s in t.sources],
        doc_splits=np.cumsum([0] + [t.binary_text.shape[0] for t in texts[:-1]]),
    )


'''
TODO
def remove(t: SourcedText, keep: np.ndarray) -> SourcedText:
    """
    Removes some positions from a SourcedText (out-of-place).
    Args:
        t: the text to remove some positions of
        keep: an np.ndarray with dtype == np.bool, that is True
          for positions that should be kept; must have the same shape
          as t.text.
    Returns:
        A SourcedText with some positions removed.
    """
    pass

'''
