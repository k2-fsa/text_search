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
            binary_text=np.frombuffer(binary_text, dtype="S1", count=len(binary_text), offset=0),
            pos=pos
        )



def _find_byte_offsets_for_utf8_symbols(binary_text: bytes) -> np.ndarray:
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

    return np.asarray(byte_offsets, dtype=np.int64)

'''
TODO


# class Transcript, let's view this as a not-very-fundamental type that is for automatically
# transcribed text.  The time marks are at the byte level.  We can easily turn automatic
# transcripts of text into this type as long as we keep track of the time for each symbol.
# we don't have to use dataclass, could have it as a regular class for flexibility of
# initialization.
@dataclass
class Transcript:
    name: str  # a filename or some kind of id.
    text: np.ndarray  # the text as a sequence of bytes or (more likely) int32 corresponding to UTF codepoints.
    # We will have to expand BPE tokens into bytes, presumably converting _ into space.
    times: np.ndarray  # gives the time in seconds for each byte of the text.  (Should be in non-decreasing order).


# we'll have a global list of text sources during the program lifetime, and indexes into this list
# (probably int32) will represent the text source.
TextSources = List[Union[TextSource, Transcript]]


@dataclass
class SourcedText:
    # represents a text with some meta-info that records from where in a collection of
    # texts it came.
    text: np.ndarray  # the text, a 1-d array, probably a sequence of bytes but could be uint32 representing utf
    # code points.
    # Note: this text array is 'owned' here so it can be modified by the user.
    pos: np.ndarray  # the text position in the original source for each element of the text.  Of type uint32;
    doc: Union[int, np.ndarray]   # the document index for each element of the text.
    # the np.ndarray would be of type uint32.
    doc_splits: Optional[np.ndarray]  # optional: an array derived from `doc` which is like the row
    # splits in k2 (or in tensorflow ragged arrays), with `doc` being
    # the row-ids.

    sources: TextSource  # for reference, the list of available text sources that this text might come from.


def texts_to_sourced_texts(sources: List[TextSource]) -> List[SourcedText]:
    pass


def append_texts(texts: List[SourcedText]) -> SourcedText:
    pass


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