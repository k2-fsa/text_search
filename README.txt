


  Plan for text searching.  I propose to create a standalone Python package for this that could have
  multiple usage.  e.g. we can call the package fasttextsearch; it would have a C++ extension, we could
  perhaps use lilcom as the starting point in terms of package structure (unless someone has another idea);
  I am thinking we could make numpy a dependency, and lilcom already handles viewing Numpy arrays
  as C arrays.

  The immediate problem this package would solve is how to locate the reference text for a piece of
  a file that we decoded, when we have the reference text as a long piece of text with no correspondence.

  The core part of the package - the part in C - would be some code for creating and manipulating
  suffix arrays.  (For flexibility, at the "C" level we should use templates to support int8/char, int16 and int32 for
  the symbol type, and int32 and int64 for the index type.)

  The most technical part, for creating the suffix array (written in C) would look like this from
  its Python interface (be careful, it is a little quirky to save a copy).
  There is some code in k2 (see nbest.h) that we can just copy for the internals of this, which has already been tested.
  In fact we don't really need it in k2 so we could remove it from there at some point.
  (this project has other potential users from k2, and I want to separate it as k2 is always going to
  be harder to install).




   def create_suffix_array(input: np.ndarray) -> np.ndarray:
      """
      Creates a suffix array from the input text and returns it as a NumPy array.  Read
      the usage carefully as it has some special requirements that will require careful data
      preparation.

      Args:
         input: an np.ndarray that must be one types uint8, uint16 or uint32.  Its shape
            should be (seq_len + 3,) where `seq_len` is the text sequence length INCLUDING
            EOS SYMBOL.
            The EOS (end of sequence) symbol must be the largest element of the
            type (i.e. of the form 2^n - 1), must be located at input[seq_len - 1] and
            must appear nowhere else in `input` (you may have to map the input
            symbols somehow to achieve this).  It must be followed by 3 zeros, for reasons
            related to how the algorithm works.
      Returns:
            Returns a suffix array of type np.uint64,
            of shape (seq_len,).  This will consist of some permutation of the elements
            0 .. seq_len - 1.
      """

==

# we might make this a regular class, as initialization would be non-trivial if
# UTF-8 decoding is required since we have to figure out the file position.
# [Note: I'd like this to be robust to UTF decoding errors if possible, but
# in the short term this might not be necessary.  What I don't want is that
# we silently ignore UTF errors that could cause mismatches in file position
# between the recorded and actual ones.]
@dataclass
class TextSource:
      name: str  # might be a filename
      text: np.ndarray  # the text, probably as a sequence of bytes but could be utf-32 i.e. np.int32.
                        # Length of the array may not be >= 2**32.
      pos: Optional[np.ndarray]  # Only used if text is an np.int32: the mapping from
                                 # UTF character position to byte position.

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


# note: for things like mapping bytes to e.g. lower-case or finding the positions to remove()
# because of, for example, repeated spaces or punctuation, the user can do this manually
# using normal Python and numpy operations.  This might get a little more complicated for
# languages that use lots of non-ASCII code points; should be doable though.
#
#

#




==
  The following part should also probably be written in C:

    def find_close_matches(suffix_array: np.ndarray, query_len: int) -> np.ndarray:
       """
       Assuming the suffix array was created from a text where the first `query_len`
       positions represented the query text and the remaining positions represent
       the reference text, return a list indicating, for each suffix position in the query
       text, the two suffix positions in the reference text that immediately precede and
       follow it lexicographically.  (I think suffix position refers to the last character
       of a suffix).     This is easy to do from the suffix array without computing,
       for example, the LCP array; and it produces exactly 2 matches per position in the
       query text, which is also convenient.

       (Note: the query and reference texts could each represent multiple separate
       sequences, but that is handled by other code; class SourcedText keeps track of that
       information.)

       Args:
        suffix_array: A suffix array as created by create_suffix_array(), of dtype
           np.uint32 or np.uint64 and shape (seq_len,).

         query_len: A number 0 <= query_len < seq_len, indicating the length in symbols
          (likely bytes) of the query part of the text that was used to create `suffix_array`.

       Returns an np.ndarray of shape (query_len * 2,), of the same dtype as suffix_array,
         in which positions 2*i and 2*i + 1 represent the two positions in the original
         text that are within the reference portion, and which immediately follow and
         precede, in the suffix array, query position i.  This means that the
         suffixes ending at those positions are reverse-lexicographically close
         to the suffix ending at position i.  As a special case, if one of these
         returned numbers would equal the EOS position (position seq_len - 1), or
         if a query position is before any reference position in the suffix aray, we
         output seq_len - 2 instead to avoid having to handle special cases later on
         (anyway, these would not represent a close match).
       """
       # This can be accomplished in a linear pass over the suffix array, e.g. (ignoring
       # edge cases):
       #   last_ref_pos = -1;
       #   for (int i = 0; i < seq_len; i++) {
       #      text_pos = suffix_array[i];
       #      if (text_pos > query_len) {  // reference position.
       #         for (int j = last_ref_pos + 1; j < i; j++) {
       #            int cur_ref_pos = text_pos;
       #            query_pos = suffix_array[j + 1];
       #            output[2 * query_pos] = last_ref_pos;
       #            output[2 * query_pos + 1] = cur_ref_pos;
       #            last_ref_pos = cur_ref_pos;
       #         }
       #      }
       #   (


 ===
     def find_candidate_matches(close_matches: np.ndarray,
                                text: SourcedText,
                                length_ratio: float = 2.0,
                                num_candidates: int = 5) -> List[List[Tuple[int, int]]]:
       """
       Find candidate regions in reference document that could be good matches for
       each query document.

       Args:
          close_matches: an np.ndarray of shape (2*tot_query_symbols,) as returned by
             find_close_matches(), indicating two close matches within the reference
             text for each symbol in the query documents.
          text:  The SourcedText corresponding to the query and reference documents
             combined; needed for its `doc` member which can be assumed to be an np.ndarray.
          length_ratio: indicates the maximum candidate-region length, which will be reduced
             if the matching reference document was shorter than this.
           num_candidates:  the number of candidate regions to find for each query
             document.
       Returns:
          A list, one per query document, of close matches, where each close match
          is a (begin, end) position within `text`.
       """

  [TODO: finish this. uses the output of find_close_matches to locate candidate matches for
  Levenshtein or Smith-Waterman alignment.  This is done separately for each query text.

  We process pieces of `close_matches` separately for each query document (the number of query
  documents can be found from the size of close_matches and from text.doc).

  It will be helpful to go through the elements of
  'text.doc' and accumulate something equivalent to k2's "row_splits" (maybe just
  for the query documents); this can be used
  for a few things in this function, such as iterating over the query documents

  We are looking for the up-to-`num_candidates` pieces of reference text of length at most
  (this_query_len * length_ratio) that have the largest number of "hits" in the output of
  find_close_matches, subject to a "non-overlapping" constraint.  This may not be formally
  very well-defined when we consider the "non-overlapping" constraint but approximate
  solutions are OK.

  The basic algorithm would be a linear-time thing where, for each query
  document, we go through the sorted close-matches, always keeping track of the next element
  of the sorted close-matches that is no more than


   for (int q = 0; q < num_query_docs; ++q) {
      matches_start_pos = 2 * row_splits[q];
      matches_end_pos = 2 * row_splits[q+1];


      int j = matches_start_pos;
      for (i = matches_start_pos;  i < matches_end_pos; ++i) {
       # find the current candidate start-position in the reference part of the full text (this is
       # just a couple of array lookups where we find the reference document index
       # and the position 'pos' within that reference document.

       # Then we keep advancing j as far as we can while ensuring that the
       # position within that reference document is not more than `reference_chunk_length`
       # greater than the position for i and the document-index is not different from that of i.

       # j - i (i.e. the number of matches ) is the metric we are trying to maximize.
       # basically we will keep up to a specified number of 'best' matches, e.g. 5, with the limitation
       # that they cannot overlap (if they overlap, we take the better one).  no need for fancy
       # algorithms here, we can just assume num_candidates is small and use iteration over
       # that number.
    }

 If the core part of this is in C++, it might be easiest to format the output as an ndarray conceptually
 of shape (num_query_docs, num_candidates, 2) which can probably be just treated as a uint64_t*
 within the C++ code.



======
  Next: stuff for Levenshtein or Smith-Waterman alignment.

   (We should give this alignment algorithm as a generic an interface as possible because we might
    want to use it in a few different contexts.)

   Once we have the candidate matches:
     We extend them slightly to the left and right, e.g. by 10 + query_length // 8, this can be done
     at a fairly outer level of the code.

     We match each of them with the query text using a Levenshtein-like alignment process that
     allows un-penalized deletions of reference text at the very beginning, or very end, of
     the query text.   (easy to do: just initialize with zeros along one side of the array, and
     once all scores are computed, take minimum from other side of the array).
     This will return the alignment and the corresponding score.
     The score can be used to find the best-matching region, and the alignment (probably
     pairs of: ref-position, query-position) can be used to find the beginning and end
     position in the reference text.  (I think we can just convert the alignments from
     np.ndarray to python objects and use Python code to find the beginning/end positions,
     since this part is linear time).


 The output that we'll be writing to disk would then be, for each query text:
    (query_length, num_errs, reference_name, reference_first_byte, reference_last_byte).

 To find the reference_first_byte and reference_last_byte, we will use the "pos" elements
 in the SourcedText objects.  This is why we added the "pos" elements, because we need
 to backtrack through the normalization process, which would otherwise be nontrivial.


=====
 A note on the larger plan:

  You might be worried about text-normalization issues, and how we will align things like digits
  that may look very diffrent between normalized and un-normalized text, i.e. between our automatic
  transcript and the reference text.
  What I am planning is that most of the time,  the query texts will be fairly long, like 30 seconds
  or a minute, and even if there are lots of digits internally, if there are words at the start and end
  we should still find the right match in the reference text.  Then we can later use algorithms
  involving the acoustics to do further cleanup of the reference text.


====
(I mentioned edlib in an issue, we can use edlib for the levenshtein code as it's already optimized.)


The alignment from edlib will, it seems consist of:
The (start, end) position within the reference, and an array of

  (0, 1, 2, 3) meaning (match, insert, delete, mismatch).


The following is how we can turn this into a segmentation.  (Note: the query
document is assumed here to be a long recording, not an already-segmented
one.)

We can work out the positions in query and ref documents corresponding
to each position in this alignment.  By "position" here, I refer to an index
into the 'text' array of the TextSource or Transcript.

Then I suggest that we create scores for each position in the alignment,
corresponding to how good a position it is to begin or end a split.
[Note: it might be better to formulate this as a penalty, you decide.]

 - begin a split (i.e. this is first position in a segment)
    - plus score equal to log(num silence frames this
      follows, up to some limit like 4.0 sec), i.e. this element
      of the Transcript's time minus the previous element's time; or some default (4.0 sec)
      if this is the first element of the Transcript.
    - good if there are few errors around this point in the alignment, i.e.
      score corresponding to number of ins,del,mismatch within a certain
      region of this position.
    - good if this reference position follows a whitespace character.
    - good if the previous non-whitespace character was a punctuation
      character.  (Lists of whitespace and punctuation characters can probably
      be passed in, or we can use some kind of isspace for utf-8.).

 - end a split (i.e. this is the last position in a segment).
    - good if more silence follows this position.
    - good if there are few errors around this point in the alignment, i.e.
      score corresponding to number of ins,del,mismatch within a certain
      region of this position.
    - good if this reference position precedes a whitespace character.
    - good if this position is a punctuation character.

Each split-begin and split-end position (corresponding to each position in the
alignment) can be assigned its time position, i.e. what time would we use as the
start-time or end-time if we (began, ended) the split here.  This can be
the midpoint between this symbol and the (previous or next) symbol, but up
to some maximum limit of how much silence we allow, e.g. 1 second; we can maybe
reduce it later if we need.

We can then create a rule to assign scores to potential segments.    This would
consist of the begin-scores, plus the end-scores, plus:
  - Some kind of penalty related to the duration of the segment, e.g.
    infinity if it's over some max-duration like 30 seconds or less than a
    min-duration like 2 seconds; else, one that encourages a duration between
    5 to 20 seconds.
  - A penalty against long silences inside segments, e.g. compute the
    longest internal silence and add increasing penalty if it exceeds 2 seconds.
  - A bonus for the number of matches in the alignment.
  - A penalty for the number of errors in the alignment (could multiply this by
    some scale depending how much we don't want to have errors, but some errors
    are expected due to both ASR errors and normalization differences.)

Next, we can do a search for a good segmentation.  You could define the problem
as getting the highest-scoring set of segments that do not overlap.  One
possible way to do it is as follows:
   For each begin_position in the top 10% of scores, find the 4 best-scoring end_positions
   For each end_position in the top 10% of scores, find the 4 best-scoring begin_positions
Append the preceding 2 sets of segments to get a list of candidate segments.

Sort the candidate segments by goodness and choose them from the best to the last,
always excluding choices that would overlap with already-chosen segments.

When we output the segments we can probably over-specify the information a little
bit, e.g. as json files, in case we need to know stuff later, but the
ultimate format of the segments will probably be something like this, e.g.
as a json file (we can later turn it into a csv if that's convenient):

 - {'id': random_id, 'audio': audio_filename, 'begin_time'=begin_time, 'end_time',
   'text': text_filename, 'begin_byte': begin_byte, 'end_byte': end_byte}
[end_byte would be last-byte-plus-one.]
