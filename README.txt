


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
            Returns a suffix array of type np.uint32 (if the size of the input permits)
            or np.uint64 if not, of shape (seq_len,).  This will consist of some permutation of the elements
            0 .. seq_len - 1.
      """

==

@dataclass
class TextSource:
      name: str  # might be a filename
      text: np.ndarray  # the text, probably as a sequence of bytes but could be utf-32.
                        # Length of the array may not be >= 2**32.


# we'll have a global list of text sources during the program lifetime, and indexes into this list
# (probably int32) will represent the text source.
TextSources = List[TextSource]


@dataclass
class SourcedText:
      # represents a text with some meta-info that records from where in a collection of
      # texts it came.
      text: np.ndarray  # the text, a 1-d array, probably a sequence of bytes but could be uint32 representing utf-32.
                        # Note: this text array is 'owned' here so it can be modified by the user.
      pos: np.ndarray  # the text position in the original source for each element of the text.  Of type uint32;
      src: Union[int, np.ndarray]   # the source-text index for each element of the text.
                                    # the np.ndarray would be of type uint32.
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
# languages that use lots of non-ASCII code points, but




==
  The following part should also probably be written in C:

    def find_close_matches(suffix_array: np.ndarray, query_len: int) -> np.ndarray:
       """
       Assuming the suffix array was created from a text where the first `query_len`
       positions represented the query text and the remaining positions represent
       the reference text, return a list indicating the immediately preceding and
       following positions in the reference text for each position in the query text.
       (Note: the query and reference texts could each represent multiple sequences,
       but that is handled by other code.)

       Args:
        suffix_array: A suffix array as created by create_suffix_array(), of dtype
           np.uint32 or np.uint64 and shape (seq_len,).

         query_len: A number 0 <= query_len < seq_len, indicating the length of
           the query part of the text that was used to create `suffix_array`.

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
     def find_candidate_regions(suffix_array: np.ndarray, ...

  [TODO: finish this. uses the output of find_close_matches to locate candidate regions for
  Levenshtein or Smith-Waterman alignment.  This is done separately for each query text.

  Basically we sort the output of find_close_matches and we are looking for pieces of
  source text with a certain maximum length (e.g. reference_chunk_length = 4 times the length of the query text), that
  have the most "hits" in the output of find_close_matches.  basic algorithm would be a linear-time
  thing where we go through the sorted close-matches, always keeping track of the end-position in

  # start_pos, end_pos  are indexes into the 'reference' part
   int start_pos = 0, end_pos = 0;  p

   int j = 0;
   for ( i = 0;  i < num_matches; ..)  {
       # find the current candidate start-position in the reference part of the full text (this is
       # just a couple of array lookups where we find the reference document src-index (i.e. document-index)
       # and the position 'pos' within that reference document.

       # Then we keep advancing j as far as we can while ensuring that the
       # position within that reference document is not more than `reference_chunk_length`
       # greater than that of i and the document-index is not different from that of i.

       # j - i (i.e. the number of matches ) is the metric we are trying to maximize.
       # basically we will keep up to a specified number of 'best' matches, e.g. 5, with the limitation
       # that they cannot overlap (if they overlap, we take the better one).
   }



====
  Next: stuff for Levenshtein or Smith-Waterman alignment.

====
