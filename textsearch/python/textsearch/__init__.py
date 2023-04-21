from _fasttextsearch import find_close_matches
from _fasttextsearch import levenshtein_distance

from .datatypes import SourcedText
from .datatypes import TextSource
from .datatypes import TextSources
from .datatypes import Transcript
from .datatypes import append_texts
from .datatypes import filter_texts
from .datatypes import texts_to_sourced_texts

from .levenshtein import get_nice_alignments

from .match import get_alignments
from .match import get_longest_increasing_pairs
from .match import find_candidate_matches
from .match import split_into_segments

from .suffix_array import create_suffix_array

from .utils import AttributeDict
from .utils import is_punctuation
from .utils import row_ids_to_row_splits
