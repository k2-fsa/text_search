# Note: an alternative name for the package could be "hermes" (Hermes in Greek mythology was the god of searches)
from fasttextsearch import TextSource, Transcript, SourcedText, infix_align

#
# Input data for the script.
#
# This is possibly a several hours long audio recording accompanied by
# a text file that contains the full text content of the recording.
# The text is expected to be:
# - encoded in UTF-8 (i.e. contains non-ASCII symbols),
# - not normalized (i.e., contains punctuation, including non-ASCII punctuation/whitespaces)
# - not a 100% accurate transcription (doesn't account for repetitions/restarts/other disfluencies)
# - not in a specific format (e.g. a line could be a whole chapter or a single sentence)
audio_path = "bible.wav"
text_path = "bible.txt"

#
# The script's output is a single JSONL file with segmentation in a format like the following:
# $ cat bible.jsonl
# {'id': random_id, 'audio': "bible.wav", 'begin_time'="00:00:00:000", 'end_time': "0:00:01:220",'text': "bible.txt", 'begin_byte': 0, 'end_byte': 7}
# {'id': random_id, 'audio': "bible.wav", 'begin_time'="00:00:01:220", 'end_time': "0:00:01:760",'text': "bible.txt", 'begin_byte': 7, 'end_byte': 11}
# ...
out_path = "bible.jsonl"

#
# The ASR model to recognize the text and timestamps
#
asr = Asr(...)

#
# Here we prepare the data structures for the alignment/segmentation task.
#

# Initializing source will read the full text from a file into CPU memory,
# and in case of UTF-8 create and internally store a mapping from UTF character
# position to byte position.
# Note: I wonder if "TextSource" class is really needed; we could simply use
#       paths of the sources and load the text directly into SourcedText.
source: TextSource = TextSource.from_file(text_path)

#
# We don't need to call "append_texts()" because this example has a single source.
# Note: In this example, I converted "texts_to_sourced_texts()"
#       to a member function on "TextSource.to_text()".
text: SourcedText = source.to_text()

# Open the output file for writing.
out_file = open(out_path, "w")

# We have only one file, so we have only one session ID. We just hash the file path to be determinstic.
session_id = md5(audio_path)

# This may or may not be needed, I'll write down some thoughts in the loop below.
alignment_state: Dict[str, Any]

# The loop below is unbatched processing of a single audio file.
# In the future once it's working, it might be batched over multiple files.
# The chunk size is hardcoded at 30 seconds in this example.
# Each chunk is a float32 torch.Tensor with shape (1, chunk_duration * resample_to)
for audio_chunk in read_audio_in_chunks(
    audio_path, chunk_duration=30, chunk_shift=30, resample_to=asr.sampling_rate
):

    # Use ASR to get the recognized text together with symbol (byte?) -level timestamps.
    results: List[Transcript] = asr.decode(audio_chunk)

    # Note: in this example batch_size = 1, so we're taking the first index
    transcript = results[0]

    # The actual processing happens here, and my understanding of this part is still quite fuzzy;
    # I imagine that as a first step, the alignment procedure applies an edit-distance based algorithm
    # and comes up with a list of possible matches of the transcript to some part of
    # the "sourced" text.
    # Then, it has to keep track of what we matched so far, select the best hypothesis,
    # and merge the previous + new ones. This is why I think we will need some sort of alignment/session
    # state variable to keep track of these things (kind of like incremental lattice in ASR).
    # I may be missing something though.
    (partial_alignment_results, alignment_state) = infix_align(
        text, transcript, alignment_state, **align_opts
    )

    # "partial_alignment_results" contains the information we want to store in the output JSONL.
    # It can be a nice data structure, but here for prototyping we just assume a list of tuples.
    for begin_byte, end_byte, begin_time, end_time in partial_alignment_results:
        print(
            {
                "id": session_id,
                "audio_path": audio_path,
                "text_path": text_path,
                "begin_byte": begin_byte,
                "end_byte": end_byte,
                "begin_time": begin_time,
                "end_time": end_time,
            },
            file=out_file
        )

