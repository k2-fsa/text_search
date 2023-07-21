## Download data

### Audio

You can download librilight audios here : https://github.com/facebookresearch/libri-light/tree/main/data_preparation#1a-downloading

It has three subsets, small (577 hours), medium (5193 hours), large (51934 hours).


### Texts

For each audio in librilight, there is a `json` manifest containing the librivox link
of current audio, you can download the text book via this link. We provide several
scripts in `scripts` directory showing how to get the cleaned texts.

```
.
|-- download_text.py    # download raw text via librivox link.
|-- clean.py            # clean the downloaded raw text, for example, removing html tags
`-- recording2book.py   # map audio to text
```

Caution: All the three scripts above are not ready for using, data downloading and
cleaning are very hassle, we changed the scripts during the process. We will try
to give a fully automatic downloading script later.

If you don't want to download and clean the data by yourself, we provide a cleaned
version of texts data which is available at https://huggingface.co/datasets/pkufool/librilight-text

The download command:

```
git lfs install
git clone https://huggingface.co/datasets/pkufool/librilight-text
```

> We provide a shell script `run.sh` to run all the following stages step by step.

## Prepare manifests (stage 1 in run.sh)

Note: You need to install [lhotse](https://github.com/lhotse-speech/lhotse) to prepare manifests.

You can do it as follows:

```
pip install lhotse
```

To prepare the manifests, run the following command:

```
python prepare_manifest.py \
    --corpus-dir CORPUS_DIR \
    --books-dir BOOKS_DIR \
    --output-dir OUTPUT-DIR \
    --num-jobs 16
```

The `CORPUS_DIR` is the audios directory, each subdirectory contains a subset of data, it looks like:
```
.
|-- large
|-- medium
`-- small
```

The `BOOKS_DIR` is the books directory, it has three subdirectories, each for a subset of librilight, it also
has three json files indicating the relationship between audios and books.

```
.
|-- output_text_large_cleaned
|-- output_text_medium_cleaned
|-- output_text_small_cleaned
|-- recording2book_large.json
|-- recording2book_medium.json
`-- recording2book_small.json
```

The recording2book_{small,medium,large}.json look like this {recordingid: book_path}:

```
{
    "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb": "output_text_small_cleaned/Sea Fairies/text.txt",
    "small/100/sea_fairies_0812_librivox_64kb_mp3/02_baum_sea_fairies_64kb": "output_text_small_cleaned/Sea Fairies/text.txt",
    "small/100/sea_fairies_0812_librivox_64kb_mp3/03_baum_sea_fairies_64kb": "output_text_small_cleaned/Sea Fairies/text.txt",
    "small/100/sea_fairies_0812_librivox_64kb_mp3/22_baum_sea_fairies_64kb": "output_text_small_cleaned/Sea Fairies/text.txt",
    "small/103/im_nobody_librivox_64kb_mp3/im_nobody_dickinson_krs_64kb": "output_text_small_cleaned/Im Nobody/text.txt",
    "small/103/nippon_libriviox_64kb_mp3/nippon_noyes_krs_64kb": "output_text_small_cleaned/Nippon/text.txt"
}
```

This stage generates three manifests, librilight_cut_{small,medium,large}.jsonl.gz

The cuts look like this (only one line of it):

```
{"id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb", "start": 0.0, "duration": 597.9425, "channel": 0, "supervisions": [{"id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb", "recording_id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb", "start": 0.0, "duration": 597.9425, "channel": 0, "language": "English", "speaker": "100"}], "recording": {"id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb", "sources": [{"type": "file", "channels": [0], "source": "/star-kw/data/libri-light/small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb.flac"}], "sampling_rate": 16000, "num_samples": 9567080, "duration": 597.9425, "channel_ids": [0]}, "custom": {"text_path": "/ceph-data3/xiaoyu/librilight_text/output_text_small_cleaned/Sea Fairies/text.txt"}, "type": "MonoCut"}
```


## Decode the audios (stage 2,3,4 in run.sh)

This stage decodes the audios to texts with a pre-trained ASR model.
We will firstly split the long audio into smaller pieces (for eaxmple 30 seconds), then decode these pieces of audios to texts, combine them together at last.

### Split

```
./tools/split_into_chunks.py \
  --manifest-in path/to/input_manifest \
  --manifest-out path/to/output_manifest \
  --chunk 30 \
  --extra 2  # Extra duration (in seconds) at both sides
```
The input_manifest is the output of previous stage.

### Transcription

```
./tools/recognize.py \
  --world-size 4 \
  --num-workers 8 \
  --manifest-in path/to/input_manifest \
  --manifest-out path/to/output_manifest \
  --nn-model-filename path/to/jit_script.pt \
  --tokens path/to/tokens.txt \
  --max-duration 2400 \
  --decoding-method greedy_search \
  --master 12345
```
The input_manifest is the output of previous stage.

### Combine

```
./tools/merge_chunks.py \
  --manifest-in path/to/input_manifest \
  --manifest-out path/to/output_manifest \
  --extra 2  # should be the same as in split stage
```
The input_manifest is the output of previous stage.

It will generate a manifest (including the transcripted text and timestamps).


## Align the decoded texts to the reference books (stage 5 in run.sh)

This stage aligns the transcripted texts to its reference books.

First, you have to install the text_search library (https://github.com/k2-fsa/text_search),
then run the following command:

```
python examples/librilight/matching.py \
    --manifest-in path/to/librilight_cuts_small.jsonl.gz \
    --manifest-out path/to/cuts_small.jsonl.gz \
    --batch-size 50
```

Or the parallel version:

```
python examples/librilight/matching_parallel.py \
    --manifest-in path/to/librilight_cuts_small.jsonl.gz \
    --manifest-out path/to/cuts_small.jsonl.gz \
    --batch-size 50 \
    --num-workers 5
```

The manifest-in is the manifests generated in the previous stage.


