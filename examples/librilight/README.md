## Download data

### Audio

You can download librilight audios here : https://github.com/facebookresearch/libri-light/tree/main/data_preparation#1a-downloading

It has three subsets, small (577 hours), medium (5193 hours), large (51934 hours).


### Texts

For each audio in librilight, there is a `json` manifest containing the librivox link
of current audio, you can download the text book via this link. We provide a eaxmple
scirpt `download_text.py`, it is not perfect though, you might need to change some
code during the downloading. We will try to give a fully automatic downloading script
later.


## Prepare manifests

**HINT:** You don't have to run this stage separately, it is already included in the next stage, the following documents just shows you
the details of this stage.

Use the lhotse changes in this branch: https://github.com/yfyeung/lhotse/tree/librilight

Run the command below:

```
lhotse prepare librilight audio-dir book-dir output-dir -j 16
```

The audio-dir is the audios directory, each subdirectory contains a subset of data, it looks like:
```
.
|-- large
|-- medium
`-- small
```

The book-dir is the books directory, it has three subdirectories, each for a subset of librilight, it also
has three json files indicating the relationship between recordings and books.

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

This stage generates two manifests, librilight_suppervisons_{small,medium,large}.jsonl.gz, librilight_recording_{small,medium,large}.jsonl.gz


The recording one looks like (only one line of it):

```
{"id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb", "sources": [{"type": "file", "channels": [0], "source": "/star-kw/data/libri-light/small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb.flac"}], "sampling_rate": 16000, "num_samples": 9567080, "duration": 597.9425, "channel_ids": [0]}
```

The supervision one looks like:
```
{"id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb", "recording_id": "small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb","start": 0.0, "duration": 597.9425, "channel": 0, "language": "English", "speaker": "100", "custom": {"book": "/ceph-data3/xiaoyu/librilight_text/output_text_small_cleaned/Sea Fairies/text.txt"}}
```


## Decode the audios

This stage decodes the audios to texts with a pre-trained ASR model.
Firstly split the long audio into smaller pieces (for eaxmple 30 seconds), then decode these pieces of audios to texts, combine them together at last.

Code is available here: https://github.com/k2-fsa/icefall/pull/980

You can run the whole pipeline with the script long_file_recog.sh

**Note:** The whole pipeline includes the stages to prepare raw manifests in the above stage (stage 1 in long_file_recog.sh).

It will generate a manifest (including the transcripted text and timestamps).


## Align the decoded texts to the reference books

This stage aligns the transcripted texts to its reference books.

First, you have to install the text_search library (https://github.com/danpovey/text_search),
then run the following command:

```
python examples/librilight/matching.py --manifest-in path/to/librilight_cuts_small.jsonl.gz  --manifest-out path/to/cuts_small.jsonl.gz --batch-size 50 --num-workers 4
```

The manifest-in is the manifests generated in the decode audios stage.

Note: You have to use the changes in https://github.com/danpovey/text_search/pull/27










