#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export CUDA_VISIBLE_DEVICES="0,1,2,3"

set -eou pipefail

# This script is used to recogize long audios. The process is as follows:
# 1) Split long audios into chunks with overlaps.
# 2) Perform speech recognition on chunks, getting tokens and timestamps.
# 3) Merge the overlapped chunks into utterances acording to the timestamps.

# Each chunk (except the first and the last) is padded with extra left side and right side.
# The chunk length is: left_side + chunk_size + right_side.
chunk=30.0
extra=2.0

stage=1
stop_stage=5

# We assume that you have downloaded the LibriLight dataset
# with audio files in $corpus_dir and texts in $text_dir
# The corpus_dir looks like:
# .
# |-- large
# |-- medium
# `-- small
#
# The text_dir looks like:
# .
# |-- output_text_large_cleaned
# |-- output_text_medium_cleaned
# |-- output_text_small_cleaned
# |-- recording2book_large.json
# |-- recording2book_medium.json
# `-- recording2book_small.json

corpus_dir=download/libri-light
text_dir=download/librilight_text
# Path to save the manifests
output_dir=data

. parse_options.sh || exit 1

world_size=4

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # We will get librilight_raw_cuts_{subset}.jsonl.gz
  # saved in $output_dir/manifests
  log "Stage 1: Prepare LibriLight manifest"
  python prepare_manifest.py \
    --corpus-dir $corpus_dir \
    --books-dir $text_dir \
    --output-dir $output_dir/manifests \
    --num-jobs 10
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  # We will get librilight_chunk_cuts_{subset}.jsonl.gz
  # saved in $output_dir/manifests
  log "Stage 2: Split long audio into chunks"
  for subset in small medium large; do
      ./tools/split_into_chunks.py \
        --manifest-in $output_dir/manifests/librilight_raw_cuts_${subset}.jsonl.gz \
        --manifest-out $output_dir/manifests/librilight_chunk_cuts_${subset}.jsonl.gz \
        --chunk $chunk \
        --extra $extra  # Extra duration (in seconds) at both sides
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  # This script loads torchscript models, exported by `torch.jit.script()`,
  # and uses it to decode waves.
  # You can download the jit model from
  # https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15

  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
  cd icefall-asr-librispeech-zipformer-2023-05-15
  git lfs pull --include "exp/jit_script.pt"
  cd ..
  ln -s icefall-asr-librispeech-zipformer-2023-05-15 exp

  # We will get librilight_asr_cuts_{subset}.jsonl.gz
  # saved in $output_dir/manifests
  log "Stage 3: Perform speech recognition on splitted chunks"
  for subset in small medium large; do
    ./tools/recognize.py \
      --world-size $world_size \
      --num-workers 8 \
      --manifest-in $output_dir/manifests/librilight_chunk_cuts_${subset}.jsonl.gz \
      --manifest-out $output_dir/manifests/librilight_asr_cuts_${subset}.jsonl.gz \
      --nn-model-filename exp/exp/jit_script.pt \
      --tokens exp/data/lang_bpe_500/tokens.txt \
      --max-duration 2400 \
      --decoding-method greedy_search \
      --master 12345
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  # Final results are saved in $output_dir/manifests/librilight_cuts_{subset}.jsonl.gz
  log "Stage 4: Merge splitted chunks into utterances."
  for subset in small medium large; do
      ./tools/merge_chunks.py \
        --manifest-in $output_dir/manifests/librilight_asr_cuts_${subset}.jsonl.gz \
        --manifest-out $output_dir/manifests/librilight_cuts_merged_${subset}.jsonl.gz \
        --extra $extra
  done
fi


if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Align the long audios to corresponding texts and split into small piece."
  for subset in small medium large; do
    ./matching_parallel.py \
      --manifest-in $output_dir/manifests/librilight_cuts_merged_${subset}.jsonl.gz \
      --manifest-out $output_dir/manifests/librilight_cuts_${subset}.jsonl.gz
  done
fi
