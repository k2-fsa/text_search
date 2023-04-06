#!/usr/bin/env bash

set -eou pipefail

stage=0
stop_stage=10

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare manifests"

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Matching"

fi
