#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file merge overlapped chunks into utterances accroding to recording ids.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from lhotse import (
    CutSet,
    MonoCut,
    SupervisionSegment,
    load_manifest_lazy,
)
from lhotse.cut import Cut
from lhotse.serialization import SequentialJsonlWriter


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="Path to directory of chunk cuts with recognition results.",
    )

    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Path to directory to save full utterance by merging overlapped chunks.",
    )

    parser.add_argument(
        "--extra",
        type=float,
        default=2.0,
        help="""Extra duration (in seconds) at both sides.""",
    )

    return parser.parse_args()


def merge_chunks(
    cuts_chunk: CutSet,
    cuts_writer: SequentialJsonlWriter,
    extra: float,
) -> int:
    """Merge chunk-wise cuts accroding to recording ids.

    Args:
      cuts_chunk:
        The chunk-wise cuts opened in a lazy mode.
      cuts_writer:
        Writer to save the cuts with recognition results.
      extra:
        Extra duration (in seconds) to drop at both sides of each chunk.
    """

    #  Background worker to add alignemnt and save cuts to disk.
    def _save_worker(utt_cut: Cut):
        cuts_writer.write(utt_cut, flush=True)

    def _merge(cut_list: List[Cut], rec_id: str, utt_idx: int):
        """Merge chunks with same recording_id."""
        # For each group with a same recording, sort it accroding to the start time
        # In fact, we don't need to do this since the cuts have been sorted
        # according to the start time
        cut_list = sorted(cut_list, key=(lambda cut: cut.start))

        assert len(cut_list) > 0, len(cut_list)
        rec = cut_list[0].recording
        alignments = []
        cur_end = 0
        for cut in cut_list:
            assert cut.recording.id == rec_id, (cut.recording.id, rec_id)
            # Get left and right borders
            left = cut.start + extra if cut.start > 0 else 0
            chunk_end = cut.start + cut.duration
            right = (
                chunk_end - extra if chunk_end < rec.duration else rec.duration
            )

            # Assert the chunks are continuous
            assert left == cur_end, (left, cur_end)
            cur_end = right

            assert len(cut.supervisions) == 1, len(cut.supervisions)
            for ali in cut.supervisions[0].alignment["symbol"]:
                t = ali.start + cut.start
                if left <= t < right:
                    alignments.append(ali.with_offset(cut.start))

        old_sup = cut_list[0].supervisions[0]
        new_sup = SupervisionSegment(
            id=rec_id,
            recording_id=rec_id,
            start=0,
            duration=rec.duration,
            alignment={"symbol": alignments},
            language=old_sup.language,
            speaker=old_sup.speaker,
        )

        utt_cut = MonoCut(
            id=rec_id,
            start=0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=[new_sup],
        )
        # Set a custom attribute to the cut
        utt_cut.text_path = cut_list[0].text_path

        return utt_cut

    last_rec_id = None
    cut_list = []
    utt_idx = 0

    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:

        for cut in cuts_chunk:
            cur_rec_id = cut.recording.id
            if len(cut_list) == 0:
                # Case of the first cut
                last_rec_id = cur_rec_id
                cut_list.append(cut)
            elif cur_rec_id == last_rec_id:
                cut_list.append(cut)
            else:
                # Case of a cut belonging to a new recording
                utt_cut = _merge(cut_list, last_rec_id, utt_idx)
                utt_idx += 1

                futures.append(executor.submit(_save_worker, utt_cut))

                last_rec_id = cur_rec_id
                cut_list = [cut]

                if utt_idx % 5000 == 0:
                    logging.info(f"Procesed {utt_idx} utterances.")

        # For the cuts belonging to the last recording
        if len(cut_list) != 0:
            utt_cut = _merge(cut_list, last_rec_id, utt_idx)
            utt_idx += 1

            futures.append(executor.submit(_save_worker, utt_cut))
            logging.info("Finished")

        for f in futures:
            f.result()

    return utt_idx


def main():
    args = get_parser()

    logging.info(f"Processing {args.manifest_in}.")

    if args.manifest_in == args.manifest_out:
        logging.error(
            f"Input manifest and output manifest share the same path : "
            f"{args.manifest_in}, the filenames should be different."
        )

    if args.manifest_out.is_file():
        logging.info(f"{args.manifest_out} already exists - skipping.")
        return

    cuts_chunk = load_manifest_lazy(args.manifest_in)

    cuts_writer = CutSet.open_writer(args.manifest_out, overwrite=True)
    num_utt = merge_chunks(
        cuts_chunk, cuts_writer=cuts_writer, extra=args.extra
    )
    cuts_writer.close()
    logging.info(f"{num_utt} cuts saved to {args.manifest_out}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
