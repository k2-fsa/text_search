#!/usr/bin/env python3
# Copyright 2024 Xiaomi Corporation (Author: Wei Kang)
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


import argparse
import gzip
import json
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        """
    Calculate the overlap ratio for given manifest, it will print out the overlap ratio
    for each recording and the overlap ratio for the whole dataset.
    """
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="""The input file in lhotse manifest format, MUST be
        a jsonl.gz file.
        """,
    )
    return parser.parse_args()


def calculate_overlap(ifile: Path):
    total_duration = 0.0
    segments = dict()
    durations = dict()
    with gzip.open(ifile, "r") as f:
        for line in f:
            cut = json.loads(line)
            duration = cut["duration"]
            total_duration += duration
            start = cut["start"]
            end = start + duration
            recording_id = cut["supervisions"][0]["recording_id"]
            if recording_id in segments:
                segments[recording_id].append((start, end))
                durations[recording_id] += duration
            else:
                segments[recording_id] = [(start, end)]
                durations[recording_id] = duration
    total_overlap = 0.0
    for k, v in segments.items():
        # sort by the start time
        values = sorted(v, key=lambda x: x[0])
        overlap = 0.0
        for i in range(len(values) - 1):
            j = i + 1
            # a segment might have overlap with its succeeding segments
            while values[j][0] < values[i][1]:
                overlap += values[i][1] - values[j][0]
                if j == len(values) - 1:
                    break
                j += 1
        # print out the overlap ratio for current recording.
        logging.info(
            f"Recording {k} overlap ratio : "
            f"{round(overlap, 2)} / {round(durations[k], 2)} "
            f"= {round(overlap / durations[k] * 100, 2)}%"
        )
        total_overlap += overlap
    # print out the overlap ratio for the whole dataset.
    logging.info(
        f"Total overlap ratio : "
        f"{round(total_overlap, 2)} / {round(total_duration, 2)} "
        f"= {round(total_overlap / total_duration * 100, 2)}%"
    )


def main():
    args = get_args()
    ifile = args.manifest
    assert ifile.is_file(), f"File not exists : {ifile}"
    assert str(ifile).endswith(
        "jsonl.gz"
    ), f"Expect a jsonl gz file, given : {ifile}"
    calculate_overlap(ifile)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
