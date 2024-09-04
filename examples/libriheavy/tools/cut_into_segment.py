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
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(
        """
    Cut the long wav into small segments according to the supervisions in cuts, and
    also generate the corresponding manifests.
    """
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="""The input file in lhotse manifest format, MUST be
        a jsonl.gz file.
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="""The directory that wavs and manifest will been written to.""",
    )

    parser.add_argument(
        "--num-segments",
        type=float,
        default=-1,
        help="The number of segments need to be processed, for debugging purpose, -1 means all.",
    )

    return parser.parse_args()


def cut_into_segments(ifile: Path, output_dir: Path, num_segments: int = -1):
    index = 0
    with gzip.open(ifile, "r") as f, open(
        output_dir / "manifests.txt", "w"
    ) as fm:
        prev_audio = ""
        for line in f:
            if num_segments != -1 and index == num_segments:
                break
            index += 1
            cut = json.loads(line)
            id = cut["id"].replace("/", "_")
            duration = cut["duration"]
            start = cut["start"]
            end = start + duration
            audio = cut["recording"]["sources"][0]["source"]
            text = cut["supervisions"][0]["custom"]["texts"][0]
            if audio != prev_audio:
                samples, sample_rate = sf.read(audio)
                prev_audio = audio
            current_samples = samples[
                int(start * sample_rate) : int((end + 0.5) * sample_rate)
            ]
            sf.write(output_dir / f"{id}.wav", current_samples, sample_rate)
            fm.write(f"{id}\t{id}.wav\t{text}\n")
            if index % 200 == 0:
                logging.info("Processed {index} segments.")


def main():
    args = get_args()
    ifile = args.manifest
    assert ifile.is_file(), f"File not exists : {ifile}"
    assert str(ifile).endswith(
        "jsonl.gz"
    ), f"Expect a jsonl gz file, given : {ifile}"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cut_into_segments(ifile, output_dir, args.num_segments)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
