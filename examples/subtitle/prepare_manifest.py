#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Wei Kang)
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
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm

from lhotse.audio import Recording
from lhotse import CutSet, MonoCut
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Pathlike

import pysrt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to the manifest file containing the mapping between audios and subtitles.",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="The name of this dataset.",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="The language of this dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the directory to saved generated manifests in lhotse format.",
    )

    parser.add_argument(
        "--extend-seconds",
        type=float,
        default=1.5,
        help="""The number of seconds extended to both side of each subtitle
        segment in case the timestamp in subtitle is inaccurate.
        """,
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="""Number of workers.""",
    )

    return parser.parse_args()


def _parse_single_audio(
    line: str, language: str, extend_seconds: float = 1.5
) -> MonoCut:
    toks = line.strip().split("\t")
    if len(toks) != 2:
        logging.warning(
            f"Each line of the manifests should contain two items separated by tab, "
            "the first one is audio path, the second one is subtitle path, "
            "given {line}, skipping this line."
        )
        return None
    audio, subtitle = Path(toks[0]), Path(toks[1])
    if not audio.is_file():
        logging.warning(f"Audio path : {audio} not exists.")
        return None
    if not subtitle.is_file():
        logging.warning(f"subtitle path : {subtitle} not exists.")
        return None

    file_name = str(audio).replace("/", "_")
    file_name = ".".join(file_name.split(".")[0:-1])

    recording = Recording.from_file(
        path=audio,
        recording_id=file_name,
    )

    segments = []
    for i, sub in enumerate(pysrt.open(subtitle, encoding="utf8")):
        start = (
            sub.start.hours * 3600
            + sub.start.minutes * 60
            + sub.start.seconds
            + sub.start.milliseconds / 1000
        )
        start -= extend_seconds
        start = 0 if start < 0 else start

        end = (
            sub.end.hours * 3600
            + sub.end.minutes * 60
            + sub.end.seconds
            + sub.end.milliseconds / 1000
        )
        end += extend_seconds
        end = recording.duration if end > recording.duration else end

        if end < start:
            continue

        seg_id = f"{file_name}_{i}"
        segments.append(
            SupervisionSegment(
                id=seg_id,
                recording_id=file_name,
                start=start,
                text=sub.text,
                duration=end - start,
                channel=0,
                language=language,
            )
        )

    return MonoCut(
        id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=segments,
    )


def prepare_subtitle(
    manifest: Pathlike,
    name: str,
    language: str,
    output_dir: Pathlike,
    extend_seconds: float = 1.5,
    num_jobs: int = 1,
):
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param manifest: Path to the mapping between audios and subtitles.
    :param name: The name of this dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    """
    manifest = Path(manifest)
    assert manifest.is_file(), f"No such file: {manifest}"

    logging.info(f"Preparing manifests from {manifest}...")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(num_jobs) as ex, open(manifest, "r") as f:
        futures = []
        cuts = []
        for line in f:
            futures.append(
                ex.submit(
                    _parse_single_audio,
                    line,
                    language,
                    extend_seconds,
                )
            )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            cuts.append(result)

        cut_set = CutSet.from_cuts(cuts)

        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )
        cut_set.to_file(output_dir / f"{name}_raw_cuts.jsonl.gz")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    logging.info(vars(args))

    prepare_subtitle(
        manifest=args.manifest,
        name=args.name,
        language=args.language,
        output_dir=args.output_dir,
        extend_seconds=args.extend_seconds,
        num_jobs=args.num_jobs,
    )

    logging.info(f"Done.")
