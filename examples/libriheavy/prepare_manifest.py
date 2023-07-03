# This file is copied and modified from
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/librilight.py

"""
About the librilight corpus

Libri-light is a benchmark for the training of automatic speech recognition (ASR)
systems with limited or no supervision.

It contains a large dataset of 60K hours of unlabelled speech from audiobooks in 
English and a small labelled dataset (10h, 1h, and 10 min) plus metrics,
trainable baseline models, and pretrained models that use these datasets.

It is covered in more detail at https://arxiv.org/abs/1912.07875.

This data is very huge - please download manually at LIBRILIGHT_URL.
"""

import argparse
import logging
import json
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm

from lhotse.audio import Recording
from lhotse import CutSet, MonoCut
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment
from lhotse.utils import Pathlike

LIBRILIGHT = ("small", "medium", "large")

LIBRILIGHT_URL = (
    "https://dl.fbaipublicfiles.com/librilight/data/small.tar",
    "https://dl.fbaipublicfiles.com/librilight/data/medium.tar",
    "https://dl.fbaipublicfiles.com/librilight/data/large.tar",
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="Path to the original audio data.",
    )

    parser.add_argument(
        "--books-dir",
        type=Path,
        help="Path to the original texts data.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the directory to saved generated manifests.",
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="""Number of workers.""",
    )

    return parser.parse_args()


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
    books_dir: Pathlike,
    books_dict: Dict,
) -> MonoCut:
    file_name = (
        str(audio_path).replace(".flac", "").replace(str(corpus_dir) + "/", "")
    )
    speaker = str(audio_path).split("/")[-3]
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=file_name,
    )
    segment = SupervisionSegment(
        id=file_name,
        recording_id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="English",
        speaker=speaker,
    )

    return MonoCut(
        id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        custom={"text_path": str(books_dir / books_dict[file_name])},
        recording=recording,
        supervisions=[segment],
    )


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    books_dir: Pathlike,
    num_jobs: int = 1,
) -> CutSet:
    """
    Returns the CutSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param books_dir: Path to the LibriLight books.
    :return: the CutSet
    """
    part_path = corpus_dir / subset
    audio_paths = list(part_path.rglob("*.flac"))

    with open(books_dir / f"recording2book_{subset}.json") as f:
        books_dict = json.load(f)

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        cuts = []
        for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
            futures.append(
                ex.submit(
                    _parse_utterance,
                    corpus_dir,
                    audio_path,
                    books_dir,
                    books_dict,
                )
            )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            cuts.append(result)
        cut_set = CutSet.from_cuts(cuts)

    return cut_set


def prepare_librilight(
    corpus_dir: Pathlike,
    books_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int = 1,
):
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the LibriLight dataset.
    :param books_dir: Path to the LibriLight books.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    books_dir = Path(books_dir)
    output_dir = Path(output_dir) if output_dir is not None else None

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert books_dir.is_dir(), f"No such directory: {books_dir}"

    logging.info("Preparing LibriLight...")

    subsets = LIBRILIGHT

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing LibriLight subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="librilight",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"LibriLight subset: {part} already prepared - skipping."
            )
            continue

        cut_set = _prepare_subset(part, corpus_dir, books_dir, num_jobs)

        if output_dir is not None:
            cut_set.to_file(output_dir / f"librilight_cut_{part}.jsonl.gz")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    logging.info(vars(args))

    prepare_librilight(
        corpus_dir=args.corpus_dir,
        books_dir=args.books_dir,
        output_dir=args.output_dir,
        num_jobs=args.num_jobs,
    )

    logging.info(f"Done.")
