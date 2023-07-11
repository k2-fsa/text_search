#!/usr/bin/env python3
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                            Zengwei Yao,
#                                            Wei Kang)
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
This script loads torchscript models, exported by `torch.jit.script()`,
and uses them to decode waves.
You can use the following command to get the exported models:

./zipformer/export.py \
  --exp-dir ./zipformer/exp \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 20 \
  --avg 10 \
  --jit 1

You can also download the jit model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
"""

import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from pathlib import Path

from asr_datamodule import AsrDataModule
from beam_search import greedy_search_batch
from utils import SymbolTable, convert_timestamp

from lhotse import CutSet, combine, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.supervision import AlignmentItem
from lhotse.serialization import SequentialJsonlWriter
from textsearch.utils import AttributeDict, setup_logger


def num_tokens(
    token_table: SymbolTable, disambig_pattern: str = re.compile(r"^#\d+$")
) -> int:
    """Return the number of tokens excluding those from
    disambiguation symbols.

    Caution:
      0 is not a token ID so it is excluded from the return value.
    """
    symbols = token_table.symbols
    ans = []
    for s in symbols:
        if not disambig_pattern.match(s):
            ans.append(token_table[s])
    num_tokens = len(ans)
    if 0 in ans:
        num_tokens -= 1
    return num_tokens


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--manifest-in",
        type=Path,
        help="Path to the input manifests.",
    )

    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Path to output manifest containing recognition results.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Path to directory to save logs.",
    )

    parser.add_argument(
        "--nn-model-filename",
        type=str,
        required=True,
        help="Path to the torchscript model cpu_jit.pt",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="path/to/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
        """,
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing decoding parameters."""
    params = AttributeDict(
        {
            "subsampling_factor": 4,
            "frame_shift_ms": 10,
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
    """Decode one batch.

    Args:
      params:
        It's the return value of :func:`get_params`.
      paramsmodel:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.

    Returns:
      Return the decoding result, timestamps, and scores.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(
        features=feature,
        feature_lengths=feature_lens,
    )

    if params.decoding_method == "greedy_search":
        res = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
    else:
        raise ValueError(
            f"Unsupported decoding method: {params.decoding_method}"
        )

    hyps = []
    timestamps = []
    scores = []
    for i in range(feature.shape[0]):
        hyps.append(res.hyps[i])
        timestamps.append(
            convert_timestamp(
                res.timestamps[i],
                params.subsampling_factor,
                params.frame_shift_ms,
            )
        )
        scores.append(res.scores[i])

    return hyps, timestamps, scores


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    token_table: SymbolTable,
    cuts_writer: SequentialJsonlWriter,
) -> None:
    """Decode dataset and store the recognition results to manifest.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      token_table:
        The table to map tokens to texts.
      cuts_writer:
        Writer to save the cuts with recognition results.

    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains five elements:
        - cut_id
        - reference transcript
        - predicted result
        - timestamps of reference transcript
        - timestamps of predicted result
    """
    #  Background worker to add alignemnt and save cuts to disk.
    def _save_worker(
        cuts: List[Cut],
        hyps: List[List[str]],
        timestamps: List[List[float]],
        scores: List[List[float]],
    ):
        for cut, symbol_list, time_list, score_list in zip(
            cuts, hyps, timestamps, scores
        ):
            symbol_list = [token_table[x] for x in symbol_list]
            ali = [
                AlignmentItem(
                    symbol=symbol, start=start, duration=None, score=score
                )
                for symbol, start, score in zip(
                    symbol_list, time_list, score_list
                )
            ]
            assert len(cut.supervisions) == 1, len(cut.supervisions)
            cut.supervisions[0].alignment = {"symbol": ali}
            cuts_writer.write(cut, flush=True)

    num_cuts = 0
    log_interval = 10
    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # We only want one background worker so that serialization is deterministic.

        for batch_idx, batch in enumerate(dl):
            cuts = batch["supervisions"]["cut"]

            hyps, timestamps, scores = decode_one_batch(
                params=params,
                model=model,
                batch=batch,
            )

            futures.append(
                executor.submit(_save_worker, cuts, hyps, timestamps, scores)
            )

            num_cuts += len(cuts)
            if batch_idx % log_interval == 0:
                logging.info(f"cuts processed until now is {num_cuts}")

        for f in futures:
            f.result()


@torch.no_grad()
def run(rank, world_size, args, in_cuts):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    setup_logger(
        f"{params.log_dir}/log-decode",
        dist=(rank, world_size) if world_size > 1 else None,
    )
    logging.info("Decoding started")

    assert params.decoding_method in ("greedy_search",), params.decoding_method

    token_table = SymbolTable.from_file(params.tokens)

    params.blank_id = token_table["<blk>"]
    params.unk_id = token_table["<unk>"]
    params.vocab_size = num_tokens(token_table) + 1

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"device: {device}")

    logging.info("Loading jit model")
    model = torch.jit.load(params.nn_model_filename)
    model.to(device)
    model.eval()

    # we will store new cuts with recognition results.
    args.return_cuts = True
    asr_data_module = AsrDataModule(args)

    if world_size > 1:
        in_cuts = in_cuts[rank]
        out_cuts_filename = params.manifest_out_dir / (
            f"split/{params.cuts_filename}_{rank}" + params.suffix
        )
    else:
        out_cuts_filename = params.manifest_out_dir / (
            f"{params.cuts_filename}" + params.suffix
        )

    dl = asr_data_module.dataloaders(in_cuts)

    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)
    decode_dataset(
        dl=dl,
        params=params,
        model=model,
        token_table=token_table,
        cuts_writer=cuts_writer,
    )
    cuts_writer.close()
    logging.info(f"Cuts saved to {out_cuts_filename}")

    logging.info("Done!")


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    if args.manifest_in == args.manifest_out:
        logging.error(
            f"Input manifest and output manifest share the same path : "
            f"{args.manifest_in}, the filenames should be different."
        )

    args.manifest_out_dir = args.manifest_out.parents[0]
    args.manifest_out_dir.mkdir(parents=True, exist_ok=True)

    assert args.manifest_in.is_file(), args.manifest_in

    args.suffix = ".jsonl.gz"
    args.cuts_filename = str(args.manifest_out.name).replace(args.suffix, "")

    if args.manifest_out.is_file():
        logging.info(f"{args.manifest_out} already exists - skipping.")
        return

    in_cuts = load_manifest_lazy(args.manifest_in)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        chunk_size = (len(in_cuts) + (world_size - 1)) // world_size
        # Each manifest is saved at: ``{output_dir}/{prefix}.{split_idx}.jsonl.gz``
        splits = in_cuts.split_lazy(
            output_dir=args.manifest_out_dir / "split",
            chunk_size=chunk_size,
            prefix=args.cuts_filename,
        )
        assert len(splits) == world_size, (len(splits), world_size)
        mp.spawn(
            run, args=(world_size, args, splits), nprocs=world_size, join=True
        )
        out_filenames = []
        for i in range(world_size):
            out_filenames.append(
                args.manifest_out_dir
                / f"split/{args.cuts_filename}_{i}{args.suffix}"
            )
        cuts = combine(*[load_manifest_lazy(x) for x in out_filenames])
        cuts.to_file(args.manifest_out)
        logging.info(f"Cuts saved to {args.manifest_out}")
    else:
        run(rank=0, world_size=world_size, args=args, in_cuts=in_cuts)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
