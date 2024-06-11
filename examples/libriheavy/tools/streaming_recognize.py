#!/usr/bin/env python3
# Copyright 2022-2023 Xiaomi Corporation (Authors: Wei Kang,
#                                                  Fangjun Kuang,
#                                                  Zengwei Yao)
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
import torch.multiprocessing as mp
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from decode_stream import DecodeStream
from lhotse import CutSet, Fbank, FbankConfig, combine, load_manifest_lazy
from beam_search import streaming_greedy_search
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor
from utils import SymbolTable, convert_timestamp
from lhotse.cut import Cut
from lhotse.supervision import AlignmentItem
from lhotse.serialization import SequentialJsonlWriter

from textsearch.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

LOG_EPS = math.log(1e-10)


@dataclass
class DecodingResult:
    timestamps: List[int]

    hyp: List[int]

    scores: List[float]


def get_params() -> AttributeDict:
    """Return a dict containing decoding parameters."""
    params = AttributeDict(
        {
            "subsampling_factor": 4,
            "frame_shift_ms": 10,
            "beam_size": 4,
        }
    )
    return params


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
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be a exported jit.script model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="The decoding chunk size",
    )

    parser.add_argument(
        "--left-context-frames",
        type=int,
        required=True,
        help="The decoding left context frames",
    )

    parser.add_argument(
        "--manifests-in",
        type=str,
        required=True,
        help="The path to the input manifests.",
    )

    parser.add_argument(
        "--manifests-out",
        type=str,
        required=True,
        help="The path to the output manifests.",
    )

    parser.add_argument(
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded parallel.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Path to directory to save logs.",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="path/to/tokens.txt",
        help="Path to the tokens.txt",
    )
    return parser


def get_init_states(
    model: nn.Module,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
    is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
    states[-2] is the cached left padding for ConvNeXt module,
    of shape (batch_size, num_channels, left_pad, num_freqs)
    states[-1] is processed_lens of shape (batch,), which records the number
    of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
    """
    states = model.encoder.get_init_states(batch_size, device)

    return states


def stack_states(state_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Stack list of zipformer states that correspond to separate utterances
    into a single emformer state, so that it can be used as an input for
    zipformer when those utterances are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance. For element-n,
        state_list[n] is a list of cached tensors of all encoder layers. For layer-i,
        state_list[n][i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1,
        cached_val2, cached_conv1, cached_conv2).
        state_list[n][-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
        state_list[n][-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Note:
      It is the inverse of :func:`unstack_states`.
    """
    batch_size = len(state_list)
    assert (len(state_list[0]) - 2) % 6 == 0, (
        len(state_list[0]),
        len(state_list),
    )
    tot_num_layers = (len(state_list[0]) - 2) // 6

    batch_states = []
    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key = torch.cat(
            [state_list[i][layer_offset] for i in range(batch_size)], dim=1
        )
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn = torch.cat(
            [state_list[i][layer_offset + 1] for i in range(batch_size)], dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1 = torch.cat(
            [state_list[i][layer_offset + 2] for i in range(batch_size)], dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2 = torch.cat(
            [state_list[i][layer_offset + 3] for i in range(batch_size)], dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1 = torch.cat(
            [state_list[i][layer_offset + 4] for i in range(batch_size)], dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2 = torch.cat(
            [state_list[i][layer_offset + 5] for i in range(batch_size)], dim=0
        )
        batch_states += [
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        ]

    cached_embed_left_pad = torch.cat(
        [state_list[i][-2] for i in range(batch_size)], dim=0
    )
    batch_states.append(cached_embed_left_pad)

    processed_lens = torch.cat(
        [state_list[i][-1] for i in range(batch_size)], dim=0
    )
    batch_states.append(processed_lens)

    return batch_states


def unstack_states(batch_states: List[Tensor]) -> List[List[Tensor]]:
    """Unstack the zipformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Note:
      It is the inverse of :func:`stack_states`.

    Args:
        batch_states: A list of cached tensors of all encoder layers. For layer-i,
          states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
          cached_conv1, cached_conv2).
          state_list[-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
          states[-1] is processed_lens of shape (batch,), which records the number
          of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Returns:
        state_list: A list of list. Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance.
    """
    assert (len(batch_states) - 2) % 6 == 0, len(batch_states)
    tot_num_layers = (len(batch_states) - 2) // 6

    processed_lens = batch_states[-1]
    batch_size = processed_lens.shape[0]

    state_list = [[] for _ in range(batch_size)]

    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key_list = batch_states[layer_offset].chunk(
            chunks=batch_size, dim=1
        )
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn_list = batch_states[layer_offset + 1].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1_list = batch_states[layer_offset + 2].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2_list = batch_states[layer_offset + 3].chunk(
            chunks=batch_size, dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1_list = batch_states[layer_offset + 4].chunk(
            chunks=batch_size, dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2_list = batch_states[layer_offset + 5].chunk(
            chunks=batch_size, dim=0
        )
        for i in range(batch_size):
            state_list[i] += [
                cached_key_list[i],
                cached_nonlin_attn_list[i],
                cached_val1_list[i],
                cached_val2_list[i],
                cached_conv1_list[i],
                cached_conv2_list[i],
            ]

    cached_embed_left_pad_list = batch_states[-2].chunk(
        chunks=batch_size, dim=0
    )
    for i in range(batch_size):
        state_list[i].append(cached_embed_left_pad_list[i])

    processed_lens_list = batch_states[-1].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(processed_lens_list[i])

    return state_list


def streaming_forward(
    features: Tensor,
    feature_lens: Tensor,
    model: nn.Module,
    states: List[Tensor],
    chunk_size: int,
    left_context_len: int,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    Returns encoder outputs, output lengths, and updated states.
    """
    cached_embed_left_pad = states[-2]
    (
        x,
        x_lens,
        new_cached_embed_left_pad,
    ) = model.encoder_embed.streaming_forward(
        x=features,
        x_lens=feature_lens,
        cached_left_pad=cached_embed_left_pad,
    )
    assert x.size(1) == chunk_size, (x.size(1), chunk_size)

    src_key_padding_mask = make_pad_mask(x_lens)

    # processed_mask is used to mask out initial states
    processed_mask = torch.arange(left_context_len, device=x.device).expand(
        x.size(0), left_context_len
    )
    processed_lens = states[-1]  # (batch,)
    # (batch, left_context_size)
    processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
    # Update processed lengths
    new_processed_lens = processed_lens + x_lens

    # (batch, left_context_size + chunk_size)
    src_key_padding_mask = torch.cat(
        [processed_mask, src_key_padding_mask], dim=1
    )

    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    encoder_states = states[:-2]
    (
        encoder_out,
        encoder_out_lens,
        new_encoder_states,
    ) = model.encoder.streaming_forward(
        x=x,
        x_lens=x_lens,
        states=encoder_states,
        src_key_padding_mask=src_key_padding_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

    new_states = new_encoder_states + [
        new_cached_embed_left_pad,
        new_processed_lens,
    ]
    return encoder_out, encoder_out_lens, new_states


def decode_one_chunk(
    params: AttributeDict,
    model: nn.Module,
    decode_streams: List[DecodeStream],
) -> List[int]:
    """Decode one chunk frames of features for each decode_streams and
    return the indexes of finished streams in a List.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      decode_streams:
        A List of DecodeStream, each belonging to a utterance.
    Returns:
      Return a List containing which DecodeStreams are finished.
    """
    device = params.device
    chunk_size = int(params.chunk_size)
    left_context_len = int(params.left_context_frames)

    features = []
    feature_lens = []
    states = []
    processed_lens = []  # Used in fast-beam-search

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(chunk_size * 2)
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    # Make sure the length after encoder_embed is at least 1.
    # The encoder_embed subsample features (T - 7) // 2
    # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
    tail_length = chunk_size * 2 + 7 + 2 * 3
    if features.size(1) < tail_length:
        pad_length = tail_length - features.size(1)
        feature_lens += pad_length
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, pad_length),
            mode="constant",
            value=LOG_EPS,
        )

    states = stack_states(states)

    encoder_out, encoder_out_lens, new_states = model.encoder.forward(
        features=features, feature_lengths=feature_lens, states=states
    )

    encoder_out = model.joiner.encoder_proj(encoder_out)

    streaming_greedy_search(
        model=model, encoder_out=encoder_out, streams=decode_streams
    )

    states = unstack_states(new_states)

    finished_streams = []
    for i in range(len(decode_streams)):
        decode_streams[i].states = states[i]
        decode_streams[i].done_frames += int(encoder_out_lens[i])
        if decode_streams[i].done:
            finished_streams.append(i)

    return finished_streams


def decode_dataset(
    cuts: CutSet,
    params: AttributeDict,
    model: nn.Module,
    token_table: SymbolTable,
    cuts_writer: SequentialJsonlWriter,
):
    """Decode dataset.

    Args:
      cuts:
        Lhotse Cutset containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    #  Background worker to add alignemnt and save cuts to disk.
    def _save_worker(
        decode_streams: List[DecodeStream],
    ):
        for stream in decode_streams:
            cut = stream.cut
            symbol_list = [token_table[x] for x in stream.decoding_result()]
            timestamps = convert_timestamp(
                stream.timestamps,
                params.subsampling_factor,
                params.frame_shift_ms,
            )

            ali = [
                AlignmentItem(
                    symbol=symbol, start=start, duration=None, score=score
                )
                for symbol, start, score in zip(
                    symbol_list, timestamps, stream.scores
                )
            ]
            assert len(cut.supervisions) == 1, len(cut.supervisions)
            cut.supervisions[0].alignment = {"symbol": ali}
            cuts_writer.write(cut, flush=True)

    device = params.device
    opts = FbankConfig()
    opts.sampling_rate = 16000
    opts.num_filters = 80

    log_interval = 50

    # Contain decode streams currently running.
    decode_streams = []
    futures = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        for num, cut in enumerate(cuts):
            # each utterance has a DecodeStream.
            initial_states = states = model.encoder.get_init_states(1, device)
            decode_stream = DecodeStream(
                params=params,
                cut=cut,
                initial_states=initial_states,
                device=device,
            )

            audio: np.ndarray = cut.load_audio()
            # audio.shape: (1, num_samples)
            assert len(audio.shape) == 2
            assert audio.shape[0] == 1, "Should be single channel"
            assert audio.dtype == np.float32, audio.dtype

            # The trained model is using normalized samples
            # - this is to avoid sending [-32k,+32k] signal in...
            # - some lhotse AudioTransform classes can make the signal
            #   be out of range [-1, 1], hence the tolerance 10
            assert (
                np.abs(audio).max() <= 10
            ), "Should be normalized to [-1, 1], 10 for tolerance..."

            samples = torch.from_numpy(audio).squeeze(0)

            fbank = Fbank(opts)
            feature = fbank.extract(samples, sampling_rate=16000).to(device)
            decode_stream.set_features(feature, tail_pad_len=30)

            decode_streams.append(decode_stream)

            while len(decode_streams) >= params.num_decode_streams:
                finished_streams = decode_one_chunk(
                    params=params, model=model, decode_streams=decode_streams
                )
                done_streams = []
                for i in sorted(finished_streams, reverse=True):
                    done_streams.append(decode_streams[i])
                    del decode_streams[i]
                if done_streams:
                    futures.append(executor.submit(_save_worker, done_streams))

            if num % log_interval == 0:
                logging.info(f"Cuts processed until now is {num}.")

        # decode final chunks of last sequences
        while len(decode_streams):
            finished_streams = decode_one_chunk(
                params=params, model=model, decode_streams=decode_streams
            )
            done_streams = []
            for i in sorted(finished_streams, reverse=True):
                done_streams.append(decode_streams[i])
                del decode_streams[i]
            if done_streams:
                futures.append(executor.submit(_save_worker, done_streams))

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

    token_table = SymbolTable.from_file(params.tokens)
    params.blank_id = token_table["<blk>"]

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"device: {device}")
    params.device = device

    logging.info("Loading jit model")
    model = torch.jit.load(params.checkpoint)
    model.to(device)
    model.eval()
    model.device = device

    params.context_size = model.decoder.context_size

    # we will store new cuts with recognition results.
    args.return_cuts = True

    if world_size > 1:
        in_cuts = in_cuts[rank]
        out_cuts_filename = params.manifests_out_dir / (
            f"split/{params.cuts_filename}_{rank}" + params.suffix
        )
    else:
        out_cuts_filename = params.manifests_out_dir / (
            f"{params.cuts_filename}" + params.suffix
        )

    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)
    decode_dataset(
        cuts=in_cuts,
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
    args = parser.parse_args()

    args.manifests_in = Path(args.manifests_in)
    args.manifests_out = Path(args.manifests_out)

    if args.manifests_in == args.manifests_out:
        print(
            f"Input manifest and output manifest share the same path : "
            f"{args.manifests_in}, the filenames should be different."
        )

    args.manifests_out_dir = args.manifests_out.parents[0]
    args.manifests_out_dir.mkdir(parents=True, exist_ok=True)

    assert args.manifests_in.is_file(), args.manifests_in

    args.suffix = ".jsonl.gz"
    args.cuts_filename = str(args.manifests_out.name).replace(args.suffix, "")

    if args.manifests_out.is_file():
        print(f"{args.manifests_out} already exists - skipping.")
        return

    in_cuts = load_manifest_lazy(args.manifests_in)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        chunk_size = (len(in_cuts) + (world_size - 1)) // world_size
        # Each manifest is saved at: ``{output_dir}/{prefix}.{split_idx}.jsonl.gz``
        splits = in_cuts.split_lazy(
            output_dir=args.manifests_out_dir / "split",
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
                args.manifests_out_dir
                / f"split/{args.cuts_filename}_{i}{args.suffix}"
            )
        cuts = combine(*[load_manifest_lazy(x) for x in out_filenames])
        cuts.to_file(args.manifests_out)
        print(f"Cuts saved to {args.manifests_out}")
    else:
        run(rank=0, world_size=world_size, args=args, in_cuts=in_cuts)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
