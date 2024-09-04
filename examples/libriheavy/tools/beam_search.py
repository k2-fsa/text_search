# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                  Xiaoyu Yang)
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

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import sentencepiece as spm

import torch
from utils import row_splits_to_row_ids
from decode_stream import DecodeStream


@dataclass
class DecodingResults:
    # timestamps[i][k] contains the frame number on which tokens[i][k]
    # is decoded
    timestamps: List[List[int]]

    # hyps[i] is the recognition results, i.e., word IDs or token IDs
    # for the i-th utterance with fast_beam_search_nbest_LG.
    hyps: List[List[int]]

    # scores[i][k] contains the log-prob of tokens[i][k]
    scores: Optional[List[List[float]]] = None


def greedy_search_batch(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
) -> DecodingResults:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
    Returns:
      Return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    # timestamp[n][i] is the frame index after subsampling
    # on which hyp[n][i] is decoded
    timestamps = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        log_probs = logits.log_softmax(dim=-1)
        assert log_probs.ndim == 2, log_probs.shape
        y = log_probs.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(log_probs[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])

    return DecodingResults(
        hyps=ans,
        timestamps=ans_timestamps,
        scores=ans_scores,
    )


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    # The log_prob of each token in ys[context_size:]
    # It is derived from the nnet_output.
    scores: List[float] = field(default_factory=list)

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(
                old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob
            )
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(
                self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys)
            )
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis."""
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


def get_hyps_row_splits(hyps: List[HypothesisList]) -> torch.Tensor:
    """Return the row_splits of hyps.

    A row_splits is a 1 dim tensor with the length of len(hyps) + 1, from which
    we can get the length of each hyp by the following formula.

       len(hyps[i]) = row_splits[i+1] - row_splits[i]

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return the row_splits of given hyps. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    return row_splits


def modified_beam_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
    Returns:
      Return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
                scores=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        # on cpu
        hyps_row_splits = get_hyps_row_splits(B)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=row_splits_to_row_ids(hyps_row_splits).to(device),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        log_probs = (logits / temperature).log_softmax(
            dim=-1
        )  # (num_hyps, vocab_size)

        log_probs_t = log_probs.reshape(-1)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_row_splits * vocab_size

        for i in range(batch_size):
            topk_log_probs, topk_indexes = log_probs[
                row_splits[i] : row_splits[i + 1]
            ].topk(beam)

            current_log_probs = log_probs_t[
                row_splits[i] : row_splits[i + 1]
            ].cpu()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()
            topk_indexes = topk_indexes.cpu()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                new_scores = hyp.scores[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    new_scores.append(current_log_probs[topk_indexes[k]].item())

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys,
                    scores=new_scores,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    sorted_scores = [h.scores for h in best_hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])
        ans_scores.append(sorted_scores[unsorted_indices[i]])

    return DecodingResults(
        hyps=ans,
        timestamps=ans_timestamps,
        scores=ans_scores,
    )


# The force alignment problem can be formulated as finding
# a path in a rectangular lattice, where the path starts
# from the lower left corner and ends at the upper right
# corner. The horizontal axis of the lattice is `t` (representing
# acoustic frame indexes) and the vertical axis is `u` (representing
# BPE tokens of the transcript).
#
# The notations `t` and `u` are from the paper
# https://arxiv.org/pdf/1211.3711.pdf
#
# Beam search is used to find the path with the highest log probabilities.
#
# It assumes the maximum number of symbols that can be
# emitted per frame is 1.


def batch_force_alignment(
    model: torch.nn.Module,
    sp: spm.SentencePieceProcessor,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ys_list: List[List[int]],
    beam_size: int = 4,
) -> List[int]:
    """Compute the force alignment of a batch of utterances given their transcripts
    in BPE tokens and the corresponding acoustic output from the encoder.

    Caution:
      This function is modified from `modified_beam_search` in beam_search.py.
      We assume that the maximum number of sybmols per frame is 1.

    Args:
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      ys_list:
        A list of BPE token IDs list. We require that for each utterance i,
        len(ys_list[i]) <= encoder_out_lens[i].
      beam_size:
        Size of the beam used in beam search.

    Returns:
      Return a list of frame indexes list for each utterance i,
      where len(ans[i]) == len(ys_list[i]).
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) == len(ys_list), (
        encoder_out.size(0),
        len(ys_list),
    )
    assert encoder_out.size(0) > 0, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    sorted_indices = packed_encoder_out.sorted_indices.tolist()
    encoder_out_lens = encoder_out_lens.tolist()
    ys_lens = [len(ys) for ys in ys_list]
    sorted_encoder_out_lens = [encoder_out_lens[i] for i in sorted_indices]
    sorted_ys_lens = [ys_lens[i] for i in sorted_indices]
    sorted_ys_list = [ys_list[i] for i in sorted_indices]

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
                scores=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]
        sorted_encoder_out_lens = sorted_encoder_out_lens[:batch_size]
        sorted_ys_lens = sorted_ys_lens[:batch_size]

        # on cpu
        hyps_row_splits = get_hyps_row_splits(B)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=row_splits_to_row_ids(hyps_row_splits).to(device),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out, decoder_out, project_input=False
        )  # (num_hyps, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)
        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)
        log_probs.add_(ys_log_probs.to(device))
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.reshape(-1)
        row_splits = hyps_row_splits * vocab_size

        for i in range(batch_size):
            current_log_probs = log_probs[
                row_splits[i] : row_splits[i + 1]
            ].cpu()

            for h, hyp in enumerate(A[i]):
                pos_u = len(hyp.timestamp)
                idx_offset = h * vocab_size
                if (sorted_encoder_out_lens[i] - 1 - t) >= (
                    sorted_ys_lens[i] - pos_u
                ):
                    # emit blank token
                    new_hyp = Hypothesis(
                        log_prob=current_log_probs[idx_offset + blank_id],
                        ys=hyp.ys[:],
                        timestamp=hyp.timestamp[:],
                        scores=hyp.scores[:],
                    )
                    B[i].add(new_hyp)
                if pos_u < sorted_ys_lens[i]:
                    # emit non-blank token
                    new_token = sorted_ys_list[i][pos_u]
                    log_prob = current_log_probs[idx_offset + new_token]
                    new_hyp = Hypothesis(
                        log_prob=log_prob,
                        ys=hyp.ys + [new_token],
                        timestamp=hyp.timestamp + [t],
                        scores=hyp.scores + [float(log_prob.exp())],
                    )
                    B[i].add(new_hyp)

            if len(B[i]) > beam_size:
                B[i] = B[i].topk(beam_size, length_norm=True)

    B = B + finalized_B
    sorted_hyps = [b.get_most_probable() for b in B]
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    hyps = [sorted_hyps[i] for i in unsorted_indices]
    ans = []
    for i, hyp in enumerate(hyps):
        assert hyp.ys[context_size:] == ys_list[i], (
            hyp.ys[context_size:],
            ys_list[i],
        )
        sym_list = [sp.id_to_piece(j) for j in ys_list[i]]
        ans.append(list(zip(sym_list, hyp.scores)))

    return ans


def streaming_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
    blank_penalty: float = 0.0,
) -> None:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      streams:
        A list of Stream objects.
    """
    assert len(streams) == encoder_out.size(0)
    assert encoder_out.ndim == 3

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = model.device
    T = encoder_out.size(1)

    decoder_input = torch.tensor(
        [stream.hyp[-context_size:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
    # decoder_out is of shape (N, 1, decoder_out_dim)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    for t in range(T):
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        current_encoder_out = encoder_out[:, t : t + 1, :]  # noqa

        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        # logits'shape (batch_size,  vocab_size)
        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0.0:
            logits[:, 0] -= blank_penalty

        assert logits.ndim == 2, logits.shape
        log_probs = logits.log_softmax(dim=-1)
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                streams[i].hyp.append(v)
                streams[i].timestamps.append(t + streams[i].done_frames)
                streams[i].scores.append(log_probs[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [stream.hyp[-context_size:] for stream in streams],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(
                decoder_input,
                need_pad=False,
            )
            decoder_out = model.joiner.decoder_proj(decoder_out)
