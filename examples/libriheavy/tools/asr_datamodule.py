# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2023  Xiaomi Corporation     (Author: Zengwei Yao)
# Copyright      2023  Xiaomi Corporation     (Author: Wei Kang)
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
from typing import Dict, List, Union

import torch
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.cut import Cut
from lhotse.dataset import (
    K2SpeechRecognitionDataset,
    DynamicBucketingSampler,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import (
    BatchIO,
    OnTheFlyFeatures,
)
from torch.utils.data import DataLoader

from textsearch.utils import str2bool


class AsrDataModule:
    """
    DataModule for k2 ASR experiments.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=600.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-mel-bins",
            type=int,
            default=80,
            help="The number of melbank bins for fbank",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--batch-size",
            type=int,
            default=10,
            help="The number of utterances in a batch",
        )

    def dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        dataset = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(
                Fbank(FbankConfig(num_mel_bins=self.args.num_mel_bins))
            ),
            return_cuts=self.args.return_cuts,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            sampler = DynamicBucketingSampler(
                cuts,
                max_duration=self.args.max_duration,
                shuffle=False,
                drop_last=False,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            sampler = SimpleCutSampler(
                cuts,
                max_cuts=self.args.batch_size,
            )

        logging.debug("About to create test dataloader")
        dl = DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )
        return dl
