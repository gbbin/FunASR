import logging
import random
from typing import Iterator
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from funasr.datasets.diar_dataset.dataset import DiarizationDataset
from funasr.fileio.read_text import read_2column_text
from funasr.iterators.abs_iter_factory import AbsIterFactory
from funasr.samplers.abs_sampler import AbsSampler


class DiarBatchSampler(AbsSampler):
    """
    Modified from UnSortedBatchSampler
    """

    def __init__(
            self,
            batch_size: int,
            key_file: str,
            drop_last: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
        self.key_file = key_file
        self.drop_last = drop_last

        utt2any = read_2column_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        keys = list(utt2any)
        random.shuffle(keys)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {key_file}")

        category2utt = {}
        category2utt["default_category"] = keys

        self.batch_list = []
        for d, v in category2utt.items():
            category_keys = v
            N = max(len(category_keys) // batch_size, 1)
            if not self.drop_last:
                cur_batch_list = [
                    category_keys[i * len(keys) // N: (i + 1) * len(keys) // N]
                    for i in range(N)
                ]
            else:
                cur_batch_list = [
                    tuple(category_keys[i * batch_size: (i + 1) * batch_size])
                    for i in range(N)
                ]
            self.batch_list.extend(cur_batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)


def custom_collate(batch):
    sample_names, xs, ts, orders = zip(*batch)
    xs = [torch.from_numpy(x.astype(np.float32)) for x in xs]
    ts = [torch.from_numpy(t.astype(np.float32)) for t in ts]
    orders = [torch.from_numpy(o.astype(np.int64)) for o in orders]
    data = dict(xs=xs, ts=ts, orders=orders)
    return sample_names, data


class DiarDataLoader(AbsIterFactory):
    def __init__(self, data_file, dataset_conf):
        self.dataset_conf = dataset_conf
        self.dataset = DiarizationDataset(data_file)
        self.data_loader = None
        self.batch_sampler = DiarBatchSampler(batch_size=self.dataset_conf.get("batch_size", 64),
                                              key_file=data_file)

    def build_iter(self, epoch, shuffle=True):
        batches = list(self.batch_sampler)
        random.shuffle(batches)
        data_loader = DataLoader(self.dataset,
                                 sampler=batches,
                                 num_workers=self.dataset_conf.get("num_workers", 8),
                                 collate_fn=custom_collate)
        return data_loader
