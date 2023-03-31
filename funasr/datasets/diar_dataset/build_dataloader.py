import numpy as np
import torch
from torch.utils.data import DataLoader

from funasr.datasets.diar_dataset.dataset import DiarizationDataset
from funasr.iterators.abs_iter_factory import AbsIterFactory


def custom_collate(batch):
    ys, ts, orders = zip(*batch)
    ys = [torch.from_numpy(y.astype(np.float32)) for y in ys]
    ts = [torch.from_numpy(t.astype(np.float32)) for t in ts]
    orders = [torch.from_numpy(o.astype(np.int64)) for o in orders]
    data = dict(ys=ys, ts=ts, orders=orders)
    return data


class DiarDataLoader(AbsIterFactory):
    def __init__(self, data_file, dataset_conf):
        self.dataset_conf = dataset_conf
        self.dataset = DiarizationDataset(data_file)
        self.data_loader = None

    def build_iter(self, epoch, shuffle=True):
        if self.data_loader is None:
            self.data_loader = DataLoader(self.dataset,
                                          batch_size=self.dataset_conf.get("batch_size", 64),
                                          num_workers=self.dataset_conf.get("num_workers", 8),
                                          shuffle=True,
                                          collate_fn=custom_collate)
            return self.data_loader
        else:
            return self.data_loader