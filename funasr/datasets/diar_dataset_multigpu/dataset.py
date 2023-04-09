import logging

import kaldiio
import numpy as np
from torch.utils.data import Dataset


class DiarizationDataset(Dataset):

    def __init__(self, data_file):
        self.data_file = data_file
        self.lines = [line.strip().split() for line in open(data_file)]
        self.chunk_inputs = dict()
        for line in self.lines:
            self.chunk_inputs[line[0]] = (line[1], line[2])
        logging.info("total chunks: {}".format(len(self.chunk_inputs.keys())))

    def __len__(self):
        return len(self.chunk_inputs)

    def __getitem__(self, uid):
        feature_path, label_path = self.chunk_inputs[uid]
        X_ss = kaldiio.load_mat(feature_path)
        T_ss = kaldiio.load_mat(label_path)
        T_ss = T_ss.reshape(X_ss.shape[0], -1)

        order = np.arange(X_ss.shape[0])
        np.random.shuffle(order)

        return uid, X_ss, T_ss, order
