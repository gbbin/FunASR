import logging

import numpy as np
from torch.utils.data import Dataset
import kaldiio

class DiarizationDataset(Dataset):

    def __init__(self, data_file):
        self.data_file = data_file
        self.chunk_indices = []

        self.lines = [line.strip().split() for line in open(data_file)]
        self.chunk_indices = [(x[0], x[1], x[2]) for x in self.lines]
        logging.info("total chunks: {}".format(len(self.chunk_indices)))

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        sample_name, feature_path, label_path = self.chunk_indices[idx]
        Y_ss = kaldiio.load_mat(feature_path)
        T_ss =  kaldiio.load_mat(label_path)

        order = np.arange(Y_ss.shape[0])
        np.random.shuffle(order)

        return Y_ss, T_ss, order
