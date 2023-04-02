import logging

import numpy as np
from torch.utils.data import Dataset
import kaldiio

class DiarizationDataset(Dataset):

    def __init__(self, data_file):
        self.data_file = data_file
        self.lines = [line.strip().split() for line in open(data_file)]
        self.chunk_inputs = [(x[0], x[1], x[2]) for x in self.lines]
        logging.info("total chunks: {}".format(len(self.chunk_inputs)))

    def __len__(self):
        return len(self.chunk_inputs)

    def __getitem__(self, idx):
        sample_name, feature_path, label_path = self.chunk_inputs[idx]
        Y_ss = kaldiio.load_mat(feature_path)
        T_ss =  kaldiio.load_mat(label_path)

        order = np.arange(Y_ss.shape[0])
        np.random.shuffle(order)

        return sample_name, Y_ss, T_ss, order
