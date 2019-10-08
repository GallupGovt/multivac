#!usr/bin/env/python
import numpy as np

import torch
import torch.utils.data as data


def read_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        data.append(l)
    return data


class GeneratorDataset(data.Dataset):
    def __init__(self, data_file):
        super(GeneratorDataset, self).__init__()
        data = torch.LongTensor(np.asarray(read_file(data_file), dtype=np.int64))
        self.X = torch.cat([torch.zeros(data.size(0), 1).long(), data], dim=1)
        self.y = torch.cat([data, torch.zeros(data.size(0), 1).long()], dim=1)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class DiscriminatorDataset(data.Dataset):
    def __init__(self, real_data_file, fake_data_file):
        super(DiscriminatorDataset, self).__init__()
        real_data = read_file(real_data_file)
        fake_data = read_file(fake_data_file)
        data = real_data + fake_data
        labels = [1 for _ in range(len(real_data))] + [0 for _ in range(len(fake_data))]
        self.X = torch.LongTensor(np.asarray(data, dtype=np.int64))
        self.y = torch.LongTensor(np.asarray(labels, dtype=np.int64))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

