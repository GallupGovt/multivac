#!usr/bin/env/python
import numpy as np

import torch
import torch.utils.data as data

from multivac.src.gan.generator.dataset import DataEntry, Vocab, DataSet
from multivac.src.gan.generator.lang.eng.eng_dataset import generate_dataset


def read_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        data.append(l)
    return data


class GeneratorDataset(DataSet):
    def __init__(self, annot_file, text_file, vocab, grammar):
        super().__init__(vocab, vocab, grammar, name='generated_samples')

        with open(data_file, "r") as f:
            for i, line in enumerate(f.readlines()):
                seed_seq, 

                query_tokens_data = [query_to_data(seed_seq, vocab)]
                example = namedtuple('example', 
                                     ['query_tokens', 'data'])(query_tokens=query, 
                                                               data=query_tokens_data)
                example = DataEntry()

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

