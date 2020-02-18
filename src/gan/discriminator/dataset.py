import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from multivac.src.gan.utilities.vocab import Vocab
from .tree import Tree

# Dataset class for MULTIVAC dataset
class MULTIVACDataset(data.Dataset):
    def __init__(self, path, vocab):
        super().__init__()
        self.vocab = vocab
        self.sentences = self.read_sentences(os.path.join(path, 'text.toks'))
        #self.trees = MULTIVACDataset.read_trees(os.path.join(path, 'text.parents'))
        self.labels = MULTIVACDataset.read_labels(os.path.join(path, 'cat.txt'))
        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sent = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        
        return (sent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]

        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split())
        try:
            result = torch.tensor(indices, dtype=torch.long, device='cpu')
        except:
            import pdb; pdb.set_trace()

        return result

    @staticmethod
    def read_trees(filename):
        with open(filename, 'r') as f:
            trees = [MULTIVACDataset.read_tree(line) for line in tqdm(f.readlines())]

        return trees

    @staticmethod
    def read_tree(line):
        if isinstance(line, list):
            parents = line
        else:
            parents = list(map(int, line.split()))
        
        trees = dict()
        root = None

        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None

                while True:
                    parent = parents[idx - 1]

                    if parent == -1:
                        break

                    tree = Tree()

                    if prev is not None:
                        tree.add_child(prev)
                    
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent

        return root

    @staticmethod
    def read_labels(filename):
        with open(filename, 'r') as f:
            labels = list(map(float, f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')

        return labels
