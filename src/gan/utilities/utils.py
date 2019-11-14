
import os
import pickle
import math

import torch

from multivac.src.gan.utilities.vocab import Vocab


# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile, lowercase=True):
    vocab = set()

    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()

                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)

    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

def deserialize_from_file(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    
    return obj

# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path, lowercase=True):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab', lower=lowercase)

        return vocab, vectors
    elif path.endswith('.pkl'):
        print('==> File found, loading to memory')

        with open(path, "rb") as f:
            glove = pickle.load(f)

        vectors = torch.from_numpy(glove['embeddings']).float()
        vocab = Vocab(data=glove['vocab'], lower=lowercase)

        return vocab, vectors

    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')


    count = sum(1 for line in open(path + '.txt', 'r', encoding='utf8', errors='ignore'))

    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])

    words = [None] * (count)
    vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')

    with open(path + '.txt', 'r', encoding='utf8', errors='ignore') as f:
        idx = 0

        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            values = list(map(float, contents[1:]))
            vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
            idx += 1

    with open(path + '.vocab', 'w', encoding='utf8', errors='ignore') as f:
        for word in words:
            f.write(word + '\n')

    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')

    return vocab, vectors

def serialize_to_file(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def typename(x):
    if isinstance(x, str):
        return x
    return x.__name__
