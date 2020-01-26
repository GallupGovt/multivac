
from collections import Counter
from itertools import chain

class Vocab(object):
    def __init__(self, filename=None, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            self.addSpecials(data)
        if filename is not None:
            self.loadFile(filename)

        self.add('<pad>')
        self.add('<unk>')
        self.add('<eos>')

    def __getitem__(self, item):
            return self.labelToIdx.get(item, self.unk)

    def __contains__(self, item):
            return item in self.labelToIdx

    def __setitem__(self, key, value):
        self.labelToIdx[key] = value

    def __len__(self):
        return len(self.labelToIdx)

    def __iter__(self):
        return iter(list(self.labelToIdx.keys()))

    def __eq__(self, other):
        return all([self.idxToLabel == other.idxToLabel, 
                    self.labelToIdx == other.labelToIdx,
                    self.lower      == other.lower, 
                    self.special    == other.special])

    @property
    def pad(self):
        return self.labelToIdx['<pad>']

    @property
    def size(self):
        return len(self.idxToLabel)

    @property
    def unk(self):
        return self.labelToIdx['<unk>']

    @property
    def eos(self):
        return self.labelToIdx['<eos>']

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label):
        label = label.lower() if self.lower else label

        if label in self.labelToIdx:
            idx = self.labelToIdx[label]
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        return idx

    def add_from_data(self, label, idx=None):
        if idx:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            idx = self.add(label)
        
    # Mark this `label` and `idx` as special
    def addSpecial(self, label):
        idx = self.add(label)
        self.special += [idx]

    # Mark all labels in `labels` as specials
    def addSpecials(self, labels):
        for label in labels:
            if isinstance(label, tuple):
                self.add_from_data(*label)
            else:
                self.addSpecial(label)

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord=None, bosWord=None, eosWord=None):
        if unkWord is None:
            unk = self.unk
        else:
            unk = self.getIndex(unkWord)

        vec = []

        if bosWord is not None:
            vec += [self.getIndex(bosWord)]

        
        vec += [self.getIndex(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.getIndex(eosWord)]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop=None):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels

    @staticmethod
    def from_corpus(corpus, size=None, freq_cutoff=0):
        vocab = Vocab()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)

        if size is not None:
            top_k_words = top_k_words[:size]

        words_not_included = []

        for word in top_k_words:
            if word_freq[word] >= freq_cutoff:
                vocab.add(word)
            else:
                words_not_included.append(word)

            if len(vocab) == size:
                break

        return vocab

    @staticmethod
    def from_dict(vocab):
        new_vocab = Vocab()

        for key, value in vocab.items():
            setattr(new_vocab, key, value)

        return new_vocab

    def getIndex(self, key, default=None):
        key = key.lower() if self.lower else key

        return self.labelToIdx.get(key, default)

    def getLabel(self, idx, default=None):
        return self.idxToLabel.get(idx, default)

    def is_unk(self, word):
        return word not in self

    # Load entries from a file.
    def loadFile(self, filename):
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            [self.add(x.strip()) for x in f.readlines() if len(x.strip()) > 0]

    def size(self):
        return len(self.idxToLabel)
