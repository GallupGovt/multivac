# vocab object from harvardnlp/opennmt-py
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
        if isinstance(item, int):
            return self.getLabel(item)
        else:
            return self.getIndex(item)

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.idxToLabel
        else:
            return item in self.labelToIdx

    @property
    def size(self):
        return len(self.idxToLabel)

    def __setitem__(self, key, value):
        self.labelToIdx[key] = value

    def __len__(self):
        return len(self.labelToIdx)

    def __iter__(self):
        return iter(list(self.labelToIdx.keys()))

    @property
    def unk(self):
        return self.labelToIdx['<unk>']

    @property
    def eos(self):
        return self.labelToIdx['<eos>']

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        idx = 0
        for line in open(filename, 'r', encoding='utf8', errors='ignore'):
            token = line.rstrip('\n')
            self.add(token)
            idx += 1

    def getIndex(self, key, default=None):
        key = key.lower() if self.lower else key

        if key in self.labelToIdx:
            return self.labelToIdx[key]
        else:
            return default

    def getLabel(self, idx, default=None):
        if idx in self.idxToLabel:
            return self.idxToLabel[idx]
        else:
            return default

    # Mark this `label` and `idx` as special
    def addSpecial(self, label, idx=None):
        idx = self.add(label)
        self.special += [idx]

    # Mark all labels in `labels` as specials
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

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

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord=None, bosWord=None, eosWord=None):
        if unkWord is None:
            unkWord = self.unk

        vec = []

        if bosWord is not None:
            vec += [self.getIndex(bosWord)]

        unk = self.getIndex(unkWord)
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
