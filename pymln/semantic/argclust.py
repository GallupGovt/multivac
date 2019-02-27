

class ArgClust(object):
    def __init__(self):
        # Dictionary mapping {int: int}
        self._argTypeIdx_cnt = {}
        # Dictionary mapping {int: int}
        self._chdClustIdx_cnt = {}
        # Dictionary mapping {int: int}
        self._argNum_cnt = {}
        self._ttlArgCnt = 0
        self._partRootTreeNodeIds = set()

    def toString(self):
        s = ''
        for k, v in self._argTypeIdx_cnt.items():
            if len(s) > 0:
                s += ' '
            s += '{}:{}'.format(ArgType.getArgType(k), c)

        return s


