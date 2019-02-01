
from . import RelType

class ArgType(object):
    argTypes = []
    # Dictionary mapping {str: int}
    argTypeStr_idx = {}

    ARGTYPEIDX_SUBJ = -1
    ARGTYPEIDX_OBJ = -1
    ARGTYPEIDX_IN = -1

    def __init__(self):
        self._dep = None
        self._relTypeIdx = -1
        self._dep2 = None
        self._str = None

    def getArgType(target):
        if isinstance(target,int):
            result = ArgType.argTypes[idx]
        else:
            s = target.toString()

            if s not in ArgType.argTypeStr_idx:
                t = ArgType()
                t._dep  = p.getDep()
                t._dep2 = p.getDep2()
                t._relTypeIdx = -1

                if p.getTreeRoot() is not None:
                    t._relTypeIdx = RelType.getRelType(p.getTreeRoot())

                ArgType.argTypes.append(t)
                ati = len(ArgType.argTypes) - 1
                ArgType.argTypeStr_idx[s] = ati

                if p.getTreeRoot() is None:
                    if p.getDep() == 'nsubj':
                        ARGTYPEIDX_SUBJ = ati
                    elif p.getDep() == 'dobj':
                        ARGTYPEIDX_OBJ = ati
                    elif p.getDep() == 'prep_in':
                        ARGTYPEIDX_IN = ati

            result = ArgType.argTypeStr_idx[s]

        return result

    def compareTo(self, z):
        if self._dep is None or z.GetDep() is None:
            return None

        this = sum([ord(x) for x in self._dep])
        that = sum([ord(x) for x in z.getDep()])
        result = this - that

        if result == 0:
            result = self._relTypeIdx - z._relTypeIdx

            if result == 0:
                if self._dep2 is not None:
                    this = sum([ord(x) for x in self._dep2])

                    try:
                        that = sum([ord(x) for x in z.getDep2()])
                    except TypeError:
                        result = -1
                    else:
                        result = this - that
        
        return result

    def equals(self, o):
        return self.compareTo(o) == 0

    def toString(self):
        if self._str is None:
            self._str = '<' + self._dep

            if self._relTypeIdx >= 0:
                self._str += ':{}:{}'.format(self._relTypeIdx,self._dep2)

            self._str += '>'

        return self._str




