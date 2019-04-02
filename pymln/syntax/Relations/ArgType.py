
from multivac.pymln.syntax.Relations import RelType

class ArgType(object):
    argTypes = []
    # Dictionary mapping {str: int}
    argTypeStr_idx = {}

    def __init__(self, target):
        s = target.toString()
        self._dep = target.getDep()
        self._dep2 = target.getDep2()
        self._str = None

        if target.getTreeRoot() is not None:
            self._relTypeIdx = RelType.getRelType(target.getTreeRoot())
        else:
            self._relTypeIdx = -1

        self._str = self.toString()
        ArgType.argTypes.append(self)
        i = len(ArgType.argTypes) - 1
        ArgType.argTypeStr_idx[s] = i

    def __hash__(self):
        return hash(self.toString())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()

    def getArgType(target):
        if isinstance(target, int):
            return ArgType.argTypes[target]
        elif not isinstance(target, str):
            s = target.toString()

            if s not in ArgType.argTypeStr_idx:
                t = ArgType(target)

        return ArgType.argTypeStr_idx[s]

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

    def toString(self):
        if self._str is None:
            self._str = '<' + self._dep

            if self._relTypeIdx >= 0:
                self._str += ':{}:{}'.format(self._relTypeIdx,self._dep2)

            self._str += '>'

        return self._str




