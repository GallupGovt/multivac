
from syntax.Relations import RelType

class Path(object):
    def __init__(self, dep, treeRoot=None, argNode=None, dep2=None):
        self._dep = dep
        self._treeRoot = treeRoot
        self._argNode = argNode
        self._dep2 = dep2
        self._argTypeIdx = -1
        self._str = None

    def getDep(self):
        return self._dep

    def getTreeRoot(self):
        return self._treeRoot

    def getArgNode(self):
        return self._argNode

    def getDep2(self):
        return self._dep2

    def getArgType(self):
        return self._argTypeIdx

    def toString(self):
        if self._str is None:
            self._str = self.genTypeStr()

        return self._str

    def genTypeStr(self):
        typ_str = '<' + self._dep

        if self._treeRoot is not None:
            rel_str = RelType.genTypeStr(self._treeRoot)
            typ_str += ':' + rel_str + ':' + self._dep2

        typ_str += '>'

        return typ_str

