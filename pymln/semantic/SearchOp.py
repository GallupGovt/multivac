
from semantic import Clust

class SearchOp(object):
    OP_MERGE_CLUST = '0'
    OP_MERGE_ROLE  = '1'
    OP_COMPOSE     = '2'

    def __init__(self):
        self._op = ''
        self._clustIdx1 = None
        self._clustIdx2 = None
        self._clustIdx = None
        self._argIdx1 = None
        self._argIdx2 = None
        self._parClustIdx = None
        self._chdClustIdx = None
        self._str = None

    def compareTo(self, z):
        this = sum([ord(x) for x in self.toString()])
        that = sum([ord(x) for x in z.toString()])
        result = this - that
        
        return result

    def equals(self, o):
        if not isinstance(o, SearchOp):
            return False
        else:
            return self.compareTo(o) == 0

    def toString(self):
        if self._str is None:
            self.genString()

        return self._str

    def genString(self):
        self._str = "OP_{}:".format(self._op)

        if self._op == OP_MERGE_CLUST:
            c1 = Clust.getClust(self._clustIdx1)
            c2 = Clust.getClust(self._clustIdx2)
            self._str += "{} == {}".format(c1.toString(), c2.toString())
        elif self._op == OP_MERGE_ROLE:
            self._str += "{}:{}:{}".format(self._clustIdx, 
                                           self._argIdx1, 
                                           self._argIdx2)
        elif self._op == OP_COMPOSE:
            rc = Clust.getClust(self._parClustIdx)
            ac = Clust.getClust(self._chdClustIdx)
            self._str += "{} ++ {}".format(rc.toString(), ac.toString())


