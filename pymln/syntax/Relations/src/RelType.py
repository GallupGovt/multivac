
from . import Token, TreeNode

class RelType(object):
    _relTypes = []
    _relTypeStr_idx = {}

    def __init__(self):
        self._str = None
        self._type = ''

    def getType(self):
        return self._type

    def getRelType(target):
        if target is None:
            return None
        elif isinstance(target,int):
            return RelType.relTypes[idx]
        else:
            s = RelType.genTypeStr(target)

            try:
                _ = _relTypeStr_idx[s]
            except KeyError:
                t = RelType()
                t._str = s
                
                if target.getToken().isContent():
                    t._type = 'C'
                else:
                    t._type = 'N'

                RelType.relTypes.append(t)
                RelType.relTypeStr_idx[s] = len(RelType.relTypes) - 1

        return RelType.relTypeStr_idx[s]

    def genTypeStr(tn):
        type_str = '('
        type_str += tn.getToken().toString()
        children = tn.getChildren()

        if len(children) > 0:
            for child in children:
                type_str += ' (' + child
                tns = children[child]

                for node in tns:
                    type_str += ' ' + genTypeStr(node)

                type_str += ')'

        type_str += ')'

        return type_str

    def compareTo(self, z):
        this = sum([ord(x) for x in self._str])
        that = sum([ord(x) for x in z.toString()])
        result = this - that

        return result

    def equals(self, o):
        return self.compareTo(o)==0

    def toString(self):
        return self._str

