
from syntax.Nodes import Token, TreeNode

class RelType(object):
    relTypes = []
    # Dictionary mapping {str: int} tracking RelType strings and 
    # their unique indices. 
    relTypeStr_idx = {}

    def __init__(self, target):
        self._str = RelType.genTypeStr(target)

        if target.getToken().isContent():
            self._type = 'C'
        else:
            self._type = 'N'

        RelType.relTypes.append(self)
        RelType.relTypeStr_idx[self._str] = len(RelType.relTypes) - 1


    def getType(self):
        return self._type

    def getRelType(target):
        if target is None:
            result = None
        elif isinstance(target,int):
            result = RelType.relTypes[target]
        else:
            t = RelType(target)
            result = RelType.relTypeStr_idx[t.toString()]

        return result

    def genTypeStr(tn):
        type_str = '('
        type_str += tn.getToken().toString()
        children = tn.getChildren()

        if len(children) > 0:
            for child in children:
                type_str += ' (' + child
                tns = children[child]

                for node in tns:
                    type_str += ' ' + RelType.genTypeStr(node)

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

