
from multivac.pymln.syntax.Nodes import Token, TreeNode

class RelType(object):
    relTypes = []
    # Dictionary mapping {str: int} tracking RelType strings and
    # their unique indices.
    relTypeStr_idx = {}

    def __init__(self, target):
        self._str = RelType.genTypeStr(target)

        if Token.isContent(target._tkn):
            self._type = 'C'
        else:
            self._type = 'N'

        RelType.relTypeStr_idx[self._str] = len(RelType.relTypes)
        RelType.relTypes.append(self)

    def __hash__(self):
        return hash(self.toString())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def getType(self):
        return self._type

    def getRelType(target):
        if target is None:
            result = None
        elif isinstance(target,int):
            result = RelType.relTypes[target]
        else:
            type_str = RelType.genTypeStr(target)

            if type_str not in RelType.relTypeStr_idx:
                t = RelType(target)

            result = RelType.relTypeStr_idx[type_str]

        return result

    def genTypeStr(tn):
        type_str = '('
        type_str += tn.toString()
        children = tn.getChildren()

        if len(children) > 0:
            for child in children:
                type_str += ' (' + child
                tree_nodes = children[child]

                for node in tree_nodes:
                    type_str += ' ' + RelType.genTypeStr(node)

                type_str += ')'

        type_str += ')'

        return type_str

    def compareTo(self, z):
        this = sum([ord(x) for x in self._str])
        that = sum([ord(x) for x in z.toString()])
        result = this - that

        return result

    def toString(self):
        return self._str

