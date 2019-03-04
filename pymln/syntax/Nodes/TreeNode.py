
from syntax.Nodes import Token

class TreeNode(object):
    def __init__(self, idx, tkn):
        self._id = idx
        self._tkn = tkn
        self._children = {}

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()


    def addChild(self, dep, child):
        if dep in self._children:
            self._children[dep] = tns.add(child)
        else:
            tns = set()
            tns.add(child)
            self._children[dep] = tns

        return None

    def getId(self):
        return self._id

    def getToken(self):
        return self._tkn

    def getChildren(self):
        return self._children

    def compareTo(self, z):
        if not isinstance(z, TreeNode):
            raise ValueError

        return self._tkn.compareTo(z.tkn_)

    def equals(self, obj):
        return self.compareTo(obj) == 0

    def toString(self):
        return self._tkn.toString()

    def getTreeStr(self):
        id_str = {}

        if (len(self._children) > 0):
            for dep, nodes in self._children.items():
                s = ''

                for node in nodes:
                    if dep.startswith('prep_') or dep.startswith('conj_'):
                        s = dep[5:] + ' '
                    s = s + node.getTreeStr()
                    id_str[node.getId()] = s

        id_str[self._id] = self._tkn.getLemma()
        result = ' '.join([id_str[x] for x in id_str.keys()])

        return result


