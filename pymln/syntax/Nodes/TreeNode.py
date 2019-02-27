
from syntax.Nodes import Token

class TreeNode(object):
    # dictionary mapping {str: TreeNode}
    id_treeNodes = {}

    def getTreeNode(idx):
        return TreeNode.id_treeNodes[idx]


    def __init__(self, idx, tkn):
        self._id = idx
        self._tkn = tkn
        self._children = {}
        TreeNode.id_treeNodes[idx] = self

    def addChild(self, dep, child):
        try:
            tns = self._children[dep]
        except KeyError:
            tns = set(child)
            self._children[dep] = tns
        else:
            self._children[dep] = tns.add(child)

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

    def equals(self, o):
        return self.compareTo(o) == 0

    def toString(self):
        return self._tkn.toString()

    def getTreeStr(self):
        id_str = {}

        if (len(self._children) > 0):
            for dep in self._children.keys():
                nodes = self._children[dep]
                s = ''
                for node in nodes:
                    if dep.startswith('prep_') or dep.startswith('conj_'):
                        s = dep[5:] + ' '
                    s = s + node.getTreeStr()
                    id_str[node.getId()] = s

        id_str[self._id] = self._tkn.getLemma()
        result = ' '.join([id_str[x] for x in id_str.keys()])

        return result


