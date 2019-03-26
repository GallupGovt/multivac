
# from collections import OrderedDict
from sortedcontainers import SortedDict, SortedSet
from syntax.Nodes import Token

class TreeNode(object):
    # map {str: TreeNode}
    id_treeNodes = {}

    def __init__(self, tree_node_id, token):
        self._id = tree_node_id
        self._tkn = token
        # map {str: set(TreeNodes)}
        self._children = SortedDict()
        TreeNode.id_treeNodes[tree_node_id] = self

    def __hash__(self):
        return hash(self.toString())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()

    def addChild(self, dep, child):
        if dep not in self._children:
            self._children[dep] = SortedSet()

        self._children[dep].add(child)

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

        return self._tkn.compareTo(z._tkn)

    def toString(self):
        return self._tkn.toString()

    def getTreeNode(tree_node_id):
        return TreeNode.id_treeNodes[tree_node_id]

    def getTreeStr(self):
        id_str = SortedDict()

        if (len(self._children) > 0):
            for dep, nodes in self._children.items():
                s = ''

                for node in nodes:
                    if dep.startswith('prep_') or dep.startswith('conj_'):
                        s = dep[5:] + ' '
                    s = s + node.getTreeStr()
                    id_str[node.getId()] = s

        id_str[self._id] = self._tkn.getLemma()
        result = ' '.join(id_str.values())

        return result


