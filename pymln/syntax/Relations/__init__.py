
from ..Nodes import Token, TreeNode

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




