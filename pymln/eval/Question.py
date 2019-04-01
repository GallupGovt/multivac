
class Question(object):
    def __init__(self, rel, arg, dep):
        self._rel = rel
        self._dep = dep
        self._arg = arg
        self._argClustIdxSeq = None

    def __hash__(self):
        return hash(self.toString())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __lt__(self):
        return self.compareTo(other) < 0

    def __str__(self):
        return self.toString()

    def getRel(self):
        return self._rel

    def getArg(self):
        return self._arg

    def getDep(self):
        return self._dep

    def compareTo(self, q):
        result = 0

        if self._dep != q.getDep():
            if self._dep < q.getDep():
                result -= 1
            else:
                result += 1
        elif self._rel != q.getRel():
            if self._rel < q.getRel():
                result -= 1
            else:
                result += 1
        elif self._arg != q.getArg():
            if self._arg < q.getArg():
                result -= 1
            else:
                result += 1

        return result

    def getPattern(self):
        if self._dep == 'nsubj':
            return ' '.join([self._arg, self._rel])
        elif self._dep == 'dobj':
            return ' '.join([self._rel, self._arg])
        else:
            return None

    def toString(self):
        if self._dep == 'nsubj':
            return "What does {} {}?".format(self._arg, self._rel)
        elif self._dep == 'dobj':
            return "What {}s {}?".format(self._rel, self._arg)
        else:
            return "{} ::: {} ::: {}".format(self._rel, self._dep, self._arg)
    
