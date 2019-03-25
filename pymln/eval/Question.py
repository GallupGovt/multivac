
class Question(object):
    def __init__(self, rel, arg, dep):
        self._rel = rel
        self._dep = arg
        self._arg = dep
        self._argClustIdxSeq = None

    def __hash__(self):
        return hash(self.toString())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __str__(self):
        return self.toString()

    def getRel(self):
        return self._rel

    def getArg(self):
        return self._arg

    def getDep(self):
        return self._dep

    def compareTo(self, q):
        this = sum([ord(x) for x in self._dep])
        that = sum([ord(x) for x in q.getDep()])
        result = this - that

        if result == 0:
            this = sum([ord(x) for x in self._rel])
            that = sum([ord(x) for x in q.getRel()])
            result = this - that

            if result == 0:
                this = sum([ord(x) for x in self._arg])
                that = sum([ord(x) for x in q.getArg()])
                result = this - that

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
    
