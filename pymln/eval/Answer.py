
class Answer(object):
    def __init__(self, sid, rst):
        self._sid = sid
        self._rst = rst

    def __str__(self):
        return self.toString()

    def getSentId(self):
        return self._sid

    def getRst(self):
        return self._rst

    def compareTo(self, a):
        this = sum([ord(x) for x in self._rst])
        that = sum([ord(x) for x in a.getRst()])
        result = this - that

        if result == 0:
            this = sum([ord(x) for x in self._sid])
            that = sum([ord(x) for x in a.getSentId()])
            result = this - that

        return result

    def equals(self, o):
        if not isinstance(o, Answer):
            return False
        else:
            return self.compareTo(o) == 0

    def toString(self):
        return ' '.join([self._sid, self._rst])

