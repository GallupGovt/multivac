
class Answer(object):
    def __init__(self, sid, rst):
        self._sid = sid
        self._rst = rst

    def __hash__(self):
        return hash(self.toString())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def __str__(self):
        return self.toString()

    def getSentId(self):
        return self._sid

    def getRst(self):
        return self._rst

    def compareTo(self, a):
        result = 0

        if self._rst != a.getRst():
            if self._rst < a.getRst():
                result -= 1
            else:
                result += 1
        elif self._sid != a.getSentId():
            if self._sid < a.getSentId():
                result -= 1
            else:
                result += 1

        return result

    def toString(self):
        return ' '.join([self._sid, self._rst])

