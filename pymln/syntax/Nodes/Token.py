
class Token(object):

    contentPOS = set(['J','R','V','N'])

    def isContent(t):
        return t._pos[0] in Token.contentPOS

    def isVerb(t):
        return t._pos[0] == 'V'

    def isNoun(t):
        return (t._pos[0] == 'N') | (self._pos.startswith('PRP'))


    def __init__(self, pos, lemma, form=None):
        self._pos = pos

        if Token.isContent(self):
            self._pos = pos[0]

        self._lemma = lemma

        if form is None:
            self._form = lemma
        else:
            self._form = form

    def __hash__(self):
        return hash(self.toString())

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __str__(self):
        return self.toString()

    def getForm(self):
        return self._form

    def getPOS(self):
        return self._pos

    def getLemma(self):
        return self._lemma

    def compareTo(self, t):
        this = sum([ord(x) for x in self._lemma])
        that = sum([ord(x) for x in t._lemma])
        result = this - that

        if result == 0:
            this = sum([ord(x) for x in self._pos])
            that = sum([ord(x) for x in t._pos])
            result = this - that
        return result

    def equals(self, t):
        return (self._pos == t._pos) & (self._lemma == t._lemma)

    def hashCode(self):
        return hash(self)

    def toString(self):
        return (self._pos + ":" + self._lemma)
