

class Token(object):
    def __init__(self, pos, lemma, form=None):
        self._pos = pos
        self._lemma = lemma

        if form is None:
            self._form = lemma
        else:
            self._form = form

        self._tkn_cnt = dict()

    def __repr__(self):
        return self.toString()


    def getForm(self):
        return self._form

    
    def getPOS(self):
        return self._pos

    
    def getLemma(self):
        return self._lemma


    def isContent(pos=None):
        if pos is None:
            pos = self._pos
        
        result = pos in ['J','R','V','N']

        return result


    def isVerb(self):
        return self._pos[0] == 'V'


    def isNoun(self):
        return (self._pos[0] == 'N') | (self._pos.startswith('PRP'))


    def compareTo(self, t):
        this = sum([ord(x) for x in self._lemma])
        that = sum([ord(x) for x in t.getLemma()])
        result = this - that

        if result == 0:
            this = sum([ord(x) for x in self.pos])
            that = sum([ord(x) for x in t.getPOS()])
            result = this - that
        return result


    def equals(self, t):
        return (self._pos == t.getPOS()) & (self._lemma == t.getLemma())


    def hashCode(self):
        return hash(self)


    def toString(self):
        return (self._pos + ":" + self._lemma)
