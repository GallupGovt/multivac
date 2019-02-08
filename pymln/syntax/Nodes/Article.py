
from . import Sentence

class Article(object):
    '''
    An Article() is merely a collection of Sentences() (represented as a list)
    and an article id, which can be of any particular type but should be unique
    in a collection of Articles. 
    '''
    def __init__(self, fn=None, sentences=[]):
        self.uid = fn
        self.sentences = sentences

    def __repr__(self):
        return str(self.__dict__)

