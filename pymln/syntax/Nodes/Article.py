
from syntax.Nodes import Sentence

class Article(object):
    '''
    An Article() is merely a collection of Sentences() (represented as a list)
    and an article id, which can be of any particular type but should be unique
    in a collection of Articles. 
    '''
    def __init__(self, fn=None):
        self.uid = fn
        self.sentences = []

    def __repr__(self):
        return str(self.__dict__)

    
