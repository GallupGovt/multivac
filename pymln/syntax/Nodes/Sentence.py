
from syntax.Nodes.Token import Token

class Sentence(object):
    def __init__(self):
        ''' 
        Each sentence consists of:
        _tokens: A list of individual tokens in the sentence, containing POS, 
        lemma, and actual form of the word/item.
        _tkn_children: A dictionary mapping parents (denoted by the integer 
        keys) to children (sets of integer, string tuples).
        _tkn_par: A dictionary mapping children (denoted by integer keys) to 
        parents (tuples of string, integer values)
        '''
        self._tokens = []

        # Dictionary mapping {int: set((int, str))}
        self._tkn_children = {0: set()}
        # Dictionary mapping {int: (str, int)}
        self._tkn_par = {}

        return None


    def __repr__(self):
        return ('Tokens: ' + str([str(x) for x in self._tokens]))

    def get_tokens(self, idx=None):
        '''
        Return Tokens at the specified indices.
        '''
        if idx is None:
            return self._tokens
        elif isinstance(idx, list):
            return [self._tokens[i] for i in idx]
        elif isinstance(idx, int):
            return self.get_token(idx)
        else:
            raise ValueError


    def get_token(self, idx):
        '''
        Return the Token() at the specified index.
        '''
        return self._tokens[idx]

    def add_token(self, tok):
        '''
        Append the Token() to the list of _tokens.
        '''
        assert isinstance(tok, Token)
        self._tokens.append(tok)

        return None

    def get_children(self, parent=None):
        '''
        Return the child/children of the parent specified by the given key. If 
        no key specified, return them all.
        '''
        if parent is not None:
            if parent in self._tkn_children:
                c = self._tkn_children[parent]
            else:
                c = None
        else:
            c = self._tkn_children

        return c

    def set_children(self, parent, kids):
        '''
        Add the child/children specified by the key/kids key/value pair.
        '''
        assert isinstance(kids, set)
        self._tkn_children[parent] = kids

        return None

    def add_child(self, parent, kid):
        '''
        Add/update the child/children specified by the key/kids key/value pair.
        '''
        assert parent in self._tkn_children
        self._tkn_children[parent].add(kid)

        return None

    def get_parent(self, kid):
        '''
        Return the parent of the child specified by the given key.
        '''
        if kid in self._tkn_par:
            return self._tkn_par[kid]
        else:
            return None

    def set_parent(self, kid, parent):
        '''
        Add/update the parent specified by the given key/parent value pair.
        '''
        assert isinstance(parent, tuple)
        self._tkn_par[kid] = parent

        return None



