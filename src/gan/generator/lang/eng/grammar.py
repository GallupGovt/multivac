"""
English grammar and typing system
"""

from NL2code.lang.grammar import Grammar

class EnglishGrammar(Grammar):
    def __init__(self, rules):
        super(EnglishGrammar, self).__init__(rules)

    def is_value_node(self, node):
       return False
