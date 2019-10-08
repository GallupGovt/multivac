"""
English grammar and typing system
"""

from multivac.src.gan.generator.lang.grammar import Grammar

class EnglishGrammar(Grammar):
    def __init__(self, rules):
        super(EnglishGrammar, self).__init__(rules)

