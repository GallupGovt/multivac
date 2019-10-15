"""
English grammar and typing system
"""

from multivac.src.gan.generator.lang.grammar import Grammar

BRACKET_TYPES = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LCB-': '{',
    '-RCB-': '}',
    '-LSB-': '[',
    '-RSB-': ']',
}

TERMINAL_TYPES = {
    'CC',    # Coordinating conjunction
    'CD',    # Cardinal number
    'DT',    # Determiner
    'EX',    # Existential there
    'FW',    # Foreign word
    'IN',    # Preposition or subordinating conjunction
    'JJ',    # Adjective
    'JJR',   # Adjective, comparative
    'JJS',   # Adjective, superlative
    'LS',    # List item marker
    'MD',    # Modals
    'NN',    # Noun, singular or mass
    'NNS',   # Noun, plural
    'NNP',   # Proper noun, singular
    'NNPS',  # Proper noun, plural
    'PDT',   # Predeterminer
    'POS',   # Possessive ending
    'PRP',   # Personal pronoun
    'PRP$',  # Possessive pronoun (prolog version PRP-S)
    'RB',    # Adverb
    'RBR',   # Adverb, comparative
    'RBS',   # Adverb, superlative
    'RP',    # Particle
    'SYM',   # Symbol
    'TO',    # to
    'UH',    # Interjection
    'VB',    # Verb, base form
    'VBD',   # Verb, past tense
    'VBG',   # Verb, gerund or present participle
    'VBN',   # Verb, past participle
    'VBP',   # Verb, non-3rd person singular present
    'VBZ',   # Verb, 3rd person singular present
    'WDT',   # Wh-determiner
    'WP',    # Wh-pronoun
    'WP$',   # Possessive wh-pronoun (prolog version WP-S)
    'WRB'    # Wh-adverb
}

class EnglishGrammar(Grammar):
    def __init__(self, rules):
        super().__init__(rules)

        self.terminal_types.update(TERMINAL_TYPES)
        self.terminal_types.update(BRACKET_TYPES)
