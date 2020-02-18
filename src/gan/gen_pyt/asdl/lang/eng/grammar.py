"""
English grammar and typing system
"""
from collections import OrderedDict
from copy import deepcopy
from multivac.src.gan.gen_pyt.asdl.lang.grammar import Grammar
from multivac.src.gan.gen_pyt.asdl.asdl import ASDLGrammar, ASDLProduction, \
                                               Field, ASDLCompositeType, \
                                               ASDLPrimitiveType, \
                                               ASDLConstructor
from multivac.src.gan.gen_pyt.asdl.lang.eng.eng_asdl_helper \
    import english_ast_to_asdl_ast, asdl_ast_to_english

BRACKET_TYPES = {
    ASDLPrimitiveType('-LRB-'): '(',
    ASDLPrimitiveType('-RRB-'): ')',
    ASDLPrimitiveType('-LCB-'): '{',
    ASDLPrimitiveType('-RCB-'): '}',
    ASDLPrimitiveType('-LSB-'): '[',
    ASDLPrimitiveType('-RSB-'): ']',
}

TERMINAL_TYPES = {
    ASDLPrimitiveType('CC'),    # Coordinating conjunction
    ASDLPrimitiveType('CD'),    # Cardinal number
    ASDLPrimitiveType('DT'),    # Determiner
    ASDLPrimitiveType('EX'),    # Existential there
    ASDLPrimitiveType('FW'),    # Foreign word
    ASDLPrimitiveType('IN'),    # Preposition or subordinating conjunction
    ASDLPrimitiveType('JJ'),    # Adjective
    ASDLPrimitiveType('JJR'),   # Adjective, comparative
    ASDLPrimitiveType('JJS'),   # Adjective, superlative
    ASDLPrimitiveType('LS'),    # List item marker
    ASDLPrimitiveType('MD'),    # Modals
    ASDLPrimitiveType('NN'),    # Noun, singular or mass
    ASDLPrimitiveType('NNS'),   # Noun, plural
    ASDLPrimitiveType('NNP'),   # Proper noun, singular
    ASDLPrimitiveType('NNPS'),  # Proper noun, plural
    ASDLPrimitiveType('PDT'),   # Predeterminer
    ASDLPrimitiveType('POS'),   # Possessive ending
    ASDLPrimitiveType('PRP'),   # Personal pronoun
    ASDLPrimitiveType('PRP$'),  # Possessive pronoun (prolog version PRP-S)
    ASDLPrimitiveType('RB'),    # Adverb
    ASDLPrimitiveType('RBR'),   # Adverb, comparative
    ASDLPrimitiveType('RBS'),   # Adverb, superlative
    ASDLPrimitiveType('RP'),    # Particle
    ASDLPrimitiveType('SYM'),   # Symbol
    ASDLPrimitiveType('TO'),    # to
    ASDLPrimitiveType('UH'),    # Interjection
    ASDLPrimitiveType('VB'),    # Verb, base form
    ASDLPrimitiveType('VBD'),   # Verb, past tense
    ASDLPrimitiveType('VBG'),   # Verb, gerund or present participle
    ASDLPrimitiveType('VBN'),   # Verb, past participle
    ASDLPrimitiveType('VBP'),   # Verb, non-3rd person singular present
    ASDLPrimitiveType('VBZ'),   # Verb, 3rd person singular present
    ASDLPrimitiveType('WDT'),   # Wh-determiner
    ASDLPrimitiveType('WP'),    # Wh-pronoun
    ASDLPrimitiveType('WP$'),   # Possessive wh-pronoun (prolog version WP-S)
    ASDLPrimitiveType('WRB')    # Wh-adverb
}

class EnglishGrammar(Grammar):
    def __init__(self, rules):
        super().__init__(rules)

        self.terminal_types.update(TERMINAL_TYPES)
        self.terminal_types.update(BRACKET_TYPES)

class EnglishASDLGrammar(ASDLGrammar):
    """
    Collection of types, constructors and productions
    """
    def __init__(self, grammar=None, productions=None):
        # productions are indexed by their head types
        self._productions = OrderedDict()
        self._constructor_production_map = dict()

        if productions is not None:
            english_prods = set(productions)

            for prod in english_prods:
                if prod.type not in self._productions:
                    self._productions[prod.type] = list()
                self._productions[prod.type].append(prod)
                self._constructor_production_map[prod.constructor.name] = prod

            self.root_type = ASDLCompositeType("ROOT")
        elif grammar is not None:
            if isinstance(grammar, ASDLGrammar):
                self = grammar
                return

            for rule in grammar.rules:
                fields = []

                for child in rule.children:
                    if grammar.is_terminal(child):
                        child_type = ASDLPrimitiveType(child.type)
                    else:
                        child_type = ASDLCompositeType(child.type)

                    fields.append(Field(child.type, child_type, 'single'))

                constructor = ASDLConstructor(rule.type, fields)
                production  = ASDLProduction(ASDLCompositeType(rule.type), 
                                             constructor)

                if production.type not in self._productions:
                    self._productions[production.type] = list()

                self._productions[production.type].append(production)
                self._constructor_production_map[constructor.name] = production

            self.root_type = ASDLCompositeType(grammar.root_node.type)

        self.size = sum(len(head) for head in self._productions.values())
        self.terminal_types = set(self.primitive_types)
        self.terminal_types.update(TERMINAL_TYPES)
        self.terminal_types.update(BRACKET_TYPES.keys())

        self._types = sorted(self.terminal_types.union(set(self.types)), 
                             key=lambda x: x.name)

        # get entities to their ids map
        self.prod2id = {prod: i for i, prod in enumerate(self.productions)}
        self.type2id = {type: i for i, type in enumerate(self.types)}
        self.field2id = {field: i for i, field in enumerate(self.fields)}

        self.id2prod = {i: prod for i, prod in enumerate(self.productions)}
        self.id2type = {i: type for i, type in enumerate(self.types)}
        self.id2field = {i: field for i, field in enumerate(self.fields)}

    @staticmethod
    def from_text(text, parser):
        productions = set()

        if isinstance(text, list):
            text = '\n'.join(text)

        for s in text:
            try:
                p = parser.get_parse(s)['sentences'][0]['parse']
            except:
                continue
            try:
                parse_tree = english_ast_to_asdl_ast(p.parse_string)
            except: 
                continue

            productions.update(parse_tree.get_productions())

        productions = sorted(productions, key=lambda x: x.__repr__)

        grammar = EnglishASDLGrammar(productions=productions)




