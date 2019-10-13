from collections import OrderedDict, defaultdict

from multivac.src.gan.generator.astnode import ASTNode
from multivac.src.gan.generator.lang.util import typename

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
    'MD',    # Modal
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

class Grammar(object):
    def __init__(self, rules):
        """
        instantiate a grammar with a set of production rules of type Rule
        """
        self.rules = rules
        self.rule_index = defaultdict(list)
        self.rule_to_id = OrderedDict()

        node_types = set()
        lhs_nodes = set()
        rhs_nodes = set()

        for rule in self.rules:
            self.rule_index[rule.parent].append(rule)

            # we also store all unique node types
            for node in rule.nodes:
                node_types.add(typename(node.type))

            lhs_nodes.add(rule.parent)

            for child in rule.children:
                rhs_nodes.add(child.as_type_node)

        root_node = lhs_nodes - rhs_nodes
        assert len(root_node) == 1
        self.root_node = next(iter(root_node))

        self.terminal_nodes = rhs_nodes - lhs_nodes
        self.terminal_types = set([n.type for n in self.terminal_nodes])
        self.terminal_types.update(TERMINAL_TYPES)

        self.node_type_to_id = OrderedDict()
        for i, type in enumerate(node_types, start=0):
            self.node_type_to_id[type] = i

        for gid, rule in enumerate(rules, start=0):
            self.rule_to_id[rule] = gid

        self.id_to_rule = OrderedDict((v, k) for (k, v) in list(self.rule_to_id.items()))

    def __iter__(self):
        return self.rules.__iter__()

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, lhs):
        key_node = ASTNode(lhs.type, None)  # Rules are indexed by types only
        if key_node in self.rule_index:
            return self.rule_index[key_node]
        else:
            KeyError('key=%s' % key_node)

    def get_node_type_id(self, node):
        from NL2code.astnode import ASTNode

        if isinstance(node, ASTNode):
            type_repr = typename(node.type)
            return self.node_type_to_id[type_repr]
        else:
            # assert isinstance(node, str)
            # it is a type
            type_repr = typename(node)
            return self.node_type_to_id[type_repr]

    def is_terminal(self, node):
        return node.type in self.terminal_types

    def is_value_node(self, node):
        return node.type in self.terminal_types
