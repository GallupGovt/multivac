
import argparse
import pickle

from multivac.src.gan.gen_pyt.asdl.lang.eng.eng_asdl_helper import \
    english_ast_to_asdl_ast
from multivac.src.gan.gen_pyt.asdl.lang.eng.grammar import (EnglishASDLGrammar,
                                                            EnglishGrammar)
from multivac.src.gan.gen_pyt.astnode import ASTNode
from multivac.src.rdf_graph.rdf_parse import (StanfordParser, check_parse,
                                              clean_queries, stanford_parse)


def find_match_paren(s):
    count = 0

    for i, c in enumerate(s):
        if c == "(":
            count += 1
        elif c == ")":
            count -= 1

        if count == 0:
            return i


def get_eng_tree(text, depth=0, debug=False):
    ''' Takes a constituency parse string of an English sentence and creates
        an ASTNode tree from it.

        Example input:
        '(ROOT (SBARQ (WHADVP (WRB Why)) (SQ (VBP do) (NP (NNS birds)) (ADVP
        (RB suddenly)) (VP (VB appear) (SBAR (WHADVP (WRB whenever)) (S (NP
        (PRP you)) (VP (VBP are) (ADJP (JJ near))))))) (. ?)))'
    '''

    if debug:
        print(("\t" * depth + "String: '{}'".format(text)))

    try:
        tree_str = text[text.index("(") + 1:text.rfind(")")]
    except ValueError:
        print(("Malformatted parse string: '{}'".format(text)))
        raise ValueError

    next_idx = tree_str.index(" ")

    tree = ASTNode(tree_str[:next_idx])
    if debug:
        print(("\t" * depth + "Type: '{}'".format(tree.type)))

    if "(" in tree_str:
        while "(" in tree_str:
            tree_str = tree_str[tree_str.index("("):]
            next_idx = find_match_paren(tree_str) + 1
            tree.add_child(get_eng_tree(tree_str[:next_idx], depth+1, debug))
            tree_str = tree_str[next_idx + 1:]
    else:
        tree.value = tree_str[next_idx + 1:]
        if debug:
            print(("\t" * depth + "Value: " + tree.value))

    return tree


def get_grammar(parse_trees, verbose=False):
    rules = set()

    for parse_tree in parse_trees:
        parse_tree_rules, rule_parents = parse_tree.get_productions()
        for rule in parse_tree_rules:
            rules.add(rule)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = EnglishGrammar(rules)

    if verbose:
        print(('num. rules: %d', len(rules)))

    return grammar


def parse_raw(parser, query):
    try:
        query = stanford_parse(parser, query)
    except Exception:
        print('Could not parse query: {}'.format(query))
        return None

    try:
        result = get_eng_tree(query.parse_string)
    except Exception:
        print("Could not interpret query parse: {}".format(query.parse_string))
        return None

    return result


def extract_grammar(source_file, output=None, clean=False, verbose=False,
                    asdl=False):
    parse_trees = list()

    if asdl:
        parse_func = english_ast_to_asdl_ast
    else:
        parse_func = get_eng_tree

    parser = StanfordParser(annots="tokenize ssplit parse")

    with open(source_file, 'r') as f:
        queries = f.readlines()

    if clean:
        queries = clean_queries(queries, verbose)

    if verbose:
        print("Performing constituency parsing of queries")

    for i, q in enumerate(queries):
        if len(q) > 0:
            try:
                query = stanford_parse(parser, q)
            except Exception:
                print('Could not parse query {}: "{}"'.format(i, q))
                continue

        if check_parse(query):
            try:
                parse_trees.append(parse_func(query.parse_string))
            except Exception:
                print(("Could not interpret query parse {}: '{}'".format(i, query)))
                continue

        if i % 100 == 0:
            print("{} queries processed.".format(i))

    if verbose:
        print(("{} queries successfully parsed.".format(len(parse_trees))))
        print("Extracting grammar production rules.")

    if asdl:
        productions = set()

        for parse_tree in parse_trees:
            productions.update(parse_tree.get_productions())

        grammar = EnglishASDLGrammar(productions=productions)
    else:
        rules = set()

        for parse_tree in parse_trees:
            parse_tree_rules, _ = parse_tree.get_productions()

            for rule in parse_tree_rules:
                rules.add(rule)

        rules = list(sorted(rules, key=lambda x: x.__repr__()))
        grammar = EnglishGrammar(rules)

    if verbose:
        print("Grammar induced successfully.")

    if output is not None:
        with open(output, 'wb') as f:
            pickle.dump(grammar, f)
    else:
        return grammar, parse_trees


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile grammar from query examples.')
    parser.add_argument('-q', '--queries', required=True,
                        help='Path to queries.')
    parser.add_argument('-o', '--output',
                        help='Filename for output.')
    parser.add_argument('-c', '--clean', action='store_true', default=False,
                        help='Pre-clean queries before populating.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print verbose output on progress.')
    parser.add_argument('-a', '--asdl', action='store_true', default=False,
                        help='Return grammar in ASDL mode.')

    args_dict = vars(parser.parse_args())

    extract_grammar(args_dict['queries'],
                    args_dict['output'],
                    args_dict['clean'],
                    args_dict['verbose'],
                    args_dict['asdl'])
