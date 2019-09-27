
import argparse
import re
import spacy
from benepar.spacy_plugin import BeneparComponent

from NL2code.lang.eng.grammar import EnglishGrammar
from NL2code.astnode import *

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))

def check_parse(query):
    parseable = True
    toks = [x.pos_.upper() for x in query]

    if 'AUX' in toks:
        pass
    elif 'VERB' in toks:
        pass
    else:
        parseable = False

    return parseable

def clean_queries(queries, verbose=False):
    clean_queries = list()

    if verbose:
        print("Performing basic clean on {} queries.".format(len(queries)))

    for text in queries:
        text = re.sub(r"^\W+|[^\w{W}(?<!?)]+$", "", text)

        if "citation" not in text.lower():
            clean_queries.append(text)

    if verbose:
        print("{} cleaned queries remaining.".format(len(queries)))

    return clean_queries

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
        '(SBARQ (WHADVP (WRB Why)) (SQ (VBP do) (NP (NNS birds)) (ADVP (RB 
        suddenly)) (VP (VB appear) (SBAR (WHADVP (WRB whenever)) (S (NP (PRP 
        you)) (VP (VBP are) (ADJP (JJ near))))))) (. ?))'
    '''

    if debug: print("\t" * depth + "String: '{}'".format(text))

    try:
        tree_str = text[text.index("(") + 1:text.rfind(")")]
    except ValueError:
        print("Malformatted parse string: '{}'".format(text))
        raise ValueError

    next_idx = tree_str.index(" ") + 1

    tree = ASTNode(tree_str[:next_idx - 1])
    if debug: print("\t" * depth + "Type: '{}'".format(tree.type))

    if "(" in tree_str:
        while "(" in tree_str:
            tree_str = tree_str[tree_str.index("("):]
            next_idx = find_match_paren(tree_str) + 2
            tree.children.append(get_eng_tree(tree_str[:next_idx - 1], 
                                              depth+1, 
                                              debug))
            tree_str = tree_str[next_idx:]
    else:
        tree.value = tree_str[next_idx:]
        if debug: print("\t" * depth + "Value: " + tree.value)

    return tree

def get_grammar(parse_trees, verbose=False):
    rules = set()
    # rule_num_dist = defaultdict(int)

    for parse_tree in parse_trees:
        parse_tree_rules, rule_parents = parse_tree.get_productions()
        for rule in parse_tree_rules:
            rules.add(rule)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = EnglishGrammar(rules)

    if verbose:
        print('num. rules: %d', len(rules))

    return grammar

def get_parse(s):
    if isinstance(s, str):
        s = list(nlp(s).sents)[0]

    parse_string = "(ROOT " + s._.parse_string + ")"

    return parse_string


def run(args_dict):
    verbose = args_dict['verbose']
    parse_trees = list()
    rules = set()

    with open(args_dict['queries'], 'r') as f:
        queries = f.readlines()

    if args_dict['clean']:
        queries = clean_queries(queries, verbose)

    if verbose:
        print("Performing constituency parsing of queries")

    doc = nlp('\n'.join(queries))

    for query in doc.sents:
        if not check_parse(query):
            continue

        try:
            parse_trees.append(get_eng_tree(get_parse(query)))
        except:
            print("Could not parse query: '{}'".format(query))
            continue

    if verbose:
        print("{} queries successfully parsed.".format(len(parse_trees)))
        print("Extracting grammar production rules.")

    for parse_tree in parse_trees:
        parse_tree_rules, _ = parse_tree.get_productions()

        for rule in parse_tree_rules:
            rules.add(rule)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = EnglishGrammar(rules)

    if verbose:
        print("Grammar induced successfully.")

    if args_dict['output'] is not None:
        with open(args_dict['output'], 'w') as f:
            for rule in rules:
                rule_kids = [str(x) for x in rule.children]
                out_str = str(rule.parent) + ' -> ' + ', '.join(rule_kids)
                f.write(out_str + '\n')
    else:
        return grammar, parse_trees


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile grammar from query examples.')
    parser.add_argument('-q', '--queries', required=True,
                        help='Path to queries.')
    parser.add_argument('-o', '--output',
                        help='Filename for output.')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Pre-clean queries before populating.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output on progress.')

    args_dict = vars(parser.parse_args())
    run(args_dict)
