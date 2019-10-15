
import argparse
import re

from multivac.src.gan.generator.astnode import *
from multivac.src.gan.generator.lang.eng.grammar import EnglishGrammar
from multivac.src.rdf_graph.rdf_parse import StanfordParser, stanford_parse


def check_parse(query):
    parseable = True

    if any([x.pos_.upper().startswith('VB') for x in query.tokens]):
        pass
    else:
        parseable = False

    return parseable

def clean_queries(queries, verbose=False):
    clean = list()

    if verbose:
        print(("Performing basic clean on {} queries.".format(len(queries))))

    for query in queries:
        # strip whitespace, and quotes
        # Remove any sentence fragments preceding question
        # Remove non-alphabetic characters at the start of the string
        query = query.strip()
        query = query.strip("\"")
        query = re.sub(r"“|”", "\"", query)
        query = re.sub(r"‘|’", "\'", query)
        query = re.sub(r"`", "\'", query)
        query = query[query.index(re.split(r"\"", query)[-1]):]
        query = query[query.index(re.split(r"NumericCitation", query, re.IGNORECASE)[-1]):]
        query = query[query.index(re.split(r"[\.\!\?]\s+", query)[-1]):]
        query = re.sub(r"^(?!\()[^a-zA-Z]+","", query)
        query = re.sub(r"^(\(.*\))?\W+","", query)
        query = re.sub(r"(\s+)([\)\]\}\.\,\?\!])", r"\2", query)
        query = re.sub(r"([\(\[\{])(\s+)", r"\1", query)

        if len(query) > 0:
            tok_chk = [len(x) for x in query.split()]

            if sum(tok_chk)/len(tok_chk) < 2:
                continue

            query = query[0].upper() + query[1:]
            clean.append(query)

    if verbose:
        print(("{} cleaned queries remaining.".format(len(queries))))

    return clean

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

    if debug: print(("\t" * depth + "String: '{}'".format(text)))

    try:
        tree_str = text[text.index("(") + 1:text.rfind(")")]
    except ValueError:
        print(("Malformatted parse string: '{}'".format(text)))
        raise ValueError

    next_idx = tree_str.index(" ")

    tree = ASTNode(tree_str[:next_idx])
    if debug: print(("\t" * depth + "Type: '{}'".format(tree.type)))

    if "(" in tree_str:
        while "(" in tree_str:
            tree_str = tree_str[tree_str.index("("):]
            next_idx = find_match_paren(tree_str) + 1
            tree.add_child(get_eng_tree(tree_str[:next_idx], depth+1, debug))
            tree_str = tree_str[next_idx + 1:]
    else:
        tree.value = tree_str[next_idx + 1:]
        if debug: print(("\t" * depth + "Value: " + tree.value))

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
        print(('num. rules: %d', len(rules)))

    return grammar

def parse_raw(parser, query):
    query = stanford_parse(parser, query)
    return get_eng_tree(query.parse_string)


def extract_grammar(source_file, output=None, clean=False, verbose=False):
    parse_trees = list()
    rules = set()

    parser = StanfordParser(annots = "tokenize pos lemma ner parse")

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
            except:
                print('Could not parse query {}: "{}"'.format(i, q))
                continue

        if check_parse(query):
            try:
                parse_trees.append(get_eng_tree(query.parse_string))
            except:
                print(("Could not interpret query parse {}: '{}'".format(i, query)))
                continue

        if i % 100 == 0:
            print("{} queries processed.".format(i))

    if verbose:
        print(("{} queries successfully parsed.".format(len(parse_trees))))
        print("Extracting grammar production rules.")

    for parse_tree in parse_trees:
        parse_tree_rules, _ = parse_tree.get_productions()

        for rule in parse_tree_rules:
            rules.add(rule)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = EnglishGrammar(rules)

    if verbose:
        print("Grammar induced successfully.")

    if output is not None:
        with open(output, 'w') as f:
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
    parser.add_argument('-c', '--clean', action='store_true', default=False,
                        help='Pre-clean queries before populating.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print verbose output on progress.')

    args_dict = vars(parser.parse_args())

    extract_grammar(args_dict['queries'],
                    args_dict['output'], 
                    args_dict['clean'], 
                    args_dict['verbose'])
