# -*- coding: UTF-8 -*-

import argparse
from itertools import chain
import nltk
import os

from multivac.src.gan.generator.dataset import \
    gen_vocab, DataSet, DataEntry, Action, Vocab, \
    APPLY_RULE, GEN_TOKEN, COPY_TOKEN, GEN_COPY_TOKEN
from multivac.src.gan.generator.lang.eng.unaryclosure import \
    apply_unary_closures, get_top_unary_closures
from multivac.src.gan.generator.nn.utils.io_utils import \
    serialize_to_file, deserialize_from_file
from multivac.src.gan.generator.query_treebank import parse_raw, get_grammar
from multivac.src.rdf_graph.rdf_parse import StanfordParser


def canonicalize_example(query, text, parser):
    '''
    If we want to do any cleaning on the text, here's where to do it.
    '''
    query_tokens = nltk.word_tokenize(query)
    parse_tree = parse_raw(parser, text)

    return query_tokens, text, parse_tree


def preprocess_eng_dataset(annot_file, text_file, write_out=False):
    if write_out:
        f = open('eng_dataset.examples.txt', 'w')

    examples = []
    parser = StanfordParser(annots = "tokenize pos lemma ner parse")

    for idx, (annot, text) in enumerate(zip(open(annot_file), open(text_file))):
        annot = annot.strip()
        text = text.strip()

        tokens, clean_text, parse_tree = canonicalize_example(annot, text, parser)

        if parse_tree is None:
            continue

        example = {'id': idx, 
                   'query_tokens': tokens, 
                   'text': clean_text, 
                   'parse_tree': parse_tree,
                   'str_map': None, 
                   'raw_text': text}
        examples.append(example)

        if write_out:
            f.write('*' * 50 + '\n')
            f.write('example# %d\n' % idx)
            f.write(' '.join(tokens) + '\n')
            f.write('\n')
            f.write(clean_text + '\n')
            f.write('*' * 50 + '\n')

    if write_out:
        f.close()

    return examples


def parse_eng_dataset(annot_file, text_file, 
                      MAX_QUERY_LENGTH=70, WORD_FREQ_CUT_OFF=1):

    DIR = os.path.dirname(annot_file)

    data = preprocess_eng_dataset(annot_file, text_file)
    parse_trees = [e['parse_tree'] for e in data]

    # apply unary closures
    # THESE RUN CHECKS, THEY DO NOT CHANGE THE DATA
    unary_closures = get_top_unary_closures(parse_trees, k=20)

    for parse_tree in parse_trees:
        apply_unary_closures(parse_tree, unary_closures)

    # build the grammar
    grammar = get_grammar(parse_trees)

    with open('eng.grammar.unary_closure.txt', 'w') as f:
        for rule in grammar:
            f.write(rule.__repr__() + '\n')

    serialize_to_file(grammar, os.path.join(DIR, "eng.grammar.pkl"))

    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_vocab = gen_vocab(annot_tokens, 
                            vocab_size=None, 
                            freq_cutoff=WORD_FREQ_CUT_OFF)

    # enumerate all terminal tokens to build up the terminal tokens vocabulary
    all_terminal_tokens = []

    for entry in data:
        parse_tree = entry['parse_tree']
        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = [t for t in terminal_str.split(' ') if len(t) > 0]

                for terminal_token in terminal_tokens:
                    assert len(terminal_token) > 0
                    all_terminal_tokens.append(terminal_token)

    terminal_vocab = gen_vocab(all_terminal_tokens, 
                               vocab_size=None, 
                               freq_cutoff=WORD_FREQ_CUT_OFF)

    # now generate the dataset!

    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'eng.train_data')
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'eng.dev_data')
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'eng.test_data')

    all_examples = []

    can_fully_reconstructed_examples_num = 0
    examples_with_empty_actions_num = 0

    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        text = entry['text']
        parse_tree = entry['parse_tree']

        rule_list, rule_parents = parse_tree.get_productions(include_value_node=True)

        actions = []
        can_fully_reconstructed = True
        rule_pos_map = dict()

        for rule_count, rule in enumerate(rule_list):
            if not grammar.is_value_node(rule.parent):
                assert rule.value is None
                parent_rule = rule_parents[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0

                rule_pos_map[rule] = len(actions)

                d = {'rule': rule, 'parent_t': parent_t, 
                     'parent_rule': parent_rule}
                action = Action(APPLY_RULE, d)
                actions.append(action)
            else:
                assert rule.is_leaf

                parent_rule = rule_parents[(rule_count, rule)][0]
                parent_t = rule_pos_map[parent_rule]

                terminal_val = rule.value
                terminal_str = str(terminal_val)
                terminal_tokens = [t for t in terminal_str.split(' ') if len(t) > 0]

                for terminal_token in terminal_tokens:
                    term_tok_id = terminal_vocab[terminal_token]
                    tok_src_idx = -1
                    try:
                        tok_src_idx = query_tokens.index(terminal_token)
                    except ValueError:
                        pass

                    d = {'literal': terminal_token, 'rule': rule, 
                         'parent_rule': parent_rule, 'parent_t': parent_t}

                    # cannot copy, only generation
                    # could be unk!
                    # import pdb; pdb.set_trace()
                    QUERY_TOO_LONG = False

                    if MAX_QUERY_LENGTH is not None:
                        if tok_src_idx >= MAX_QUERY_LENGTH:
                            QUERY_TOO_LONG = True
                    
                    if tok_src_idx < 0 or QUERY_TOO_LONG:
                            action = Action(GEN_TOKEN, d)
                            if terminal_token not in terminal_vocab:
                                if terminal_token not in query_tokens:
                                    # print terminal_token
                                    can_fully_reconstructed = False
                    else:  # copy
                        if term_tok_id != terminal_vocab.unk:
                            d['source_idx'] = tok_src_idx
                            action = Action(GEN_COPY_TOKEN, d)
                        else:
                            d['source_idx'] = tok_src_idx
                            action = Action(COPY_TOKEN, d)

                    actions.append(action)

                d = {'literal': '<eos>', 'rule': rule, 
                     'parent_rule': parent_rule, 'parent_t': parent_t}
                actions.append(Action(GEN_TOKEN, d))

        if len(actions) == 0:
            examples_with_empty_actions_num += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, text, actions, 
                            {'str_map': None, 'raw_text': entry['raw_text']})

        if can_fully_reconstructed:
            can_fully_reconstructed_examples_num += 1

        # train, valid, test splits

        if idx % 3 == 0:
            train_data.add(example)
        elif idx % 3 == 1:
            dev_data.add(example)
        else:
            test_data.add(example)

        all_examples.append(example)

    # print statistics

    if MAX_QUERY_LENGTH is not None:
        max_query_len = MAX_QUERY_LENGTH
    else:
        max_query_len = max(len(e.query_tokens) for e in all_examples)

    max_actions_len = max(len(e.actions) for e in all_examples)

    train_data.init_data_matrices(max_query_length=max_query_len, 
                                  max_example_action_num=max_actions_len)
    dev_data.init_data_matrices(max_query_length=max_query_len, 
                                max_example_action_num=max_actions_len)
    test_data.init_data_matrices(max_query_length=max_query_len, 
                                 max_example_action_num=max_actions_len)

    fp = "eng.freq{}.max_actions{}.pre_suf.unary_closure.bin".format(WORD_FREQ_CUT_OFF, 
                                                                     max_actions_len)
    fp = os.path.join(DIR, fp)

    serialize_to_file((train_data, dev_data, test_data), fp)

    return train_data, dev_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annots', required=True, 
                        help='File to pull source annotations from.')
    parser.add_argument('-t', '--texts', required=True, 
                        help='File to pull target texts from.')
    parser.add_argument('-l', '--max_query_length', required=False, 
                        help='Maximum length of query to parse. If not set, no'
                             ' maximum.')
    parser.add_argument('-f', '--min_freq', required=False, default='1',
                        help='Minimum word frequency for inclusion in '
                             'generated vocabulary; default is 1.')
    args = vars(parser.parse_args())

    parse_eng_dataset(args['annots'], args['texts'], 
                      MAX_QUERY_LENGTH=args['max_query_length'], 
                      WORD_FREQ_CUT_OFF=int(args['min_freq']))

