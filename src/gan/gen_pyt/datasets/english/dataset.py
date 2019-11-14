# coding=utf-8

import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from multivac.src.gan.gen_pyt.asdl.hypothesis import *
from multivac.src.gan.gen_pyt.asdl.lang.eng.grammar import EnglishASDLGrammar
from multivac.src.gan.gen_pyt.asdl.lang.eng.eng_asdl_helper \
    import english_ast_to_asdl_ast
from multivac.src.gan.gen_pyt.asdl.lang.eng.eng_transition_system \
    import EnglishTransitionSystem

from multivac.src.gan.gen_pyt.components.action_info \
    import ActionInfo, get_action_infos
from multivac.src.gan.gen_pyt.components.dataset import Example
from multivac.src.gan.gen_pyt.components.vocab import Vocab

from multivac.src.gan.gen_pyt.utils.io_utils \
    import serialize_to_file, deserialize_from_file

from multivac.src.rdf_graph.rdf_parse import tokenize_text, StanfordParser, stanford_parse

class English(object):
    @staticmethod
    def canonicalize_example(text, parser, verbose=False):
        '''
        If we want to do any cleaning on the text, here's where to do it.
        '''
        try:
            parse = stanford_parse(parser, text)
        except:
            print('Could not parse query: {}'.format(text))
            return None

        # try:
        parse_tree = english_ast_to_asdl_ast(parse.parse_string)
        # except: 
        #     print("Could not interpret query parse: {}".format(parse.parse_string))
        #     return None

        return parse_tree

    @staticmethod
    def check_parse(query):
        parseable = True

        if any([x.pos_.upper().startswith('VB') for x in query.tokens]):
            pass
        else:
            parseable = False

        return parseable

    @staticmethod
    def clean_queries(queries, verbose=False):
        clean = list()

        if verbose:
            print(("Performing basic clean on {} queries.".format(len(queries))))

        for query in queries:
            # strip whitespace, and convert and strip 'smart' quotes
            query = query.strip()
            query = re.sub(r"“|”", "\"", query)
            query = re.sub(r"‘|’", "\'", query)
            query = re.sub(r"`", "\'", query)
            query = query.strip("\"")
            query = query.strip("\'")

            # Remove any sentence fragments preceding question
            query = query[query.index(re.split(r"\"", query)[-1]):]
            query = query[query.index(re.split(r"NumericCitation", query, re.IGNORECASE)[-1]):]
            query = query[query.index(re.split(r"[\.\!\?]\s+", query)[-1]):]

            # Remove non-alphabetic characters at the start of the string
            query = re.sub(r"^(?!\()[^a-zA-Z]+","", query)
            query = re.sub(r"^(\(.*\))?\W+","", query)

            # Remove whitespace preceding right-hand-side punctuation and
            #  following left-hand-side puncuation. I.e., "( we like it )"
            #  becomes "(we like it)"
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

    @staticmethod
    def find_match_paren(s):
        count = 0

        for i, c in enumerate(s):
            if c == "(":
                count += 1
            elif c == ")":
                count -= 1

            if count == 0:
                return i

    @staticmethod
    def preprocess_dataset(annot_file, text_file, verbose=False):
        parser = StanfordParser(annots = "tokenize ssplit parse")

        processed_examples = []
        productions = set()

        for idx, (src_query, tgt_text) in enumerate(zip(open(annot_file), 
                                                        open(text_file))):
            query_toks = src_query.strip().split()
            tgt_text = tgt_text.strip()

            tree = English.canonicalize_example(tgt_text, parser)

            if tree is not None:
                productions.update(tree.get_productions())
                processed_examples.append((query_toks, tgt_text, tree))

        productions = sorted(productions, key=lambda x: x.__repr__())

        return processed_examples, productions

    @staticmethod
    def parse_english_dataset(annot_file, text_file, grammar=None,
                              max_query_len=70, vocab_freq_cutoff=1,
                              train_size=.8, dev_size=.1, test_size=.1):

        processed_examples, productions = English.preprocess_dataset(annot_file, 
                                                                     text_file)

        if grammar is None:
            grammar = EnglishASDLGrammar(productions=productions)

        transition_system = EnglishTransitionSystem(grammar)

        serialize_to_file(grammar, 
                          os.path.join(os.path.dirname(annot_file), 
                                       "eng.grammar.pkl"))

        all_examples = []
        action_len = []

        for idx, example in enumerate(processed_examples):
            toks, text, tree = example

            if max_query_len is not None:
                toks = toks[:max_query_len]

            tgt_actions = transition_system.get_actions(tree)
            tgt_action_infos = get_action_infos(toks, tgt_actions)
            action_len.append(len(tgt_action_infos))

            all_examples.append(Example(idx=idx,
                                        src_sent=toks,
                                        tgt_actions=tgt_action_infos,
                                        tgt_text=text,
                                        tgt_ast=tree,
                                        meta={'raw_text': text, 
                                              'str_map': None}))

        train_examples, dev_examples = train_test_split(all_examples, 
                                                        train_size=train_size)
        test_examples, dev_examples = train_test_split(dev_examples,
                                                       train_size=test_size/(dev_size+test_size))

        print('Max action len: {}'.format(max(action_len)))
        print('Avg action len: {}'.format(np.average(action_len)))
        print('Actions larger than 100: {}'.format(len(list(filter(lambda x: x > 100, action_len)))))

        # generate vocabulary for the code tokens!
        tokens = [tokenize_text(e.tgt_text) for e in train_examples]
        vocab = Vocab.from_corpus(tokens, freq_cutoff=vocab_freq_cutoff)

        print('generated vocabulary {}'.format(repr(vocab)))

        return (train_examples, dev_examples, test_examples), vocab, grammar

    @staticmethod
    def generate_dataset(annot_file, text_file, grammar=None, max_query_len=None,
                         vocab_freq_cutoff=1, verbose=False):
        processed_examples, productions = English.preprocess_dataset(annot_file, 
                                                                     text_file,
                                                                     verbose)
        if grammar is None:
            grammar = EnglishASDLGrammar(productions=productions)

        transition_system = EnglishTransitionSystem(grammar)
        all_examples = []
        action_len = []

        for idx, example in tqdm(enumerate(processed_examples), desc='Generating Dataset... '):
        # for idx, example in enumerate(processed_examples):
            toks, text, tree = example

            if max_query_len is not None:
                toks = toks[:max_query_len]

            tgt_actions = transition_system.get_actions(tree)
            tgt_action_infos = get_action_infos(toks, tgt_actions)
            action_len.append(len(tgt_action_infos))

            all_examples.append(Example(idx=idx,
                                        src_sent=toks,
                                        tgt_actions=tgt_action_infos,
                                        tgt_text=text,
                                        tgt_ast=tree,
                                        meta={'raw_text': text, 
                                              'str_map': None}))

        # generate vocabulary for the tgt_text tokens!
        tokens = [tokenize_text(e.tgt_text) for e in all_examples]
        vocab = Vocab.from_corpus(tokens, freq_cutoff=vocab_freq_cutoff)

        return all_examples, vocab, grammar

if __name__ == '__main__':
    English.parse_english_dataset()
