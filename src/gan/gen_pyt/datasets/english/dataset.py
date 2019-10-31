# coding=utf-8

import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from multivac.src.gan.gen_pyt.asdl.hypothesis import *
from multivac.src.gan.gen_pyt.asdl.lang.eng.eng_asdl_helper \
    import english_ast_to_asdl_ast, asdl_ast_to_english_ast
from multivac.src.gan.gen_pyt.asdl.lang.eng.eng_transition_system \
    import EnglishTransitionSystem

from multivac.src.gan.gen_pyt.components.action_info \
    import ActionInfo, get_action_infos
from multivac.src.gan.gen_pyt.components.dataset import Example
from multivac.src.gan.gen_pyt.components.vocab import Vocab, VocabEntry

from multivac.src.gan.gen_pyt.utils.io_utils \
    import serialize_to_file, deserialize_from_file

from multivac.src.rdf_graph.rdf_parse import tokenize_text

class English(object):
    @staticmethod
    def canonicalize_example(query, text, parser):
        '''
        If we want to do any cleaning on the text, here's where to do it.
        '''
        try:
            query = stanford_parse(parser, query)
        except:
            print('Could not parse query: {}'.format(query))
            return None, None, None

        try:
            parse_tree = english_ast_to_asdl_ast(query.parse_string)
        except: 
            print("Could not interpret query parse: {}".format(query.parse_string))
            return None, None, None

        query_tokens = [x.text for x in query.tokens]
        parse_str = query.parse_string

        return query_tokens, text, parse_str, parse_tree

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
            # strip whitespace, and quotes
            # Remove any sentence fragments preceding question
            # Remove non-alphabetic characters at the start of the string
            query = query.strip()
            query = re.sub(r"“|”", "\"", query)
            query = re.sub(r"‘|’", "\'", query)
            query = re.sub(r"`", "\'", query)
            query = query.strip("\"")
            query = query.strip("\'")
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
    def preprocess_dataset(annot_file, text_file):
        parser = StanfordParser(annots = "tokenize ssplit parse")

        processed_examples = []
        productions = set()

        for idx, (src_query, tgt_text) in enumerate(zip(open(annot_file), 
                                                        open(text_file))):
            src_query = src_query.strip()
            tgt_text = tgt_text.strip()

            toks, text, parse_str, tree = English.canonicalize_example(src_query, 
                                                                       tgt_text, 
                                                                       parser)
            productions.update(tree.get_productions())
            processed_examples.append((toks, text, parse_str, tree))

        productions = sorted(productions, key=lambda x: x.__repr__)

        return processed_examples, productions

    @staticmethod
    def parse_english_dataset(annot_file, text_file, 
                              max_query_len=70, vocab_freq_cutoff=1,
                              train_size=.8, dev_size=.1, test_size=.1):

        processed_examples, productions = English.preprocess_dataset(annot_file, 
                                                                     text_file)

        grammar = EnglishASDLGrammar(productions=productions)
        transition_system = EnglishTransitionSystem(grammar)

        serialize_to_file(grammar, 
                          os.path.join(os.path.dirname(annot_file), 
                                       "eng.grammar.pkl"))

        all_examples = []
        action_len = []

        for idx, example in enumerate(processed_examples):
            toks, text, parse_str, tree = example

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

        src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], 
                                           size=5000, 
                                           freq_cutoff=vocab_freq_cutoff)

        primitive_tokens = [map(lambda a: a.action.token,
                                filter(lambda a: isinstance(a.action, 
                                                            GenTokenAction), e.tgt_actions))
                            for e in train_examples]

        primitive_vocab = VocabEntry.from_corpus(primitive_tokens, 
                                                 size=5000, 
                                                 freq_cutoff=vocab_freq_cutoff)

        # generate vocabulary for the code tokens!
        tokens = [tokenize_text(e.tgt_text) for e in train_examples]
        tgt_vocab = VocabEntry.from_corpus(tokens, 
                                           size=5000, 
                                           freq_cutoff=vocab_freq_cutoff)

        vocab = Vocab(source=src_vocab, 
                      primitive=primitive_vocab, 
                      code=tgt_vocab)
        print('generated vocabulary {}'.format(repr(vocab)))

        return (train_examples, dev_examples, test_examples), vocab

    @staticmethod
    def generate_dataset(annot_file, text_file, grammar):
        processed_examples, productions = English.preprocess_dataset(annot_file, 
                                                                     text_file)
        transition_system = EnglishTransitionSystem(grammar)
        all_examples = []
        action_len = []

        for idx, example in enumerate(processed_examples):
            toks, text, parse_str, tree = example

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

        return all_examples

def generate_vocab_for_paraphrase_model(vocab_path, save_path):
    from components.vocab import VocabEntry, Vocab

    vocab = pickle.load(open(vocab_path))
    para_vocab = VocabEntry()
    for i in range(0, 10):
        para_vocab.add('<unk_%d>' % i)
    for word in vocab.source.word2id:
        para_vocab.add(word)
    for word in vocab.code.word2id:
        para_vocab.add(word)

    pickle.dump(para_vocab, open(save_path, 'w'))


if __name__ == '__main__':
    English.parse_english_dataset()
