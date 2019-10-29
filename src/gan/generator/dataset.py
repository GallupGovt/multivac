
import copy

import nltk
from collections import OrderedDict, defaultdict
import logging
import collections
import numpy as np
import re
import astor
# from itertools import chain

from multivac.src.gan.generator.nn.utils.generic_utils import init_logging
from multivac.src.gan.generator.nn.utils.io_utils import serialize_to_file, deserialize_from_file

# import config
from multivac.src.gan.generator.lang.eng.unaryclosure import get_top_unary_closures, apply_unary_closures

# define actions
APPLY_RULE = 0
GEN_TOKEN = 1
COPY_TOKEN = 2
GEN_COPY_TOKEN = 3

ACTION_NAMES = {APPLY_RULE: 'APPLY_RULE',
                GEN_TOKEN: 'GEN_TOKEN',
                COPY_TOKEN: 'COPY_TOKEN',
                GEN_COPY_TOKEN: 'GEN_COPY_TOKEN'}

class Action(object):
    def __init__(self, act_type, data):
        self.act_type = act_type
        self.data = data

    def __repr__(self):
        data_str = self.data if not isinstance(self.data, dict) else \
            ', '.join(['%s: %s' % (k, v) for k, v in list(self.data.items())])
        repr_str = 'Action{%s}[%s]' % (ACTION_NAMES[self.act_type], data_str)

        return repr_str


class Vocab(object):
    def __init__(self):
        self.token_id_map = OrderedDict()
        self.insert_token('<pad>')
        self.insert_token('<unk>')
        self.insert_token('<eos>')
        self.id2word = {v: k for k, v in self.token_id_map.items()}

    @property
    def unk(self):
        return self.token_id_map['<unk>']

    @property
    def eos(self):
        return self.token_id_map['<eos>']

    def __getitem__(self, item):
        if item in self.token_id_map:
            return self.token_id_map[item]
        else:
            return self.unk

    def __contains__(self, item):
        return item in self.token_id_map

    @property
    def size(self):
        return len(self.token_id_map)

    def __setitem__(self, key, value):
        self.token_id_map[key] = value
        self.id2word[value] = key

    def __len__(self):
        return len(self.token_id_map)

    def __iter__(self):
        return iter(list(self.token_id_map.keys()))

    def iteritems(self):
        return iter(list(self.token_id_map.items()))

    def complete(self):
        self.id_token_map = dict((v, k) for (k, v) in list(self.token_id_map.items()))

    def get_token(self, token_id):
        return self.id_token_map[token_id]

    def insert_token(self, token):
        if token in self.token_id_map:
            return self[token]
        else:
            idx = len(self)
            self[token] = idx

            return idx


# replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))


# def tokenize(str):
#     str = str.translate(replace_punctuation)
#     return nltk.word_tokenize(str)


def gen_vocab(tokens, vocab_size=None, freq_cutoff=5):
    # Changed this to allow for unlimited vocab size
    # and remove print statements

    word_freq = defaultdict(int)

    for token in tokens:
        word_freq[token] += 1

    words_freq_cutoff = [w for w in word_freq if word_freq[w] >= freq_cutoff]

    if vocab_size is not None:
        ranked_words = sorted(words_freq_cutoff, 
                              key=word_freq.get, 
                              reverse=True)[:vocab_size-2]
        ranked_words = set(ranked_words)
    else:
        ranked_words = words_freq_cutoff

    vocab = Vocab()

    for token in tokens:
        if token in ranked_words:
            vocab.insert_token(token)

    vocab.complete()

    return vocab


class DataEntry:
    def __init__(self, raw_id, query_tokens, parse_tree, text, actions, meta_data=None):
        self.raw_id = raw_id
        self.eid = -1
        # FIXME: rename to query_token
        self.query_tokens = query_tokens
        self.parse_tree = parse_tree
        self.actions = actions
        self.text = text
        self.meta_data = meta_data

    @property
    def data(self):
        if not hasattr(self, '_data'):
            assert self.dataset is not None, 'No associated dataset for the example'

            self._data = self.dataset.get_prob_func_inputs([self.eid])

        return self._data

    def copy(self):
        e = DataEntry(self.raw_id, self.query_tokens, self.parse_tree, self.text, self.actions, self.meta_data)

        return e


class DataSet:
    def __init__(self, annot_vocab, terminal_vocab, grammar, name='train_data'):
        self.annot_vocab = annot_vocab
        self.terminal_vocab = terminal_vocab
        self.name = name
        self.examples = []
        self.data_matrix = dict()
        self.grammar = grammar

    def add(self, example):
        example.eid = len(self.examples)
        example.dataset = self
        self.examples.append(example)

    def get_dataset_by_ids(self, ids, name):
        dataset = DataSet(self.annot_vocab, self.terminal_vocab,
                          self.grammar, name)
        for eid in ids:
            example_copy = self.examples[eid].copy()
            dataset.add(example_copy)

        for k, v in list(self.data_matrix.items()):
            dataset.data_matrix[k] = v[ids]

        return dataset

    @property
    def count(self):
        if self.examples:
            return len(self.examples)

        return 0

    def get_examples(self, ids):
        if isinstance(ids, collections.Iterable):
            return [self.examples[i] for i in ids]
        else:
            return self.examples[ids]

    def get_prob_func_inputs(self, ids):
        order = ['query_tokens', 'tgt_action_seq', 'tgt_action_seq_type',
                 'tgt_node_seq', 'tgt_par_rule_seq', 'tgt_par_t_seq']

        max_src_seq_len = max(len(self.examples[i].query_tokens) for i in ids)
        max_tgt_seq_len = max(len(self.examples[i].actions) for i in ids)

        logging.debug('max. src sequence length: %d', max_src_seq_len)
        logging.debug('max. tgt sequence length: %d', max_tgt_seq_len)

        data = []
        for entry in order:
            if entry == 'query_tokens':
                data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            else:
                data.append(self.data_matrix[entry][ids, :max_tgt_seq_len])

        return data


    def init_data_matrices(self, max_query_length=70, max_example_action_num=100):
        logging.info('init data matrices for [%s] dataset', self.name)
        annot_vocab = self.annot_vocab
        terminal_vocab = self.terminal_vocab

        query_tokens = self.data_matrix['query_tokens'] = np.zeros((self.count, max_query_length), dtype='int32')
        tgt_node_seq = self.data_matrix['tgt_node_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_par_rule_seq = self.data_matrix['tgt_par_rule_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_par_t_seq = self.data_matrix['tgt_par_t_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_action_seq = self.data_matrix['tgt_action_seq'] = np.zeros((self.count, max_example_action_num, 3), dtype='int32')
        tgt_action_seq_type = self.data_matrix['tgt_action_seq_type'] = np.zeros((self.count, max_example_action_num, 3), dtype='int32')

        for eid, example in enumerate(self.examples):
            exg_query_tokens = example.query_tokens[:max_query_length]
            exg_action_seq = example.actions[:max_example_action_num]

            for tid, token in enumerate(exg_query_tokens):
                token_id = annot_vocab[token]

                query_tokens[eid, tid] = token_id

            assert len(exg_action_seq) > 0

            for t, action in enumerate(exg_action_seq):
                if action.act_type == APPLY_RULE:
                    rule = action.data['rule']
                    tgt_action_seq[eid, t, 0] = self.grammar.rule_to_id[rule]
                    tgt_action_seq_type[eid, t, 0] = 1
                elif action.act_type == GEN_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1
                elif action.act_type == COPY_TOKEN:
                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                elif action.act_type == GEN_COPY_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1

                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                else:
                    raise RuntimeError('wrong action type!')

                # parent information
                rule = action.data['rule']
                parent_rule = action.data['parent_rule']
                tgt_node_seq[eid, t] = self.grammar.get_node_type_id(rule.parent)
                if parent_rule:
                    tgt_par_rule_seq[eid, t] = self.grammar.rule_to_id[parent_rule]
                else:
                    assert t == 0
                    tgt_par_rule_seq[eid, t] = -1

                # parent hidden states
                parent_t = action.data['parent_t']
                tgt_par_t_seq[eid, t] = parent_t

            example.dataset = self


class DataHelper(object):
    @staticmethod
    def canonicalize_query(query):
        return query


QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


def canonicalize_query(query):
    """
    canonicalize the query, replace strings to a special place holder
    """
    str_count = 0
    str_map = dict()

    matches = QUOTED_STRING_RE.findall(query)
    # de-duplicate
    cur_replaced_strs = set()
    for match in matches:
        # If one or more groups are present in the pattern,
        # it returns a list of groups
        quote = match[0]
        str_literal = quote + match[1] + quote

        if str_literal in cur_replaced_strs:
            continue

        # FIXME: substitute the ' % s ' with
        if str_literal in ['\'%s\'', '\"%s\"']:
            continue

        str_repr = '_STR:%d_' % str_count
        str_map[str_literal] = str_repr

        query = query.replace(str_literal, str_repr)

        str_count += 1
        cur_replaced_strs.add(str_literal)

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls like foo.bar.func
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    query = ' '.join(new_query_tokens)

    return query, str_map


if __name__== '__main__':
    init_logging('parse.log')
    parse_eng_dataset()
