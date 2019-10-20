
import argparse
import ast
# import config
import configparser
import cProfile
import os
import logging
import numpy as np
import traceback
# from vprof import profiler

from astnode import ASTNode
from components import Hyp
from dataset import DataEntry, DataSet, Vocab, Action
from decoder import decode_english_dataset
from evaluation import *
from learner import Learner
from model import Generator
from multivac.src.gan.generator.nn.utils.io_utils import deserialize_from_file, serialize_to_file

def run(args_dict):
    verbose = args_dict['verbose']

    if not os.path.exists(args_dict['output_dir']):
        os.makedirs(args_dict['output_dir'])

    np.random.seed(int(args_dict['random_seed']))

    if verbose:
        print('Extracting dataset from ' + args_dict['data'])

    train_data, dev_data, test_data = deserialize_from_file(args_dict['data'])

    if not args_dict['source_vocab_size']:
        args_dict['source_vocab_size'] = train_data.annot_vocab.size
    if not args_dict['target_vocab_size']:
        args_dict['target_vocab_size'] = train_data.terminal_vocab.size
    if not args_dict['rule_num']:
        args_dict['rule_num'] = len(train_data.grammar.rules)
    if not args_dict['node_num']:
        args_dict['node_num'] = len(train_data.grammar.node_type_to_id)

    if args_dict['operation'] in ['train', 'decode', 'interactive']:
        if verbose:
            print("Initiating generator model...")
        model = Generator(args_dict)
        model.build()

        if verbose:
            print("Done.")

        if args_dict['model']:
            model.load(args_dict['model'])

    if args_dict['operation'] == 'train':
        if verbose:
            print("Training!")
        learner = Learner(args_dict, model, train_data, dev_data)
        learner.train()

        if verbose:
            print('Done.')

    if args_dict['operation'] == 'decode':
        # ==========================
        # investigate short examples
        # ==========================

        # short_examples = [e for e in test_data.examples if e.parse_tree.size <= 2]
        # for e in short_examples:
        #     print(e.parse_tree)
        # print('short examples num: ', len(short_examples))

        # dataset = test_data # test_data.get_dataset_by_ids([1,2,3,4,5,6,7,8,9,10], name='sample')
        # cProfile.run('decode_dataset(model, dataset)', sort=2)

    if args_dict['operation'] == 'evaluate':
        dataset = eval(args_dict['type'])
        if config.mode == 'self':
            decode_results_file = args_dict['input']
            decode_results = deserialize_from_file(decode_results_file)

            evaluate_decode_results(dataset, decode_results)
        elif config.mode == 'seq2tree':
            from evaluation import evaluate_seq2tree_sample_file
            evaluate_seq2tree_sample_file(config.seq2tree_sample_file, 
                                          config.seq2tree_id_file, dataset)
        elif config.mode == 'seq2seq':
            from evaluation import evaluate_seq2seq_decode_results
            evaluate_seq2seq_decode_results(dataset, 
                                            config.seq2seq_decode_file, 
                                            config.seq2seq_ref_file, 
                                            is_nbest=config.is_nbest)
        elif config.mode == 'analyze':
            from evaluation import analyze_decode_results

            decode_results_file = args_dict['input']
            decode_results = deserialize_from_file(decode_results_file)
            analyze_decode_results(dataset, decode_results)

    if args_dict['operation'] == 'interactive':
        from dataset import canonicalize_query, query_to_data
        from collections import namedtuple
        from lang.py.parse import decode_tree_to_python_ast
        assert model is not None

        while True:
            cmd = eval(input('example id or query: '))
            if args_dict['mode'] == 'dataset':
                try:
                    example_id = int(cmd)
                    example = [e for e in test_data.examples if e.raw_id == example_id][0]
                except:
                    print('something went wrong ...')
                    continue
            elif args_dict['mode'] == 'new':
                # we play with new examples!
                query, str_map = canonicalize_query(cmd)
                vocab = train_data.annot_vocab
                query_tokens = query.split(' ')
                query_tokens_data = [query_to_data(query, vocab)]
                example = namedtuple('example', 
                                     ['query', 'data'])(query=query_tokens, 
                                                        data=query_tokens_data)

            if hasattr(example, 'parse_tree'):
                print('gold parse tree:')
                print((example.parse_tree))

            cand_list = model.decode(example, train_data.grammar, 
                                     train_data.terminal_vocab,
                                     beam_size=args_dict['beam_size'], 
                                     max_time_step=args_dict['decode_max_time_step'], 
                                     log=True)

            has_grammar_error = any([c for c in cand_list if c.has_grammar_error])
            print(('has_grammar_error: ', has_grammar_error))

            for cid, cand in enumerate(cand_list[:5]):
                print(('*' * 60))
                print(('cand #%d, score: %f' % (cid, cand.score)))

                try:
                    ast_tree = decode_tree_to_python_ast(cand.tree)
                    code = astor.to_source(ast_tree)
                    print(('code: ', code))
                    print(('decode log: ', cand.log))
                except:
                    print("Exception in converting tree to code:")
                    print(('-' * 60))
                    print(('raw_id: %d, beam pos: %d' % (example.raw_id, cid)))
                    traceback.print_exc(file=sys.stdout)
                    print(('-' * 60))
                finally:
                    print('* parse tree *')
                    print((cand.tree.__repr__()))
                    print(('n_timestep: %d' % cand.n_timestep))
                    print(('ast size: %d' % cand.tree.size))
                    print(('*' * 60))


if __name__ == '__main__':
    # Good god, this all needs to be in a config file and use 
    # configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, 
                        help='Config file with updated parameters for generator;'
                             'defaults to "config.cfg" in this directory '
                             'otherwise.')
    parser.add_argument('-d', '--data', required=True, 
                        help='Dataset file with for training the Generator.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print verbose reporting to standard out.')

    all_args = parser.parse_known_args()
    args = vars(all_args[0])

    i = 0

    while i < len(all_args[1]):
        if all_args[1][i].startswith('--'):
            args[all_args[1][i][2:]] = all_args[1][i+1]
            i += 2
        else:
            i += 1

    cfg = configparser.ConfigParser()
    cfgDIR = os.path.dirname(os.path.realpath(__file__))

    if args['config'] is not None:
        cfg.read(args['config'])
    else:
        cfg.read(os.path.join(cfgDIR, 'config.cfg'))

    cfg_dict = cfg._sections['ARGS']
    cfg_dict.update(args)

    for carg in cfg_dict:
        # Cast all arguments to proper types
        if cfg_dict[carg] == 'None':
            cfg_dict[carg] = None
            continue

        try:
            cfg_dict[carg] = int(cfg_dict[carg])
        except:
            try:
                cfg_dict[carg] = float(cfg_dict[carg])
            except:
                if cfg_dict[carg] in ['True','False']:
                    cfg_dict[carg] = eval(cfg_dict[carg])

    run(cfg_dict)
