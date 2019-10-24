#!usr/bin/env/python
import argparse
from collections import namedtuple
import configparser
import copy
import math
import random
import numpy as np
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from generator.model import Generator
from discriminator.treelstm import QueryGAN_Discriminator, MULTIVACDataset, Trainer
from rollout import Rollout

from multivac.src.gan.generator.lang.eng.eng_dataset import get_actions
from multivac.src.gan.discriminator.treelstm import utils
from multivac.src.gan.generator.nn.utils.io_utils import deserialize_from_file, \
                                                         serialize_to_file
from multivac.src.rdf_graph.rdf_parse import StanfordParser
from generator.learner import Learner
from generator.components import Hyp
from generator.dataset import DataEntry, DataSet, Vocab, Action
from generator.decoder import decode_tree_to_string

def DiscriminatorDataset(real_dir, fake_dir, vocab):
    '''
    Take real examples from existing training dataset and add them to the 
    Generated dataset for adversarial training.
    '''
    real_file = MULTIVACDataset(real_dir, vocab)
    combined_file = MULTIVACDataset(fake_dir, vocab)

    labels = torch.cat((combined_file.labels, 
                        real_file.labels[real_file.labels==1]), dim=0)

    for i, item in enumerate(real_file.labels):
        if item == 1:
            combined_file.trees.append(real_file.trees[i])
            combined_file.sentences.append(real_file.sentences[i])

    combined_file.size = combined_file.labels.size(0)

    return combined_file

def disc_trainer(model, glove_emb, glove_vocab, use_cuda=False):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    criterion = nn.MSELoss()
    emb = torch.zeros(glove_vocab.size(), glove_emb.size(1), dtype=torch.float, 
                      device=device)
    emb.normal_(0, 0.05)

    for word in glove_vocab.labelToIdx.keys():
        if glove_vocab.getIndex(word) < glove_emb.size(0):
            emb[glove_vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        else:
            emb[glove_vocab.getIndex(word)].zero_()

    # plug these into embedding matrix inside model
    model.emb.weight.data.copy_(emb)
    model.to(device), criterion.to(device)

    if model.cfg['optim'] == 'adam':
        opt = optim.Adam
    elif model.cfg['optim'] == 'adagrad':
        opt = optim.Adagrad
    elif model.cfg['optim'] == 'sgd':
        opt = optim.SGD

    optimizer = opt(filter(lambda p: p.requires_grad,
                                  model.parameters()), 
                           lr=model.cfg['lr'], weight_decay=model.cfg['wd'])

    return Trainer(model.cfg, model, criterion, optimizer, device)

def generate_samples(net, grammar, vocab, seq_len, 
                     generated_num, dst_dir, oracle=False):
    parser = StanfordParser(annots="depparse")
    samples = [''] * generated_num
    ds = DataSet(vocab, vocab, grammar, name='train_data')
    max_query_len = 0
    max_actions_len = 0

    for i in tqdm(range(generated_num), desc='Generating Samples... '):
        query = vocab.convertToLabels(random.sample(range(vocab.size()), 
                                      seq_len))
        seed_seq = ' '.join(query)
        query_tokens_data = [query_to_data(seed_seq, vocab)]
        example = namedtuple('example', 
                             ['query_tokens', 'data'])(query_tokens=query, 
                                                       data=query_tokens_data)
        sample = net.decode(example, 
                            grammar, 
                            vocab,
                            beam_size=net.cfg['beam_size'], 
                            max_time_step=net.cfg['decode_max_time_step'])[0]
        text = decode_tree_to_string(sample.tree)

        samples.append(text)

        if oracle:
            actions = get_actions(sample.tree, query, grammar, vocab)
            example = DataEntry(i, query, sample.tree, text, actions)
            ds.add(example)
            #ds.examples[-1]._data = ds.get_prob_func_inputs([len(ds.examples)-1])

            if len(example.query_tokens) > max_query_len:
                max_query_len = len(example.query_tokens)

            if len(example.actions) > max_actions_len:
                max_actions_len = len(example.actions)

    if oracle:
        ds.init_data_matrices(max_query_length=max_query_len, 
                              max_example_action_num=max_actions_len)
        serialize_to_file(ds, os.path.join(dst_dir, "samples.pkl"))
    else:
        sample_parses = parser.get_parse('?\n'.join(samples))

        for i, parse in enumerate(sample_parses['sentences']):
            tokens = [x['word'] for x in parse['tokens']]
            deps = sorted(parse['basicDependencies'], 
                          key=lambda x: x['dependent'])
            parents = [x['governor'] for x in deps]
            samples[i] = (tokens, parents)

        with open(os.path.join(dst_dir, 'text.toks'), 'w') as tokfile, \
                open(os.path.join(dst_dir, 'text.parents'), 'w') as parfile, \
                open(os.path.join(dst_dir, 'cat.txt'), 'w') as catfile:

            for tokens, parents in samples:
                parfile.write(' '.join([str(x) for x in parents]) + '\n')
                tokfile.write(' '.join(tokens) + '\n')
                catfile.write('0' + '\n')

def query_to_data(query, annot_vocab):
    if isinstance(query, str):
        query_tokens = query.split(' ')
    else:
        query_tokens = query

    data = np.zeros((1, len(query_tokens)), dtype='int32')

    for tid, token in enumerate(query_tokens):
        token_id = annot_vocab[token]

        data[0, tid] = token_id

    return data

def run(cfg_dict):
    # Set up model and training parameters based on config file and runtime
    # arguments

    args = cfg_dict['ARGS']
    gargs = cfg_dict['GENERATOR']
    dargs = cfg_dict['DISCRIMINATOR']
    gan_args = cfg_dict['GAN']

    seed = gan_args['seed']
    batch_size = gan_args['batch_size']
    total_epochs = gan_args['total_epochs']
    generated_num = gan_args['generated_num']
    vocab_size = gan_args['vocab_size']
    sequence_len = gan_args['sequence_len']

    # rollout params
    rollout_update_rate = gan_args['rollout_update_rate']
    rollout_num = gan_args['rollout_num']

    g_steps = gan_args['g_steps']
    d_steps = gan_args['d_steps']
    k_steps = gan_args['k_steps']
    
    use_cuda = gan_args['device'] == 'cuda'

    if not torch.cuda.is_available():
        use_cuda = False

    random.seed(seed)
    np.random.seed(seed)

    # 
    # NEED A DATASET FIRST, TO DEFINE EMBEDDINGS/RULES SIZES
    # 
    #   - given a grammar file
    #   - given GloVe vocab list

    grammar = deserialize_from_file(gargs['grammar'])
    glove_vocab, glove_emb = utils.load_word_vectors(
        os.path.join(gan_args['glove_dir'], gan_args['glove_file']))

    gargs['rule_num'] = len(grammar.rules)
    gargs['node_num'] = len(grammar.node_type_to_id)
    gargs['source_vocab_size'] = glove_vocab.size()
    gargs['target_vocab_size'] = glove_vocab.size()

    # Set up Generator component with given parameters
    netG = Generator(gargs)
    netG.build()

    # Set up Discriminator component with given parameters

    dargs['vocab_size'] = glove_vocab.size()
    netD = QueryGAN_Discriminator(dargs)
    trainer = disc_trainer(netD, glove_emb, glove_vocab, use_cuda)

    # Set up Oracle component with given parameters
    # ### This is super expensive memory wise. let's figure something else out
    # oracle = copy.deepcopy(netG)
    # oracle.oracle = True

    # Generate starting samples
    seq_len = 6
    generate_samples(netG, grammar, glove_vocab, seq_len, generated_num, 
                     netG.cfg['sample_dir'], oracle=True)

    # 
    # PRETRAIN GENERATOR
    # 

    gen_set = deserialize_from_file(os.path.join(netG.cfg['sample_dir'],
                                                 "samples.pkl"))

    print('\nPretraining generator...\n')
    # Pre-train epochs are set in config.cfg file
    learner = Learner(gargs, netG, gen_set)
    learner.pretrain()

    # pretrain discriminator
    print('\nPretraining discriminator...\n')
    # for epoch in range(d_steps):
        # generate_samples(netG, grammar, glove_vocab, seq_len, generated_num, 
        #                  netG.cfg['sample_dir'])
    dis_set = MULTIVACDataset(os.path.join(netD.cfg['data'], "train"), 
                              glove_vocab)
    # disloader = DataLoader(dataset=dis_set,
    #                        batch_size=batch_size,
    #                        shuffle=True)
    
    for epoch in range(k_steps):
        loss = trainer.train(dis_set)
        print('Epoch {} pretrain discriminator training loss: {}'.format(epoch + 1, loss))

    # adversarial training
    rollout = Rollout(netG, update_rate=rollout_update_rate, rollout_num=rollout_num)
    print('\n#####################################################')
    print('Adversarial training...\n')

    for epoch in range(total_epochs):
        for _ in range(g_steps):
            netG.pgtrain(batch_size, sequence_len, rollout, netD)

        for d_step in range(d_steps):
            # train discriminator
            generate_samples(netG, batch_size, generated_num, FAKE_FILE)
            dis_set = DiscriminatorDataset(os.path.join(netD.cfg['data'], "train"), 
                                           netG.cfg['sample_dir'],
                                           glove_vocab)
            # disloader = DataLoader(dataset=dis_set,
            #                        batch_size=BATCH_SIZE,
            #                        shuffle=True)
        
            for k_step in range(k_steps):
                loss = netD.dtrain(dis_set)
                print('D_step {}, K-step {} adversarial discriminator training loss: {}'.format(d_step + 1, k_step + 1, loss))          
        rollout.update_params()

        generate_samples(netG, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        val_set = GeneratorDataset(EVAL_FILE)
        valloader = DataLoader(dataset=val_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
        loss = oracle.val(valloader)
        print('Epoch {} adversarial generator val loss: {}'.format(epoch + 1, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', 
                        help='Enable GPU training.')
    parser.add_argument('-c', '--config', required=False, 
                        help='Config file with updated parameters for generator;'
                             'defaults to "config.cfg" in this directory '
                             'otherwise.')

    all_args = parser.parse_known_args()
    args = vars(all_args[0])

    i = 0

    # while i < len(all_args[1]):
    #     if all_args[1][i].startswith('--'):
    #         args[all_args[1][i][2:]] = all_args[1][i+1]
    #         i += 2
    #     else if 
    #         i += 1

    cfg = configparser.ConfigParser()
    cfgDIR = os.path.dirname(os.path.realpath(__file__))

    if args['config'] is not None:
        cfg.read(args['config'])
    else:
        cfg.read(os.path.join(cfgDIR, 'config.cfg'))

    cfg_dict = cfg._sections

    # NEED TO IMPLEMENT ACTUALLY OVERRIDING THE config.cfg SETTINGS
    cfg_dict['ARGS'] = args

    for name, section in cfg_dict.items():
        for carg in section:
            # Cast all arguments to proper types
            if section[carg] == 'None':
                section[carg] = None
                continue

            try:
                section[carg] = int(section[carg])
            except:
                try:
                    section[carg] = float(section[carg])
                except:
                    if section[carg] in ['True','False']:
                        section[carg] = eval(section[carg])

    run(cfg_dict)
