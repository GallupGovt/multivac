#!usr/bin/env/python
import argparse
import configparser
import math
import random
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from models.generator import Generator
from models.discriminator import Discriminator
# from models.oracle import Oracle
from models.rollout import Rollout

from data_utils import GeneratorDataset, DiscriminatorDataset

from model import Model
from nn.utils.io_utils import deserialize_from_file, serialize_to_file
from learner import Learner
from components import Hyp
from dataset import DataEntry, DataSet, Vocab, Action


def generate_samples(net, batch_size, generated_num, output_file):
    samples = []
    for _ in range(generated_num // batch_size):
        sample = net.sample(batch_size, SEQUENCE_LEN).cpu().data.numpy().tolist()
        samples.extend(sample)
    
    with open(output_file, 'w') as f:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            f.write('{}\n'.format(string))


def run(args):
    # Set up model and training parameters based on config file and runtime
    # arguments
    SEED = int(args['SEED'])
    BATCH_SIZE = int(args['BATCH_SIZE'])
    TOTAL_EPOCHS = int(args['TOTAL_EPOCHS']) 
    GENERATED_NUM = int(args['GENERATED_NUM'])
    VOCAB_SIZE = int(args['VOCAB_SIZE'])
    SEQUENCE_LEN = int(args['SEQUENCE_LEN'])

    # generator params
    PRE_G_EPOCHS = int(args['PRE_G_EPOCHS'])
    G_EMB_SIZE = int(args['G_EMB_SIZE'])
    G_HIDDEN_SIZE = int(args['G_HIDDEN_SIZE'])
    G_LR = float(args['G_LR'])

    # discriminator params
    D_EMB_SIZE = int(args['D_EMB_SIZE'])
    D_NUM_CLASSES = int(args['D_NUM_CLASSES'])
    D_FILTER_SIZES = eval(args['D_FILTER_SIZES'])
    D_NUM_FILTERS = eval(args['D_NUM_FILTERS'])
    DROPOUT = float(args['DROPOUT'])
    D_LR = float(args['D_LR'])
    D_L2_REG = float(args['D_L2_REG'])
    D_WGAN = eval(args['D_WGAN'])

    # rollout params
    ROLLOUT_UPDATE_RATE = float(args['ROLLOUT_UPDATE_RATE'])
    ROLLOUT_NUM = int(args['ROLLOUT_NUM'])

    G_STEPS = int(args['G_STEPS'])
    D_STEPS = int(args['D_STEPS'])
    K_STEPS = int(args['K_STEPS'])
    
    use_cuda = eval(args['cuda'])

    if not torch.cuda.is_available():
        use_cuda = False

    random.seed(SEED)
    np.random.seed(SEED)

    # Set this up so Generator gets NL2Code inputs
    # netG = Generator(VOCAB_SIZE, G_EMB_SIZE, G_HIDDEN_SIZE, G_LR, use_cuda)
    model = Model()
    model.build()

    # Set this up so the Discriminator gets Tree.LSTM inputs
    netD = Discriminator(VOCAB_SIZE, D_EMB_SIZE, D_NUM_CLASSES, D_FILTER_SIZES, D_NUM_FILTERS, DROPOUT, D_LR, D_L2_REG, use_cuda)
    # Does this need to change?
    oracle = Oracle(VOCAB_SIZE, G_EMB_SIZE, G_HIDDEN_SIZE, use_cuda)

    # generating synthetic data
    print('Generating data...')

    # Generate starting samples
    generate_samples(oracle, BATCH_SIZE, GENERATED_NUM, REAL_FILE)

    # 
    # PRETRAIN GENERATOR
    # 

    gen_set = GeneratorDataset(REAL_FILE)
    genloader = DataLoader(dataset=gen_set, 
                           batch_size=BATCH_SIZE, 
                           shuffle=True)

    print('\nPretraining generator...\n')
    for epoch in range(PRE_G_EPOCHS):
        loss = netG.pretrain(genloader)
        print('Epoch {} pretrain generator training loss: {}'.format(epoch, loss))

        generate_samples(netG, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        val_set = GeneratorDataset(EVAL_FILE)
        valloader = DataLoader(dataset=val_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
        loss = oracle.val(valloader)
        print('Epoch {} pretrain generator val loss: {}'.format(epoch + 1, loss))

    # pretrain discriminator
    print('\nPretraining discriminator...\n')
    for epoch in range(D_STEPS):
        generate_samples(netG, BATCH_SIZE, GENERATED_NUM, FAKE_FILE)
        dis_set = DiscriminatorDataset(REAL_FILE, FAKE_FILE)
        disloader = DataLoader(dataset=dis_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True)
        
        for _ in range(K_STEPS):
            loss = netD.dtrain(disloader)
            print('Epoch {} pretrain discriminator training loss: {}'.format(epoch + 1, loss))

    # adversarial training
    rollout = Rollout(netG, update_rate=ROLLOUT_UPDATE_RATE, rollout_num=ROLLOUT_NUM)
    print('\n#####################################################')
    print('Adversarial training...\n')

    for epoch in range(TOTAL_EPOCHS):
        for _ in range(G_STEPS):
            netG.pgtrain(BATCH_SIZE, SEQUENCE_LEN, rollout, netD)

        for d_step in range(D_STEPS):
            # train discriminator
            generate_samples(netG, BATCH_SIZE, GENERATED_NUM, FAKE_FILE)
            dis_set = DiscriminatorDataset(REAL_FILE, FAKE_FILE)
            disloader = DataLoader(dataset=dis_set,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)
        
            for k_step in range(K_STEPS):
                loss = netD.dtrain(disloader)
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
    args = vars(parser.parse_args())

    cfg = configparser.ConfigParser()
    cfgDIR = os.path.dirname(os.path.realpath(__file__))

    if args['config'] is not None:
        cfg.read(args['config'])
    else:
        cfg.read(os.path.join(cfgDIR, 'config.cfg'))

    cfg_dict = cfg['ARGS']

    for carg in cfg_dict:
        if carg in args:
            cfg_dict[carg] = str(args.get(carg))

    run(cfg_dict)
