#!usr/bin/env/python
import argparse
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from models.oracle import Oracle
from models.rollout import Rollout

from data_utils import GeneratorDataset, DiscriminatorDataset


SEED = 88
BATCH_SIZE = 64
TOTAL_EPOCHS = 200 
GENERATED_NUM = 10000
VOCAB_SIZE = 5000
SEQUENCE_LEN = 20

REAL_FILE = 'data/real.data'
FAKE_FILE = 'data/fake.data'
EVAL_FILE = 'data/eval.data'

# generator params
PRE_G_EPOCHS = 120
G_EMB_SIZE = 32
G_HIDDEN_SIZE = 32
G_LR = 1e-3

# discriminator params
D_EMB_SIZE = 32
D_NUM_CLASSES = 2
D_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
D_NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
DROPOUT = 0.75
D_LR = 1e-3
D_L2_REG = 0.0

# rollout params
ROLLOUT_UPDATE_RATE = 0.8
ROLLOUT_NUM = 16

G_STEPS = 1
D_STEPS = 5
K_STEPS = 3


def generate_samples(net, batch_size, generated_num, output_file):
    samples = []
    for _ in range(generated_num // batch_size):
        sample = net.sample(batch_size, SEQUENCE_LEN).cpu().data.numpy().tolist()
        samples.extend(sample)
    
    with open(output_file, 'w') as f:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            f.write('{}\n'.format(string))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    args = parser.parse_args()
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    random.seed(SEED)
    np.random.seed(SEED)

    netG = Generator(VOCAB_SIZE, G_EMB_SIZE, G_HIDDEN_SIZE, G_LR, use_cuda)
    netD = Discriminator(VOCAB_SIZE, D_EMB_SIZE, D_NUM_CLASSES, D_FILTER_SIZES, D_NUM_FILTERS, DROPOUT, D_LR, D_L2_REG, use_cuda)
    oracle = Oracle(VOCAB_SIZE, G_EMB_SIZE, G_HIDDEN_SIZE, use_cuda)

    # generating synthetic data
    print('Generating data...')
    generate_samples(oracle, BATCH_SIZE, GENERATED_NUM, REAL_FILE)

    # pretrain generator
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
    main()
                


