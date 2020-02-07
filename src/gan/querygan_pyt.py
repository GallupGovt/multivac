#!usr/bin/env/python
import argparse
import configparser
from itertools import compress
import numpy as np
import os
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from discriminator import QueryGAN_Discriminator_CNN, MULTIVACDataset, Trainer
from gen_pyt.datasets.english.dataset import English
from gen_pyt.asdl.lang.eng.eng_asdl_helper import asdl_ast_to_english
from gen_pyt.asdl.lang.eng.eng_transition_system import EnglishTransitionSystem
from gen_pyt.components.action_info import get_action_infos
from gen_pyt.components.dataset import Example, Dataset
from gen_pyt.model import nn_utils
from gen_pyt.model.parser import Parser
from utilities.rollout import Rollout
from utilities.utils import load_word_vectors, deserialize_from_file

from multivac.src.rdf_graph.rdf_parse import StanfordParser


def DiscriminatorDataset(DIR, fake, vocab, limit=None):
    '''
    Take real examples from existing training dataset and add them to the 
    Generated dataset for adversarial training.
    '''
    data_file = MULTIVACDataset(DIR, vocab)

    if not fake:
        true_items = data_file.labels==1
        data_file.sentences = list(compress(data_file.sentences, true_items))
        data_file.labels = data_file.labels[true_items]
        data_file.size = data_file.labels.size(0)

        if limit:
            idx = random.sample(range(data_file.size), limit)
            data_file.sentences = list(compress(data_file.sentences, idx))
            data_file.labels = data_file.labels[idx]
            data_file.size = data_file.labels.size(0)

    y_onehot = torch.zeros(data_file.size, 2)
    y_onehot.scatter_(1, data_file.labels.long().unsqueeze(1), 1)

    data_file.labels = y_onehot

    maxlen = 150 # to match CNN classifier architecture
    sents = torch.full((len(data_file.sentences), maxlen), 
                        data_file.vocab.pad)

    for i, s in enumerate(data_file.sentences):
        sents[i, :len(s)] = s[:150]

    data_file.sentences = sents.long()

    return data_file

def disc_trainer(model, glove_emb, glove_vocab, use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
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

    if model.args['optim'] == 'adam':
        opt = optim.Adam
    elif model.args['optim'] == 'adagrad':
        opt = optim.Adagrad
    elif model.args['optim'] == 'sgd':
        opt = optim.SGD

    optimizer = opt(filter(lambda p: p.requires_grad,
                                  model.parameters()), 
                           lr=model.args['lr'], weight_decay=model.args['wd'])

    return Trainer(model.args, model, criterion, optimizer, device)

def generate_samples(net, seq_len, generated_num, parser, oracle=False, 
                     writeout=False):
    samples = []
    examples = []
    states = []
    texts = [''] * generated_num
    max_query_len = 0
    max_actions_len = 0
    dst_dir = net.args['sample_dir']
    net.eval()
    print("Generating Samples...")
    pbar = tqdm(total=generated_num)

    while len(samples) < generated_num:
        samps = []
        sts = []

        while True:
            query = net.vocab.convertToLabels(random.sample(range(net.vocab.size()), 
                                          seq_len))
            if oracle:
                samps, sts = net.parse(query, return_states=True, 
                                          beam_size=net.args['beam_size'])
            else:
                samps, _ = net.parse(query, beam_size=net.args['beam_size'])

            if samps[0].completed:
                break

        s = samps[0]
        samples.append(s)

        if oracle:
            states.append(sts[0])

        text = asdl_ast_to_english(s.tree)
        actions = net.transition_system.get_actions(s.tree)
        tgt_actions = get_action_infos(query, actions)
        example = Example(src_sent=query, tgt_actions=tgt_actions, 
                          tgt_text=text,  tgt_ast=s.tree, idx=len(samples))

        if len(example.src_sent) > max_query_len:
            max_query_len = len(example.src_sent)
        if len(example.tgt_actions) > max_actions_len:
            max_actions_len = len(example.tgt_actions)

        examples.append(example)
        pbar.update(1)

    pbar.close()

    if oracle:
        return samples, states, examples
    elif writeout:
        sample_parses = parser.get_parse('\n'.join([e.tgt_text for e in examples]))

        with open(os.path.join(dst_dir, 'text.toks'   ), 'w') as tokfile, \
             open(os.path.join(dst_dir, 'cat.txt'     ), 'w') as catfile:

            for i, parse in enumerate(sample_parses['sentences']):
                tokens = [x['word'] for x in parse['tokens']]
                tokfile.write(' '.join(tokens) + '\n')
                catfile.write('0' + '\n')

        return None
    else:
        return samples, examples

def emulate_embeddings(embeds, shape, device='cpu'):
    samples = torch.zeros(*shape, dtype=torch.float)
    samples.normal_(torch.mean(embeds), torch.std(embeds))
    return samples

def load_to_layer(layer, embeds, vocab, words=None):

    if words is None:
        words = vocab

    words = sorted(words.labelToIdx.items(), key=lambda x: x[1])

    new_tensor = layer.weight.data.new
    layer_rows = set(range(layer.num_embeddings))

    assert len(words) == layer.num_embeddings

    for word, idx in words:
        if word in vocab and vocab.getIndex(word) < embeds.size(0):
            word_id = vocab.getIndex(word)
            layer.weight[idx].data = new_tensor(embeds[word_id])
            layer_rows.remove(idx)

    layer.weight.requires_grad = False

    # layer_rows = list(layer_rows)
    # layer.weight[layer_rows].data = new_tensor(emulate_embeddings(embeds=embeds, 
    #                                                               shape=(len(layer_rows), 
    #                                                                      layer.embedding_dim)))

def run(cfg_dict):
    # Set up model and training parameters based on config file and runtime
    # arguments

    args = cfg_dict['ARGS']

    if args['continue']:
        continue_training(cfg_dict, args['gen_chk'], args['disc_chk'])

    gargs = cfg_dict['GENERATOR']
    dargs = cfg_dict['DISCRIMINATOR']
    gan_args = cfg_dict['GAN']

    seed = gan_args['seed']
    batch_size = gan_args['batch_size']
    total_epochs = gan_args['total_epochs']
    generated_num = gan_args['generated_num']
    vocab_size = gan_args['vocab_size']
    seq_len = gan_args['sequence_len']

    # rollout params
    rollout_update_rate = gan_args['rollout_update_rate']
    rollout_num = gan_args['rollout_num']

    g_steps = gan_args['g_steps']
    d_steps = gan_args['d_steps']
    k_steps = gan_args['k_steps']
    
    use_cuda = args['cuda']

    if not torch.cuda.is_available():
        print("No GPU available, running on CPU.")
        use_cuda = False

    gargs['cuda'] = use_cuda
    gargs['verbose'] = gan_args['verbose']
    dargs['cuda'] = use_cuda
    dargs['device'] = "cuda" if use_cuda else "cpu"
    dargs['verbose'] = gan_args['verbose']

    random.seed(seed)
    np.random.seed(seed)

    if gan_args['verbose']: print("Initializing Stanford Parser...")

    parser = StanfordParser(annots="tokenize ssplit")

    # Load input files for Generator: grammar and transition system, vocab,
    # word embeddings

    if gan_args['verbose']: print("Checking for existing grammar...")

    if gargs['grammar']:
        grammar = deserialize_from_file(gargs['grammar'])
    else:
        grammar = None

    glove_vocab, glove_emb = load_word_vectors(os.path.join(gan_args['glove_dir'], 
                                                            gan_args['glove_file']),
                                               lowercase=gan_args['glove_lower'])

    if gan_args['verbose']: print("Generating training dataset and grammar...")

    samples_data, prim_vocab, grammar = English.generate_dataset(gargs['annot_file'],
                                                                 gargs['texts_file'],
                                                                 grammar)
    transition_system = EnglishTransitionSystem(grammar)

    if gan_args['verbose']: print("Grammar and language transition system initiated.")


    # Build Generator model

    netG = Parser(gargs, glove_vocab, prim_vocab, transition_system)
    optimizer_cls = eval('torch.optim.%s' % gargs['optimizer'])
    netG.optimizer = optimizer_cls(netG.parameters(), lr=gargs['lr'], 
                                   betas = (gargs['beta_1'], 0.999),
                                   weight_decay=gargs['lr_decay'])

    if gargs['uniform_init']:
        if gan_args['verbose']: 
            print('uniformly initialize parameters [-{}, +{}]'.format(gargs['uniform_init'], 
                                                                  gargs['uniform_init']))
        nn_utils.uniform_init(-gargs['uniform_init'], gargs['uniform_init'], netG.parameters())
    elif gargs['glorot_init']:
        if gan_args['verbose']: print('use glorot initialization')
        nn_utils.glorot_init(netG.parameters())

    if gan_args['verbose']: print("Loading GloVe vectors as Generator embeddings...")

    load_to_layer(netG.src_embed, glove_emb, glove_vocab)
    load_to_layer(netG.primitive_embed, glove_emb, glove_vocab, prim_vocab)

    if gargs['cuda']: 
        netG.cuda()
        netG.optimizer.cuda()

    # Set up Discriminator component with given parameters

    if gan_args['verbose']: print("Loading Discriminator component...")
    dargs['vocab_size'] = glove_vocab.size()

    netD = QueryGAN_Discriminator_CNN(dargs, glove_vocab, glove_emb, 2) # CNN classifier

    # 
    # PRETRAIN GENERATOR & DISCRIMINATOR
    # 

    if gan_args['verbose']: print('\nPretraining generator...\n')
    # Pre-train epochs are set in config.cfg file
    netG.pretrain(Dataset(samples_data))
    rollout = Rollout(rollout_num=rollout_num, vocab=glove_vocab)

    # pretrain discriminator
    if gan_args['verbose']: print('Loading Discriminator pretraining dataset.')
    dis_set = MULTIVACDataset(netD.args['data'], glove_vocab)

    y_onehot = torch.zeros(dis_set.size, 2)
    y_onehot.scatter_(1, dis_set.labels.long().unsqueeze(1), 1)

    dis_set.labels = y_onehot

    maxlen = 150 # to match CNN classifier architecture
    sents = torch.full((dis_set.size, maxlen), dis_set.vocab.pad)

    for i, s in enumerate(dis_set.sentences):
        sents[i, :len(s)] = s[:150]

    dis_set.sentences = sents.long()
    
    if gan_args['verbose']: print("Pretraining discriminator...")

    #for i in tqdm(range(k_steps), desc='Pretraining discriminator ... '):
    for epoch in range(k_steps):
        loss = netD.train_single_code(dis_set)
        print('Epoch {} pretrain discriminator training loss: {}'.format(epoch + 1, loss))

    save_progress(netD, netG, [], -1, [], [])


    #
    # ADVERSARIAL TRAINING
    # 

    print('\n#####################################################')
    print('Adversarial training...\n')

    discriminator_losses = []
    generator_losses = []

    for epoch in range(total_epochs):
        for step in range(g_steps):
            # train generator
            hyps, states, examples = generate_samples(netG, seq_len, generated_num, parser, oracle=True)
            # samples = generate_samples(netG, seq_len, generated_num, parser)
            # hyps, examples = list(zip(*samples))
            step_begin = time.time()

            pgloss = netG.pgtrain(hyps, states, examples, rollout, netD)
            print('[Generator {}]  step elapsed {}s'.format(step, 
                                                            time.time() - step_begin))
            print('Generator adversarial loss={}, epoch={}'.format(pgloss, epoch))
            generator_losses.append(pgloss)

        for d_step in range(d_steps):
            # train discriminator
            generate_samples(netG, seq_len, generated_num, parser, writeout=True)
            real_set = DiscriminatorDataset(netD.args['data'], fake=False, vocab=glove_vocab, limit=generated_num+1)
            fake_set = DiscriminatorDataset(netG.args['sample_dir'], fake=True, vocab=glove_vocab)
        
            for k_step in range(k_steps):
                loss_r = netD.train_single_code(real_set)
                print('D_step {}, K-step {} Discriminator loss on real set: {}'.format(d_step + 1, k_step + 1, loss_r))
                loss_f = netD.train_single_code(fake_set)
                print('D_step {}, K-step {} Discriminator loss on fake set: {}'.format(d_step + 1, k_step + 1, loss_f))
                discriminator_losses.append((loss_r + loss_f)/2)
                
        save_progress(netD, netG, examples, epoch, discriminator_losses, generator_losses)
        

def continue_training(cfg_dict, gen_chk, disc_chk, epoch=0, gen_loss=None, disc_loss=None, use_cuda=False):
    args = cfg_dict['ARGS']
    gargs = cfg_dict['GENERATOR']
    dargs = cfg_dict['DISCRIMINATOR']
    gan_args = cfg_dict['GAN']

    seed = gan_args['seed']
    batch_size = gan_args['batch_size']
    total_epochs = gan_args['total_epochs']
    generated_num = gan_args['generated_num']
    vocab_size = gan_args['vocab_size']
    seq_len = gan_args['sequence_len']

    # rollout params
    rollout_update_rate = gan_args['rollout_update_rate']
    rollout_num = gan_args['rollout_num']

    g_steps = gan_args['g_steps']
    d_steps = gan_args['d_steps']
    k_steps = gan_args['k_steps']
    
    use_cuda = args['cuda']

    if not torch.cuda.is_available():
        use_cuda = False

    gargs['cuda'] = use_cuda
    gargs['verbose'] = gan_args['verbose']
    dargs['cuda'] = use_cuda
    dargs['verbose'] = gan_args['verbose']

    random.seed(seed)
    np.random.seed(seed)

    parser = StanfordParser(annots="depparse")

    if disc_loss is None:
        discriminator_losses = []
    
    if gen_loss is None:
        generator_losses = []

    if isinstance(gen_chk, str):
        gen_params = torch.load(gen_chk)
        netG = Parser.load(gen_chk)
        optimizer_cls = eval('torch.optim.%s' % netG.args['optimizer'])
        netG.optimizer = optimizer_cls(netG.parameters(), lr=netG.args['lr'])
        netG.optimizer.load_state_dict(gen_params['optimizer'])
    else:
        netG = gen_chk

    glove_vocab, glove_emb = load_word_vectors(os.path.join(gan_args['glove_dir'], 
                                                            gan_args['glove_file']),
                                               lowercase=gan_args['glove_lower'])

    if isinstance(disc_chk, str):
        device = torch.device("cuda" if use_cuda else "cpu")
        disc_params = torch.load(disc_chk)
        netD = QueryGAN_Discriminator_CNN(disc_params['args'], glove_vocab, glove_emb, 2)
        netD.load_state_dict(disc_params['state_dict'])

        if epoch == 0:
            epoch = disc_params['epoch']

        if netD.args['optim'] == 'adam':
            opt = optim.Adam
        elif netD.args['optim'] == 'adagrad':
            opt = optim.Adagrad
        elif netD.args['optim'] == 'sgd':
            opt = optim.SGD

        netD.optimizer = opt(filter(lambda p: p.requires_grad, netD.parameters()),
                             lr=netD.args['lr'], 
                             weight_decay=netD.args['wd'])
        netD.optimizer.load_state_dict(disc_params['optimizer'])
    else:
        netD = disc_chk

    rollout = Rollout(rollout_num=rollout_num, vocab=glove_vocab)

    print('\n#####################################################')
    print('Restarting adversarial training from epoch {}...\n'.format(epoch))

    for ep in range(epoch, total_epochs):
        for step in range(g_steps):
            # train generator
            hyps, states, examples = generate_samples(netG, seq_len, generated_num, parser, oracle=True)
            # hyps, examples = list(zip(*samples))
            step_begin = time.time()
            pgloss = netG.pgtrain(hyps, states, examples, rollout, netD)
            print('[Generator {}]  step elapsed {}s'.format(step, 
                                                            time.time() - step_begin))
            print('Generator adversarial loss={}'.format(pgloss))
            generator_losses.append(pgloss)

        for d_step in range(d_steps):
            # train discriminator
            generate_samples(netG, seq_len, generated_num, parser, writeout=True)
            real_set = DiscriminatorDataset(netD.args['data'], fake=False, vocab=glove_vocab, limit=generated_num+1)
            fake_set = DiscriminatorDataset(netG.args['sample_dir'], fake=True, vocab=glove_vocab)
        
            for k_step in range(k_steps):
                loss_r = netD.train_single_code(real_set)
                print('D_step {}, K-step {} Discriminator loss on real set: {}'.format(d_step + 1, k_step + 1, loss_r))
                loss_f = netD.train_single_code(fake_set)
                print('D_step {}, K-step {} Discriminator loss on fake set: {}'.format(d_step + 1, k_step + 1, loss_f))
                discriminator_losses.append((loss_r + loss_f)/2)
                
        save_progress(netD, netG, examples, ep, discriminator_losses, generator_losses)


def save_progress(netD, netG, examples, epoch, discriminator_losses, generator_losses):
    # Save Generator model state and metadata
    gen_save = os.path.join(netG.args['output_dir'], "gen_checkpoint.pth")
    gen_checkpoint = {'epoch': epoch,
                      'state_dict': netG.state_dict(),
                      'args': netG.args,
                      'transition_system': netG.transition_system,
                      'vocab': netG.vocab.__dict__,
                      'prim_vocab': netG.prim_vocab.__dict__,
                      'optimizer': netG.optimizer.state_dict()}
    torch.save(gen_checkpoint, gen_save)

    # Save Discriminator model state and metadata
    disc_save = os.path.join(netG.args['output_dir'], "disc_checkpoint.pth")
    dis_checkpoint = {'epoch': epoch,
                      'state_dict': netD.state_dict(),
                      'args': netD.args,
                      'optimizer': netD.optimizer.state_dict()}
    torch.save(dis_checkpoint, disc_save)

    # Save loss histories
    with open(os.path.join(netG.args['output_dir'], 
                           "generator_losses.csv"), "a") as f:
        for l in generator_losses:
            f.write("{},{}\n".format(epoch, l.item()))

    with open(os.path.join(netG.args['output_dir'], 
                           "discriminator_losses.csv"), "a") as f:
        for l in discriminator_losses:
            f.write("{},{}\n".format(epoch, l))

    # Save example generator outputs for qualitative assessment of progress
    if len(examples) > 0:
        save_examples = random.sample(examples, 10)

        with open(os.path.join(netG.args['output_dir'], 
                               "samples_{}.csv".format(epoch)), "w") as f:
            for e in save_examples:
                f.write(e.tgt_text + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build and train a Generative Adversarial Network to '
                    'produce coherent, well-formed English language questions. '
                    'Provide a config file with parameters for the system, and '
                    'optionally override any of these with commandline '
                    'arguments. To override a config parameter, pass an argument'
                    ' matching the parameter, but prefaced with the part of the'
                    ' system it refers to. '
                    'I.e., "--generator_pretrain_epochs 120" would override the'
                    'generator parameter "pretrain_epochs" and set this value'
                    'to "120". Valid system parts are "gan", "generator", and '
                    '"discriminator".')
    parser.add_argument('--cuda', default=False, action='store_true', 
                        help='Enable GPU training.')
    parser.add_argument('-c', '--config', required=False, 
                        help='Config file with updated parameters for generator;'
                             'defaults to "config.cfg" in this directory '
                             'otherwise.')
    parser.add_argument('-o', '--continue', default=False, action='store_true', 
                        help='Continue training from a previous checkpoint.')
    parser.add_argument('-g', '--gen_chk', required=False,
                        help='Path to Generator component checkpoint file.')
    parser.add_argument('-d', '--disc_chk', default=False, 
                        help='Path to Discriminator component checkpoint file.')

    all_args = parser.parse_known_args()
    args = vars(all_args[0])
    overrides = {}

    i = 0

    while i < len(all_args[1]):
        if all_args[1][i].startswith('--'):
            key = all_args[1][i][2:]
            value = all_args[1][i+1]

            if value.startswith('--'):
                overrides[key] = True
                i += 1
                continue
            else:
                overrides[key] = value
                i += 2
        else:
            i += 1
    
    cfg = configparser.ConfigParser()
    cfgDIR = os.path.dirname(os.path.realpath(__file__))

    if args['config'] is not None:
        cfg.read(args['config'])
    else:
        cfg.read(os.path.join(cfgDIR, 'config.cfg'))

    cfg_dict = cfg._sections
    cfg_dict['ARGS'] = args

    for arg in overrides:
        section, param = arg.split("_", 1)
        try:
            cfg[section.upper()][param] = overrides[arg]
        except KeyError:
            print("Section " + section.upper() + "not found in "
                  "" + args['config'] + ", skipping.")
            continue

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
