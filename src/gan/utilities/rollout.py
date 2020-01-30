import copy
import numpy as np
import os
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from discriminator import MULTIVACDataset, Tree
from gen_pyt.asdl.lang.eng.eng_asdl_helper import asdl_ast_to_english
from gen_pyt.model.parser import Parser
# from .tree_rollout import rollout_samples
from multivac.src.gan.gen_pyt.components.decode_hypothesis import DecodeHypothesis

from multivac.src.rdf_graph.rdf_parse import StanfordParser

class RolloutDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.size = self.data.shape[0]
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return deepcopy(self.data[index])

class Rollout(object):
    def __init__(self, rollout_num, vocab):
        #self.new_net = copy.deepcopy(net)
        self.vocab = vocab
        self.tokenizer = Tokenizer(Vocab(strings=list(vocab.labelToIdx.keys())))
        self.rollout_num = rollout_num
        self.parser = StanfordParser(annots='tokenize')

    def hyp_to_parse(self, hyp, vocab):
        if isinstance(hyp,str):
            text = hyp
        else:
            text = asdl_ast_to_english(hyp.tree)

        parse = self.parser.get_parse(text)['sentences']

        if len(parse) > 0:
            tokens = [x['word'] for x in parse[0]['tokens']]
            deps = sorted(parse[0]['basicDependencies'], 
                          key=lambda x: x['dependent'])
            parents = [x['governor'] for x in deps]
            tree = MULTIVACDataset.read_tree(parents)
            inp = torch.tensor(vocab.convertToIdx(tokens, '<unk>'), 
                               dtype=torch.long, device='cpu')
        else:
            tree = Tree()
            inp = torch.tensor([])

        return tree, inp

    @staticmethod
    def parse_tokens(tree):
        text = asdl_ast_to_english(tree)
        tokens = [x for x in Tokenizer(text)]
        result = torch.tensor(vocab.convertToIdx(tokens, '<unk>'), 
                              dtype=torch.long, 
                              device='cpu')
        return result

    @staticmethod
    def parse_to_trees(parses, vocab):
        results = [''] * len(parses)

        for idx, parse in enumerate(parses):
            tokens = [x['word'] for x in parse['tokens']]
            deps = sorted(parse['basicDependencies'], 
                          key=lambda x: x['dependent'])
            parents = [x['governor'] for x in deps]
            tree = MULTIVACDataset.read_tree(parents)
            results[idx] = (tree, torch.tensor(vocab.convertToIdx(tokens, '<unk>'), 
                                               dtype=torch.long, device='cpu'))

        return results

    @staticmethod
    def ffwd_hyp(hyp, j):
        new_hyp = DecodeHypothesis()

        for i in range(j):
            if i < len(hyp.action_infos):
                new_hyp.apply_action_info(hyp.action_infos[i])

        return new_hyp

    def get_tree_reward(self, hyps, states, examples, 
                        netG, netD, vocab, verbose=False):
        batch_size = len(hyps)
        src_sents = [e.src_sent for e in examples]
        rewards = []
        max_action_len = max([len(hyp.actions) for hyp in hyps])

        netD.eval()

        for i in range(self.rollout_num):
            if verbose: print("Rollout step {}".format(i))

            samples = [[0] * batch_size] * max_action_len
            inputs  = [[0] * batch_size] * max_action_len
            # texts   = [[0] * batch_size] * max_action_len

            for j in tqdm(range(1, max_action_len)):
                for n in range(batch_size):
                    src = src_sents[n]
                    hyp = Rollout.ffwd_hyp(hyps[n], j)
                    state = states[n][:j]
                    samples[j-1][n] = netG.sample(src, hyp, state)

            if verbose: print("Samples generated of shape "
                              "({},{})".format(max_action_len, batch_size))

            for x in tqdm(range(max_action_len), "Translating trees..."):
                for h, hyp in enumerate(samples[x]):
                    inputs[x][h] = parse_tokens(hyp.tree)

            for j in range(max_action_len):
                samps = torch.full((len(inputs[x]), 150), vocab.pad)

                for idx, x in enumerate(inputs[x]):
                    samps[idx, :len(x)] = x[:150]

                samps = RolloutDataset(samps.long())
                roll_loader = DataLoader(samps, batch_size=netD.args['batch_size'], 
                                         shuffle=True, num_workers=4)
                preds = []

                for k, x in tqdm(enumerate(roll_loader)):
                    if netD.args['cuda']:
                        x = x.cuda()

                    preds.append(netD.predict(x).mean().item())

                if i == 0:
                    rewards.append(sum(preds)/len(preds))
                else:
                    rewards[j] += sum(preds)/len(preds)

            originals = [parse_tokens(hyp.tree) for hyp in hyps]

            for j in tqdm(range(batch_size), desc="Rating action step {}...".format(max_action_len)):
                samps = torch.full((len(originals), 150), vocab.pad)

                for idx, x in enumerate(originals):
                    samps[idx, :len(x)] = x[:150]

                samps = RolloutDataset(samps.long())
                roll_loader = DataLoader(samps, batch_size=netD.args['batch_size'], 
                                         shuffle=True, num_workers=4)

                preds = []

                for k, x in tqdm(enumerate(roll_loader)):
                    if netD.args['cuda']:
                        x = x.cuda()

                    preds.append(netD.predict(x).mean().item())

            if i == 0:
                rewards.append(sum(preds)/len(preds))
            else:
                rewards[-1] += sum(preds)/len(preds)

        rewards = np.array(rewards) / (1.0 * self.rollout_num)

        return rewards
