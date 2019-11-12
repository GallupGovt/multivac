import copy
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from discriminator.treelstm import MULTIVACDataset
from gen_pyt.asdl.lang.eng.eng_asdl_helper import asdl_ast_to_english
from gen_pyt.model.parser import Parser
from tree_rollout import rollout_samples

from multivac.src.rdf_graph.rdf_parse import StanfordParser

class Rollout(object):
    def __init__(self, net, update_rate, rollout_num):
        self.ori_net = net
        #self.new_net = copy.deepcopy(net)
        self.rollout_num = rollout_num
        self.update_rate = update_rate
        self.parser = StanfordParser(annots='tokenize ssplit pos depparse')

    def hyp_to_parse(self, hyp, vocab):
        if isinstance(hyp,str):
            text = hyp
        else:
            text = asdl_ast_to_english(hyp.tree)

        parse = self.parser.get_parse(text)['sentences'][0]
        tokens = [x['word'] for x in parse['tokens']]
        deps = sorted(parse['basicDependencies'], 
                      key=lambda x: x['dependent'])
        parents = [x['governor'] for x in deps]
        tree = MULTIVACDataset.read_tree(parents)

        return tree, torch.tensor(vocab.convertToIdx(tokens, '<unk>'), 
                                  dtype=torch.long, device='cpu')

    # Need to figure out if the structure needs to change with the new
    # models, or if this just works as is because they're sequences?
    def get_tree_reward(self, examples, hyps, netD, vocab, verbose=False):
        batch_size = len(hyps)
        netD.eval()
        rewards = []
        src_sents = [e.src_sent for e in examples]
        
        for i in range(self.rollout_num):
            if verbose: print("Rollout step {}".format(i))
            # results is list of lists, (max_action_len-1, len(hyps))
            samples = rollout_samples(self.ori_net, src_sents, hyps)
            if verbose: print("Samples generated of shape ({},{})".format(len(samples), len(samples[0])))

            inputs = [[]] * len(samples)

            for x in tqdm(range(len(samples)), "Translating trees to discriminator..."):
                for n, hyp in enumerate(samples[x]):
                    inputs[x][n] = self.hyp_to_parse(hyp, vocab)

            seq_len = len(inputs)

            for j in range(seq_len):
                preds = np.zeros(len(samples[0]))

                for k in tqdm(range(len(samples[0])), desc="Rating action step {}...".format(j)):
                    tree, inp = inputs[j][k]

                    if netD.args['cuda']:
                        inp = inp.cuda()

                    preds[k] = netD(tree, inp).item()

                if i == 0:
                    rewards.append(preds)
                else:
                    rewards[j] += preds

            texts = [self.hyp_to_parse(e.tgt_text, vocab) for e in examples]

            for k in tqdm(range(len(samples[0])), desc="Rating action step {}...".format(seq_len)):
                tree, inp = texts[k]

                if netD.args['cuda']:
                    inp = inp.cuda()

                preds[k] = netD(tree, inp).item()

            if i == 0:
                rewards.append(preds)
            else:
                rewards[-1] += preds

        rewards = np.array(rewards) / (1.0 * self.rollout_num)

        return rewards

    def get_reward(self, x, netD):
        batch_size = x.size(0)
        seq_len = x.size(1)

        netD.eval()

        rewards = []

        for i in range(self.rollout_num):
            for j in range(1, seq_len):
                data = x[:, 0:j]
                samples = self.ori_net.sample(batch_size, seq_len, data)
                pred = netD(samples)
                pred = pred.cpu().data[:, 1].numpy()

                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[j - 1] += pred

            pred = netD(x)
            pred = pred.cpu().data[:, 1].numpy()

            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len - 1] += pred

        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)

        return rewards

    # def update_params(self):
    #     dct = {}

    #     for name, param in self.ori_net.named_parameters():
    #         dct[name] = param.data

    #     for name, param in self.new_net.named_parameters():
    #         if name.startswith('emb'):
    #             param.data = dct[name]
    #         else:
    #             param.data = self.update_rate * param.data + (1 - self.update_rate) * dct[name]

