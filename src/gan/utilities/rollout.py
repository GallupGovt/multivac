import copy
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from discriminator import MULTIVACDataset, Tree
from gen_pyt.asdl.lang.eng.eng_asdl_helper import asdl_ast_to_english
from gen_pyt.model.parser import Parser
# from .tree_rollout import rollout_samples
from multivac.src.gan.gen_pyt.components.decode_hypothesis import DecodeHypothesis

from multivac.src.rdf_graph.rdf_parse import StanfordParser

class Rollout(object):
    def __init__(self, update_rate, rollout_num):
        #self.new_net = copy.deepcopy(net)
        self.rollout_num = rollout_num
        self.update_rate = update_rate
        self.parser = StanfordParser(annots='tokenize ssplit pos depparse')

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
            texts   = [[0] * batch_size] * max_action_len

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
                    texts[x][h] = asdl_ast_to_english(hyp.tree)
                    inputs[x][h] = self.hyp_to_parse(hyp, vocab)

                alltext = '\n'.join(texts[x])

                if len(alltext) > 100000:
                    step_parsed = []
                    
                    while len(alltext) > 0:
                        offset = alltext.rfind('\n',0,100000)
                        chunk = alltext[:offset if offset > 0 else len(alltext)]
                        alltext = alltext[len(chunk):]
                        step_parsed.extend(self.parser.get_parse(chunk)['sentences'])
                else:
                    step_parsed = self.parser.get_parse(alltext)['sentences']

                inputs[x].extend(Rollout.parse_to_trees(step_parsed, vocab))

            for j in range(max_action_len):
                preds = np.zeros(batch_size)

                for k in tqdm(range(batch_size), desc="Rating action step {}...".format(j)):
                    tree, inp = inputs[j][k]

                    if netD.args['cuda']:
                        inp = inp.cuda()

                    preds[k] = netD(tree, inp).item()

                if i == 0:
                    rewards.append(preds)
                else:
                    rewards[j] += preds

            texts = [self.hyp_to_parse(e.tgt_text, vocab) for e in examples]

            for k in tqdm(range(batch_size), desc="Rating action step {}...".format(max_action_len)):
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
