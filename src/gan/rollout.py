import copy
import numpy as np

import torch
import torch.nn as nn


class Rollout(object):
    def __init__(self, net, update_rate, rollout_num):
        self.ori_net = net
        self.new_net = copy.deepcopy(net)
        self.rollout_num = rollout_num
        self.update_rate = update_rate

    # Need to figure out if the structure needs to change with the new
    # models, or if this just works as is because they're sequences?
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

    def update_params(self):
        dct = {}
        for name, param in self.ori_net.named_parameters():
            dct[name] = param.data
        for name, param in self.new_net.named_parameters():
            if name.startswith('emb'):
                param.data = dct[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dct[name]

