#!usr/bin/env/python
import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    '''Reward-refined NLLLoss for adversarial training of generator'''
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target, reward):
        '''
        Args:
            pred: (batch_size, seq_len), 
            target : (batch_size, seq_len), 
            reward : (batch_size, ); reward of each whole sentence
        '''
        one_hot = torch.zeros(pred.size(), dtype=torch.bool)
        if pred.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view(-1, 1), True)
        loss = torch.masked_select(pred, one_hot)
        loss = loss * reward.contiguous().view(-1)
        loss = -torch.sum(loss)
        return loss


class WassersteinLoss(nn.Module):
    def __init__(self, wgan_reg_lambda=1.0):
        super(WassersteinLoss, self).__init__()
        self.wgan_reg_lambda = wgan_reg_lambda

    def forward(self, pred, target):
        neg = (target == 0).nonzero().view(-1)
        pos = (target != 0).nonzero().view(-1)
        
        pred_neg = pred[neg]
        pred_pos = pred[pos]

        wgan_loss = torch.abs(
            torch.sum(pred_neg) / pred_neg.size(0).float() - 
            torch.sum(pred_pos) / pred_pos.size(0).float()
        )

        loss = self.wgan_reg_lambda * wgan_loss
        return loss
