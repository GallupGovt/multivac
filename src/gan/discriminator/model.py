import torch
import torch.nn as nn
import torch.nn.functional as F



# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


class QueryGAN_Discriminator(nn.Module):
    def __init__(self, args, vocab): #vocab_size, in_dim, mem_dim, sparsity, freeze):
        super().__init__()
        self.args = args
        self.vocab_size = len(vocab)
        self.in_dim = self.args['input_dim']
        self.mem_dim = self.args['mem_dim']
        self.hidden_dim = self.args['hidden_dim']
        self.sparsity = self.args['sparse']
        self.freeze = self.args['freeze_embed']
        self.emb = nn.Embedding(self.vocab_size, 
                                self.in_dim, 
                                padding_idx=vocab.pad, 
                                sparse=self.sparsity)
        if self.freeze:
            self.emb.weight.requires_grad = False
        #self.childsumtreelstm = ChildSumTreeLSTM(self.in_dim, self.mem_dim)
        
        self.discriminator_cnn = nn.Sequential(
                    nn.Conv1d(in_channels = self.in_dim, out_channels = 1,  
                        kernel_size = 1, stride = 1, padding = 1),
                    nn.ReLU(inplace = True),
                    nn.BatchNorm1d(1)) 
        

    def forward(self, tree, inputs):
        inputs_embed = self.emb(inputs)      
        inputs_unsqeeze = inputs_embed.unsqueeze_(-1)
        inputs_cnn = inputs_embed.permute(2,1,0)
        outputs = self.discriminator_cnn(inputs_cnn)
        output = torch.max(outputs)
        return output
