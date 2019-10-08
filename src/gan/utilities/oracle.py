import torch
import torch.nn as nn
import torch.nn.functional as F


class Oracle(nn.Module):
    '''Target LSTM as Oracle to generate training samples'''
    def __init__(self, vocab_size, emb_size, hidden_size, use_cuda):
        super(Oracle, self).__init__()
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        if self.use_cuda:
            self.criterion = nn.NLLLoss().cuda()
        else:
            self.criterion = nn.NLLLoss()

        self.reset_parameters()

    def forward(self, x):
        self.lstm.flatten_parameters()
        emb = self.emb(x)
        h_0, c_0 = self.init_hidden(x.size(0))
        out, _ = self.lstm(emb, (h_0, c_0))
        pred = F.log_softmax(self.linear(out.contiguous().view(-1, self.hidden_size)), dim=1)
        
        return pred 

    def step(self, x, h, c):
        '''
        Args:
            x: (batch_size, 1); sequence of generated tokens
            h: (1, batch_size, hidden_dim); lstm hidden state
            c: (1, batch_size, hidden_dim); lstm cell state
        Returns:
            pred: (batch_size, vocab_size); predicted prob for next tokens
            h: (1, batch_size, hidden_dim); lstm new hidden state
            c: (1, batch_size, hidden_dim); lstm new cell state
        '''
        self.lstm.flatten_parameters()
        emb = self.emb(x)
        out, (h, c) = self.lstm(emb, (h, c))
        pred = F.log_softmax(self.linear(out.view(-1, self.hidden_size)), dim=1)

        return pred, h, c
            
    def sample(self, batch_size, seq_len):
        '''
        Creates sequence of length seq_len to be added to training set
        '''
        x = torch.zeros(batch_size, 1, dtype=torch.int64)
        h, c = self.init_hidden(batch_size)
        if self.use_cuda:
            x = x.cuda()

        samples = []
        for _ in range(seq_len):
            out, h, c = self.step(x, h, c)
            prob = torch.exp(out)
            x = torch.multinomial(prob, 1)
            samples.append(x)

        return torch.cat(samples, dim=1)

    def init_hidden(self, batch_size):
        '''
        Initializes hidden and cell states
        '''
        h = torch.zeros((1, batch_size, self.hidden_size))
        c = torch.zeros((1, batch_size, self.hidden_size))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()

        return h, c

    def reset_parameters(self):
        '''
        Resets parameters to be drawn from the standard normal distribution
        '''
        for param in self.parameters():
            param.data.normal_(0, 1)

    def val(self, dataloader):
        self.eval()

        total_loss = 0.0
        with torch.no_grad():
            for X, y in dataloader:
                if self.use_cuda:
                    X, y = X.cuda(), y.cuda()

                y = y.contiguous().view(-1)
                pred = self(X)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        
        return total_loss / (len(dataloader.dataset) // dataloader.batch_size)