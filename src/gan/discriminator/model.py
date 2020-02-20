import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class QueryGAN_Discriminator_CNN(nn.Module):

    def __init__(self, args, vocab, vectors, output_shape):
        super(QueryGAN_Discriminator_CNN, self).__init__()

        self.args = args
        self.filter_sizes = eval(self.args['filter_sizes'])
        self.num_filters = self.args['num_filters']
        self.hidden_dims = self.args['hidden_dims']
        self.dropout_prob1 = self.args['dropout_prob1']
        self.dropout_prob2 = self.args['dropout_prob2']
        self.num_classes = output_shape
        self.channels_out = sum([((150-(k-1))//2)*self.num_filters
                                 for k in self.filter_sizes])
        self.vocab = vocab

        self.emb = nn.Embedding(vocab.size(), vectors.size(1))
        emb = torch.zeros(vocab.size(), vectors.size(1), dtype=torch.float,
                          device=args['device'])
        emb.normal_(0, 0.05)

        for word in vocab.labelToIdx.keys():
            if vocab.getIndex(word) < vectors.size(0):
                emb[vocab.getIndex(word)] = vectors[vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)].zero_()

        self.emb.weight.data.copy_(emb)
        del emb

        self.emb.weight.requires_grad = False
        self.dropout1 = nn.Dropout(self.dropout_prob1)

        self.vocab_size = len(vocab)
        self.batchsize = self.args['batch_size']
        self.num_epochs = self.args['num_epochs']

        self.conv_blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=vectors.shape[1],
                          out_channels=self.num_filters,
                          kernel_size=sz,
                          stride=1,
                          padding=0),
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()) for sz in self.filter_sizes]
        )

        self.out = nn.Sequential(
                        nn.Dropout(self.dropout_prob2),
                        nn.Linear(self.channels_out, self.hidden_dims),
                        nn.Linear(self.hidden_dims, self.num_classes)
                      )

        for block in self.conv_blocks:
            block.apply(self.init_weights)

        self.out.apply(self.init_weights)

        if self.args['optim'] == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                               self.parameters()), 
                                        betas = (self.args['beta_1'], 0.999),
                                        lr=self.args['lr'], 
                                        weight_decay=self.args['wd'])
        elif self.args['optim'] == 'adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, 
                                                  self.parameters()), 
                                           lr=self.args['lr'], 
                                           weight_decay=self.args['wd'])
        elif self.args['optim'] == 'sgd':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, 
                                              self.parameters()), 
                                       lr=self.args['lr'], 
                                       weight_decay=self.args['wd'])

    def init_weights(self, m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, verbatim_indices):
        embeddings = self.emb(verbatim_indices)
        embeddings = embeddings.permute(0, 2, 1)
        X = self.dropout1(embeddings)

        X = [conv(embeddings) for conv in self.conv_blocks]
        X_cat = torch.cat(X, 1)

        return self.out(X_cat)

    def predict(self, X):
        self.eval()

        with torch.no_grad():
            yhat = self(X).softmax(dim=-1)

            scores, labels = yhat.topk(1, -1, True, True)
            return scores, labels

    def train_single_code(self, train):

        if self.args['label_smoothing']:
            criterion = SmoothedCrossEntropy(self.args['label_smoothing'])
        else:
            criterion = nn.CrossEntropyLoss()

        return self.trainer(train, criterion)


    def trainer(self, train, criterion):
        trainloader = DataLoader(train, batch_size=self.args['batch_size'], 
                                 shuffle=True, num_workers=4)
        steps = len(trainloader)

        if self.args['device'] == 'cuda':
            self.cuda()
            self.optimizer.cuda()

        self.train()

        for i, (x, y) in enumerate(tqdm(trainloader)):
            verbs = x.to(self.args['device'])
            labels = y.to(self.args['device'])

            # Forward pass
            outputs = self(verbs)

            if not self.args['label_smoothing']:
                labels = labels.argmax(1)

            loss = criterion(outputs, labels)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

class SmoothedCrossEntropy(nn.Module):
    '''
    Adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38
    '''
    def __init__(self, smoothing):
        super(SmoothedCrossEntropy, self).__init__()

        self.smoothing = smoothing
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, output, target):
        '''
            output: Tensor of predictions for class labels of size 
                    batchsize * n_classes
            target: Onehot Tensor indicating actual class labels of size
                    batchsize * n_classes
        '''
        target = target * smoothing + (1 - target) * (1 - smoothing) / (n_class - 1)
        return -(one_hot * self.softmax(output)).mean(dim=1)



