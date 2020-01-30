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
        self.channels_out = sum([((150-(k-1))//2)*self.num_filters \
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
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()) for sz in self.filter_sizes
            ]
        )

        self.out = nn.Sequential(
                        nn.Dropout(self.dropout_prob2),
                        nn.Linear(self.channels_out, self.hidden_dims),
                        nn.Linear(self.hidden_dims, self.num_classes)
                      )
        
        for block in self.conv_blocks:
            block.apply(self.init_weights)

        self.out.apply(self.init_weights)

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
            labels.apply_(lambda x: self.args['labels'][x])

            return scores, labels

    def train_single_code(self, train):
        criterion = nn.CrossEntropyLoss()
        #acc_metric = accuracy

        return self.trainer(train, criterion, 
                     self.args['early_stopping'])


    def trainer(self, train, criterion, early_stopping=True,
                n_epochs_stop=10):

        trainloader = DataLoader(train, batch_size=self.args['batch_size'], shuffle=True, num_workers=4)
        steps = len(trainloader)

        #acc = [0] * steps
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                                 self.parameters()), 
                                                 #lr=0.0001,
                                                 amsgrad=True)
        #epochs_no_improve = 0
        #max_test_acc = 0

        self.train()

        for i, (x, y) in enumerate(tqdm(trainloader)):
            verbs = x.to(self.args['device'])
            labels = y.to(self.args['device'])
        
            # Forward pass
            outputs = self.forward(verbs)
            loss = criterion(outputs, labels.argmax(1))
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

            # self.eval()

            # with torch.no_grad():
            #     correct = 0
            #     test_loader = DataLoader(test, 
            #                              batch_size=self.args['batch_size'], 
            #                              shuffle=True, num_workers=4)
            #     test_steps = len(test_loader)
            #     test_acc = [0] * test_steps
            #     test_loss = [0] * test_steps
                
            #     if early_stopping:
            #         min_loss = None

            #     for i, (x, y) in enumerate(test_loader):
            #         verbs = x.to(self.args['device'])
            #         labels = y.to(self.args['device'])
            #         outputs = self.forward(verbs)
            #         test_loss[i] = criterion(outputs, labels.float())
            #         test_acc[i] = acc_metric(outputs, labels, 1)

            #     test_acc = sum(test_acc)/len(test_acc)
            #     test_loss = sum(test_loss)/len(test_loss)

            #     if early_stopping:
            #         if min_loss is None:
            #             min_loss = test_loss
            #         elif 1 - test_loss/min_loss > 0.001:
            #             # Save the model
            #             self.save_checkpoint()
            #             epochs_no_improve = 0
            #         else:
            #             epochs_no_improve += 1

            #             # Check early stopping condition
            #             if epochs_no_improve == n_epochs_stop:
            #                 print('Test Loss: {:.4f}, topk_acc: '
            #                       '{:.4f}%'.format(test_loss, 100 * test_acc))
            #                 print('Early stopping! Reloading best weights '
            #                       'and saving.')
            #                 # Load in the best model
            #                 if self.best_model_state_dict is not None:
            #                     self.load_state_dict(self.best_model_state_dict)
            #                     self.optimizer.load_state_dict(self.opt_dict)
            #                     self.save_checkpoint(write=True)
            #                 break

            #     print('Test Loss: {:.4f}, topk_acc: '
            #           '{:.4f}%'.format(test_loss, 100 * test_acc))
    

# class QueryGAN_Discriminator_CNN(nn.Module):
#     def __init__(self, args, vocab): #vocab_size, in_dim, mem_dim, sparsity, freeze):
#         super().__init__()
#         self.args = args
#         self.vocab = vocab
#         self.vocab_size = len(vocab)
#         self.batchsize = self.args['batch_size']
#         self.num_epochs = self.args['num_epochs']

#     def forward(self, inputs):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         X = inputs.sentences
#         Y = inputs.labels

#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, stratify=Y)

#         model = CNN_Classifier(args_dict, vocab, glove_vectors, Y.shape[1])
#         model.args['device'] = device
#         model.to(device)

#         model.train_single_code(train, test)

#         probs, labels = model.predict(X)
#         output = probs.numpy()
#         return output


