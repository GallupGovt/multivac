from tqdm import tqdm

import torch

from . import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device=self.device)

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, inputs, label = dataset[indices[idx]]
            inputs = inputs.to(self.device)
            label = label.to(self.device)
            output = self.model(tree, inputs)
            try:
                loss = self.criterion(output, label)
            except:
                import pdb; pdb.set_trace()
            total_loss += loss.item()
            loss.backward()

            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()

        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device=self.device)
            indices = torch.arange(0, 2, dtype=torch.float, device=self.device)

            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                tree, inputs, label = dataset[idx]
                inputs, label = inputs.to(self.device), label.to(self.device)
                output = self.model(tree, inputs)
                loss = self.criterion(output, label)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.round(output)

        return total_loss / len(dataset), predictions
