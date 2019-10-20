import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import QueryGAN_Discriminator
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR MULTIVAC DATASET
from treelstm import MULTIVACDataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args


# MAIN BLOCK
def main():
    global args
    args = parse_args()

    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    formatter = logging.Formatter(log_format)

    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', 
                             mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()

    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    vocab_file = os.path.join(args.data, 'multivac.vocab')

    if not os.path.isfile(vocab_file):
        dirs = [train_dir, dev_dir, test_dir]
        token_files = [os.path.join(split, 'text.toks') for split in dirs]
        vocab_file = os.path.join(args.data, 'multivac.vocab')

        utils.build_vocab(token_files, vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> MULTIVAC vocabulary size : %d ' % vocab.size())

    # load dataset splits
    train_file = os.path.join(args.data, 'multivac_train.pth')

    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = MULTIVACDataset(train_dir, vocab)
        torch.save(train_dataset, train_file)

    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data, 'multivac_dev.pth')

    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = MULTIVACDataset(dev_dir, vocab)
        torch.save(dev_dataset, dev_file)

    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data, 'multivac_test.pth')

    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = MULTIVACDataset(test_dir, vocab)
        torch.save(test_dataset, test_file)

    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    # initialize model, criterion/loss_function, optimizer
    dargs = vars(args)
    dargs['vocab_size'] = vocab.size()
    model = QueryGAN_Discriminator(dargs)
    criterion = nn.MSELoss()

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'multivac_embed.pth')

    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab

        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), 
                          glove_emb.size(1), 
                          dtype=torch.float, 
                          device=device)
        emb.normal_(0, 0.05)

        # zero out the embeddings for padding and other special words 
        # if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()

        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]

        torch.save(emb, emb_file)

    # plug these into embedding matrix inside model
    model.emb.weight.data.copy_(emb)

    model.to(device), criterion.to(device)

    if args.optim == 'adam':
        opt = optim.Adam
    elif args.optim == 'adagrad':
        opt = optim.Adagrad
    elif args.optim == 'sgd':
        opt = optim.SGD

    optimizer = opt(filter(lambda p: p.requires_grad,
                                  model.parameters()), 
                           lr=args.lr, weight_decay=args.wd)

    metrics = Metrics(2)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)
    best = -float('inf')

    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)
        dev_loss, dev_pred = trainer.test(dev_dataset)
        test_loss, test_pred = trainer.test(test_dataset)

        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, train_loss, train_pearson, train_mse))

        dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, dev_loss, dev_pearson, dev_mse))

        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, test_loss, test_pearson, test_mse))

        if best < test_pearson:
            best = test_pearson
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': test_pearson, 'mse': test_mse,
                'args': args, 'epoch': epoch
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))

if __name__ == "__main__":
    main()
