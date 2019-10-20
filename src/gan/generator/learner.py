from multivac.src.gan.generator.nn.utils.config_factory import config
from multivac.src.gan.generator.nn.utils.generic_utils import *

import logging
import numpy as np
import sys, os
import time

from . import decoder
from . import evaluation
from .dataset import *
# import config


class Learner(object):
    def __init__(self, cfg, model, train_data, val_data=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg

        if self.cfg['verbose']:
            print('initial learner with training set '
                  '[{}] ({} examples)'.format(train_data.name, train_data.count))
            if val_data:
                print('validation set [{}] ({} examples)'.format(val_data.name, 
                                                                 val_data.count))

    def train(self):
        dataset = self.train_data
        nb_train_sample = dataset.count
        index_array = np.arange(nb_train_sample)

        nb_epoch = self.cfg['max_epoch']
        batch_size = self.cfg['batch_size']

        if self.cfg['verbose']:
            print('begin training')

        cum_updates = 0
        patience_counter = 0
        early_stop = False
        history_valid_perf = []
        history_valid_bleu = []
        history_valid_acc = []
        best_model_params = best_model_by_acc = best_model_by_bleu = None

        for epoch in range(nb_epoch):
            # train_data_iter.reset()
            # if shuffle:
            np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)

            # epoch begin
            sys.stdout.write('Epoch %d' % epoch)
            begin_time = time.time()
            cum_nb_examples = 0
            loss = 0.0

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                cum_updates += 1

                batch_ids = index_array[batch_start:batch_end]
                examples = dataset.get_examples(batch_ids)
                cur_batch_size = len(examples)

                inputs = dataset.get_prob_func_inputs(batch_ids)

                if not self.cfg['enable_copy']:
                    tgt_action_seq = inputs[1]
                    tgt_action_seq_type = inputs[2]

                    for i in range(cur_batch_size):
                        for t in range(tgt_action_seq[i].shape[0]):
                            if tgt_action_seq_type[i, t, 2] == 1:
                                # can only be copied
                                if tgt_action_seq_type[i, t, 1] == 0:
                                    tgt_action_seq_type[i, t, 1] = 1
                                    tgt_action_seq[i, t, 1] = 1  # index of <unk>

                                tgt_action_seq_type[i, t, 2] = 0

                train_func_outputs = self.model.train_func(*inputs)
                batch_loss = train_func_outputs[0]

                if self.cfg['verbose']:
                    print('prob_func finished computing')

                cum_nb_examples += cur_batch_size
                loss += batch_loss * batch_size

                if self.cfg['verbose']:
                    print('Batch {}, avg. loss = {}'.format(batch_index, batch_loss, 4))

                if batch_index == 4:
                    elapsed = time.time() - begin_time
                    eta = nb_train_sample / (cum_nb_examples / elapsed)
                    print((', eta %ds' % (eta)))
                    sys.stdout.flush()

                if cum_updates % self.cfg['valid_per_batch'] == 0:
                    if self.cfg['verbose']:
                        print('begin validation')

                    # [[(cid, cand, text), ()...], [()...]...]
                    # list of lists of tuples -> id, example, decoded text
                    decode_results = decoder.decode_english_dataset(self.model, 
                                                                    self.val_data, 
                                                                    self.cfg)
                    bleu, accuracy = evaluation.evaluate_decode_results(self.val_data, 
                                                                        decode_results, 
                                                                        self.cfg)

                    val_perf = eval(self.cfg['valid_metric'])

                    if self.cfg['verbose']:
                        print('avg. example bleu: {}'.format(bleu))
                        print('accuracy: {}'.format(accuracy))

                    if len(history_valid_acc) == 0 or accuracy > np.array(history_valid_acc).max():
                        best_model_by_acc = self.model.pull_params()

                    history_valid_acc.append(accuracy)

                    if len(history_valid_bleu) == 0 or bleu > np.array(history_valid_bleu).max():
                        best_model_by_bleu = self.model.pull_params()

                    history_valid_bleu.append(bleu)

                    if len(history_valid_perf) == 0 or val_perf > np.array(history_valid_perf).max():
                        best_model_params = self.model.pull_params()
                        patience_counter = 0

                        if self.cfg['verbose']:
                            print('save current best model')

                        self.model.save(os.path.join(self.cfg['output_dir'], 'model.npz'))
                    else:
                        patience_counter += 1

                        if self.cfg['verbose']:
                            print('hitting patience_counter: {}'.format(patience_counter))

                        if patience_counter >= self.cfg['train_patience']:
                            if self.cfg['verbose']:
                                print('Early Stop!')

                            early_stop = True
                            break

                    history_valid_perf.append(val_perf)

                if cum_updates % self.cfg['save_per_batch'] == 0:
                    self.model.save(os.path.join(self.cfg['output_dir'], 
                                                 'model.iter%d' % cum_updates))

            if self.cfg['verbose']:
                print('[Epoch {}] cumulative loss = {}, (took {}s)'.format(epoch,
                                                                           loss / cum_nb_examples,
                                                                           time.time() - begin_time))

            if early_stop:
                break

        if best_model_params is None:
            best_model_params = self.model.pull_params()

        if self.cfg['verbose']:
            print('training finished, save the best model')

        np.savez(os.path.join(self.cfg['output_dir'], 'model.npz'), **best_model_params)

        if self.cfg['verbose']:
            print('save the best model by accuracy')

        np.savez(os.path.join(self.cfg['output_dir'], 'model.best_acc.npz'), **best_model_by_acc)

        if self.cfg['verbose']:
            print('save the best model by bleu')

        np.savez(os.path.join(self.cfg['output_dir'], 'model.best_bleu.npz'), **best_model_by_bleu)
