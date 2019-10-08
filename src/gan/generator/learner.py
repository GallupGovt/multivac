from nn.utils.config_factory import config
from nn.utils.generic_utils import *

import logging
import numpy as np
import sys, os
import time

import decoder
import evaluation
from dataset import *
import config


class Learner(object):
    def __init__(self, model, train_data, val_data=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

        logging.info('initial learner with training set [%s] (%d examples)',
                     train_data.name,
                     train_data.count)
        if val_data:
            logging.info('validation set [%s] (%d examples)', val_data.name, val_data.count)

    def train(self):
        dataset = self.train_data
        nb_train_sample = dataset.count
        index_array = np.arange(nb_train_sample)

        nb_epoch = config.max_epoch
        batch_size = config.batch_size

        logging.info('begin training')
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

                if not config.enable_copy:
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
                logging.debug('prob_func finished computing')

                cum_nb_examples += cur_batch_size
                loss += batch_loss * batch_size

                logging.debug('Batch %d, avg. loss = %f', batch_index, batch_loss)

                if batch_index == 4:
                    elapsed = time.time() - begin_time
                    eta = nb_train_sample / (cum_nb_examples / elapsed)
                    print((', eta %ds' % (eta)))
                    sys.stdout.flush()

                if cum_updates % config.valid_per_batch == 0:
                    logging.info('begin validation')

                    # Need ENGLISH version of this!!
                    decode_results = decoder.decode_python_dataset(self.model, 
                                                                   self.val_data, 
                                                                   verbose=False)
                    bleu, accuracy = evaluation.evaluate_decode_results(self.val_data, 
                                                                        decode_results, 
                                                                        verbose=False)

                    val_perf = eval(config.valid_metric)

                    logging.info('avg. example bleu: %f', bleu)
                    logging.info('accuracy: %f', accuracy)

                    if len(history_valid_acc) == 0 or accuracy > np.array(history_valid_acc).max():
                        best_model_by_acc = self.model.pull_params()
                        # logging.info('current model has best accuracy')
                    history_valid_acc.append(accuracy)

                    if len(history_valid_bleu) == 0 or bleu > np.array(history_valid_bleu).max():
                        best_model_by_bleu = self.model.pull_params()
                        # logging.info('current model has best accuracy')
                    history_valid_bleu.append(bleu)

                    if len(history_valid_perf) == 0 or val_perf > np.array(history_valid_perf).max():
                        best_model_params = self.model.pull_params()
                        patience_counter = 0
                        logging.info('save current best model')
                        self.model.save(os.path.join(config.output_dir, 'model.npz'))
                    else:
                        patience_counter += 1
                        logging.info('hitting patience_counter: %d', patience_counter)
                        if patience_counter >= config.train_patience:
                            logging.info('Early Stop!')
                            early_stop = True
                            break
                    history_valid_perf.append(val_perf)

                if cum_updates % config.save_per_batch == 0:
                    self.model.save(os.path.join(config.output_dir, 'model.iter%d' % cum_updates))

            logging.info('[Epoch %d] cumulative loss = %f, (took %ds)',
                         epoch,
                         loss / cum_nb_examples,
                         time.time() - begin_time)

            if early_stop:
                break

        logging.info('training finished, save the best model')
        np.savez(os.path.join(config.output_dir, 'model.npz'), **best_model_params)

        if config.data_type == 'django' or config.data_type == 'hs':
            logging.info('save the best model by accuracy')
            np.savez(os.path.join(config.output_dir, 'model.best_acc.npz'), **best_model_by_acc)

            logging.info('save the best model by bleu')
            np.savez(os.path.join(config.output_dir, 'model.best_bleu.npz'), **best_model_by_bleu)
