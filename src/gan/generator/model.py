import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from collections import OrderedDict
import copy
import sys

from multivac.src.gan.discriminator.treelstm import Constants

from multivac.src.gan.generator.nn.layers.embeddings import Embedding
from multivac.src.gan.generator.nn.layers.core import Dense, Dropout, WordDropout
from multivac.src.gan.generator.nn.layers.recurrent import BiLSTM, LSTM
import multivac.src.gan.generator.nn.optimizers as optimizers
import multivac.src.gan.generator.nn.initializations as initializations
from multivac.src.gan.generator.nn.activations import softmax
from multivac.src.gan.generator.nn.utils.theano_utils import *

from multivac.src.gan.generator.lang.grammar import Grammar
from multivac.src.gan.generator.astnode import ASTNode, Rule, DecodeTree
from multivac.src.gan.generator.dataset import DataEntry
from multivac.src.rdf_graph.rdf_parse import tokenize_text
from multivac.src.gan.generator.components import Hyp, PointerNet, CondAttLSTM

sys.setrecursionlimit(50000)

class Generator():
    '''Generator'''
    def __init__(self, config, oracle=False):
        self.cfg = config
        self.oracle = oracle

        if self.cfg['verbose']:
            print("Query embedding...")

        self.query_embedding = Embedding(self.cfg['source_vocab_size'], 
                                         self.cfg['word_embed_dim'], 
                                         name='query_embed')

        if self.cfg['verbose']:
            print("Query encoder Bi-LSTM...")
        self.query_encoder_lstm = BiLSTM(self.cfg['word_embed_dim'], 
                                         int(self.cfg['encoder_hidden_dim'] / 2), 
                                         return_sequences=True,
                                         name='query_encoder_lstm')

        cond_att_lstm_dims =  self.cfg['rule_embed_dim'] \
                            + self.cfg['node_embed_dim'] \
                            + self.cfg['rule_embed_dim']

        if self.cfg['verbose']:
            print("Decoder LSTM with Conditional Attention...")
        self.decoder_lstm = CondAttLSTM(cond_att_lstm_dims,
                                        self.cfg['decoder_hidden_dim'], 
                                        self.cfg['encoder_hidden_dim'], 
                                        self.cfg['attention_hidden_dim'],
                                        self.cfg['parent_hidden_state_feed'],
                                        self.cfg['tree_attention'],
                                        name='decoder_lstm')

        if self.cfg['verbose']:
            print("Pointer Net layers...")
        self.src_ptr_net = PointerNet(self.cfg)


        if self.cfg['verbose']:
            print("Terminal softmax layer...")
        self.terminal_gen_softmax = Dense(self.cfg['decoder_hidden_dim'], 2, 
                                          activation='softmax', 
                                          name='terminal_gen_softmax')

        if self.cfg['verbose']:
            print("Rule embeddings...")
        self.rule_embedding_W = initializations.get('normal')((self.cfg['rule_num'], 
                                                               self.cfg['rule_embed_dim']), 
                                                              name='rule_embedding_W', 
                                                              scale=0.1)
        self.rule_embedding_b = shared_zeros(self.cfg['rule_num'], 
                                             name='rule_embedding_b')

        if self.cfg['verbose']:
            print("Node embeddings...")
        self.node_embedding = initializations.get('normal')((self.cfg['node_num'], 
                                                             self.cfg['node_embed_dim']), 
                                                            name='node_embed', 
                                                            scale=0.1)

        if self.cfg['verbose']:
            print("Vocab embedding...")
        self.vocab_embedding_W = initializations.get('normal')((self.cfg['target_vocab_size'], 
                                                                self.cfg['rule_embed_dim']), 
                                                               name='vocab_embedding_W', 
                                                               scale=0.1)
        self.vocab_embedding_b = shared_zeros(self.cfg['target_vocab_size'], 
                                              name='vocab_embedding_b')


        if self.cfg['verbose']:
            print("Hidden State layers...")
        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = Dense(self.cfg['decoder_hidden_dim'], 
                                                 self.cfg['rule_embed_dim'], 
                                                 name='decoder_hidden_state_W_rule')
        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_token= Dense(self.cfg['decoder_hidden_dim'] + self.cfg['encoder_hidden_dim'], 
                                                 self.cfg['rule_embed_dim'],
                                                 name='decoder_hidden_state_W_token')

        # self.rule_encoder_lstm.params
        self.params =  self.query_embedding.params \
                     + self.query_encoder_lstm.params \
                     + self.decoder_lstm.params \
                     + self.src_ptr_net.params \
                     + self.terminal_gen_softmax.params \
                     + [self.rule_embedding_W, 
                        self.rule_embedding_b, 
                        self.node_embedding, 
                        self.vocab_embedding_W, 
                        self.vocab_embedding_b] \
                     + self.decoder_hidden_state_W_rule.params \
                     + self.decoder_hidden_state_W_token.params

        if self.cfg['verbose']:
            print("Random Streams...")
        self.srng = RandomStreams()

    def build(self):
        if self.cfg['verbose']:
            print("Build encoder system...")
        #
        # Build Query Encoder LSTM Component
        #
        # tokens: (batch_size, max_query_length) -> 
        #    embed: (batch_size, max_query_length, query_token_embed_dim)
        #    mask:  (batch_size, max_query_length)
        if self.cfg['verbose']:
            print("\tQuery token embeddings...")
        query_tokens = ndim_itensor(2, 'query_tokens')
        query_token_embed, query_token_embed_mask = \
            self.query_embedding(query_tokens, mask_zero=True)

        # input shape:  (batch_size, max_query_length, query_token_embed_dim)
        # output shape: (batch_size, max_query_length, query_embed_dim)
        query_embed = self.query_encoder_lstm(query_token_embed, 
                                              mask=query_token_embed_mask,
                                              dropout=self.cfg['dropout'], 
                                              srng=self.srng)

        #
        # Build Decoder LSTM Component
        # 
        # Dim 3 tensors: (batch_size, max_example_action_num, action_type)
        # Dim 2 tensors: (batch_size, max_example_action_num)
        if self.cfg['verbose']:
            print("\tAction embeddings...")
        tgt_action_seq = ndim_itensor(3, 'tgt_action_seq')
        tgt_action_seq_type = ndim_itensor(3, 'tgt_action_seq_type')
        tgt_par_rule_seq = ndim_itensor(2, 'tgt_par_rule_seq')
        tgt_par_t_seq = ndim_itensor(2, 'tgt_par_t_seq')
        tgt_node_seq = ndim_itensor(2, 'tgt_node_seq')

            # Action Sequence Embeddings
            #
        # in:  (batch_size, max_example_action_num, action_type) ->
        # out: (batch_size, max_example_action_num, action_embed_dim)
        tgt_action_seq_embed = \
            T.switch(
                T.shape_padright(tgt_action_seq[:, :, 0] > 0), 
                                 self.rule_embedding_W[tgt_action_seq[:,:,0]], 
                                 self.vocab_embedding_W[tgt_action_seq[:,:,1]]
                    )

        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)
        # (batch_size, max_example_action_num)
        tgt_action_seq_mask = T.any(tgt_action_seq_type, axis=-1)
        
            # Rule Embeddings
            #
        # parent rule application embeddings
        if self.cfg['verbose']:
            print("\tRule  embeddings...")
        tgt_par_rule_embed = T.switch(tgt_par_rule_seq[:, :, None] < 0,
                                      T.alloc(0., 1, self.cfg['rule_embed_dim']),
                                      self.rule_embedding_W[tgt_par_rule_seq])

        if not self.cfg['parent_action_feed']:
            tgt_par_rule_embed *= 0.

            # Node Embeddings
            #
        # node seq: (batch_size, max_example_action_num) ->
        #    embeds: (batch_size, max_example_action_num, symbol_embed_dim)
        if self.cfg['verbose']:
            print("\tNode embeddings...")
        tgt_node_embed = self.node_embedding[tgt_node_seq]

        if not self.cfg['frontier_node_type_feed']:
            tgt_node_embed *= 0.

        decoder_input = T.concatenate([tgt_action_seq_embed_tm1, 
                                       tgt_node_embed, 
                                       tgt_par_rule_embed], axis=-1)

        # input dims: (batch_size, 
        #              max_example_action_num, 
        #              action_embed_dim + symbol_embed_dim + action_embed_dim)
        # output dims: 
        # decoder_hidden_states: (batch_size, max_example_action_num, lstm_hidden_state)
        # ctx_vectors: (batch_size, max_example_action_num, encoder_hidden_dim)
        if self.cfg['verbose']:
            print("\tDecoder processing...")

        H, _, C = self.decoder_lstm(decoder_input,
                                    context=query_embed,
                                    context_mask=query_token_embed_mask,
                                    mask=tgt_action_seq_mask,
                                    parent_t_seq=tgt_par_t_seq,
                                    dropout=self.cfg['dropout'],
                                    srng=self.srng)

        # ====================================================
        # apply additional non-linearity transformation before
        # predicting actions
        # ====================================================

        if self.cfg['verbose']:
            print("\tAdditional non-linearity transformations...")
        batch_size = tgt_action_seq.shape[0]
        max_example_action_num = tgt_action_seq.shape[1]
        ptr_net_decoder_state = T.concatenate([H, C], axis=-1)

        decoder_hidden_state_trans_rule = self.decoder_hidden_state_W_rule(H)
        decoder_hidden_state_trans_token = self.decoder_hidden_state_W_token(ptr_net_decoder_state)

        # (batch_size, max_example_action_num, rule_num)
        rule_predict = softmax(T.dot(decoder_hidden_state_trans_rule, 
                                     T.transpose(self.rule_embedding_W)) \
                               + self.rule_embedding_b)

        # (batch_size, max_example_action_num)
        rule_tgt_prob = rule_predict[T.shape_padright(T.arange(batch_size)),
                                     T.shape_padleft(T.arange(max_example_action_num)),
                                     tgt_action_seq[:, :, 0]]

        # (batch_size, max_example_action_num, target_vocab_size)
        vocab_predict = softmax(T.dot(decoder_hidden_state_trans_token, 
                                      T.transpose(self.vocab_embedding_W)) \
                                + self.vocab_embedding_b)

        # (batch_size, max_example_action_num)
        vocab_tgt_prob = vocab_predict[T.shape_padright(T.arange(batch_size)),
                                       T.shape_padleft(T.arange(max_example_action_num)),
                                       tgt_action_seq[:, :, 1]]

        # (batch_size, max_example_action_num, max_query_length)
        copy_prob = self.src_ptr_net(query_embed, 
                                     query_token_embed_mask, 
                                     ptr_net_decoder_state)

        # (batch_size, max_example_action_num)
        copy_tgt_prob = copy_prob[T.shape_padright(T.arange(batch_size)),
                                  T.shape_padleft(T.arange(max_example_action_num)),
                                  tgt_action_seq[:, :, 2]]

        # (batch_size, max_example_action_num, 2)
        terminal_gen_action_prob = self.terminal_gen_softmax(H)

        # (batch_size, max_example_action_num)
        tgt_prob = tgt_action_seq_type[:, :, 0] * rule_tgt_prob + \
                   tgt_action_seq_type[:, :, 1] * terminal_gen_action_prob[:, :, 0] * vocab_tgt_prob + \
                   tgt_action_seq_type[:, :, 2] * terminal_gen_action_prob[:, :, 1] * copy_tgt_prob

        # Defining MLE loss - I think.
        likelihood = T.log(tgt_prob + 1.e-7 * (1 - tgt_action_seq_mask))
        loss = - (likelihood * tgt_action_seq_mask).sum(axis=-1)
        loss = T.mean(loss)

        # let's build the function!

        if self.cfg['verbose']:
            print("\tBuilding theano.train function...")
        train_inputs = [query_tokens, tgt_action_seq, tgt_action_seq_type,
                        tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq]
        optimizer = optimizers.get(self.cfg['optimizer'])
        optimizer.clip_grad = self.cfg['clip_grad']
        updates, grads = optimizer.get_updates(self.params, loss)
        self.train_func = theano.function(train_inputs, [loss], updates=updates)

        self.build_decoder(query_tokens, 
                           query_token_embed, 
                           query_token_embed_mask)

    def build_decoder(self, 
                      query_tokens, query_token_embed, query_token_embed_mask):
        if self.cfg['verbose']:
            print("Build decoder system...")
        # Re-setup the encoder
        query_embed = self.query_encoder_lstm(query_token_embed, 
                                              mask=query_token_embed_mask,
                                              dropout=self.cfg['dropout'], train=False)

        # (batch_size, n_timestep, decoder_state_dim)
        hist_h = ndim_tensor(3, name='hist_h')

        # ([time_step])
        time_steps = T.ivector(name='time_steps')

        # (batch_size) -> (batch_size, 1, node_embed_dim)
        node_id = T.ivector(name='node_id')
        node_embed = self.node_embedding[node_id]
        node_embed_reshaped = node_embed.dimshuffle((0, 'x', 1))

        if not self.cfg['frontier_node_type_feed']:
            node_embed_reshaped *= 0.

        # (batch_size) -> (batch_size, 1, node_embed_dim)
        par_rule_id = T.ivector(name='par_rule_id')
        par_rule_embed = T.switch(par_rule_id[:, None] < 0,
                                  T.alloc(0., 1, self.cfg['rule_embed_dim']),
                                  self.rule_embedding_W[par_rule_id])
        par_rule_embed_reshaped = par_rule_embed.dimshuffle((0, 'x', 1))

        if not self.cfg['parent_action_feed']:
            par_rule_embed_reshaped *= 0.

        # (batch_size) -> (batch_size, 1)
        parent_t = T.ivector(name='parent_t')
        parent_t_reshaped = T.shape_padright(parent_t)

        # (batch_size, decoder_state_dim)
        decoder_prev_state = ndim_tensor(2, name='decoder_prev_state')
        decoder_prev_cell = ndim_tensor(2, name='decoder_prev_cell')
        prev_action_embed = ndim_tensor(2, name='prev_action_embed')

        # (batch_size, 1, decoder_state_dim)
        prev_action_embed_reshaped = prev_action_embed.dimshuffle((0, 'x', 1))

        decoder_input = T.concatenate([prev_action_embed_reshaped,
                                       node_embed_reshaped, 
                                       par_rule_embed_reshaped], axis=-1)

        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, field_token_encode_dim)
        next_state_dim3, next_cell_dim3, C = self.decoder_lstm(decoder_input,
                                                               init_state=decoder_prev_state,
                                                               init_cell=decoder_prev_cell,
                                                               hist_h=hist_h,
                                                               context=query_embed,
                                                               context_mask=query_token_embed_mask,
                                                               parent_t_seq=parent_t_reshaped,
                                                               dropout=self.cfg['dropout'],
                                                               train=False,
                                                               time_steps=time_steps)

        decoder_next_state = next_state_dim3.flatten(2)
        decoder_next_cell = next_cell_dim3.flatten(2)

        decoder_next_state_trans_rule = \
            self.decoder_hidden_state_W_rule(decoder_next_state)
        decoder_next_state_trans_token = \
            self.decoder_hidden_state_W_token(T.concatenate([decoder_next_state, 
                                                            C.flatten(2)], axis=-1))

        rule_prob = softmax(T.dot(decoder_next_state_trans_rule, 
                                  T.transpose(self.rule_embedding_W)) \
                            + self.rule_embedding_b)

        gen_action_prob = self.terminal_gen_softmax(decoder_next_state)

        vocab_prob = softmax(T.dot(decoder_next_state_trans_token, 
                                   T.transpose(self.vocab_embedding_W)) \
                             + self.vocab_embedding_b)

        ptr_net_decoder_state = T.concatenate([next_state_dim3, C], axis=-1)

        copy_prob = self.src_ptr_net(query_embed, 
                                     query_token_embed_mask, 
                                     ptr_net_decoder_state)

        copy_prob = copy_prob.flatten(2)

        inputs = [query_tokens]
        outputs = [query_embed, query_token_embed_mask]

        self.decoder_func_init = theano.function(inputs, outputs)

        inputs = [time_steps, decoder_prev_state, 
                  decoder_prev_cell, hist_h, prev_action_embed,
                  node_id, par_rule_id, parent_t,
                  query_embed, query_token_embed_mask]

        outputs = [decoder_next_state, decoder_next_cell,
                   rule_prob, gen_action_prob, vocab_prob, copy_prob]

        self.decoder_func_next_step = theano.function(inputs, outputs)

    def decode(self, example, grammar, terminal_vocab, 
               beam_size, max_time_step):
        # beam search decoding

        eos = 1
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W.get_value(borrow=True)
        rule_embedding = self.rule_embedding_W.get_value(borrow=True)

        query_tokens = example.data[0]

        query_embed, query_token_embed_mask = self.decoder_func_init(query_tokens)

        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp(grammar)
        root_hyp.state = np.zeros(self.cfg['decoder_hidden_dim']).astype('float32')
        root_hyp.cell = np.zeros(self.cfg['decoder_hidden_dim']).astype('float32')
        root_hyp.action_embed = np.zeros(self.cfg['rule_embed_dim']).astype('float32')
        root_hyp.node_id = grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]

        # source word id in the terminal vocab
        src_token_id = [terminal_vocab[t] for t in example.query_tokens]
        
        if self.cfg['max_query_length'] is not None:
            src_token_id = src_token_id[:self.cfg['max_query_length']]

        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()

        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else: token_set.add(tid)

        for t in range(max_time_step):
            print("{}\n".format(t) + '\n'.join([x.__repr__() for x in hyp_samples]))
            hyp_num = len(hyp_samples)
            decoder_prev_state = np.array([hyp.state for hyp in hyp_samples]).astype('float32')
            decoder_prev_cell = np.array([hyp.cell for hyp in hyp_samples]).astype('float32')

            hist_h = np.zeros((hyp_num, max_time_step, self.cfg['decoder_hidden_dim'])).astype('float32')

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = hyp.hist_h

            prev_action_embed = np.array([hyp.action_embed for hyp in hyp_samples]).astype('float32')
            node_id = np.array([hyp.node_id for hyp in hyp_samples], dtype='int32')
            parent_rule_id = np.array([hyp.parent_rule_id for hyp in hyp_samples], dtype='int32')
            parent_t = np.array([hyp.get_action_parent_t() for hyp in hyp_samples], dtype='int32')
            query_embed_tiled = np.tile(query_embed, [live_hyp_num, 1, 1])
            query_token_embed_mask_tiled = np.tile(query_token_embed_mask, [live_hyp_num, 1])

            inputs = [np.array([t], dtype='int32'), decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, \
            rule_prob, gen_action_prob, vocab_prob, copy_prob  = self.decoder_func_next_step(*inputs)

            new_hyp_samples = []

            cut_off_k = beam_size
            score_heap = []

            # iterating over items in the beam

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = np.array([hyp.score for hyp in hyp_samples])

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in range(live_hyp_num):
                hyp = hyp_samples[k]

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                assert hyp, 'none hyp!'

                # if it's not a leaf
                if not grammar.is_value_node(frontier_nt):
                    # iterate over all the possible rules
                    if self.cfg['head_nt_constraint']:
                        rules = grammar[frontier_nt.as_type_node] 
                    else:
                        rules = grammar

                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt

                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = np.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf that holds values
                    cand_copy_prob = 0.0

                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]

                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score

                        unk_word = example.query_tokens[unk_pos]
                        unk_words.append(unk_word)

                        cand_copy_prob = gen_action_prob[k, 1]

                    word_gen_hyp_ids.append(k)
                    cand_copy_probs.append(cand_copy_prob)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = np.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores =  hyp_scores[word_gen_hyp_ids, None] \
                                      + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, 
                                              word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]
            #import pdb; pdb.set_trace()

            # expand_cand_num = 0
            for cand_id in top_cand_ids:
                # cand is rule application
                new_hyp = None

                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp(hyp)
                    new_hyp.apply_rule(rule)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    word_gen_hyp_id = int(word_gen_hyp_id)
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]

                    new_hyp = Hyp(hyp)
                    new_hyp.append_token(token)

                    if self.cfg['verbose']:
                        cand_copy_prob = cand_copy_probs[word_gen_hyp_id]

                        # if cand_copy_prob > 0.5:
                        #     print(str(new_hyp.frontier_nt()) + ''
                        #           '::copy[{}][p={}]::'.format(token, 
                        #                                       cand_copy_prob))

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]
                    new_hyp.node_id = grammar.get_node_type_id(frontier_nt)


                # get the new frontier nt after rule application
                new_frontier_nt = new_hyp.frontier_nt()

                # if new_frontier_nt is None, then we have a new completed hyp!
                if new_frontier_nt is None:
                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1
                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]
                    new_hyp_samples.append(new_hyp)

                # cand is word generation

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)

            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)
        print('Completed hyps:')
        print(completed_hyps)

        return completed_hyps

    def sample(self, parser, seed_seq, grammar, vocab):
        example = DataEntry(raw_id=None, 
                            query_tokens=tokenize_text(seed_seq, parser), 
                            parse_tree=None, 
                            text=seed_seq, 
                            actions=None, 
                            meta_data=None)
        example._data = [np.array([vocab.convertToIdx(example.query_tokens)], 
                                  dtype='int32')]

        cand_list = self.decode(example, 
                                grammar, 
                                vocab,
                                beam_size=self.cfg['beam_size'], 
                                max_time_step=self.cfg['decode_max_time_step'])

        cand = cand_list[0]
        text = decode_tree_to_string(cand.tree)
        parse = parser.get_parse(text)['sentences'][0]
        tokens = [x['word'] for x in parse['tokens']]

        deps = sorted(parse['basicDependencies'], 
                      key=lambda x: x['dependent'])
        parents = [x['governor'] for x in deps]

        return tokens, parents

    @staticmethod
    def get_parents(parser, text):
        parse = parser.get_parse(text)['sentences'][0]

    @property
    def params_name_to_id(self):
        name_to_id = dict()
        for i, p in enumerate(self.params):
            assert p.name is not None
            # print 'parameter [%s]' % p.name

            name_to_id[p.name] = i

        return name_to_id

    @property
    def params_dict(self):
        assert len(set(p.name for p in self.params)) == len(self.params), 'param name clashes!'
        return OrderedDict((p.name, p) for p in self.params)

    def pull_params(self):
        return OrderedDict([(p_name, p.get_value(borrow=False)) \
                            for (p_name, p) \
                            in list(self.params_dict.items())])

    def save(self, model_file, **kwargs):

        weights_dict = self.pull_params()
        for k, v in list(kwargs.items()):
            weights_dict[k] = v

        np.savez(model_file, **weights_dict)

    def load(self, model_file):
        weights_dict = np.load(model_file)

        # assert len(weights_dict.files) == len(self.params_dict)

        for p_name, p in list(self.params_dict.items()):
            if p_name not in weights_dict:
                raise RuntimeError('parameter [%s] not in saved weights file', p_name)
            else:
                assert np.array_equal(p.shape.eval(), weights_dict[p_name].shape), \
                    'shape mis-match for [%s]!, %s != %s' % (p_name, p.shape.eval(), weights_dict[p_name].shape)

                p.set_value(weights_dict[p_name])
