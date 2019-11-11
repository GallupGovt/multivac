
from collections import OrderedDict
import numpy as np

import torch
import torch.nn.functional as F

from multivac.src.gan.gen_pyt.asdl.transition_system import ApplyRuleAction, ReduceAction, Action, GenTokenAction
from multivac.src.gan.gen_pyt.components.action_info import ActionInfo
from multivac.src.gan.gen_pyt.components.decode_hypothesis import DecodeHypothesis
from multivac.src.gan.gen_pyt.model import nn_utils

# hypotheses - batch of Hypothesis objects

def get_hyp_states(mod, src_encodings, dec_init_vec, hyps, max_action_len):
    # hyp    :: DecodeHypothesis()

    T = torch.cuda if mod.args['cuda'] else torch
    max_action_len = max([len(hyp.actions) for hyp in hyps])

    # (1, src_sent_len, hidden_size)
    src_encodings_att_linear = mod.att_src_linear(src_encodings)

    if mod.args['lstm'] == 'parent_feed':
        h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                mod.new_tensor(mod.args['hidden_size']).zero_(), \
                mod.new_tensor(mod.args['hidden_size']).zero_()
    else:
        h_tm1 = dec_init_vec

    hyp_states = {}
    x = mod.new_tensor(len(hyps), mod.decoder_lstm.input_size).zero_()
    zero_action_embed = mod.new_tensor(mod.args['action_embed_size']).zero_()

    for i in range(max_action_len):
        if i == 0:
            if mod.args['no_parent_field_type_embed'] is False:
                offset  = mod.args['action_embed_size']  # prev_action
                offset += mod.args['att_vec_size']      * (not mod.args['no_input_feed'])
                offset += mod.args['action_embed_size'] * (not mod.args['no_parent_production_embed'])
                offset += mod.args['field_embed_size']  * (not mod.args['no_parent_field_embed'])

                x[0, offset: offset + mod.args['type_embed_size']] = \
                    mod.type_embed.weight[mod.grammar.type2id[mod.grammar.root_type]]
        else:
            actions_tm1 = [hyp.actions[i] if i < len(hyp.actions) else None for hyp in hyps]

            a_tm1_embeds = []

            for a_tm1 in actions_tm1:
                if a_tm1:
                    if isinstance(a_tm1, ApplyRuleAction):
                        a_tm1_embed = mod.production_embed.weight[mod.grammar.prod2id[a_tm1.production]]
                    elif isinstance(a_tm1, ReduceAction):
                        a_tm1_embed = mod.production_embed.weight[len(mod.grammar)]
                    else:
                        a_tm1_embed = mod.primitive_embed.weight[mod.prim_vocab[a_tm1.token]]

                    a_tm1_embeds.append(a_tm1_embed)
                else:
                    a_tm1_embeds.append(zero_action_embed)

            a_tm1_embeds = torch.stack(a_tm1_embeds)

            inputs = [a_tm1_embeds]


            if mod.args['no_input_feed'] is False:
                inputs.append(hyp_states[i-1]['att_t'])

            if mod.args['no_parent_production_embed'] is False:
                # frontier production
                frontier_prods = [hyp.frontier_node.production if hyp.frontier_node else None for hyp in hyps]
                frontier_prod_embeds = mod.production_embed(mod.new_long_tensor(
                    [mod.grammar.prod2id[prod] if prod else 0 for prod in frontier_prods]))
                inputs.append(frontier_prod_embeds)

            if mod.args['no_parent_field_embed'] is False:
                # frontier field
                frontier_fields = [hyp.frontier_field.field if hyp.frontier_field else None for hyp in hyps]
                frontier_field_embeds = mod.field_embed(mod.new_long_tensor([
                    mod.grammar.field2id[field] if field else 0 for field in frontier_fields]))

                inputs.append(frontier_field_embeds)

            if mod.args['no_parent_field_type_embed'] is False:
                # frontier field type
                frontier_field_types = [hyp.frontier_field.type if hyp.frontier_field else None for hyp in hyps]
                frontier_field_type_embeds = mod.type_embed(mod.new_long_tensor([
                    mod.grammar.type2id[type] if type else 0 for type in frontier_field_types]))
                inputs.append(frontier_field_type_embeds)

            # parent states
            if mod.args['no_parent_state'] is False:
                p_ts = [hyp.frontier_node.created_time if hyp.frontier_node else 0 for hyp in hyps]
                parent_states = torch.stack([hyp_states[p_t]["h_t"][hyp_id] for hyp_id, p_t in enumerate(p_ts)])
                parent_cells = torch.stack([hyp_states[p_t]["c_t"][hyp_id] for hyp_id, p_t in enumerate(p_ts)])

                if mod.args['lstm'] == 'parent_feed':
                    h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                else:
                    inputs.append(parent_states)

            x = torch.cat(inputs, dim=-1)

        (h_t, c_t), att_t = mod.step(x, h_tm1, src_encodings,
                                     src_encodings_att_linear,
                                     src_token_mask=None)
        hyp_states[i] = {"att_t": att_t,
                         "h_t": h_t,
                         "c_t": c_t}

    return hyp_states


def rollout_samples(mod, src_sents, samples):
    """Perform beam search to infer the target AST given a source utterance

    Args:
        src_sents: lists of source utterance tokens
        samples: reference Hypotheses
        beam_size: beam size

    Returns:
        A list of lists of `DecodeHypothesis`, each representing an AST
        shape => (max_action_length, len(samples))
        i.e., if there are two samples and max_action_length == 4:
        computed_hyps = [[hyp, hyp, hyp, hyp], [hyp, hyp, hyp, hyp]]
    """
    T = torch.cuda if mod.args['cuda'] else torch
    hypotheses = [DecodeHypothesis()] * len(samples)
    max_action_len = max([len(sample.actions) for sample in samples])
    computed_hyps = [[]] * max_action_len

    # Variable(batch_size, src_sent_len, hidden_size * 2)
    src_sent_vars = nn_utils.to_input_variable(src_sents, 
                                               mod.vocab, 
                                               cuda=mod.args['cuda'], 
                                               training=False)
    src_encodings, (last_state, last_cell) = mod.encode(src_sent_vars, 
                                                         [len(s) for s in src_sents])
    dec_init_vec = mod.init_decoder_state(last_state, last_cell)
    
    ref_hyp_states = get_hyp_states(mod, src_encodings, dec_init_vec, 
                                    samples, max_action_len)

    # for t in range(max_action_len):
    #     aggregated_primitive_tokens = OrderedDict()

    #     for token_pos, token in enumerate(src_sent):
    #         aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

    for t in range(1, max_action_len):
        computed_hyps[t] = []
        att_t = ref_hyp_states[t]['att_t']

        # Variable(batch_size, grammar_size)
        apply_rule_log_prob = F.log_softmax(mod.production_readout(att_t), dim=-1)

        # Variable(batch_size, mod.vocab_size)
        gen_from_vocab_prob = F.softmax(mod.tgt_token_readout(att_t), dim=-1)

        if mod.args['no_copy']:
            primitive_prob = gen_from_vocab_prob
        else:
            # Variable(batch_size, src_sent_len)
            primitive_copy_prob = mod.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

            # Variable(batch_size, 2)
            primitive_predictor_prob = F.softmax(mod.primitive_predictor(att_t), dim=-1)

            # Variable(batch_size, mod.vocab_size)
            primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

            # if src_unk_pos_list:
            #     primitive_prob[:, mod.vocab.unk] = 1.e-10
        for hyp_id, h in tqdm(enumerate(hypotheses), desc="Computing hypotheses from step {}...".format(t)):
            hyp = h.clone_and_apply_action_info(samples[hyp_id].action_infos[0])
            aggregated_primitive_tokens = OrderedDict()

            for token_pos, token in enumerate(src_sents[hyp_id]):
                aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

            for j in range(1, t):
                if j < len(samples[hyp_id].actions):
                    hyp.apply_action_info(samples[hyp_id].action_infos[j])

            for step in range(t, max_action_len):
                if hyp.completed:
                    continue

                # generate new continuations
                action_types = mod.transition_system.get_valid_continuation_types(hyp)
                best_action = None
                best_score = None

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = mod.transition_system.get_valid_continuating_productions(hyp)

                        for production in productions:
                            prod_id = mod.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                            new_hyp_score = hyp.score + prod_score

                            if best_score is None or new_hyp_score > best_score:
                                best_score = new_hyp_score
                                best_action = ApplyRuleAction(production)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(mod.grammar)].data.item()
                        new_hyp_score = hyp.score + action_score

                        if best_score is None or new_hyp_score > best_score:
                            best_score = new_hyp_score
                            best_action = ReduceAction()
                    else:
                        # GenToken action
                        hyp_copy_info = dict()  # of (token_pos, copy_prob)
                        hyp_unk_copy_info = []

                        if mod.args['no_copy'] is False:
                            for token, token_pos_list in aggregated_primitive_tokens.items():
                                sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, T.LongTensor(token_pos_list)).sum()
                                gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                                if token in mod.prim_vocab:
                                    token_id = mod.prim_vocab[token]
                                    primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob

                                    hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data.item())
                                else:
                                    hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                              'copy_prob': gated_copy_prob.data.item()})

                        if mod.args['no_copy'] is False and len(hyp_unk_copy_info) > 0:
                            unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                            token = hyp_unk_copy_info[unk_i]['token']
                            primitive_prob[hyp_id, mod.prim_vocab.unk] = hyp_unk_copy_info[unk_i]['copy_prob']

                            hyp_copy_info[token] = (hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])

                        log_prob = torch.log(primitive_prob[hyp_id])
                        new_hyp_score = hyp.score + torch.max(log_prob)

                        if best_score is None or new_hyp_score > best_score:
                            best_score = new_hyp_score
                            token = mod.prim_vocab.idxToLabel[torch.argmax(log_prob).item()]
                            best_action = GenTokenAction(token)

                if best_action is None:
                    continue

                action_info = ActionInfo()

                if isinstance(best_action, GenTokenAction):
                    if best_action.token in aggregated_primitive_tokens:
                        action_info.copy_from_src = True
                        action_info.src_token_position = aggregated_primitive_tokens[token]

                action_info.action = best_action
                action_info.t = t

                if t > 0:
                    action_info.parent_t = hyp.frontier_node.created_time
                    action_info.frontier_prod = hyp.frontier_node.production
                    action_info.frontier_field = hyp.frontier_field.field

                hyp.apply_action_info(action_info)
                hyp.score = new_hyp_score

            computed_hyps[t].append(hyp)

    return computed_hyps[1:]
