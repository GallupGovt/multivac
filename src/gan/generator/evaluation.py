# -*- coding: UTF-8 -*-


import logging
import os
import traceback

from multivac.src.rdf_graph.rdf_parse import tokenize_text, StanfordParser
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from .model import *
from multivac.src.gan.generator.nn.utils.generic_utils import init_logging

def tokenize_for_bleu_eval(text):
    text = re.sub(r'([^A-Za-z0-9_])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '`')
    text = text.replace('\'', '`')
    tokens = [t for t in text.split(' ') if t]

    return tokens


def evaluate(model, dataset, verbose=True):
    if verbose:
        logging.info('evaluating [%s] dataset, [%d] examples' % (dataset.name, dataset.count))

    exact_match_ratio = 0.0

    for example in dataset.examples:
        logging.info('evaluating example [%d]' % example.eid)
        hyps, hyp_scores = model.decode(example, max_time_step=config.decode_max_time_step)
        gold_rules = example.rules

        if len(hyps) == 0:
            logging.warning('no decoding result for example [%d]!' % example.eid)
            continue

        best_hyp = hyps[0]
        predict_rules = [dataset.grammar.id_to_rule[rid] for rid in best_hyp]

        assert len(predict_rules) > 0 and len(gold_rules) > 0

        exact_match = sorted(gold_rules, key=lambda x: x.__repr__()) == sorted(predict_rules, key=lambda x: x.__repr__())
        if exact_match:
            exact_match_ratio += 1

    exact_match_ratio /= dataset.count

    logging.info('exact_match_ratio = %f' % exact_match_ratio)

    return exact_match_ratio


def evaluate_decode_results(dataset, decode_results, cfg):
    verbose = cfg['verbose']
    assert dataset.count == len(decode_results)

    parser = StanfordParser()

    f = f_decode = None

    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()

        for ex in dataset.examples:
            eid_to_annot[ex.raw_id] = ' '.join(ex.query_tokens).strip()

        f_bleu_eval_ref = open(dataset.name + '.ref', 'w')
        f_bleu_eval_hyp = open(dataset.name + '.hyp', 'w')
        f_generated_code = open(dataset.name + '.geneated_code', 'w')

        print('evaluating [{}] set, [{}] examples'.format(dataset.name, 
                                                         dataset.count))

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if all(len(cand) == 0 for cand in decode_results):
        print('Empty decoding results for the current dataset!')
        print(decode_results)
        return -1, -1

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_text = example.text
        refer_source = ref_text
        refer_tokens = tokenize_text(ref_text, parser)
        cur_example_correct = False

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        text = ''
        i = 0

        while i < len(decode_cands) and len(text) == 0:
            cid, cand, text = decode_cands[i]
            i += 1

        if text == '':
            predict_tokens = []
            continue
        else:
            predict_tokens = tokenize_text(text, parser)

        if refer_tokens == predict_tokens:
            cum_acc += 1
            cur_example_correct = True

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(text + '\n')
                f.write('-' * 60 + '\n')

        if cfg['data_type'] == 'eng':
            ref_text_for_bleu = example.text
            pred_text_for_bleu = text

        refer_tokens_for_bleu = tokenize_text(ref_text_for_bleu, parser)
        pred_tokens_for_bleu = predict_tokens

        shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

        all_references.append([refer_tokens_for_bleu])
        all_predictions.append(pred_tokens_for_bleu)

        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], 
                                    pred_tokens_for_bleu, 
                                    weights=ngram_weights, 
                                    smoothing_function=sm.method3)
        cum_bleu += bleu_score

        if verbose:
            print(('raw_id: %d, bleu_score: %f' % (example.raw_id, bleu_score)))

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            if cfg['data_type'] == 'eng':
                f_decode.write(eid_to_annot[example.raw_id] + '\n')

            f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
            f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

            f_decode.write('canonicalized reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('canonicalized prediction: \n')
            f_decode.write(text + '\n')
            f_decode.write('reference text for bleu calculation: \n')
            f_decode.write(ref_text_for_bleu + '\n')
            f_decode.write('predicted text for bleu calculation: \n')
            f_decode.write(pred_text_for_bleu + '\n')
            f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
            f_decode.write('-' * 60 + '\n')

            # for Hiro's evaluation
            f_generated_code.write(pred_text_for_bleu.replace('\n', 
                                                              '#NEWLINE#') + '\n')

        # compute oracle
        best_score = 0.
        cur_oracle_acc = 0.

        for decode_cand in decode_cands[:cfg['beam_size']]:
            cid, cand, text = decode_cand

            try:
                predict_tokens = tokenize_text(text, parser)

                if predict_tokens == refer_tokens:
                    cur_oracle_acc = 1

                if config.data_type == 'eng':
                    pred_text_for_bleu = text
                # if config.data_type == 'django':
                #     pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
                #     # convert canonicalized code to raw code
                #     for literal, place_holder in list(example.meta_data['str_map'].items()):
                #         pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # elif config.data_type == 'hs':
                #     pred_code_for_bleu = code

                # we apply Ling Wang's trick when evaluating BLEU scores
                pred_tokens_for_bleu = tokenize_text(pred_text_for_bleu, parser)

                ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
                bleu_score = sentence_bleu([refer_tokens_for_bleu], 
                                           pred_tokens_for_bleu,
                                           weights=ngram_weights,
                                           smoothing_function=sm.method3)

                if bleu_score > best_score:
                    best_score = bleu_score

            except:
                continue

        cum_oracle_bleu += best_score
        cum_oracle_acc += cur_oracle_acc

    cum_bleu /= dataset.count
    cum_acc /= dataset.count
    cum_oracle_bleu /= dataset.count
    cum_oracle_acc /= dataset.count

    if verbose:
        print('corpus level bleu: {}'.format(corpus_bleu(all_references, 
                                                         all_predictions, 
                                                         smoothing_function=sm.method3)))
        print('sentence level bleu: {}'.format(cum_bleu))
        print('accuracy: {}'.format(cum_acc))
        print('oracle bleu: {}'.format(cum_oracle_bleu))
        print('oracle accuracy: {}'.format(cum_oracle_acc))

    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

        f_bleu_eval_ref.close()
        f_bleu_eval_hyp.close()
        f_generated_code.close()

    return cum_bleu, cum_acc

