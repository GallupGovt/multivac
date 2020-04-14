#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import numpy as np
import os
import re
import tensorflow as tf

from datetime import datetime

from OpenKE import config, models


def predicted_object(con, query, num_top_rel=25, max_digits_ent=1000,
                     threshold=.1):
    """
        This function takes a query to the knowledge graph and outputs
        predicted relationships that have the highest accuracy. First, this
        function finds subjects that relate to the query. Second, it iterates
        through all the relationships and finds the highest probable object
        based on the relationships. Finally, it outputs the predicted
        relationships with the highest accuracy. The user can also specify the
        number of top relationships (num_top_rel) based on the relationships
        with highest accuracy. In addition, max_digits_ent (default=1000) will
        specify the maximum length of a head or tail in a triple. Finally, the
        threshold argument (default=0.1) determines the similarity cutoff to
        identify relationships in the network graph.
    """
    top_rel = []
    net_rel = []

    # identify files for use
    if args_dict['filetime']:
        relation = 'relation2id-{}.txt'.format(args_dict['filetime'])
        entity = 'entity2id-{}.txt'.format(args_dict['filetime'])
    else:
        files = glob.glob('{}/*.txt'.format(args_dict['dir']))
        relation = max([file for file in files if re.search('relation', file)],
                       key=os.path.getctime)
        entity = max([file for file in files if re.search('entity', file)],
                       key=os.path.getctime)

    with open(relation, 'r') as rel_file:
        # skips first line in text
        next(rel_file)
        for line_rel in rel_file:
            rel_id = int(line_rel.split()[-1])
            rel = line_rel.split()[:-1]
            string1 = ' '.join(rel)

            with open(entity, 'r') as searchfile:
                for line_ent in searchfile:
                    left, sep, right = line_ent.partition(query.lower())

                    if sep:
                        # find all head ids that correspond to the query
                        find_head = right[:max_digits_ent]
                        head_id = int(find_head[-max_digits_ent:].split()[-1])
                        head = line_ent.split()[:-1]
                        string0 = ' '.join(head)

                        # Find the predicted tails 
                        tail0 = con.predict_tail_entity(head_id, rel_id, 2)
                        tail_init = int(tail0[0])

                        if tail_init == head_id:
                            tail_id = int(tail0[1])
                        else:
                            tail_id = tail_init

                        with open(entity, 'r') as tailfile:
                            for line_tail in tailfile:
                                left_tail, sep_tail, right_tail = (
                                    #line_tail.partition('\t'+str(tail_id)+' ')
                                    line_tail.partition('\t'+str(tail_id))
                                )
                                if sep_tail:
                                    tail = line_tail.split()[:-1]
                                    string2 = ' '.join(tail)
                                    print('({}, {}, {})'.format(string0,
                                                                string1,
                                                                string2))
                                    acc = con.predict_triple(head_id, tail_id,
                                                             rel_id)
                                    top_rel.append((string0, string1, string2,
                                                    round(float(acc),3)))
                                    net_rel.append((head_id, tail_id,
                                                    float(acc)))
            continue

    # sort by second 4th element the accuracy
    lis = sorted(top_rel, key=lambda x: x[3], reverse=False)
    out = lis[0:num_top_rel]

    # develop network data
    nets = [x for x in net_rel if x[2] > threshold]
    with open('{}network_{}_{}.json'
              .format(args_dict['out'], args_dict['search'].replace(' ', '-'),
                      args_dict['timestamp']), 'w') as f:
        json.dump(nets, f)


    print(out)
    with open('{}prediction_{}_{}.json'
              .format(args_dict['out'], args_dict['search'].replace(' ', '-'),
                      args_dict['timestamp']), 'w') as f:
        json.dump(out, f)


def run(args_dict):
    # setup
    timestamp = datetime.now().strftime('%d%b%Y-%H:%M:%S')

    # check if output directory exists
    if not os.path.isdir(args_dict['out']):
        os.mkdir(args_dict['out'])

    # instantiate connection to OpenKE
    con = config.Config()

    # set global parameters
    con.set_in_path(args_dict['dir'])
    con.set_work_threads(8)
    con.set_dimension(100)

    # fit run-determined parameters
    if 'fit' in args_dict['run']:
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        con.set_export_files('{}model.vec.{}.tf'.format(args_dict['out'],
                                                         timestamp), 0)

        # save out model parameters
        con.set_out_files('{}embedding.vec.{}.json'.format(args_dict['out'],
                                                            timestamp))
    else:
        con.set_test_link_prediction(True)
        con.set_test_triple_classification(True)

        files = glob.glob('{}*tf*'.format(args_dict['out']))
        if not files:
            raise Exception('No models to predict on; generate one first.')
        else:
            times = list(set([file.split('.')[2] for file in files]))
            ifile = max([datetime.strptime(x, '%d%b%Y-%H:%M:%S') for
                        x in times]).strftime('%d%b%Y-%H:%M:%S')
            con.set_import_files('{}model.vec.{}.tf'.format(args_dict['out'],
                                                            ifile))
            args_dict.update({'timestamp': ifile})

    # initialize settings
    con.init()

    # set knowledge embedding model
    kem = set_model_choice(args_dict['model'])
    con.set_model(kem)

    # determine action
    if 'fit' in args_dict['run']:
        # model training
        con.run()
    else:
        # predict objects
        if not args_dict['search']:
            raise Exception('You need to provide a search term.')
        else:
            predicted_object(con, args_dict['search'])

            # create network graph
            with open('{}network_{}_{}.json'
                      .format(args_dict['out'],
                              args_dict['search'].replace(' ', '-'),
                              args_dict['timestamp']), 'r') as f:
                net_data = json.load(f)

            el = {}
            [el[str(item[0])].append(item[1]) if str(item[0]) in
             el.keys() else el.update({str(item[0]): [item[1]]}) for
             item in net_data]


def set_model_choice(model):
    if 'analogy' in model:
        kem = models.Analogy
    elif 'complex' in model:
        kem = models.ComplEx
    elif 'distmult' in model:
        kem = models.DistMult
    elif 'hole' in model:
        kem = models.HolE
    elif 'rescal' in model:
        kem = models.RESCAL
    elif 'transd' in model:
        kem = models.TransD
    elif 'transe' in model:
        kem = models.TransE
    elif 'transh' in model:
        kem = models.TransH
    else:
        kem = models.TransR

    return kem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASKE Link prediction.')
    parser.add_argument('-d', '--dir', required=True, help='Path to data '
                        'directory.')
    parser.add_argument('-f', '--filetime', required=False, help='Timestamp '
                        'appended to desired file for use; if none, the '
                        'system will use the most current file.')
    parser.add_argument('-m', '--model', required=True, choices=['analogy',
                        'complex', 'distmult', 'hole', 'rescal', 'transd',
                        'transe', 'transh', 'transr'], help='Model selection '
                        'for knowledge embedding.')
    parser.add_argument('-o', '--out', required=True, help='Path to output '
                        'directory.')
    parser.add_argument('-r', '--run', required=True, choices=['fit', 'model'],
                        help='Identify choice of action, fitting a model or '
                        'predicting from it.')
    parser.add_argument('-s', '--search', required=False, help='Search phrase '
                        'for execution.')
    args_dict = vars(parser.parse_args())

    run(args_dict)
