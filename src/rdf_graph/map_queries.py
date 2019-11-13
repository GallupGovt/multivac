
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import re
import tensorflow as tf

import pandas as pd
import sys

from datetime import datetime
#from multivac.settings import models_dir
from numpy import array
from OpenKE.openke import config
from OpenKE.openke.module import model
from rdf_parse import StanfordParser, stanford_parse

def get_best_score(x):
    if isinstance(x, tuple):
        return np.nan

    scores = [tup[-1] for tup in x if tup[0] != tup[2]]

    return min(scores)

def get_avg_score(x):
    if isinstance(x, tuple):
        return np.nan
        
    scores = [tup[-1] for tup in x if tup[0] != tup[2]]

    return np.mean(scores)

def loadGloveModel(gloveFile=None, verbose=False):
    if gloveFile is None:
        gloveFile = os.path.join(models_dir, "glove.42B.300d.txt")

    model = {}

    with open(gloveFile,'r') as f:
        if verbose:
            print("Reading GloVe embeddings...")
        lines = f.readlines()
        if verbose:
            print("Done.")
            print("Indexing embeddings...")

    indices = [x[:x.index(' ')] for x in lines]
    deciles = set([x/10 for x in range(1, 10)])

    for i, word in enumerate(indices):
        if verbose and round(i/len(indices), 2) in deciles:
            deciles.remove(round(i/len(indices), 2))
            print("\r{}% done".format(round(i/len(indices)*100)), end=' ')

        model[word] = np.fromstring(lines[i][lines[i].index(' '):], sep=' ')

    if verbose:
        print("\rDone.             ")

    return model

def avg_embed(x, glove):
    if isinstance(x, str):
        x = x.split()

    if not isinstance(x, list):
        return np.zeros(len(next(iter(glove.values()))))

    rows = len(x)
    cols = len(next(iter(glove.values())))
    result = np.zeros((rows, cols))

    for i, word in enumerate(x):
        if word.lower() in glove:
            result[i, :] = glove[word.lower()]

    return np.average(np.vstack(result), axis=0)

def predict(con, h, r, t, num_top_rel=10, threshold=.1):
    if sum([x == -1 for x in (h, r, t)]) > 1:
        result = [((h,r,t), np.nan)]
    elif h == -1:
        heads = con.predict_head_entity(t, r, num_top_rel)
        result = []

        for head in heads:
            result.append(((head, r, t), con.predict_triple(head, t, r, threshold)))

    elif r == -1:
        rels = con.predict_relation(h, t, num_top_rel)
        result = []

        for rel in rels:
            result.append(((h, rel, t), con.predict_triple(h, t, rel, threshold)))
    elif t == -1:
        tails = con.predict_tail_entity(h, r, num_top_rel)
        result = []

        for tail in tails:
            result.append(((h, r, tail), con.predict_triple(h, tail, r)))
    else:
        result = [((h,r,t), con.predict_triple(h, t, r, threshold))]

    return result

def cos_sim(u, v):

    if all([x==0 for x in u]) or all([x==0 for x in v]):
        return 0

    result = np.dot(u,v) / \
        (np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v))))
    
    return result

def get_answers(con, query, glove, entities, relations, 
                num_top_rel=10, threshold=.75):

    if 'subject' not in query:
        query['subject'] = ""
    if 'object' not in query:
        query['object'] = ""
    if 'relation' not in query:
        query['relation'] = ""

    top_subj = np.ones(num_top_rel)*-1
    top_rel  = np.ones(num_top_rel)*-1
    top_obj  = np.ones(num_top_rel)*-1

    if len(query['subject']) > 0:
        subj = avg_embed(query['subject'], glove)
        subj_scores = entities.Ent.apply(lambda x: cos_sim(avg_embed(x, glove), 
                                                           subj))

        if subj_scores.max() < threshold:
            subj_id = -1
        else:
            # Potentially use np.argmax() for multiple max results
            try:
                subj_id = entities.Id.iloc[subj_scores.idxmax()]
            except:
                print(query['subject'])
                print(subj)
                print(subj_scores)
                raise
    else:
        subj_id = -1

    if len(query['object']) > 0:
        obj = avg_embed(query['object'], glove)
        obj_scores = entities.Ent.apply(lambda x: cos_sim(avg_embed(x, glove), 
                                                          obj))

        if obj_scores.max() < threshold:
            obj_id = -1
        else:
            # Potentially use np.argmax() for multiple max results
            obj_id = entities.Id.iloc[obj_scores.idxmax()]
    else:
        obj_id = -1

    if len(query['relation']) > 0:
        rel = avg_embed(query['relation'], glove)
        rel_scores = relations.Rel.apply(lambda x: cos_sim(avg_embed(x, glove), 
                                                           rel))

        if rel_scores.max() < threshold:
            rel_id = -1
        else:
            # Potentially use np.argmax() for multiple max results
            rel_id = relations.Id.iloc[rel_scores.idxmax()]
    else:
        rel_id = -1

    result = predict(con, subj_id, rel_id, obj_id, num_top_rel)

    if any([x == -1 for x in result[0][0]]):
        return result[0]

    readable_result = []

    for tup in result:
        subj = entities.Ent[entities.Id==tup[0][0]].iloc[0]
        rel = relations.Rel[relations.Id==tup[0][1]].iloc[0]
        obj = entities.Ent[entities.Id==tup[0][2]].iloc[0]
        score = float(tup[1])
        readable_result.append((subj, rel, obj, score))

    return readable_result


def predicted_object(con, query, num_top_rel=10, max_digits_ent=1000,
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
        files = glob.glob(os.path.join(args_dict['dir'],'*.txt'))
        relation = max([file for file in files if re.search('relation', file)],
                       key=os.path.getctime)
        entity = max([file for file in files if re.search('entity', file)],
                       key=os.path.getctime)

    with open(relation, 'r') as rel_file:
        # skips first line in text
        _ = next(rel_file)

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
                                    line_tail.partition('\t'+str(tail_id)+' ')
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

    # sort by 4th element, the accuracy
    lis = sorted(top_rel, key=lambda x: x[3], reverse=False)
    out = lis[0:num_top_rel]

    # develop network data
    nets = [x for x in net_rel if x[2] > threshold]
    with open(os.path.join(args_dict['out'],'network_{}_{}.json'
              .format(args_dict['search'].replace(' ', '-'),
                      args_dict['timestamp'])), 'w') as f:
        json.dump(nets, f)


    print(out)
    with open(os.path.join(args_dict['out'],'prediction_{}_{}.json'
              .format(args_dict['search'].replace(' ', '-'),
                      args_dict['timestamp'])), 'w') as f:
        json.dump(out, f)

def get_con(args):
    con = config.Config()

    if args_dict['dir'].endswith(os.path.sep):
        con.set_in_path(args_dict['dir'])
    else:
        con.set_in_path(args_dict['dir']+os.path.sep)

    con.set_work_threads(8)
    con.set_dimension(100)
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)

    files = glob.glob(os.path.join(args_dict['out'],'*tf*'))

    if not files:
        raise Exception('No models to predict on; generate one first.')

    times = list(set([file.split('.')[2] for file in files]))
    ifile = max([datetime.strptime(x, '%d%b%Y-%H:%M:%S') for
                x in times]).strftime('%d%b%Y-%H:%M:%S')
    con.set_import_files(os.path.join(args_dict['out'],
                                      'model.vec.{}.tf'.format(ifile)))
    con.init()
    kem = set_model_choice(args_dict['model'])
    con.set_model(kem)


def run(args_dict):
    # setup
    timestamp = datetime.now().strftime('%d%b%Y-%H:%M:%S')
    verbose = args_dict['verbose']

    if args_dict['threshold'] is None:
        threshold = 0.1
    else:
        threshold = float(args_dict['threshold'])

    if args_dict['num_top_rel'] is None:
        num_top_rel = 10
    else:
        num_top_rel = args_dict['num_top_rel']

    # check if output directory exists
    if not os.path.isdir(args_dict['out']):
        os.mkdir(args_dict['out'])

    # instantiate connection to OpenKE
    con = config.Config()

    # set global parameters
    if args_dict['dir'].endswith(os.path.sep):
        con.set_in_path(args_dict['dir'])
    else:
        con.set_in_path(args_dict['dir']+os.path.sep)
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
        con.set_export_files(os.path.join(args_dict['out'],
                                          'model.vec.{}.tf'.format(timestamp)), 
                             0)

        # save out model parameters
        con.set_out_files(os.path.join(args_dict['out'],
                                       'embedding.vec.{}.json'.format(timestamp)))
    else:
        con.set_test_link_prediction(True)
        con.set_test_triple_classification(True)

        files = glob.glob(os.path.join(args_dict['out'],'*tf*'))
        if not files:
            raise Exception('No models to predict on; generate one first.')
        else:
            if verbose:
                print("Loading files...")

            times = list(set([file.split('.')[2] for file in files]))
            ifile = max([datetime.strptime(x, '%d%b%Y-%H:%M:%S') for
                        x in times]).strftime('%d%b%Y-%H:%M:%S')
            con.set_import_files(os.path.join(args_dict['out'],
                                              'model.vec.{}.tf'.format(ifile)))
            args_dict.update({'timestamp': ifile})

    # initialize settings
    if verbose:
        print("Initializing OpenKE system...")
                
    con.init()

    # set knowledge embedding model
    if verbose:
        print("Setting model...")
    kem = set_model_choice(args_dict['model'])
    con.set_model(kem)

    # determine action
    if 'fit' in args_dict['run']:
        # model training
        con.run()
    else:
        if verbose:
            print("Beginning predictions...")

        # predict objects
        if not args_dict['search']:
            raise Exception('You need to provide a search term.')
        else:
            parser = StanfordParser()

            glove = loadGloveModel(args_dict['glove'], verbose)

            # identify files for use
            files = glob.glob(os.path.join(con.in_path,'*.txt'))
            rel_file = max([file for file in files if re.search('relation', file)],
                           key=os.path.getctime)
            ent_file = max([file for file in files if re.search('entity', file)],
                           key=os.path.getctime)

            entities = pd.read_csv(ent_file, sep='\t', 
                                   names=["Ent","Id"], skiprows=1)
            relations = pd.read_csv(rel_file, sep='\t', 
                                    names=["Rel","Id"], skiprows=1)

            if os.path.exists(args_dict['search']):
                queries = pd.read_csv(args_dict['search'])

                parse = lambda z: stanford_parse(parser, z).get_rdfs(use_tokens=False, 
                                                                     how='list')[0]
                triples = queries.Query.apply(parse)

                results = triples.apply(lambda x: get_answers(con, x, glove, 
                                                              entities, 
                                                              relations,
                                                              num_top_rel, 
                                                              threshold))
                queries['results'] = results
                queries.to_csv(os.path.join(args_dict['dir'],
                               "query_results.csv"), index=False)
            else:
                predicted_object(con, 
                                 args_dict['search'], 
                                 num_top_rel, 
                                 threshold=threshold)



            #
            # Keeping in comments for now - not sure what visuals we may want 
            # with our output.
            # 
            
            # # create network graph
            # with open(os.path.join(args_dict['out'],'network_{}_{}.json'
            #           .format(args_dict['search'].replace(' ', '-'),
            #                   args_dict['timestamp'])), 'r') as f:
            #     net_data = json.load(f)

            # el = {}
            # [el[str(item[0])].append(item[1]) if str(item[0]) in
            #  el.keys() else el.update({str(item[0]): [item[1]]}) for
            #  item in net_data]

            # net = nx.Graph(el)
            # nx.draw(net, pos=nx.spring_layout(net), alpha=.75, node_size=100)
            # plt.savefig(os.path.join(args_dict['out'],'network_{}_{}.png'
            #             .format(args_dict['search'].replace(' ', '-'),
            #                     args_dict['timestamp'])),
            #             pad_inches=.05, orientation='landscape',
            #             papertype='letter')


def set_model_choice(model):
    if 'analogy' in model:
        kem = model.Analogy
    elif 'complex' in model:
        kem = model.ComplEx
    elif 'distmult' in model:
        kem = model.DistMult
    elif 'hole' in model:
        kem = model.HolE
    elif 'rescal' in model:
        kem = model.RESCAL
    elif 'transd' in model:
        kem = model.TransD
    elif 'transe' in model:
        kem = model.TransE
    elif 'transh' in model:
        kem = model.TransH
    else:
        kem = model.TransR

    return kem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Map queries to knowledge graph.')
    parser.add_argument('-d', '--dir', required=True, help='Path to index data '
                        'directory.')
    parser.add_argument('-f', '--filetime', required=False, help='Timestamp '
                        'appended to desired file for use; if none, the '
                        'system will use the most current file.')
    parser.add_argument('-m', '--model', required=True, choices=['analogy',
                        'complex', 'distmult', 'hole', 'rescal', 'transd',
                        'transe', 'transh', 'transr'], help='Model selection '
                        'for knowledge embedding.')
    parser.add_argument('-o', '--out', required=True, help='Path to models '
                        'output directory.')
    parser.add_argument('-g', '--glove', required=False, help='Path to GloVe '
                        'embeddings model file.')
    parser.add_argument('-r', '--run', required=True, choices=['fit', 'model'],
                        help='Identify choice of action, fitting a model or '
                        'predicting from it.')
    parser.add_argument('-t', '--threshold', required=False, 
                        help='Threshold accuracy for matches; default is 0.1.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output on progress.')
    parser.add_argument('-n', '--num_top_rel', required=False, 
                        help='Number of top matches to return; default is 10.')
    parser.add_argument('-s', '--search', required=False, 
                        help='Searches to execute. Either a path to a CSV '
                        'containing triples or a triple in the format '
                        '"subject terms ::: relation terms ::: object terms"')
    args_dict = vars(parser.parse_args())

    run(args_dict)
