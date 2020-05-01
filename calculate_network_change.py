#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is meant to identify relevant nodes based on differences of
centrality measure of real and estimated networks.
"""
import argparse
import json
import networkx as nx
import numpy as np
import os

from datetime import datetime

from multivac.get_kg_query_params import build_network, read_txt

def build_comparison_metrics(n1, n2, mtype):
    if 'degree' in mtype:
        n1x = nx.degree_centrality(n1)
        n2x = nx.degree_centrality(n2)
    else:
        tol = 1.0e-6

        while True:
            try:
                n1x = nx.eigenvector_centrality(n1, tol=tol)
                n2x = nx.eigenvector_centrality(n2, tol=tol)
                break
            except:
                tol *= 10
                print("Increasing tolerance to {}".format(tol))
                continue

    net = {**n1x, **n2x}
    for k, v in net.items():
        if k in n1x and k in n2x:
            net[k] = [n1x[k], v]
        elif k in n1x and k not in n2x:
            net[k] = [v, np.nan]
        else:
            net[k] = [np.nan, v]

    return net


def generate_node_changes(net):
    res = {}
    for k, v in net.items():
        pct_change = (net[k][1] - net[k][0]) / (net[k][0] + 1)

        if not np.isnan(pct_change):
            res.update({k: pct_change})

    return res


def generate_result_lists(net, num, ctype=['top', 'bottom']):
    res = {}
    if 'top' in ctype:
        keys = list(net.keys())[-num:]
    else:
        keys = list(net.keys())[:num]
    for key in keys:
        res.update({key: net[key]})

    return res

def get_items(fpath):
    items = {}

    for item, idx in read_txt(fpath):
        items[int(idx)] = item

    return items

def triple_to_labels(triple, ents, rels):
    head, tail, rel = trip
    return " ".join([ents[head], rels[rel], ents[tail]])


def get_top_triples(ofile, nfile, kg_dir, measure='eigenvector', num_results=100, out=None):
    ents = get_items(os.path.join(kg_dir, 'entity2id.txt'))
    rels = get_items(os.path.join(kg_dir, 'relation2id.txt'))
    triples = read_txt(os.path.join(kg_dir, 'train2id.txt'))
    triples = np.array(triples).astype(int)

    # read in new file for comparison
    new = read_txt(nfile)

    # create networks
    neto = build_network(triples)
    netn = build_network(triples + new)
    net = build_comparison_metrics(neto, netn, measure)

    # calculate node changes
    result = generate_node_changes(net)
    result = {k: v for k, v in sorted(result.items(),
                                      key=lambda item: item[1])}

    # generate results of interest
    gains = generate_result_lists(result, len(result), 'top')

    trip_scores = np.zeros(triples.shape[0])

    for i, trip in enumerate(triples):
        headgain = tailgain = 0
        head, tail, _ = trip
        trip_scores[i] = gains.get(str(head), 0) + gains.get(str(tail), 0)

    idxs = trip_scores.argsort()[::-1]
    top = triples[idxs,][:num_results,:]

    results = {}

    for i, t in enumerate(top):
        triple_id = idxs[i]
        h, t, r = t
        score = trip_scores[triple_id]

        try:
            label = " ".join([ents[h], rels[r], ents[t]])
        except:
            label = "missing RDF-triple"
    
        results[triple_id] = {'label': label, 'score': score}

    if out:
        with open('{}/key_triples.json'.format(out), 'w') as f:
            json.dump(results, f)

        return True
    else:
        return results


def run(args_dict):
    # read in files for comparison
    orig = read_txt(args_dict['files'][0])
    new = read_txt(args_dict['files'][1])

    # create networks
    neto = build_network(orig)
    netn = build_network(orig + new)
    net = build_comparison_metrics(neto, netn, args_dict['measure'])

    # calculate node changes
    result = generate_node_changes(net)
    result = {k: v for k, v in sorted(result.items(),
                                      key=lambda item: item[1])}

    # generate results of interest
    top_gain = generate_result_lists(result, args_dict['num_results'], 'top')
    top_loss = generate_result_lists(result, args_dict['num_results'], 'bottom')

    # dump results to disk
    time = datetime.now().strftime('%d%b%Y-%H:%M:%S')
    with open('{}/top_gains_{}.json'.format(args_dict['output'], time), 'w') as f:
        json.dump(top_gain, f)
    with open('{}/top_losses_{}.json'.format(args_dict['output'], time), 'w') as f:
        json.dump(top_loss, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate differences '
                                     'between networks.')
    parser.add_argument('-f', '--files', nargs=2, required=True, help='Two '
                        'files -- the real network then estimated network -- '
                        'over which to calculate differences.')
    parser.add_argument('-m', '--measure', required=False,
                        default='eigenvector', choices=['degree',
                        'eigenvector'], help='Select which network centrality '
                        'measure is required.')
    parser.add_argument('-n', '--num_results', required=False, default=10,
                        type=int, help='Number of results to return from '
                        'centrality calculation.')
    parser.add_argument('-o', '--output', required=True, help='Path to '
                        'directory to write results to disk.')
    args_dict = vars(parser.parse_args())

    run(args_dict)

