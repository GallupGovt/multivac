#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is meant to use network centrality measures to identify and select
nodes and edges (entities and relations) that would make good prediction
starting points for the MULTIVAC system. It does this by using eigenvector
centrality but can be extended to include additional network centrality
measures.
"""
import argparse
import sys

import networkx as nx


def analyze_network(net, args_dict):
    if 'degree' in args_dict['measure']:
        ans = nx.degree_centrality(net)
    elif 'eigenvector' in args_dict['measure']:
        ans = nx.eigenvector_centrality(net)
    else:
        sys.exit('Whoops; you must provide a valid network centrality measure.')
    ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

    return ans[:args_dict['num_results']]


def build_network(data):
    tmp = [tuple(x[:2]) for x in data]
    g = nx.Graph()
    g.add_edges_from(tmp)

    return g


def read_txt(file):
    with open(file) as f:
        tmp = f.readlines()[1:]

    return [x.rstrip(' \n').split('\t') for x in tmp]


def run(args_dict):
    # read in data
    entities = read_txt(args_dict['files'][0])
    network = read_txt(args_dict['files'][1])

    # construct/analyze network
    net = build_network(network)
    results = analyze_network(net, args_dict)

    # return results
    named_entities = ['{}\n'.format(entity[0]) for entity in entities if
                      entity[1] in [res[0] for res in results]]

    with open('search_terms_{}.txt'.format(args_dict['measure']), 'w') as f:
        f.writelines(named_entities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run network centrality '
                                     'measures on data.')
    parser.add_argument('-f', '--files', nargs=2, required=True, help='Two '
                        'files -- entities then train -- that are parsed to '
                        'create a network.')
    parser.add_argument('-m', '--measure', required=False,
                        default='eigenvector', choices=['degree', 'eigenvector'],
                        help='Select which network centrality '
                        'measure is required.')
    parser.add_argument('-n', '--num_results', required=False, default=10,
                        type=int, help='Number of results to return from '
                        'centrality calculation.')
    args_dict = vars(parser.parse_args())

    run(args_dict)
