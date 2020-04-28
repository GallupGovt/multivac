#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calls out to KromEM and KronGen to emulate and generate a hypothetical 
"complete" knowledge graph using Kronecker graph models. 
"""
import argparse
from multiprocessing import Pool
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import ndcg_score
import subprocess
import time

from get_kg_query_params import build_network, read_txt
from calculate_network_change import generate_node_changes, build_comparison_metrics


def call_kronem(call_str, verbose=False):
    fname = call_str[call_str.index('-i:')+3:]

    if fname.startswith('"'):
        fname = fname[1:fname[1:].index('"')+1]
    elif " -" in fname:
        fname = fname[:fname.index(" -")]
    else:
        fname = fname.strip()

    out_dir, base_name = os.path.split(fname)
    base, _ = os.path.splitext(base_name)

    if not out_dir:
        out_dir = os.getcwd()
        fname = os.path.join(out_dir, fname)

    start_time = time.time()
    outstr = subprocess.run(call_str, shell=True, capture_output=True).stdout
    ellapsed = round(time.time() - start_time, 2)

    if verbose:
        print("Call: \n" + call_str + "\n" + "Runtime: {}s".format(ellapsed))

    outstr = outstr.decode("utf-8")
    params = outstr[outstr.rfind('FITTED PARAMS'):outstr.rfind("(sum")]
    params = [x.split() for x in params.split("\n")][1:-1]
    param_str = '"' + '; '.join([' '.join(x) for x in params]) + '"'

    return param_str


def test_params_par(fname, snap_dir, em_iter=30, mat_max = 2, verbose=False):
    p = Pool()

    strs = []

    for i in range(2,mat_max+1):
        call_str = "{}/kronem/kronem -i:{} -ei:{} -n0:{} -m:R".format(snap_dir, 
                                                                      fname, 
                                                                      em_iter, 
                                                                      i)
        strs.append(call_str)

    param_strs = p.map(call_kronem, strs)

    return param_strs

def get_call_str(fname, snap_dir, em_iter=30, mat_size = 2):
    out_dir, base_name = os.path.split(fname)
    base, _ = os.path.splitext(base_name)

    if not out_dir:
        out_dir = os.getcwd()
        fname = os.path.join(out_dir, fname)

    call_str = "{}/kronem/kronem -i:{} -ei:{} -n0:{} -m:R".format(snap_dir, 
                                                                  fname, 
                                                                  em_iter, 
                                                                  mat_size)

    return call_str

def get_network_params(fname, snap_dir, em_iter=30, mat_size = 2, verbose=False):
    out_dir, base_name = os.path.split(fname)
    base, _ = os.path.splitext(base_name)

    if not out_dir:
        out_dir = os.getcwd()
        fname = os.path.join(out_dir, fname)

    call_str = get_call_str(fname, snap_dir, em_iter=30, mat_size = 2)

    if verbose: 
        print("Calculating Kronecker graph initialization matrix...")
        subprocess.call(call_str, shell=True)

        with open(os.path.join(out_dir, "KronEM-{}.tab".format(base)), "r") as f:
            params = f.readlines()[-1]

        params = params[params.index("[")+1:params.index("]")].replace(",","")
        params = '"'+params+'"'
    else:
        params = call_kronem(call_str)

    return params

def get_k(network):
    v = len(np.unique(network))
    k = 0

    while v > 1: 
            v /= 2 
            k += 1

    return k

def generate_graph(network, params, snap_dir, fname, verbose=False):
    k = get_k(network)
    call_str = "{}/krongen/krongen -o:{} -m:{} -i:{}".format(snap_dir, 
                                                             fname,
                                                             params, k)

    if verbose: 
        print("Projecting unobserved portion of graph...")
        subprocess.call(call_str, shell=True)
    else:
        _ = subprocess.run(call_str, shell=True, capture_output=True)

def emulate_network(args_dict):
    wd = os.getcwd()
    in_dir, base_name = os.path.split(args_dict['fname'])

    if not in_dir:
        in_dir = os.getcwd()
        args_dict['file'] = os.path.join(in_dir, args_dict['file'])

    if args_dict['fname']:
        out_dir, base_name = os.path.split(args_dict['fname'])

        if not out_dir:
            out_dir = wd
            args_dict['fname'] = os.path.join(out_dir, args_dict['fname'])
        else:
            os.chdir(out_dir)

    network = read_txt(args_dict['file'])
    network = np.array(network).astype(int)[:,:2]
    network = np.unique(network, axis=0)

    new_out = os.path.join(out_dir, "simplified_"+base_name)
    np.savetxt(new_out, network, fmt='%u', delimiter='\t')

    params = get_network_params(new_out, 
                                snap_dir=args_dict['snap_dir'], 
                                em_iter=args_dict['em_iter'], 
                                mat_size=args_dict['mat_size'], 
                                verbose=args_dict['verbose'])
    
    generate_graph(network, params, args_dict['snap_dir'], 
                   args_dict['fname'], args_dict['verbose'])

    new_net = read_txt(args_dict['fname'])
    new_net = [x for x in new_net if not x[0].startswith("#")]
    new_net = np.array(new_net).astype(int)
    new_net = np.unique(np.vstack((network, new_net)), axis=0)

    if wd != os.getcwd():
        os.chdir(wd)

    if args_dict['save']:
        np.savetxt(args_dict['fname'], new_net, fmt='%u', delimiter='\t')
    else:
        return new_net

def compare_nets(base_net, g, g_prime, measure='degree'):
    # get subsetted centrality measure
        # compare by node for biggest changes (only for nodes contained in both)
    net = build_comparison_metrics(base_net, g, measure)

    # calculate node changes
    result_actual = generate_node_changes(net)
    result_actual = {k: v for k, v in sorted(result_actual.items(),
                                      key=lambda item: item[1])}

    # generate results of interest
    ra = np.array(list(result_actual.items()))

    # get centrality measures on re-completed network
    # compare by node for biggest changes
    net = build_comparison_metrics(base_net, g_prime, measure)

    result_pred = generate_node_changes(net)

    # generate results of interest
    rp = np.array(list(result_pred.items()))

    return ra, rp

def test_matrices(args_dict):
    scores = np.zeros((args_dict['runs'], 2))
    param_history = [''] * args_dict['runs']

    # Load knowledge graph network

    network = read_txt(args_dict['file'])
    network = np.array(network).astype(int)[:,:2]
    network = np.unique(network, axis=0)

    g = nx.Graph()
    g.add_edges_from(network)

    out_dir, base_name = os.path.split(args_dict['fname'])

    if not out_dir:
        out_dir = os.getcwd()

    # create subset network
    drop_nodes = random.sample(g.nodes(), 
                               int(g.number_of_nodes()*args_dict['percentage']))
    # drop_edges = random.sample(g.edges(), int(len(g.edges())*args_dict['percentage']))
    masked_net = g.copy()
    masked_net.remove_nodes_from(drop_nodes)

    # write_graph(fname, masked_net)
    fname = os.path.join(out_dir, "masked_{}".format(base_name))
    np.savetxt(fname, np.array(list(masked_net.edges())), fmt='%u', delimiter='\t')

    for i in range(args_dict['runs']):
        # run KronEM/KronGen to predict missing portion
        params = get_network_params(fname=fname, 
                                    snap_dir=args_dict['snap_dir'], 
                                    em_iter=args_dict['em_iter'], 
                                    mat_size=i+2, 
                                    verbose=args_dict['verbose'])
        param_history[i] = params

        generate_graph(masked_net, params, args_dict['snap_dir'], 
                       args_dict['fname'], args_dict['verbose'])

        new_net = read_txt(new_out)
        new_net = [x for x in new_net if not x[0].startswith("#")]
        new_net = np.array(new_net).astype(int)

        g_prime = masked_net.copy()

        for edge in new_net:
            g_prime.add_edge(*edge)

        for measure in ['degree', 'eigenvector']:
            ra, rp = compare_nets(masked_net, g, g_prime, measure)

            # Score differences between this and above comparisons using 
            # Normalized Discount Cumulative Gain
            if measure == 'degree':
                col = 0
            else:
                col = 1

            scores[i,col] = ndcg(ra, rp)

    with open(os.path.join(out_dir, "param_history.txt"), "w") as f:
        f.writelines(param_history)

    np.savetxt(os.path.join(out_dir, "scores.csv"), scores)


def test_predictions(args_dict):
    scores = np.zeros((args_dict['runs'], 2))
    param_history = [''] * args_dict['runs']

    # Load knowledge graph network

    network = read_txt(args_dict['file'])
    network = np.array(network).astype(int)

    if network.shape[1] > 2:
        network = network[:,:2]

    g = nx.Graph()
    g.add_edges_from(network)

    out_dir, base_name = os.path.split(args_dict['fname'])

    if not out_dir:
        out_dir = os.getcwd()

    for i in range(args_dict['runs']):
        # create subset network
        drop_nodes = random.sample(g.nodes(), 
                                   int(g.number_of_nodes()*args_dict['percentage']))
        masked_net = g.copy()
        masked_net.remove_nodes_from(drop_nodes)

        fname = os.path.join(out_dir, "run_{}_{}".format(i, base_name))
        np.savetxt(fname, np.array(list(masked_net.edges())), fmt='%u', delimiter='\t')

        # run KronEM/KronGen to predict missing portion
        params = get_network_params(fname=fname, 
                                    snap_dir=args_dict['snap_dir'], 
                                    em_iter=args_dict['em_iter'], 
                                    mat_size=args_dict['mat_size'], 
                                    verbose=args_dict['verbose'])
        param_history[i] = params

        generate_graph(masked_net, params, args_dict['snap_dir'], 
                       args_dict['fname'], args_dict['verbose'])

        new_net = read_txt(new_out)
        new_net = [x for x in new_net if not x[0].startswith("#")]
        new_net = np.array(new_net).astype(int)

        g_prime = masked_net.copy()

        for edge in new_net:
            g_prime.add_edge(*edge)

        for measure in ['degree', 'eigenvector']:
            ra, rp = compare_nets(masked_net, g, g_prime, measure)

            # Score differences between this and above comparisons using 
            # Normalized Discount Cumulative Gain
            if measure == 'degree':
                col = 0
            else:
                col = 1

            scores[i,col] = ndcg(ra, rp)


    with open(os.path.join(out_dir, "param_history.txt"), "w") as f:
        f.writelines(param_history)

    np.savetxt(os.path.join(out_dir, "scores.csv"), scores)



def ndcg(actual, predicted, k=100):
    if k is None:
        k = min((actual.shape[0], predicted.shape[0]))

    # Match on node IDs
    act = pd.DataFrame(actual)
    act.columns = ['idx','value']
    act = act.sort_values('value')
    act.value = list(range(act.shape[0]))
    act.value /= max(act.value)

    pre = pd.DataFrame(predicted)
    pre.columns = ['idx','value']
    pre = pre.sort_values('value')
    pre.value = list(range(pre.shape[0]))
    pre.value /= max(pre.value)

    comp = act.merge(pre, on='idx', how='outer')
    comp = comp.fillna(0)

    # Convert back to numpy arrays
    actual    = comp[['idx','value_x']].values
    predicted = comp[['idx','value_y']].values

    # get the indices of the nodes in descending order of score
    act_order = np.argsort(actual, axis=0)[::-1][:,-1]
    pred_order = np.argsort(predicted, axis=0)[::-1][:,-1]

    # get the top k scores in order; first the "ideal" line up, which is the 
    # actual line up, then the predicted scores from those same nodes
    ideal = np.take(actual[:,1], act_order[:k])
    results = np.take(predicted[:,1], act_order[:k])

    # Calculate ideal DCG and the DCG for these predictions
    ideal_gain = 2 ** ideal - 1
    gain = 2 ** results - 1

    discounts = np.log2(np.arange(k) + 2)
    ideal_dcg = np.sum(ideal_gain / discounts)
    this_dcg = np.sum(gain/discounts)

    # Return the normalized DCG metric
    return this_dcg/ideal_dcg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate unobserved sections'
                                                 ' of Knowledge Graph.')
    parser.add_argument('-f', '--file', required=True, help='File to parse into'
                        ' a network.')
    parser.add_argument('-n', '--fname', help='File name for network output.')
    parser.add_argument('-d', '--snap_dir', required=True, 
                        help='Directory of Kronecker modeling programs.')
    parser.add_argument('-s', '--save', default=True, action='store_true', 
                        help='Save the new network data files instead of '
                             'returning.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', 
                        help='Verbose output.')
    parser.add_argument('-t', '--test', default=False, action='store_true', 
                        help='Test system performance.')
    parser.add_argument('-T', '--test_mat', default=False, action='store_true', 
                        help='Test different matrix sizes on performance.')
    parser.add_argument('-r', '--runs', type=int, default=5, 
                        help='Number of emulation runs to perform in testing.')
    parser.add_argument('-p', '--percentage', type=float, default=0.25, 
                        help='Percentage of network to mask in testing.')
    parser.add_argument('-e', '--em_iter', type=int, default=30, 
                        help='Number of EM iterations for the KronEM algorithm.')
    parser.add_argument('-m', '--mat_size', type=int, default=2, 
                        help='Size of the Kronecker Graph parameter matrix '
                             '(m * m). If using the `test_mat` flag, this is '
                             'ignored.')
    args_dict = vars(parser.parse_args())

    if args_dict['test']:
        test_predictions(args_dict)
    elif args_dict['test_mat']:
        test_matrices(args_dict)
    else:
        emulate_network(args_dict)





