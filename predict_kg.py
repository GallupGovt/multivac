#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calls out to KromEM and KronGen to emulate and generate a hypothetical 
"complete" knowledge graph usin Kronecker graph models. 
"""
import argparse
import numpy as np
import os
import re
import subprocess
import sys


def get_network_params(network, fname, verbose):
    out_dir, base_name = os.path.split(fname)
    base, _ = os.path.splitext(base_name)

    if not out_dir:
        out_dir = os.getcwd()
        fname = os.path.join(out_dir, fname)

    v = vcount = len(np.unique(network))
    k = 0

    while vcount > 1: 
            vcount /= 2 
            k += 1

    if verbose: print("Calculating Kronecker graph initialization matrix...")
    subprocess.call("{}/kronem/kronem -i:{} -m:R".format(snap_dir, fname),
                    shell=True)

    with open(os.path.join(out_dir, "KronEM-{}.tab".format(base)), "r") as f:
        params = f.readlines()[-1]

    init_mat = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", 
                          params)
    init_mat = '"{} {}; {} {}"'.format(*init_mat)

    return k, init_mat, v

def emulate_network(file, snap_dir, fname='multivac.txt', verbose=False, save=False):
    out_dir, base_name = os.path.split(fname)

    if not out_dir:
        out_dir = os.getcwd()
        fname = os.path.join(out_dir, fname)

    network = read_txt(file)
    network = np.array(network).astype(int)[:,:-1]
    np.savetxt(fname, network, fmt='%u', delimiter='\t')

    k, init_mat, vcount = get_network_params(network, fname, verbose)
    
    if verbose: print("Projecting unobserved portion of graph...")
    new_out = os.path.join(out_dir, "new_"+base_name)
    subprocess.call("{}/krongen/krongen -o:{} -m:{} -i:{}".format(snap_dir, 
                                                                  new_out,
                                                                  init_mat,
                                                                  k),
                    shell=True)

    new_net = read_txt(new_out)
    new_net = [x for x in new_net if not x[0].startswith("#")]
    new_net = np.array(new_net).astype(int)
    new_edges = np.apply_along_axis(lambda x: sum(x > vcount)>0, 1, new_net)
    new_net = new_net[new_edges]

    if save:
        np.savetxt(new_out, new_net, fmt='%u', delimiter='\t')
    else:
        return network, new_net

def read_txt(file):
    with open(file) as f:
        tmp = f.readlines()[1:]

    return [x.rstrip(' \n').split('\t') for x in tmp]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate unobserved sections'
                                                 ' of Knowledge Graph.')
    parser.add_argument('-f', '--file', required=True, help='File to parse into'
                        ' a network.')
    parser.add_argument('-n', '--name', required=False, help='File name for '
                                                        'network output.')
    parser.add_argument('-d', '--snap_dir', required=True, 
                        help='Directory of Kronecker modeling programs.')
    parser.add_argument('-s', '--save', default=False, action='store_true', 
                        help='Save the new network data files instead of '
                             'returning.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', 
                        help='Verbose output.')
    args_dict = vars(parser.parse_args())

    file = args_dict['file']
    verbose = args_dict['verbose']
    save = args_dict['save']
    snap_dir = args_dict['snap_dir']

    if args_dict['name']:
        emulate_network(file=file, snap_dir=snap_dir, name=args_dict['name'], 
                        verbose=verbose, save=save)
    else:
        emulate_network(file=file, snap_dir=snap_dir, verbose=verbose, save=save)

