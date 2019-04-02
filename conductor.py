#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script conducts the entire flow of the multivac system to date. it has the
following flow:
1. collect data
    a. these data come from arxiv, springer, and pubmed in this instance, but
        could be modified to include more
    b. it saves the downloaded pdf's to a directory and creates a json object
        for further use
2. parse data
    a. the json objects that are saved from the collection step are processed
       for dependencies, input (word position), and morphology (lemma) [dim]
    b. it also identifies and notates equations throughout articles
3. run glove models
    a. take article collection that is parsed and create glove word embeddings
    b. develops both domain-general and domain-specific models
4. build the query generation (qg) network
    a. uses context/answers as inputs to create questions as output
    b. builds off of the domain-adapted glove models to produces robust
       questions around a topic of interest (in this case, epidemiology)
5. build markov logic network (mln)
    a. compile parsed dim files into trees and semantically cluster
    b. produce a graphical model based on first-order logic for
"""
import argparse

from multivac.src.data.glove import glove_main
from multivac.src.data.make import collect_main
from multivac.src.data.parsing import nlp_parse_main
from multivac.src.data.qgnet import qgnet_main
from multivac.pymln.pymln import mln_main


def conduct(args_dict):
    # step 1: collect data
    collect_main()

    # step 2:
    nlp_parse_main(args_dict)

    # step 3: run glove models
    glove_main()

    # step 4: build qg network
    qgnet_main(args_dict)

    # step 5: build mln
    mln_main(args_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Orchestrate pipeline for '
                                     'MULTIVAC processing and modeling.')
    parser.add_argument('-bp', '--nlp_bp', required=False, type=int,
                        help='Which document to start parsing with.')
    parser.add_argument('-js', '--nlp_newjson', action='store_true',
                        help='Boolean; indicates whether to create new JSON '
                        'file for glove embedding.')
    parser.add_argument('-an', '--subset', type=int, help='Number of articles '
                        'for MLN run.')
    parser.add_argument('-pc', '--prior_num_conj', default=10, type=int,
                        help='Prior on number of conjunctive parts assigned to '
                        'same cluster in MLN.')
    parser.add_argument('-pp', '--prior_num_param', default=5, type=int,
                        help='Prior on number of parameters for cluster '
                        'merges.')
    parser.add_argument('-qp', '--qgnet_path', required=True, help='The '
                        'top-level qgnet directory to create folders for '
                        'models and data.')
    parser.add_argument('-v', "--verbose", action='store_true', help='Give '
                        'verbose output during MLN modeling.')
    args_dict = vars(parser.parse_args())

    conduct(args_dict)
