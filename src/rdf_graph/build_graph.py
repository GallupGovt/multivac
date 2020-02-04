#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from datetime import datetime

from rdf_graph import RDFGraph


def run(args_dict):
    # create timestamp
    timestamp = datetime.now().strftime('%d%b%Y-%H:%M:%S')

    # instantiate class
    knowledge_graph = RDFGraph(args_dict['sources'])

    # build knowledge graph
    knowledge_graph.build_graph()

    # output text files that will be used openke for knowledge graph creation
    # and embedding output .txt files for openke output
    print('\nSaving final tuples to .txt files for input to OpenKE')
    knowledge_graph.output_to_openke(timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetcher to retrieve articles '
                                     'for modeling.')
    parser.add_argument('-s', '--sources', required=True,
                        help='Select a source for article retrieval.')
    args_dict = vars(parser.parse_args())

    run(args_dict)
