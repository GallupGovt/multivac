#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from datetime import datetime

from src.rdf_graph.rdf_graph import RDFGraph


def run(args_dict):

    # create timestamp
    timestamp = datetime.now().strftime('%d%b%Y-%H:%M:%S')

    # instantiate class
    knowledge_graph = RDFGraph()

    # Associate a JSON file of source documents from which to induce
    # the knowledge graph.
    knowledge_graph.set_source(args_dict['sources'])

    print('\nExtracting relation triples from abstracts')
    knowledge_graph.extract_raw_tuples()

    # pre-process extracted tuples
    print('\nPreprocessing raw relation triples')
    knowledge_graph.preprocess_raw_tuples()

    # cluster all entities using fast
    # agglomerative clustering and cosine distance of averaged word embeddings
    print('\nClustering entities from relation triples')
    knowledge_graph.cluster_entities()
    print('\n{} entity clusters were found'
          .format(len(knowledge_graph.entity_cluster_results['cluster_members'])))

    # output text files that will be used openke for knowledge graph creation
    # and embedding output .txt files for openke output
    print('\nSaving final tuples to .txt files for input to OpenKE')
    knowledge_graph.output_to_openke(timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetcher to retrieve articles '
                                     'for modeling.')
    parser.add_argument('-s', '--sources', required=True, choices=['pubmed',
                        'arxiv'], nargs='*', help='Select a source for article '
                        'retrieval.')
    args_dict = vars(parser.parse_args())

    run(args_dict)
