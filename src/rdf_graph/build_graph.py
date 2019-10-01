#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json

from datetime import datetime

from rdf_graph import rdf_graph


def run(args_dict):

    # create timestamp
    timestamp = datetime.now().strftime('%d%b%Y-%H:%M:%S')

    # define terms dictionary
    terms = {}
    for source in args_dict['sources']:
        with open('data/{}_search.json'.format(source), 'r') as f:
            d = json.load(f)
        terms.update({source: d})

    # instantiate class
    knowledge_graph = rdf_graph()

    #
    ### DEFINE WHERE THIS GETS THE FILES TO PROCESS NOW THAT IT'S NOT DOING 
    ### THE ACTUAL PULLING.
    #

    # extract relation tuples from cleaned abstracts w/corenlp
    '''The Stanford CoreNLP runs in Java. However, the 'pycorenlp' is a python
    wrapper to the CoreNLP program.  To use it, you need to download the
    Stanford coreNLP program library on to your computer, plus an installation
    of Java 8+ to run the program. To download go here:
    https://stanfordnlp.github.io/CoreNLP/. Before you can call the CoreNLP
    program from 'pycorenlp', you need to have the Stanford CoreNLP server
    running. To do that navigate to the downloaded CoreNLP folder on your
    terminal and run the following command:

    java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

    Once you've done that, you should be ready to go! If you run into problems
    running the CoreNLP server, see the following webpage
    (https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started)'''
    print('\nExtracting relation triples from abstracts')
    knowledge_graph.extract_raw_tuples()
  
    # pre-process extracted tuples
    print('\nPreprocessing raw relation triples')
    knowledge_graph.preprocess_raw_tuples()

    # cluster all entities using fast
    # agglomerative clustering and string distance
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
    parser.add_argument('-p', '--pubmed', required=False, nargs=2, help='API '
                        'keys needed to access Pubmed; enter as email address '
                        'and API key.')
    parser.add_argument('-s', '--sources', required=True, choices=['pubmed',
                        'arxiv'], nargs='*', help='Select a source for article '
                        'retrieval.')
    args_dict = vars(parser.parse_args())

    run(args_dict)
