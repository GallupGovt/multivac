
import argparse
from corenlp import CoreNLPClient
import pandas as pd
import re

from multivac.src.rdf_graph.rdf_parse import StanfordParser, stanford_parse


def process_queries(parser, queries, clean=False, verbose=False, how='list'):
    if clean:
        cl = lambda text: re.sub(r"^\W+|[^\w{W}(?<!?)]+$", "", text)
        queries = queries.apply(cl)

    questions = [stanford_parse(parser, query) for query in queries]
    processed = []

    for question in questions:
        if len(question.rdfs) > 0:
            processed.append(question.get_rdfs(use_tokens=False, how=how))
        else:
            processed.append([])

    return processed

def run(args_dict):
    if args_dict['how'] is None:
        args_dict['how'] = 'list'
    if args_dict['out_file'] is None:
        args_dict['out_file'] = args_dict['query_file']

    parser = StanfordParser()
    queries = pd.read_csv(args_dict['query_file'])
    contents = process_queries(parser, 
                               queries['Query'], 
                               args_dict['clean'], 
                               args_dict['verbose'],
                               args_dict['how'])
    queries['Annotations'] = pd.Series(contents)
    queries.to_csv(args_dict['out_file'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process queries into key semantic components.')
    parser.add_argument('-q', '--query_file', required=True,
                        help='Path to queries.')
    parser.add_argument('-o', '--out_file',
                        help='Filename for output. If not supplied, output '
                             'will save back to the source query file.')
    parser.add_argument('-H', '--how', choices=['asis','longest','all','list'],
                        help='Method for returning components of queries. '
                             'Default is "list."')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Pre-clean queries before populating.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output on progress.')

    args_dict = vars(parser.parse_args())
    run(args_dict)
