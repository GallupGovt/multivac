

#
# Python implementation of Unsupervised Semantic Parsing system, from:
# 
#   Hoifung Poon and Pedro Domingos (2009). "Unsupervised Semantic Parsing", 
#   in Proceedings of the Conference on Empirical Methods in Natural Language 
#   Processing (EMNLP), 2009. http://alchemy.cs.washington.edu/usp.
# 

import argparse
import os

from multivac import settings

from semantic import Parse
from semantic.MLN import MLN

def read_input_files(DIR):
    '''Read files given by list of names '''
    files = set()
    for file in os.listdir(DIR):
        if file.endswith(".dep"):
            files.add(file)

    return files


def run(params):
    if os.path.isabs(params['data_dir']):
        data_dir = params['data_dir']
    else:
        data_dir = os.path.join(os.getcwd(), params['data_dir'])

    if os.path.isabs(params['results_dir']):
        results_dir = params['results_dir']
    else:
        results_dir = os.path.join(os.getcwd(), params['results_dir'])

    priorNumParam = params['priorNumParam']
    priorNumConj = params['priorNumConj']

    parser = Parse.Parse()

    # Get files
    input_files = read_input_files(data_dir)

    # Parse files into MLN knowledge base
    parser.parse(input_files)

    # Save knowledge base files to disk
    MLN.printModel(results_dir)

    return None


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description='Parse scientific articles into'
                                     ' Markov Logic Network knowledge base. \n'
                                     'Usage: python -m pymln.py [-d dataDir] '
                                     '[-r resultDir] [-p priorNumParam] [-c '
                                     'priorNumConj]')
    prs.add_argument('-d', '--data_dir', 
                        help='Directory of source files. If not specified, '
                        'defaults to the current working directory.')
    prs.add_argument('-r', '--results_dir', 
                        help='Directory to save results files. If not specified,'
                        ' defaults to the current working directory.')
    prs.add_argument('-p', '--priorNumParam', 
                        help='Prior on parameter number. If not specified,'
                        ' defaults to 5.')
    prs.add_argument('-c', '--priorNumConj', 
                        help='Prior on number of conjunctive parts assigned to '
                        'same cluster. If not specified, defaults to 10.')

    args = vars(prs.parse_args())

    # Default argument values
    params = {'priorNumParam': 5, 
              'priorNumConj': 10, 
              'data_dir': settings.processed_dir,
              'results_dir': settings.processed_dir / 'results_dir'}

    # If specified in call, override defaults
    for par in params:
        if args[par] is not None:
            params[par] = args[par]

    run(params)






