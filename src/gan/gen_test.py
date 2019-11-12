import argparse

from gen_pyt.asdl.lang.eng.eng_asdl_helper import asdl_ast_to_english
from gen_pyt.model.parser import Parser

def run(args):
    '''
    Load GAN generator model
    Apply query items
    Return beam search results
    '''

    if isinstance(args['model'], str):
        netG = Parser.load(args['model'])
    else:
        netG = args['model']

    if isinstance(args['query'], str):
        query = args['query'].split()
    elif args['query'] is None:
        query = [] # get list of query tokens from KG graph 
    else:
        query = args['query']

    results = netG.parse(query, beam_size=netG.args['beam_size'])
    texts = [asdl_ast_to_english(x.tree) for x in results]

    return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', 
                        help='Path to model checkpoint file.')
    parser.add_argument('-q', '--query', nargs='+', required=False,
                        help='Query tokens for generating a question.')
    parser.add_argument('-l', '--length', nargs='+', required=False,
                        help='Number of query tokens to use for generating '
                             'a question.')

    args = vars(parser.parse_args())

    results = run(args)
    print(results)
