import argparse
import random

from gen_pyt.asdl.lang.eng.eng_asdl_helper import asdl_ast_to_english
from gen_pyt.model.parser import Parser

from multivac.get_kg_query_params import build_network, analyze_network
from multivac.src.rdf_graph.map_queries import predicted_object, get_con


def get_query_tokens(args):
    folder = os.listdir(args['dir'])
    files = [x for x in folder if ('2id' in x and x.endswith('.txt'))]
    rel_file = sorted([(os.path.getctime(os.path.join(folder, x)), x)
                        for x in files if 'relation' in x.lower()])
    rel_file = rel_file = os.path.join(folder, rel_file[-1][1])

    ent_file = sorted([(os.path.getmtime(os.path.join(folder, x)), x)
                        for x in files if 'entity' in x.lower()])
    ent_file = ent_file = os.path.join(folder, ent_file[-1][1])
    
    trn_file = sorted([(os.path.getmtime(os.path.join(folder, x)), x)
                        for x in files if 'train' in x.lower()])
    trn_file = trn_file = os.path.join(folder, trn_file[-1][1])
    
    entities = pd.read_csv(ent_file, sep='\t', 
                           names=["Ent","Id"], skiprows=1)
    relations = pd.read_csv(rel_file, sep='\t', 
                            names=["Rel","Id"], skiprows=1)
    train = pd.read_csv(trn_file, sep='\t', 
                           names=["Ent","Id"], skiprows=1)


    entities = read_txt(args_dict['files'][0])
    network = read_txt(args_dict['files'][1])

    # construct/analyze network
    net = build_network(network)
    results = analyze_network(net, args_dict)

    # return results
    named_entities = ['{}\n'.format(entity[0]) for entity in entities if
                      entity[1] in [res[0] for res in results]]

    seed = random.sample(named_entities).strip()
    con = get_con(args)

    out_list = predicted_object(con, seed)

    return out_list[0]

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
        query = get_query_tokens(args)
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
    parser.add_argument('-d', '--dir', required=True, help='Path to index data '
                        'directory.')

    args = vars(parser.parse_args())

    results = run(args)
    print(results)
