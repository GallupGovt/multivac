
import argparse
import os 
from datetime import datetime
from multivac import settings 
from semantic import Parse, MLN, Clust, ArgClust 
from syntax.Nodes import Article, Sentence, Token, TreeNode 
from syntax.StanfordParseReader import StanfordParseReader 

def run(args):

    def read_input_files(DIR):
        files = []
        for file in os.listdir(DIR):
            if file.endswith(".dep"):
                files.append(file)

        return files

    verbose = args['verbose']
    data_dir = args['data_dir']

    if 'results_dir' in args:
        results_dir = args['results_dir']
    else:
        results_dir = data_dir

    priorNumParam = args['priorNumParam']
    priorNumConj = args['priorNumConj']

    parser = Parse(priorNumParam, priorNumConj)

    input_files = read_input_files(data_dir)
    input_files.sort()

    articles = [] 
    if 'subset' in args:
        subset = int(args['subset'])
    else:
        subset = len(input_files)

    for i, fileName in enumerate(input_files):
        try:
            a = StanfordParseReader.readParse(fileName, data_dir)
        except:
            print("Error on {}, {}".format(i, fileName))
            raise Exception

        if i%100 == 0:
            print("{} articles parsed.".format(i))

        if i >= subset:
            break

        articles.append(a)


    if verbose:
        print("{} Initializing...".format(datetime.now())) 

    parser.initialize(articles, verbose)
    
    if verbose:
        print("{}: {} articles parsed, of {} sentences and {} total tokens.".format(datetime.now(), len(articles), parser.numSents, parser.numTkns))
    num_arg_clusts = sum([len(x._argClusts) for x in Clust.clusts.values()])
    
    if verbose:
        print("{}: {} initial clusters, with {} argument clusters.".format(datetime.now(), len(Clust.clusts), num_arg_clusts))

    if verbose:
        print("{} Merging arguments...".format(datetime.now()))
    parser.mergeArgs() 
    num_arg_clusts = sum([len(x._argClusts) for x in Clust.clusts.values()])
    
    if verbose:
        print("Now with {} initial clusters, {} argument clusters.".format(len(Clust.clusts), num_arg_clusts))

    if verbose:
        print("{} Creating agenda...".format(datetime.now()))
    parser.agenda.createAgenda(verbose)

    if verbose:
        print("{}: {} possible operations in queue, {} merges and {} composes.".format(datetime.now(),
                                                                                   len(parser.agenda._agendaToScore), 
                                                                                   len(parser.agenda._mc_neighs), 
                                                                                   len(parser.agenda._compose_cnt)))
    
    if verbose:
        print("{} Processing agenda...".format(datetime.now()))
    parser.agenda.procAgenda(verbose)

    num_arg_clusts = sum([len(x._argClusts) for x in Clust.clusts.values()])

    if verbose:
        print("{}: {} final clusters, with {} argument clusters.".format(datetime.now(), len(Clust.clusts), num_arg_clusts))

    MLN.save_mln(data_dir + "/mln.pkl")
    MLN.printModel(results_dir)

    if verbose:
        print("{} Induced MLN saved.".format(datetime.now()))


if __name__ == '__main__':
    prs = argparse.ArgumentParser(description='DO it.')
    prs.add_argument('-d', '--data_dir', 
                        help='Directory of source files. If not specified, '
                        'defaults to the current working directory.')
    prs.add_argument('-r', '--results_dir', 
                        help='Directory to save results files. If not specified,'
                        ' defaults to the current working directory.')
    prs.add_argument('-v', "--verbose", dest='verbose', action='store_true',
                        help='Give verbose output.')
    prs.add_argument('-c', '--priorNumConj', 
                        help='Prior on number of conjunctive parts assigned to '
                        'same cluster. If not specified, defaults to 10.')
    prs.add_argument('-p', '--priorNumParam', 
                        help='Prior on number of conjunctive parts assigned to '
                        'same cluster. If not specified, defaults to 10.')
    prs.add_argument('-n', '--subset', 
                        help='Number of articles for a rest run.')

    args = vars(prs.parse_args())

    run(args)


