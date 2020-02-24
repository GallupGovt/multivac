"""
Preprocessing script for MULTIVAC data.

"""
import argparse
import glob
import os
import re

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from multivac.src.gan.discriminator import MULTIVACDataset
from multivac.src.gan.utilities.utils import build_vocab
from multivac.src.rdf_graph.rdf_parse import StanfordParser


def dep_parse(filepath, parser):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'r') as f:
        examples = f.readlines()

    with open(os.path.join(dirpath, 'text.toks'   ), 'w') as tokfile, \
         open(os.path.join(dirpath, 'text.parents'), 'w') as parfile:

        for example in tqdm(examples):
            text = example.strip()

            if not text.endswith("?"):
                text = re.sub(r"\?","",text)
                text += "?"

            sample_parse = parser.get_parse(text)['sentences'][0]
            tokens = [x['word'] for x in sample_parse['tokens']]
            deps = sorted(sample_parse['basicDependencies'], 
                          key=lambda x: x['dependent'])
            parents = [x['governor'] for x in deps]
            tree = MULTIVACDataset.read_tree(parents)

            parfile.write(' '.join([str(x) for x in parents]) + '\n')
            tokfile.write(' '.join(tokens) + '\n')


def gen_tokens(filepath, parser):
    print('\nTokenizing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'r') as f:
        examples = f.readlines()

    with open(os.path.join(dirpath, 'text.toks'   ), 'w') as tokfile:

        for example in tqdm(examples):
            text = example.strip()

            if not text.endswith("?"):
                text = re.sub(r"\?","",text)
                text += "?"

            sample_parse = parser.get_parse(text)
            tokens = [x['word'] for x in sample_parse['tokens']]
            tokfile.write(' '.join(tokens) + '\n')


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def split(filepath, dst_dir):
    '''
    Input datafiles now have form:
    id \t sentence \t category (0, 1)
    id = id number
    sentence = text of sentence/query
    category = whether this is a "real" or "fake" sentence
    '''
    with open(filepath) as datafile, \
            open(os.path.join(dst_dir, 'text.txt'), 'w') as textfile, \
            open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'cat.txt'), 'w') as catfile:
        datafile.readline()

        for line in datafile:
            i, text, cat = line.strip().split('\t')
            idfile.write(i + '\n')
            textfile.write(text + '\n')
            catfile.write(cat + '\n')

def train_dev_test_split(filepath, dst_dir, 
                         train=0.7, dev=0.2, test=0.1):
    test = test/(train + test)

    with open(filepath, "r") as datafile:
        data = datafile.readlines()

    header = data[0]

    x_train, x_dev = train_test_split(data[1:], test_size=dev, shuffle=True)
    x_train, x_test = train_test_split(x_train, test_size=test, shuffle=True)

    with open(os.path.join(dst_dir, "MULTIVAC_train.txt"), "w") as f:
        f.write(header)
        for line in x_train:
            f.write(line)

    with open(os.path.join(dst_dir, "MULTIVAC_test_annotated.txt"), "w") as f:
        f.write(header)
        for line in x_dev:
            f.write(line)

    with open(os.path.join(dst_dir, "MULTIVAC_trial.txt"), "w") as f:
        f.write(header)
        for line in x_test:
            f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing of MULTIVAC data for QueryGAN '
                    'discriminator training.')
    # data arguments
    parser.add_argument('-d', '--data', required=False,
                        help='Path to source dataset.')

    args = vars(parser.parse_args())

    print('=' * 80)
    print('Preprocessing MULTIVAC dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    multivac_dir = os.path.join(data_dir, 'multivac')
    # lib_dir = os.path.join(base_dir, 'lib')
    # train_dir = os.path.join(multivac_dir, 'train')
    # dev_dir = os.path.join(multivac_dir, 'dev')
    # test_dir = os.path.join(multivac_dir, 'test')
    # make_dirs([train_dir, dev_dir, test_dir])

    prs = StanfordParser(annots='tokenize')

    split(os.path.join(multivac_dir, 'extracted_questions_labels.txt'), multivac_dir)
    gen_tokens(os.path.join(multivac_dir, 'text.txt'), prs)

    # if args['data']:
    #     train_dev_test_split(args['data'], multivac_dir)

    # split into separate files
    # split(os.path.join(multivac_dir, 'MULTIVAC_train.txt'), train_dir)
    # split(os.path.join(multivac_dir, 'MULTIVAC_trial.txt'), dev_dir)
    # split(os.path.join(multivac_dir, 'MULTIVAC_test_annotated.txt'), test_dir)

    # parse sentences
    # dep_parse(os.path.join(train_dir, 'text.txt'), prs)
    # dep_parse(os.path.join(dev_dir, 'text.txt'), prs)
    # dep_parse(os.path.join(test_dir, 'text.txt'), prs)

    # get vocabulary
    build_vocab(glob.glob(os.path.join(multivac_dir, '*/*.toks')),
                os.path.join(multivac_dir, 'vocab.txt'))
    build_vocab(glob.glob(os.path.join(multivac_dir, '*/*.toks')),
                os.path.join(multivac_dir, 'vocab-cased.txt'),
                lowercase=False)
