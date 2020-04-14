#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cross_decomposition import CCA
from unidecode import unidecode

from multivac import settings
from rpy2.robjects import numpy2ri, pandas2ri, r


def domain_adapted_CCA(DG_embed, DS_embed, NC=100):
    # calculate the z-score
    DG_embed_norm = zscore(DG_embed)
    DS_embed_norm = zscore(DS_embed)

    # Initialize CCA Model
    cca = CCA(n_components=NC)
    cca.fit(DG_embed_norm, DS_embed_norm)

    DA_embeddings = (cca.x_scores_ + cca.y_scores_)/2

    return cca, DA_embeddings


def glove_main():
    # Load data from nlp parsing
    with open('{}/articles-with-equations.json'.format(settings.data_dir), 'r',
              encoding='utf-8') as jf:
        src_data = json.load(jf)

    texts = [src_data[art]['text'] for art in src_data if
             src_data[art]['text'] is not None]

    # The "unidecode" step simplifies non-ASCII chars which
    # mess up the R GloVe engine.
    texts_df = pd.Series(texts).apply(lambda x: unidecode(x))
    texts_df = pd.DataFrame({'text': texts_df})

    # Source all the functions contained in the 'trainEmbeddings' R file
    r("source('{}/trainEmbeddings.R'.format('src/data'))")

    # Call the main GloVe-embedding function from the R script
    trainEmbeddings_R = r("trainEmbeddings")

    # Train domain-specific GloVe embedding model and ouput as a Numpy Matrix
    pandas2ri.activate()
    DS_embeddings_R = trainEmbeddings_R(texts_df)
    pandas2ri.deactivate()

    DS_embeddings = numpy2ri.rpy2py(DS_embeddings_R[0])

    # Get domain-specific GloVe vocabulary
    domain_spec_vocab = list(DS_embeddings_R[1])

    # Load in Stanford's 'Common Crawl' domain-general Glove Embedding Model
    # Only pull out the words that are contained in our corpus
    # * This can take a while (~30min) - could use some optimization *
    DG_embeddings = loadGloveModel(
        '{}/glove.42B.300d.txt'.format(settings.data_dir),
        domain_spec_vocab
    )

    # Processing to ensure rows match between the domain-general and
    # domain-specific embeddings
    # Convert domain-general embedding from dictionary to array
    domain_gen_vocab = np.array([DG_embeddings[i] for i in
                                DG_embeddings.keys()])

    # Find the indices of matching words
    both = set(domain_gen_vocab).intersection(domain_spec_vocab)
    indices_gen = [domain_gen_vocab.index(x) for x in both]
    indices_spec = [domain_spec_vocab.index(x) for x in both]
    indices_spec_notDG = [domain_spec_vocab.index(x) for x in
                          domain_spec_vocab if x not in both]

    # Sort and subset domain-specific array to match indices of domain-general
    # array
    DS_embeddings_subset = DS_embeddings[indices_spec, :].copy()
    DG_embeddings_subset = DG_embeddings[indices_gen, :].copy()

    # fit cca model
    cca_res, DA_embeddings = domain_adapted_CCA(DG_embeddings_subset,
                                                DS_embeddings_subset, NC=100)

    DS_embeddings_notinDG = DS_embeddings[indices_spec_notDG, :]
    DS_embeddings_notinDG_norm = zscore(DS_embeddings_notinDG)

    DA_notinDG_embeddings = cca_res.y_weights_.T @ DS_embeddings_notinDG_norm.T
    DA_embeddings_final = np.append(DA_embeddings, DA_notinDG_embeddings.T,
                                    axis=0)

    # write data to disk
    np.savetxt('{}/da_embeddings.txt'.format(settings.models_dir),
               DA_embeddings_final, fmt='%d')


def loadGloveModel(gloveFile, vocab):
    f = open(gloveFile, ' r')

    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        if word in vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding

    return model


if __name__ == '__main__':
    glove_main()
