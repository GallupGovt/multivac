#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import corenlp
import os
import subprocess

from multivac import settings
from multivac.src.data.parsing import load_data
from qgnet.test.testinput.preprocessing_pdf import (
    create_tf_idf, preprocess_pdf
)


os.environ["CORENLP_HOME"] = ('{}/stanford-corenlp-full-2018-10-05'
                              .format(settings.models_dir))


def qgnet_main(args_dict):
    # first, run shell script, if necessary, in qgnet to create model
    subprocess.call([
        '../{}/download_QG-Net.sh'.format(settings.qgnet_dir),
        args_dict['qgnet_path']
    ])

    # second, pre-process the pdfs
    jsonObj, allDocs = load_data('{}/da_embeddings.txt'
                                 .format(settings.models_dir))
    abstracts = []
    for value in jsonObj.values():
        if "summary" in value['metadata']:
            abstracts.append(value['metadata']["summary"])
        elif "abstract" in value['metadata']:
            abstracts.append(value['metadata']["abstract"])

    nlp = corenlp.CoreNLPClient(output_format='json', properties={
        'timeout': '50000'})

    features, tfidf = create_tf_idf(abstracts, False)

    for i, abstract in enumerate(abstracts):
        preprocess_pdf(abstract, features[i,:].toarray(), tfidf, nlp)

    # third, generate qg-net questions
    subprocess.call([
        '../{}/qg_reproduce_LS.sh'.format(settings.qgnet_dir),
        args_dict['qgnet_path'],
        settings.models_dir
    ])


if __name__ == '__main__':
    qgnet_main(args_dict)
