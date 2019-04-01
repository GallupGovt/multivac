#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import json
import os
import pickle
import pubmed_parser
import slate

from bs4 import BeautifulSoup as bs
from collections import OrderedDict

from multivac.src import utilities
from multivac import settings


def aggregate_pubmed(srcs, verbose=False):
    """Aggregate a set of Pubmed article text and metadata."""
    pubmed_data = OrderedDict()
    pubmed_metadata = OrderedDict()
    for src in srcs:
        if verbose:
            print(src)
        try:
            temp = OrderedDict()
            metadata, text = parse_pubmed(str(src.absolute()))
            temp['metadata'] = metadata
            temp['metadata']['source'] = 'pubmed'
            temp['text'] = text
            try:
                k = metadata['doi']
            except AttributeError:
                k = src.strip('.xml')
            if len(text) > 0:
                pubmed_data[k] = temp
                pubmed_metadata[k] = metadata
            print(src)
        except:
            if verbose:
                print('Error: %s' % src)
            pass
    dst = settings.metadata_dir / 'pubmed.pkl'
    with open(dst, 'w') as f:
        pickle.dump(pubmed_metadata, f)
    return pubmed_data


def collect_process_main():
    output = {}
    for source in settings.sources:
        data_raw_dir = settings.raw_dir / source
        if source in ['arxiv', 'springer']:
            data = parse_articles_data(source, data_raw_dir)
        elif source == 'pubmed':
            srcs = [data_raw_dir / x for x in os.listdir(data_raw_dir)]
            data = aggregate_pubmed(srcs)
        if len(output) == 0:
            output = copy.deepcopy(data)
        else:
            output.update(data)
    arxiv_drops = [x.split()[0] for x in settings.arxiv_drops]
    filtered_output = filter_arxiv(output, arxiv_drops)
    save_outputs(filtered_output)
    return True


def filter_arxiv(output, arxiv_drops):
    filtered_output = OrderedDict()
    for k, v in output.items():
        if v['metadata']['source'] == 'arxiv':
            for term in v['metadata']['tags']:
                if term['term'] not in arxiv_drops:
                    filtered_output[copy.deepcopy(k)] = copy.deepcopy(v)
        else:
            filtered_output[copy.deepcopy(k)] = copy.deepcopy(v)
    return filtered_output


def parse_articles_data(source, data_raw_dir, verbose=False):
    """Parse Arxiv and Springer article data."""
    # load metadata
    fn = source + '.pkl'
    metadata_src = settings.metadata_dir / fn
    with open(metadata_src, 'rb') as f:
        metadata_ = pickle.load(f)

    # we'll just add the text to a new arxiv object, an ordered dict keyed on
    # doi or other id
    data = OrderedDict()
    srcs = [data_raw_dir / x for x in os.listdir(data_raw_dir)]
    for ix, article_metadata in enumerate(metadata_):

        # initialize temp dictionary
        temp = OrderedDict()
        temp['metadata'] = copy.deepcopy(article_metadata)
        temp['metadata']['source'] = source
        article_fn = article_metadata['fn']
        if verbose:
            print(article_fn)
        src = data_raw_dir / article_fn

        # define key and value
        if source == 'arxiv':
            k = article_metadata['fn'].strip('.pdf')
            temp['text'] = parse_pdf(src)
        elif source =='springer':
            k = article_metadata['doi']
            temp['text'] = parse_html(src)
        elif source == 'pubmed':
            raise ValueError('pubmed not supported. Only "arxiv" and "springer" supported. Try "parse_pubmed() function"')
        else:
            raise ValueError('Only "arxiv" and "springer" supported as sources.')

        # populate interim dictionary
        data[k] = temp

    # save intermediate outputs
    data_interim_dst = settings.interim_dir / fn
    with open(data_interim_dst, 'wb') as f:
        pickle.dump(data, f)
    return data


def parse_html(src):
    """Parse research paper HTML and return text."""
    with open(src, 'r', encoding='utf-8') as f:
        raw_data_ = f.read()
    soup = bs(raw_data_)
    try:
        text = ' '.join(soup.find('article').get_text().split())
    except AttributeError:
        text = None
    return text


def parse_pdf(src):
    """Parse research paper PDF and return text."""
    try:
        # try to open file
        with open(src, 'rb') as f:
            doc = slate.PDF(f)

        # get text: strip out newlines and extra spaces
        doc = ' '.join([' '.join(x.split()) for x in doc])
        text = (doc
            .split(' Abstract ')[-1]
            .split(' Acknowledgments ')[0]
            .split(' ∗ ∗ ∗ ')[0]
            .strip()
        )

    except:  #  PDFSyntaxError
        text = None

    return text


def parse_pubmed(src):
    """Parse pubmed xml article data and return metadata and text."""
    metadata = pubmed_parser.parse_pubmed_xml(src)
    text = pubmed_parser.parse_pubmed_paragraph(src, all_paragraph=True)
    text = ' '.join(' '.join([x['text'] for x in text]).split())
    return metadata, text


def save_outputs(output, dst_dir=None, fn_prefix=None):
    if dst_dir is None:
        dst_dir = settings.processed_dir / 'data'
    utilities.mkdir(dst_dir)
    fn = 'data.json'
    if fn_prefix is not None:
        fn = fn_prefix + '_' + fn
    dst = dst_dir / fn
    with open(dst, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    collect_process_main()
