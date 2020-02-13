
"""
Do the actual docuemnt cleaning, given source document directories.
"""
import argparse
import json
import re
from collections import defaultdict
from os import getcwd, listdir
from os.path import abspath, exists, join

import lxml.html
from lxml import etree
from tqdm import tqdm

import pubmed_parser as pp
from pubmed_parser.utils import remove_namespace

PUNCT = frozenset([',', '.', ';', ':'])

REMOVE_CITATIONS1 = re.compile(r'\s*\[[1-9][0-9,–\-\s]*\]')
REMOVE_CITATIONS2 = re.compile(r'\s*\(.+?[12][0-9]{3}\)')
REMOVE_CITATIONS3 = re.compile(r'[0-9]{1,3}(,\s[0-9]{1,3})+')
REMOVE_STARTING = re.compile(r'^\(.+?\)[.\s]*')
REMOVE_NUMERALS = re.compile(r'\([ivx]+?\)\s*')

MATCH_WEIRD_NUMERALS = re.compile(r'[a-z][0-9]+')
REMOVE_EMPTY_PARENTS = re.compile(r'\(\)[\s,]*')
REMOVE_EMPTY_BRACKET = re.compile(r'\[\][\s,]*')
REMOVE_WEIRD_ELIPSES = re.compile(r'\.\s\.\s\.')

EXPECTED_EXTENSIONS = {'springer': ['.html'],
                       'pubmed': ['.xml'],
                       'arxiv': ['.xml'],
                       }


def clean_doc(doc):
    """
    Clean the document using regular expressiong.
    This cleaning should be done after parsing.

    Parameters
    ----------
    doc : list of dict
        A list of dictionaries. Each dictionary
        must contain a 'text' key, whose value is a string
        of text.

    Returns
    -------
    text : str
        A single string of cleaned text.
    """

    texts = []
    for line in doc:

        # apply some regular expressions and other replacements
        text = line['text'].strip().replace('\n', '').replace('\xa0', ' ')
        for regex in [REMOVE_CITATIONS1,
                      REMOVE_CITATIONS2,
                      REMOVE_CITATIONS3,
                      REMOVE_STARTING,
                      REMOVE_NUMERALS]:
            text = re.sub(regex, '', text)

        # match all weird numerals, and replace just the numeral part
        all_found = re.findall(MATCH_WEIRD_NUMERALS, text)
        for found in all_found:
            text = text.replace(found[1:], '')

        # apply some additional regular expressions
        text = re.sub(REMOVE_EMPTY_PARENTS, '', text)
        text = re.sub(REMOVE_EMPTY_BRACKET, '', text)
        text = re.sub(REMOVE_WEIRD_ELIPSES, '...', text)
        text = text.strip()

        # skip text that doesn't begin with a capital, or starts with 'click'
        if (text and text[0] == text[0].upper() and not text.lower().startswith('click')):

            # if text ends with a colon, make it a period
            if text.endswith(':'):
                text = text[:-1] + '.'
            texts.append(text)

    return ' '.join(texts)


def stringify_children(node):
    """
    Convert the HTML or XML document into a single string
    of text. If formula tags are found, convert them.

    Parameters
    ----------
    node : lxml ElementTree node
        The top node from the xpath.

    Returns
    -------
    parts : str
        The various parts of the document text,
        combined together into one string.
    """

    # start with the top modes, and check if there are any formulas
    if (node.tag in ['formula'] or node.get('class') in ['InlineEquation']):
        text = node.text
        punct = text[-1]
        punct = punct if punct in PUNCT else ''
        parts = ['<LONG_FORMULA>' + punct]
    else:
        parts = [node.text]

    # then loop through the child nodes, and check for formulas as well
    # as references and handle those separately
    for child_node in node.getchildren():

        if child_node.tag in ['ref']:
            parts.append(child_node.tail)
            continue

        elif ((child_node.tag in ['formula'] or
               child_node.get('class') in ['InlineEquation']) and
              child_node.text is not None):
            text = child_node.text
            punct = text[-1]
            punct = punct if punct in PUNCT else ''
            parts.append('<FORMULA>' + punct)
            parts.append(child_node.tail)

        else:
            parts.append(child_node.text)
            parts.append(child_node.tail)

    parts.append(node.tail)
    parts = filter(None, parts)
    parts = ''.join([part for part in parts])
    return parts


def do_processing(path, source):
    """
    Do the processing for a single file.

    Parameters
    ----------
    path : str
        The path to the file.
    source : {'pubmed', 'springer', 'arxiv'}
        The original source academic repository.

    Returns
    -------
    dict
        A dictionarty with text, source, and filename.
    """

    # for PubMed, just use the pubmed parser library
    if source == 'pubmed':
        doc = pp.parse_pubmed_paragraph(path, all_paragraph=True)

    # springer and arXiv have similar parsing rules, but for XML vs HTML
    elif source == 'springer':
        soup = lxml.html.parse(path)
        remove_namespace(soup)
        paragraphs = soup.xpath("//body//p[@class='Para']")
        doc = []
        for paragraph in paragraphs:
            doc.append({'text': stringify_children(paragraph)})

    elif source == 'arxiv':
        soup = etree.parse(path)
        remove_namespace(soup)
        paragraphs = soup.xpath("//body//p|//body//formula")
        doc = []
        for paragraph in paragraphs:
            doc.append({'text': stringify_children(paragraph)})

    # do the actual cleaning
    cleaned = clean_doc(doc)

    return {'text': cleaned, 'file': path, 'source': source}


def clean_documents(arxiv_dir=None,
                    pubmed_dir=None,
                    springer_dir=None,
                    output_file=None):
    """
    Clean all the documents.

    Parameters
    ----------
    arxiv_dir : str
        The directory where arXiv files live.
    pubmed_dir : str
        The directory where PubMed files live.
    springer_dir : str
        The directory where Springer files live.
    output_file : str
        The path to an output JSON file.
    """

    directories_dict = {'arxiv': arxiv_dir,
                        'pubmed': pubmed_dir,
                        'springer': springer_dir,
                        }

    directories_dict = {k: v for k, v in directories_dict.items() if v is not None}

    for directory in directories_dict.values():
        directory = abspath(directory)
        if not exists(directory):
            raise FileNotFoundError('The directory {} cannot be located.'.format(directory))

    docs_dict = defaultdict(list)
    for name, directory in directories_dict.items():
        for file in listdir(directory):
            if any(file.lower().endswith(ext) for ext in EXPECTED_EXTENSIONS[name]):
                docs_dict[name].append(file)

    final_docs_list = []
    for source, paths in docs_dict.items():
        print('Processing {}...'.format(source))
        for path in tqdm(paths):
            try:
                final_docs_list.append(do_processing(path))
            except Exception as error:
                print('Problem with {}: {}'.format(path, str(error)))
                final_docs_list.append({'text': None, 'file': path, 'source': source})
                continue

    if output_file is not None:
        with open(output_file, 'w') as fb:
            json.dump(final_docs_list, fb)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='clean_documents')

    parser.add_argument('path_to_springer', default=join(getcwd(), 'springer'))

    parser.add_argument('path_to_pubmed', default=join(getcwd(), 'pubmed'))

    parser.add_argument('path_to_arxiv', default=join(getcwd(), 'arxiv'))

    parser.add_argument('-o', '--output_file', default=join(getcwd(), 'docs.json'))

    args = parser.parse_args()

    clean_documents(args.path_to_arxiv,
                    args.path_to_pubmed,
                    args.path_to_springer,
                    args.output_file)
