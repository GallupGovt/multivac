import argparse
import bs4
import json
import os


def get_abstract(soup):

    abstract_element = soup.find('abstract').find('p')
    abstract_text = abstract_element.text if abstract_element is not None else '**NONE**'

    # some cleanup.. this is not smart, rather very naive
    if 'Abstract' in abstract_text:
        abstract_text = abstract_text.replace('Abstract', '').strip()
    if abstract_text.startswith('.'):
        abstract_text = abstract_text[1:].strip()

    return abstract_text


def get_authors(soup):

    author_elements = soup.find_all('author')

    authors = []
    for author in author_elements:
        firstname = author.find('forename')
        lastname = author.find('surname')

        first = firstname.text if firstname else '**NONE**'
        last = lastname.text if lastname else '**NONE**'
        out = f'{first} {last}'

        # some cleanup.. this is not smart, rather very naive
        if out.startswith('&amp;'):
            out = out.replace('&amp;', '').strip()

        authors.append(out)

    authors_list = list(set(authors))

    return authors_list


def get_content(soup):

    paragraph_elements = soup.find_all('p')

    paragraphs_list = []
    for para in paragraph_elements:
        paragraphs_list.append(para.text)

    return paragraphs_list


def get_references(soup):

    reference_elements = soup.find_all('ref')

    references_list = []
    for ref in reference_elements:
        references_list.append(ref.text)

    return references_list


def get_formulas(soup):

    formula_elements = soup.find_all('formula')

    formulas_list = []
    for formula in formula_elements:
        formulas_list.append(formula.text)

    return formulas_list


def run(args_dict):

    fin = open(args_dict['infile'], 'r')
    content = fin.read()
    fin.close()

    soup = bs4.BeautifulSoup(content, 'xml')

    tmp_content = get_content(soup)
    content = ' '.join(tmp_content)
    raw_content = '////'.join(tmp_content)

    abstract = get_abstract(soup)
    authors = get_authors(soup)
    references = get_references(soup)
    formulas = get_formulas(soup)

    # need to strip all the fluff from the
    # cleaned content
    for ref in references:
        content = content.replace(ref, '')

    for frm in formulas:
        content = content.replace(frm, '')

    for atr in authors:
        content = content.replace(atr, '')

    content = content.replace(abstract, '')

    data = {
        'abstract': abstract,
        'authors': authors,
        'clean_content': content,
        'raw_content': raw_content,
        'references': references,
        'formulas': formulas
    }

    outdir = os.path.abspath(args_dict['outdir'])

    tmp = args_dict['infile'].split('/')[-1]
    fname = f'{tmp}.json'
    fout = f'{outdir}/{fname}'

    f = open(fout, 'w')
    json.dump(data, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser for XMLized scholarly publications."
    )
    parser.add_argument(
        "--infile",
        required=True,
        help="Path to the directory containing XML to process."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to output directory for processed files."
    )

    args_dict = vars(parser.parse_args())

    run(args_dict)
