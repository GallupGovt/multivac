import argparse
import bs4
import datetime
import json
import os


def get_abstract(soup):

    abstract_element = soup.find('abstract').find('p')
    abstract_text = abstract_element.text if abstract_element else '**NONE**'

    # needs to be smarter
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

        # needs to be smarter
        if out.startswith('&amp;'):
            out = out.replace('&amp;', '').strip()

        authors.append(out)

    authors_list = list(set(authors))

    return authors_list


def get_content(soup):

    paragraph_elements = soup.find_all('p')
    paragraphs_list = [e.text for e in paragraph_elements]
    # potentially more cleaning here

    return paragraphs_list


def get_references(soup):

    reference_elements = soup.find_all('ref')
    references_list = [e.text for e in reference_elements]
    # potentially more cleaning here

    return references_list


def get_formulas(soup):

    formula_elements = soup.find_all('formula')
    formulas_list = [e.text for e in formula_elements]
    # potentially more cleaning here

    return formulas_list


def get_title(soup):

    title_element = soup.find('titleStmt')
    title = title_element.text.strip('\n')

    return title


def run(args_dict):

    indir = os.path.abspath(args_dict['indir'])

    # get all files in specified directory
    files = [x for x in os.walk(indir)][0][2]

    # temporary placeholder for all data
    complete_list = []
    for f in files:

        # full path to input file
        fin = f'{indir}/{f}'

        # only operate on proper files from extract_text module
        if fin.endswith('.tei.xml'):

            tmpf = open(fin, 'r')
            content = tmpf.read()
            tmpf.close()

            soup = bs4.BeautifulSoup(content, 'xml')

            # gather all parsed data
            abstract = get_abstract(soup)
            authors = get_authors(soup)
            references = get_references(soup)
            formulas = get_formulas(soup)
            title = get_title(soup)

            # comes in as list, combine to full text
            tmp_content = get_content(soup)
            content = ' '.join(tmp_content)
            raw_content = '////'.join(tmp_content)

            # cleaning fluff from main content
            for ref in references: content = content.replace(ref, '')
            for frm in formulas: content = content.replace(frm, '')
            for atr in authors: content = content.replace(atr, '')
            content = content.replace(abstract, '')

            structure = {
                f: {
                    'meta': {
                        'abstract': abstract,
                        'authors': authors,
                        'title': title
                    },
                    'text': content
                }
            }

            complete_list.append(structure)

        else:

            pass

    # file outpu handling
    outdir = os.path.abspath(args_dict['outdir'])
    stamp = datetime.datetime.now().strftime('%Y%M%d_%H%M%S')
    fname = f'output_{stamp}.json'
    fout = f'{outdir}/{fname}'

    f = open(fout, 'w')
    json.dump(complete_list, f)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser for XMLized scholarly publications."
    )
    parser.add_argument(
        "--indir",
        required=True,
        help="Path to the directory containing XMLs to process."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to output directory for processed files."
    )

    args_dict = vars(parser.parse_args())

    run(args_dict)
