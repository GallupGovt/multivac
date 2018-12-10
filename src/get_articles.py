from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import utilities
import settings
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from collections import OrderedDict
import copy
import re
import feedparser
import pubmed_parser
import time
import re
import pickle
import os
import feedparser


env_path = Path('.') / '.env'
load_dotenv(env_path)

# load environment variables from .env
springer_api_key = os.environ.get('SPRINGER_API_KEY')
ieee_api_key = os.environ.get('IEEE_API_KEY')
user_email = os.environ.get('USER_EMAIL')  # courtesy to NIH to include your email

wait_time = 3

# ------------------------------------------------------------------------------
# arxiv

def prep_terms(terms):
    return '+AND+'.join(['all:' + term for term in terms])


def get_total_number_of_results(url, params):
    xml_text = requests.get(url, params=params).text
    return int(bs(xml_text, 'lxml').find('opensearch:totalresults').contents[0])


def query_api(url, terms, params, wait_time=3, verbose=False):
    
    # get total number of results
    n_results = get_total_number_of_results(url, {'start': 0, 'max_results': 1})
    if verbose: 
        print('%s total results, %s second wait time between each call' % (str(n_results), str(wait_time)))
    
    # build list to iterate over
    starts = list(range(0, n_results, params['max_results']))  # start, stop, step

    metadata = []

    # iterate over list to get all results
    for ix, start in enumerate(starts):
        params_ = copy.deepcopy(params) 
        params_['start'] = start

        # ping api and retrieve xml for all articles in page
        xml_text = requests.get(url, params=params_).text

        # process xml page feed 
        page_feed = feedparser.parse(xml_text)
        entries = page_feed['entries']
        
        if ix == 0:
            metadata = entries
        else:
            metadata.extend(entries)
        time.sleep(wait_time)
    if verbose: print('')
    return metadata


# build query and get metadata of articles matching our search criteria
params = {'start': 0, 'max_results': 20, 'sortBy': 'relevance', 'sortOrder': 'descending'}
li = [x.replace('-', ' ').split(' ') for x in settings.terms]
q = 'OR'.join(['%28' + prep_terms(x) + '%29' for x in li])
url = 'http://export.arxiv.org/api/query?search_query=' + q
arxiv_metadata = query_api(url, q, params, verbose=True)

# save pdfs of articles that matched our search criteria
# we use doi as the filename when that id is present; otherwise we use the arxiv id
for ix, md in enumerate(arxiv_metadata):
    url = md['id']
    pdf_url = url.replace('/abs/', '/pdf/')
    fn = url.split('/abs/')[-1]
    fn = '_'.join(fn.split('/')) + '.pdf'
    arxiv_metadata[ix]['fn'] = fn  # specify filename so we can associate each pdf with its metadata down the road
    dst = settings.arxiv_dir / fn
    if not os.path.exists(dst):
        r = requests.get(pdf_url)
        with open(dst, 'wb') as f:
            f.write(r.content)
        time.sleep(0.3)
        
# save arxiv metadata
dst = settings.metadata_dir / 'arxiv.pkl'
with open(dst, 'wb') as f:
    pickle.dump(arxiv_metadata, f)

# ------------------------------------------------------------------------------
# springer

# build query to retrieve metadata
make_q = lambda li: '(' + ' OR '.join(['"' + s + '"' for s in li]) + ')'
q = make_q(settings.terms)
base = 'http://api.springernature.com/openaccess/json?q='
url = base + q
params = {'source': 'springer', 'openaccess': 'true', 'api_key': springer_api_key, 'p': 20, 's': 1}
params_ = copy.deepcopy(params)
# r = requests.get(url, params_)

# retrieve metadata
springer_metadata = []
while True:
    r = requests.get(url, params_)
    if len(r.json()['records']) == 0:
        break
    params_['s'] = params_['s'] + params_['p']
    springer_metadata += r.json()['records']
    time.sleep(wait_time)
print(len(springer_metadata))

# iterate over springer metadata and download html for each article
waits = (2**x for x in range(0,6))  # we use a generator to increase wait times with each connection error
for ix, md in enumerate(springer_metadata):
    fn = md['doi'].replace('/', '-')
    if len(fn) == 0:
        fn = md['identifier']
    fn = fn + '.html'
    springer_metadata[ix]['fn'] = fn
    dst = settings.springer_dir / fn
    if not os.path.exists(dst):
        try:
            r = requests.get(md['url'][0]['value'])
        except ConnectionError:
            time.sleep(waits.__next__)
            r = requests.get(md['url'][0]['value'])
        html = bs(r.text).encode('utf-8').decode('utf-8')
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(html)
        time.sleep(3)

# save springer metadata
dst = settings.metadata_dir / 'springer.pkl'
with open(dst, 'wb') as f:
    pickle.dump(springer_metadata, f)

# ------------------------------------------------------------------------------
# pubmed

# search pubmed central for free full text articles containing selected query

# get the ids which we then use to get the xml text data
replace = lambda s: s.replace(' ', '+')
quote = lambda s: '%22' + s + '%22'
terms = [quote(replace(s)) for s in settings.terms]
term = 'term='+ '%28'+ '+OR+'.join(terms) + '%29'
fulltext = 'free+fulltext%5bfilter%5d'
retmax = 'retmax=2000'
base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc'
params = {'retmax': 2000, 'email': user_email}
url = base + '&' + term + '+' + fulltext + '&' + retmax
r = requests.get(url)
ids = [x.contents[0] for x in bs(r.text).find_all('id')]

# get xml text data and save to disk
for i in ids:
    pmc_id = 'pmc' + str(i)
    fn = (pmc_id + '.xml')
    dst = settings.pubmed_dir / fn
    if not os.path.exists(dst):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=' + str(i)
        r = requests.get(url, params={'id': i})
        xml = r.text
        with open(dst, 'w') as f:
            f.write(xml)
        time.sleep(0.5)
