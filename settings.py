from src import utilities
from pathlib import Path
from dotenv import load_dotenv
import os

# define search terms
terms = ['sir model', 'susceptible-infected-recovered', 'irSIR model']

# define data directories
data_dir = Path('.') /'data'

# raw directories
raw_dir = data_dir / 'raw'
arxiv_dir = raw_dir / 'arxiv' 
springer_dir = raw_dir / 'springer'
pubmed_dir = raw_dir / 'pubmed'

# interim / in-between directory
interim_dir = data_dir / 'interim'

# processed / finished data ready to be input into model
processed_dir = data_dir / 'processed'
metadata_dir = processed_dir / 'metadata'

# models directory
models_dir = Path('.') / 'models'

dirs = [data_dir, raw_dir, arxiv_dir, springer_dir, pubmed_dir, interim_dir
       ,processed_dir, metadata_dir, models_dir]

# make data directories if they don't already exist
for _dir in dirs:
    utilities.mkdir(_dir)
