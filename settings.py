from multivac.src import utilities
from pathlib import Path
from dotenv import load_dotenv
import os

# define search terms
terms = ['sir model', 'susceptible-infected-recovered', 'irSIR model'
        ,'susceptible-infected', 'seir model'
        ,'susceptible-exposed-infected-recovered']

# specify sources
sources = ['arxiv', 'pubmed', 'springer']

root_dir = Path('.') / 'multivac'

# define data directories
data_dir = root_dir / 'data'

# raw directories
raw_dir = data_dir / 'raw'

# interim / in-between directory
interim_dir = data_dir / 'interim'

# processed / finished data ready to be input into model
processed_dir = data_dir / 'processed'
metadata_dir = processed_dir / 'metadata'

# models directory
models_dir = root_dir / 'models'

# make data directories if they don't already exist
dirs = [data_dir, raw_dir, interim_dir, processed_dir, metadata_dir, models_dir]
dirs += [raw_dir / x for x in sources]
for _dir in dirs:
    utilities.mkdir(_dir)
