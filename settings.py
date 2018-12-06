from src import utilities
from pathlib import Path
from dotenv import load_dotenv
import os


# define data directories
data_dir = Path('.') /'data'
raw_dir = data_dir / 'raw'
interim_dir = data_dir / 'interim'
processed_dir = data_dir / 'processed'

# make data directories if they don't already exist
for _dir in [raw_dir, interim_dir, processed_dir]:
    utilities.mkdir(_dir)