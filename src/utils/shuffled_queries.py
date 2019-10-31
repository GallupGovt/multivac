import argparse
import numpy as np
import pandas as pd
from random import shuffle

def run(args_dict):
	with open('{}/{}'.format(args_dict['output_dir'], args_dict['file'])) as f:
		clean_txt = f.readlines()
		f.close()

	df_clean = pd.DataFrame(clean_txt)
	df_clean.columns = ['query']
	# clean queries will contain a label of 1
	df_clean['label'] = 1

	# Tokenizing and shuffling each query
	# These will be labeled as 0
	shuffles = []
	for txt in clean_txt:
		txt = txt.split(" ")
		shuffle(txt)
		shuffles.append(" ".join(txt))

	shuffled_df = pd.DataFrame()
	shuffled_df['query'] = shuffles
	shuffled_df['label'] = 0

	final_df = pd.concat([df_clean, shuffled_df]).reset_index()
	final_df.rename(columns={'index': 'id'}, inplace=True)
	final_df['query'] = final_df['query'].apply(lambda x: x.replace("\n", ""))

	np.savetxt("{}/extracted_questions_labels.txt".format(
				args_dict['output_dir']), final_df.values, newline='\n', 
				fmt=["%s", "%s", "%s"], header='id query label')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create shuffled queries'
    								'from .txt file for GAN.')
    parser.add_argument('-f', '--file', required=True, 
    					help='Name of .txt file')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory to dump final output')

    args_dict = vars(parser.parse_args())

    run(args_dict)

