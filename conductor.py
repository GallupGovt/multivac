#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script conducts the entire flow of the multivac system to date. it has the
following flow:
1. collect data
    a. these data come from arxiv, springer, and pubmed in this instance, but
        could be modified to include more
    b. it saves the downloaded pdf's to a directory and creates a json object
        for further use
2. parse data
    a. the json objects that are saved from the collection step are processed
       for dependencies, input (word position), and morphology (lemma) [dim]
    b. it also identifies and notates equations throughout articles
"""
import argparse

from multivac.src.data.make import collect_main
from multivac.src.data.parsing import nlp_parse_main


def conduct(args_dict):
    # step 1: collect data
    collect_main()

    # step 2:
    nlp_parse_main(args_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse texts for natural '
                                     'language and equations.')
    parser.add_argument('-bp', '--nlp_bp', required=False, help='Which '
                        'document to start with', type=int)
    parser.add_argument('-js', '--nlp_newjson', action='store_true',
                        help='Boolean; indicates whether to create new JSON '
                        'file for glove embedding.')
    args_dict = vars(parser.parse_args())

    conduct(args_dict)
