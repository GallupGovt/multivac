#!/usr/bin/env python

"""
A script to perform post-processing and clean-up of questions.

:date: 1-25-2020
:author: Jeremy Biggs (jeremy_biggs@gallup.com)
:organization: Gallup
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

try:
    NLP = spacy.load('en_core_web_lg')
except OSError:
    print("No 'en_core_web_lg' model was found. Please download the "
          "model using the following command: `python -m spacy download`.")
    sys.exit()

REGEXES_LIST = [r'^\(?[a-zA-Z0-9\.\-]+\)\)?,?\s*',
                r'^[0-9\.\s\-]+\s*',
                r'^\*\s*',
                r'^[a-zA-Z]+\.\s+',
                r'^\)/s*',
                r'^P[0-9]{1,3}\s*',
                r'\s*(Numeric)?Citation*']
REGEXES_LIST = [re.compile(regex) for regex in REGEXES_LIST]

REGEXES_DICT = {'?': r'\s*\?$', r'\1': r'^.*?([a-zA-Z])'}
REGEXES_DICT = {k: re.compile(v) for k, v in REGEXES_DICT.items()}

REGEX_SENTENCES = re.compile(r'.*[.?!]\s+(.+[.?!])$')

REGEXES_LIST_FINAL = [r'^.+\([1-2][0-9]{3}\)']
REGEXES_LIST_FINAL = [re.compile(regex) for regex in REGEXES_LIST_FINAL]


def clean_text(text,
               min_char_len=2,
               min_word_threshold=0.20,
               max_word_len=100,
               removed_token="<REMOVED>"):
    """
    Clean the text by doing the following ::

        * Make sure everything is UTF-8, and replace any non-UTF-8 characters
        * Remove messy "prefix" text corresponding to certain regex patterns
        * Clean up "suffix" text (e.g. spaces before question marks) corresponding to certin regex patterns
        * Tokenize words, and then do the following comparison (this happens twice):
          - Count the number of tokens that are alpha numeric, greater than 2 charactres and spaCy vocabulary
          - If n_tokens / n_words < 0.2, remove the text altogether
        * Try to grab the last "sentence", given the sentence tokenizer and some regex rules.

    Parameters
    ----------
    text : str
        The text to clean.
    min_char_len : int, optional
        The minimum number of characters to consider,
        when determining whether to remove text.
        Defaults to 2.
    min_word_threshold : float, optional
        The proportion of actual words to consider,
        when determining whether to remove text.
        Defaults to 0.2.
    max_sent_len : int, optional
        The maximum number of words before we decided
        to take the last sentence.
        Defaults to 100.
    removed_token : str, optional
        A token to use in place of removed text.
        Defaults to '<REMOVED>'.

    Returns
    -------
    cleaned_text : str
        The cleaned text.
    """

    # before we do anything, make sure the text is UTF-8
    cleaned_text = text.encode('utf-8', 'replace')
    cleaned_text = cleaned_text.decode('utf-8')

    # remove everything in the `REGEXES_LIST`
    for regex in REGEXES_LIST:
        cleaned_text = re.sub(regex, '', cleaned_text)

    # remove everything in the `REGEXES_DICT`
    for update, regex in REGEXES_DICT.items():
        cleaned_text = re.sub(regex, update, cleaned_text)

    # we tokenize the text, and then check the number of words
    # in the vocabulary that are greater than `min_char_len`;
    # if this percentage is larger than `min_word_threshold`,
    # we keep the text; otherwise, we remove it
    tokens = NLP(cleaned_text)
    n_tokens = len(tokens)
    words = [token.text for token in tokens
             if token.is_alpha
             and len(token) > min_char_len
             and token.text.lower() in NLP.vocab]

    n_words = len(words)
    if (n_words / n_tokens) < min_word_threshold:
        return removed_token

    # if the number of tokens is larger than `max_word_len`,
    # we only take the last sentence from spaCy; otherwise, we find the
    # punctuation and take the last sentence using a regular expression
    if n_tokens >= max_word_len:
        cleaned_text = list(tokens.sents)[-1].text

    else:
        sents = re.findall(REGEX_SENTENCES, cleaned_text)
        if sents:
            cleaned_text = sents[-1].strip()

    # remove everything in the `REGEXES_LIST_FINAL`
    for regex in REGEXES_LIST_FINAL:
        cleaned_text = re.sub(regex, '', cleaned_text)

    # check one last time, to make sure the sentence isn't too short
    tokens = NLP(cleaned_text)
    if len(tokens) <= min_char_len:
        return removed_token

    # finally, we strip off any leading or ending white space and make
    # sure that the first letter is capitalized
    cleaned_text = cleaned_text.strip()
    cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]

    return cleaned_text


def main():
    """
    Run the script from the command line.
    """

    parser = argparse.ArgumentParser(prog='clean_questions')

    parser.add_argument('-f', '--force', dest='force_write',
                        action='store_true', default=False,
                        help="If true, rsmtool will not check if the "
                             "output directory already contains the "
                             "output of another rsmtool experiment. ")

    parser.add_argument('input_file',
                        help="The path to the input file (.txt). "
                             "Should have one line per question.")

    parser.add_argument('output_file',
                        help="The path to the output file (.txt or .xlsx). "
                             "If .txt, then write out the cleaned text. "
                             "If .xlsx, then write out both original and cleaned text.")

    parser.add_argument('-c', '--min_char_len', dest='min_char_len',
                        type=int, default=2,
                        help="The minimum number of characters to consider, "
                             "when determining whether to remove text")

    parser.add_argument('-t', '--min_word_threshold', dest='min_word_threshold',
                        type=float, default=0.20,
                        help="The proportion of actual words to consider, "
                             "when determining whether to remove text.")

    parser.add_argument('-w', '--max_word_len', dest='max_word_len',
                        type=int, default=100,
                        help="The maximum number of words before we decided "
                             "to take the last sentence.")

    parser.add_argument('-r', '--removed_token', dest='removed_token',
                        default="<REMOVED>",
                        help="A token to use in place of removed text.")

    args = parser.parse_args()

    path_to_questions = Path(args.input_file)

    error_msg = f'The `input_file`, {args.input_file}, does not exist.'
    assert path_to_questions.exists(), error_msg

    with open(path_to_questions) as fb:
        questions = [line.strip() for line in fb.readlines()]

    # loop through all of the questions and perform cleaning
    cleaned_questions = []
    for question in tqdm(questions):
        cleaned_questions.append(clean_text(question,
                                            min_char_len=args.min_char_len,
                                            min_word_threshold=args.min_word_threshold,
                                            max_word_len=args.max_word_len,
                                            removed_token=args.removed_token))

    # if the output file is Excel, then write out a formatted file
    # with both the original and cleaned questions in separate columns
    if args.output_file.lower().endswith('xlsx'):

        df_output = pd.DataFrame({'original': questions, 'cleaned': cleaned_questions})

        writer = pd.ExcelWriter(args.output_file, engine='xlsxwriter')
        df_output.to_excel(writer, sheet_name='questions', index=False, encoding='utf-8')

        workbook = writer.book
        worksheet = writer.sheets['questions']
        formatting = workbook.add_format({'text_wrap': True, 'font_size': 18})

        worksheet.set_column('A:A', 100, formatting)
        worksheet.set_column('B:B', 100, formatting)
        worksheet.freeze_panes(1, 0)

        writer.save()

    # otherwise, just put the cleaned questions in each row
    else:

        df_output = pd.DataFrame({'questions': cleaned_questions})
        df_output.to_csv(args.output_file, index=False, header=False)


if __name__ == '__main__':

    main()
