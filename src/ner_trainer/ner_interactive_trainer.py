from PyInquirer import prompt
import pandas as pd
import json
import argparse
import re
import os
import pickle
import numpy as np
import nltk

entity_question = [
    {
        'type': 'input',
        'name': 'entity_check',
        'message': 'Is there an entity in this sentence? Enter "y" for yes,'
        '"n" for no. Enter "p" if you would like a report of your progress'
    }
]

entity_user_response = [
    {
        'type': 'input',
        'name': 'entity_user_response',
        'message': 'Type entity string (must be exact). '
                   'If multiple, separate by "," (no spaces). '
                   'If there is there is no entity, type "n". '
    }
]

overwrite_pickle = [
    {
        'type': 'confirm',
        'name': 'overwrite_pickle',
        'message': 'there is an existing "training" pickle '
                   'file in your output. Would you like to continue '
                   'appending to that?'
    }
]


def analyze_chosen_sample(sent, label, progress_report):
    progress = 1
    while progress == 1:
        print('{}\n'.format(sent))
        answer_1 = prompt(entity_question)
        if answer_1['entity_check'].lower() == 'y':
            print('{}\n'.format(sent))
            answer_2 = prompt(entity_user_response)
            # If the response was an entry string
            if answer_2['entity_user_response'] != 'n':
                entity_strings = answer_2['entity_user_response'].split(',')
                match_indices = match_user_response(sent, entity_strings)
                training_output = format_output_spacy(sent, match_indices, label)
                return training_output
            # if response was 'n' for a no
            else:
                return
        # If the response was for a progress report
        elif answer_1['entity_check'].lower() == 'p':
            generate_progress_report(progress_report)
            progress = 1
        # if response was 'n' for a no
        else:
            return


def format_output_spacy(sent, match_indices, label):
    output = (sent,
              {'entities': [(match[0], match[1], label)
                            for match in match_indices]})
    return output


def generate_progress_report(progress_report):
    print('Progress Report: \n')
    if len(progress_report) == 0:
        print('No progress to report :( \n')
    else:
        print('Number of samples generated: {} \n'.format(
            progress_report['n_samples']
        ))
        print('Number of sentences viewed: {} \n'.format(
            progress_report['n_tries']
        ))


def incorrect_response_func(string):
    incorrect_response = [
        {
            'type': 'input',
            'name': 'incorrect_user_response',
            'message': 'The input string {} cannot be found. Type again,'
                       'or type "n" for no entity'.format(string)
        }
    ]
    return incorrect_response


def load_pickle_data(pickle_flag, output_dir):
    # If flagged, load in existing training file
    output_pickle = '{}/training_samples.pickle'.format(output_dir)
    if pickle_flag == 'true':
        previous_run = pickle.load(open(output_pickle, "rb"))
        progress_report = previous_run['metadata']
        training_samples = previous_run['training_samples']
    else:
        # if training pickle file exists, and user didn't specify, ask them if
        # they want to load it in, rather than overwrite
        if os.path.isfile(output_pickle):
            answer = prompt(overwrite_pickle)
            if answer['overwrite_pickle']:
                previous_run = pickle.load(open(output_pickle, "rb"))
                progress_report = previous_run['metadata']
                training_samples = previous_run['training_samples']
            else:
                training_samples = []  # initialize list that we add training samples to
                progress_report = {}
        else:
            training_samples = []  # initialize list that we add training samples to
            progress_report = {}
    return progress_report, training_samples


def match_user_response(sent, response):
    match_indices = []
    for string in response:
        matches = list(re.finditer(re.escape(string), sent))
        while len(matches) == 0:
            incorrect_response = incorrect_response_func(string)
            answer = prompt(incorrect_response)
            retry = answer['incorrect_user_response']
            if retry == 'n':
                break
            else:
                matches = list(re.finditer(re.escape(retry), sent))
        for match in matches:
            match_indices.append([match.start(), match.end()])

    return match_indices


def run_main(input, output_dir,
             n_samples, label, seed_file='none',
             pickle_file='false'):
    # Set up inputs necessary to run main function
    text_list, training_samples, sampling_index, \
    search_seeds, progress_report = set_up_inputs(input, output_dir,
                                                  seed_file,
                                                  pickle_file)

    # Start the training loop with a while loop that continues
    # until the number of specified training samples.
    if 'n_tries' in progress_report.keys():
        loop_count = progress_report['n_tries']
    else:
        loop_count = 0  # Initialize loop counter
    # Start sampling at top level of hierarchy, if any
    while len(training_samples) < n_samples:
        # Run a training instance - i.e. get a single training sample
        training_sample = run_training_instance(text_list,
                                                sampling_index,
                                                loop_count, search_seeds,
                                                label, progress_report)
        if training_sample is not None:
            training_samples.append(training_sample)
            write_output(training_samples, output_dir, progress_report)
        loop_count += 1
        progress_report = update_progress_report(progress_report,
                                                 loop_count,
                                                 training_samples,
                                                 sampling_index)


def run_training_instance(txt_list, sampler, loop_count,
                          search_seeds, label, progress_report):
    # grab first column - first column should be text, not
    # hierarchical level info - and select random sentence

    rand_sent = select_sentence_sample(
        txt_list[sampler[loop_count]]['text'], search_seeds)
    if rand_sent is None:
        return
    # analyze the chosen sample for any entities
    training_sample = analyze_chosen_sample(rand_sent, label, progress_report)

    return training_sample


def select_sentence_sample(text, search_seeds):
    # parse text into sentences with nltk
    if text is None:
        return
    text_sents = nltk.sent_tokenize(text)
    # identify sentences containing search seeds, or pull all
    if search_seeds is not None:
        positive_sents = [sent for sent in text_sents
                          if any([seed in sent for seed in search_seeds])]
    else:
        positive_sents = text_sents
    if len(positive_sents) == 0:
        return
    # select random sentence from selected sentences
    rand_select = np.random.choice(len(positive_sents))
    return positive_sents[rand_select]


def set_up_inputs(input, output_dir,
                  seed_file='none',
                  pickle_file='false'):
    # Ensure the user supplies a json file as input
    _, ext = os.path.splitext(input)
    if ext == '.json':
        text_json = json.load(open(input, 'r'))
    else:
        raise Exception('.json must be supplied as input')
    # Pull text from json into one list
    text_list = [text_json[key] for key in text_json.keys()]
    # If flagged, or already exists, get previous runs from pickle file
    progress_report, training_samples = load_pickle_data(pickle_file, output_dir)
    # simply take a permutation of all rows to randomly sample
    # create sampling index, if no previous run was loaded
    if 'sampling_index' not in progress_report.keys():
        sampling_index = np.random.permutation(len(text_json) - 1)
    else:
        sampling_index = progress_report['sampling_index']

    # Set search seeds to guide selection of sentences
    if seed_file == 'none':
        search_seeds = None
    else:
        with open(seed_file) as f:
            search_seeds = f.read().splitlines()

    return text_list, training_samples, sampling_index, search_seeds, progress_report


def update_progress_report(progress_report, loop_count, training_samples,
                           sampling_index):
    progress_report.update(
        {
            'n_tries': loop_count,
            'n_samples': len(training_samples),
            'sampling_index': sampling_index
        }
    )
    return progress_report


def write_output(training_samples, output_dir, progress_report):
    output_dict = {'metadata': progress_report,
                   'training_samples': training_samples}
    pickle.dump(output_dict,
                open('{}/training_samples.pickle'.format(output_dir), "wb"))


if __name__ == '__main__':
    """Retrieve list of text data and start CL interface to create 
    training examples for Named Entity Recognition model"""
    parser = argparse.ArgumentParser(description='Create training examples '
                                                 'for NER model')
    parser.add_argument('-i', '--input',
                        help='<Required> Path to .json file '
                             'containing training text OR path to'
                             'existing training model file',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to directory '
                             'to write output pickle file to',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--n_samples',
                        help='<Required> num of training samples to'
                             'pull from text',
                        required=True,
                        type=int)
    parser.add_argument('-e', '--entity_label',
                        help='<Required> name of the entity type you are '
                               'training the NER model for',
                        required=True,
                        type=str)
    parser.add_argument('-s', '--search_seeds',
                        help='Path to .txt file providing a set of seed strings'
                               'to guide selection of training samples. The .txt'
                               'file should have each string on a separate line',
                        required=False,
                        default='none',
                        type=str)
    parser.add_argument('-p', '--pickle_file',
                        help='If pre-trained existing file exists in '
                               'output directory, flag it here w/ "true',
                        required=False,
                        default='false',
                        choices=['true', 'false'],
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['input'], args_dict['output_dir'],
             args_dict['n_samples'],
             args_dict['entity_label'],
             args_dict['search_seeds'],
             args_dict['pickle_file'])


