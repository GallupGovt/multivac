import os
import argparse
import pickle
import re
import spacy

from numpy import random


def detect_ent_overlap(ent):
    entity_spans = ent[1]['entities']
    if len(entity_spans) == 1:
        return ent
    else:
        overlapping = [[x, y] for x in entity_spans for y in entity_spans
                       if x is not y and x[1] >= y[0] and x[0] <= y[0]]
        if len(overlapping) > 0:
            return None
        else:
            return ent


def trim_entity_spans(data):
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data


def run_main(input_pickle, output_dir):
    training_dict = pickle.load(open(input_pickle, 'rb'))
    training_data = training_dict['training_samples']
    training_data_no_overlap = [sample for sample in
                                list(map(detect_ent_overlap, training_data))
                                if sample is not None]
    training_data_final = trim_entity_spans(training_data_no_overlap)

    ner_spacy_model = train_spacy(training_data_final, 50)
    pickle.dump(ner_spacy_model,
                open('{}/ner_spacy_model.pickle'.format(output_dir), 'wb'))


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.load('en_core_web_sm')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.1,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


if __name__ == '__main__':
    """Train a SpaCy NER model on the supplied training examples"""
    parser = argparse.ArgumentParser(description='Run SpaCy NER model training on'
                                                 'supplied examples')
    parser.add_argument('-p', '--pickle_input',
                        help='<Required> Path to .pickle file containing'
                             'training examples (formatted for SpaCy input)',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to directory '
                             'to write trained model to',
                        required=True,
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['pickle_input'], args_dict['output_dir'])
