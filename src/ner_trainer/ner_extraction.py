import argparse
import json
import pickle
import spacy


def extract_entities(spacy_model, text):
    if text is not None:
        # If text is too long, it causes a ValueError
        if len(text) < 1000000:
            spacy_obj = spacy_model(text)
        else:
            return []
        ents = [str(ent) for ent in spacy_obj.ents]
        return ents


def load_text(file_path):
    text_json = json.load(open(file_path, 'r'))
    return text_json


def load_pretrained_model(model_path):
    pretrained_model = pickle.load(open(model_path, 'rb'))
    return pretrained_model


def run_main(model_path, file_path, output_dir):
    text_json = load_text(file_path)
    spacy_model = load_pretrained_model(model_path)
    text_json_keys = [key for key in text_json.keys()]
    for indx, key in enumerate(text_json_keys):
        print(indx)
        if 'meta' in text_json[key]:
            text_json[key]['abstract_ents'] = \
                extract_entities(spacy_model, text_json[key]['meta']['abstract'])
        else:
            text_json[key]['abstract_ents'] = []
        if 'text' in text_json[key]:
            text_json[key]['fulltext_ents'] = \
                extract_entities(spacy_model, text_json[key]['text'])
        else:
            text_json[key]['fulltext_ents'] = []
    with open(output_dir + '/data_entities.json', 'w') as fp:
        json.dump(text_json, fp)


if __name__ == '__main__':
    """Extract entities from corpus using trained spacy model"""
    parser = argparse.ArgumentParser(description='Extract entities using trained'
                                                 ' SpaCy NER model')
    parser.add_argument('-m', '--model_path',
                        help='<Required> Path to .pickle file containing'
                             'pretrained NER SpaCy model',
                        required=True,
                        type=str)
    parser.add_argument('-t', '--text_path',
                        help='<Required> Path to directory '
                             'to write trained model to',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to output directory'
                             ' to write to',
                        required=True,
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['model_path'], args_dict['text_path'],
             args_dict['output_dir'])