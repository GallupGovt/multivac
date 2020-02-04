import argparse
import json
import pickle
import spacy
from OpenNRE import opennre


def extraction(spacy_model, extractor, text):
    if text is not None:
        if len(text) >= 1000000:
            return None
        spacy_obj = spacy_model(text)
        ent_rel_tuples = []
        for sent in spacy_obj.sents:
            if len(sent.ents) > 0:
                tuples = sent_relation_extraction(sent, extractor)
                ent_rel_tuples.append(tuples)
        ent_rel_tuples_flat = [tuple_x for sent_x in ent_rel_tuples for tuple_x in sent_x]
        return ent_rel_tuples_flat


def extract_relation(extractor, sent, entity1_span, entity2_span, thres=0.6):
    h_pos = (entity1_span.start_char + sent.start_char, (entity1_span.end_char + sent.start_char) - 1)
    t_pos = (entity2_span.start_char + sent.start_char, (entity2_span.end_char + sent.start_char) - 1)
    result = extractor.infer({'text': str(sent), 'h': {'pos': h_pos}, 't': {'pos': t_pos}})
    if result[1] > thres:
        tuple_x = [(str(entity1_span), result[0], str(entity2_span)), result[1]]
        return tuple_x


def load_text(file_path):
    text_json = json.load(open(file_path, 'r'))
    return text_json


def load_pretrained_ner_model(model_path):
    pretrained_model = pickle.load(open(model_path, 'rb'))
    return pretrained_model


def load_pretrained_extractor_model():
    extractor = opennre.get_model('wiki80_cnn_softmax')
    return extractor


def run_main(model_path, file_path, output_dir):
    text_json = load_text(file_path)
    spacy_model = load_pretrained_ner_model(model_path)
    extractor = load_pretrained_extractor_model()
    text_json_keys = [key for key in text_json.keys()]
    for indx, key in enumerate(text_json_keys):
        print(indx)
        if 'meta' in text_json[key]:
            text_json[key]['abstract_ents'] = \
                extraction(spacy_model, extractor, text_json[key]['meta']['abstract'])
        else:
            text_json[key]['abstract_ents'] = []
        if 'text' in text_json[key]:
            text_json[key]['fulltext_ents'] = \
                extraction(spacy_model, extractor, text_json[key]['text'])
        else:
            text_json[key]['fulltext_ents'] = []
    with open(output_dir + '/data_entities.json', 'w') as fp:
        json.dump(text_json, fp)


def sent_relation_extraction(sent, extractor):
    sent_ents = sent.ents
    ent_pairs = zip(sent_ents, sent_ents[1:] + sent_ents[:1])
    tuples = []
    for ent_pair in ent_pairs:
        tuple_x = extract_relation(extractor, sent, ent_pair[0], ent_pair[1])
        tuples.append(tuple_x)
    return tuples


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
                        help='<Required> Path to corpus text file',
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