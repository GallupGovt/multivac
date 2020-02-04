import json
import numpy as np
import os

from collections import Counter
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize


class RDFGraph:
    def __init__(self, src_path,
                 openke_output_folder=os.curdir,
                 verbose=False):
        '''Inputs:
        a) src_path = the path to the corpus file
        c) top_n_ent = the same interpretation as 'top_n_rel', except for
               entities (i.e. subjects & objects)
        d) clust_dist_clust = the minimum 'correlation' distance in the
               agglomerative, hiearchical clustering algorithm for two clusters to
               be joined. This is intentionally set at a very low distance
               threshold, which provides a more conservative clustering approach.
        e) openke_output_folder = the output folder to output text files for
               OpenKE input
         '''

        # Define Inputs
        self.all_texts = {}
        self.all_tuples = {}
        self.openke_output_folder = openke_output_folder
        self.verbose = verbose
        self.set_source(src_path)

    def build_graph(self):
        self.all_texts = self.load_texts(self.source_path)
        self.article_tuples = self.get_article_tuples()
        self.unique_entities = self.get_unique_entities(self.article_tuples)
        self.unique_relations = self.get_unique_relations(self.article_tuples)
        self.clear_texts()

    def clear_texts(self):
        del(self.all_texts)
        self.all_texts = []

    def get_article_tuples(self):
        article_tuples = [self.all_texts[key]['fulltext_ents']
                          for key in self.all_texts.keys()]
        article_tuples = [tuple_list for tuple_list in article_tuples
                          if tuple_list is not None]
        tuple_list = [tuple_x[0] for entity_list in article_tuples
                      for tuple_x in entity_list if tuple_x is not None]
        return tuple_list

    @staticmethod
    def get_unique_entities(tuple_list):
        entity_pairs = [[tuple_x[0], tuple_x[2]] for tuple_x in tuple_list]
        entity_list = [entity for pair in entity_pairs for entity in pair]
        unique_entities = Counter(entity_list).most_common()
        unique_entities = [ent[0] for ent in unique_entities]
        return unique_entities

    @staticmethod
    def get_unique_relations(tuple_list):
        relation_list = [tuple_x[1] for tuple_x in tuple_list]
        unique_relations = Counter(relation_list).most_common()
        unique_relations = [rel[0] for rel in unique_relations]
        return unique_relations

    def load_texts(self, src=None):
        '''
        Loads data into object from a JSON source file or existing dict.
        Result should be formatted as:
            {article_id: {"meta": dict, "text": str}}
        '''
        if isinstance(src, dict):
            all_texts = src
        elif isinstance(src, str):
            with open(src, "r") as f:
                all_texts = json.load(f)
        else:
            with open(self.source_path, "r") as f:
                all_texts = json.load(f)

        if isinstance(all_texts, list):
            all_texts = {next(iter(x.keys())): next(iter(x.values()))
                         for x in all_texts}
        return all_texts

    def output_to_openke(self, timestamp=datetime.now()):
        relation_2_id = {relation: num_indx for num_indx, relation in
                         enumerate(self.unique_relations)}
        entity_2_id = {entity: num_indx for num_indx, entity in
                       enumerate(self.unique_entities)}
        tuples_2_id = [[entity_2_id[tuple_x[0]], relation_2_id[tuple_x[1]],
                        entity_2_id[tuple_x[2]]] for tuple_x in
                       self.article_tuples]
        final_tuple_ids = {'relation_2_id': relation_2_id,
                           'entity_2_id': entity_2_id,
                           'tuples_2_id': tuples_2_id}

        with open('{}/train2id.{}.txt'.format(self.openke_output_folder,
                                              timestamp), 'w') as f:
            line = '{}\n'.format(str(len(tuples_2_id)))
            f.write(line)
            for tuple_x in tuples_2_id:
                line = "{}\t{}\t{} \n".format(tuple_x[0], tuple_x[2],
                                              tuple_x[1])
                f.write(line)

        with open('{}/entity2id.{}.txt'.format(self.openke_output_folder,
                                               timestamp), 'w') as f:
            line = '{}\n'.format(str(len(entity_2_id)))
            f.write(line)
            for ls in [[ent, indx] for ent, indx in entity_2_id.items()]:
                line = "{}\t{} \n".format(ls[0], ls[1])
                f.write(line)

        with open('{}/relation2id.{}.txt'.format(self.openke_output_folder,
                                                 timestamp), 'w') as f:
            line = '{}\n'.format(str(len(relation_2_id)))
            f.write(line)
            for ls in [[rel, indx] for rel, indx in relation_2_id.items()]:
                line = "{}\t{} \n".format(ls[0], ls[1])
                f.write(line)

    def set_source(self, src):
        '''
        Accepts a path to a JSON source file and saves it in the object.
        Result should be formatted as:
            {article_id: {"meta": dict, "text": str}}
        '''

        if os.path.exists(src) and src.lower().endswith(".json"):
            self.source_path = src
        else:
            self.source_path = None
            print("Invalid source file, ignored.")
