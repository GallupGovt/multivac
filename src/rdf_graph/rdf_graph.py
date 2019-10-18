import fastcluster
import json
import numpy as np
import os
import re
import string
import pickle

# import xmltodict

# from collections import Counter
from corenlp import CoreNLPClient
from datetime import datetime
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from src.rdf_graph.rdf_parse import StanfordParser, stanford_parse
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

nlp = CoreNLPClient()
parser = StanfordParser(nlp)


class RDFGraph:
    def __init__(self, top_tfidf=20000, top_n_rel=None,
                 top_n_ent=None, clust_dist_thres=0.2, coref_opt=False,
                 openke_output_folder=os.curdir):
        '''Inputs:
        a) top_tfidf = number of top TF-IDF triples to use. To extract novel
               knowledge statements, we sort tuples by their mean TF-IDF scores and
               only extract the top TF-IDF tuples. This parameter controls how many
               of those you extract in decreasing order.
        b) top_n_rel = CoreNLP can sometimes pull out some relation verbs.
               Thus, it's a good idea to only use 'top' relations, in terms of the
               counts of that relation across all relation tuples. If you want to
               extract all, replace with: None.
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
        self.top_tfidf = top_tfidf
        self.top_n_rel = top_n_rel
        self.top_n_ent = top_n_ent
        self.clust_dist_thres = clust_dist_thres
        self.coref_opt = coref_opt
        self.openke_output_folder = openke_output_folder
        self.source_path = None

    @staticmethod
    def clean_out_html_tags(texts):
        tag_re = re.compile(r'<[^>]+>')

        if isinstance(texts, list):
            result = [
                tag_re.sub('', text).replace('\n', ' ') for text in
                texts
            ]
        else:
            result = tag_re.sub('', texts)

        return result

    def clear_texts(self):
        del(self.all_texts)
        self.all_texts = []

    def cluster_entities(self, embeddings_path, link_method='average'):
        embeddings_dict = self.load_embeddings(embeddings_path,
                                               self.unique_entities)
        # Create distance matrix (vector) using cosine similarity
        # between all entity strings
        embeddings_array = np.array([embedding for embedding in embeddings_dict.values()])
        dist_vec = pdist(embeddings_array, 'cosine')

        # Cluster distance matrix to find co-referring entities
        Z = fastcluster.linkage(dist_vec, method=link_method)
        cluster_labels = fcluster(Z, t=self.clust_dist_thres,
                                  criterion='distance')
        cluster_members_all = []

        entity_list = np.array([entity for entity in embeddings_dict.keys()])
        for clus_label in np.unique(cluster_labels):
            clus_indx = cluster_labels == clus_label
            cluster_members = list(entity_list[clus_indx])
            cluster_members_all.append(cluster_members)

        output = {'cluster_members': cluster_members_all,
                  'cluster_labels': cluster_labels,
                  'cluster_rep': self.get_cluster_representatives(cluster_members_all)}

        self.entity_cluster_results = output

    def extract_raw_tuples(self):
        if len(self.all_texts) == 0:
            self.load_texts()

        parser = StanfordParser()
        i = 0

        # Loop through articles or article batches:
        for Id, text in self.all_texts.items():
            print(i)
            # Start information extraction

            tuples = []
            sentences = sent_tokenize(text['text'])

            for sentence in sentences:
                try:
                    sentence = stanford_parse(parser, sentence)
                except:
                    print(sentence)
                    continue
                rdfs = [(' '.join(rel['subject']),
                         ' '.join(rel['relation']),
                         ' '.join(rel['object'])) for rel in
                         sentence.get_rdfs(use_tokens=False, how='list')]
                tuples.append(rdfs)

            self.all_tuples.update({Id: tuples})
            i += 1

    @staticmethod
    def filter_tuples(tuples, entities, relations):
        # Check if entity input is a nested list, meaning there are
        # multiple indictors of one entity from the clustering algorithm above
        if isinstance(entities, list):
            entity_dict = {entity: entity for entity in entities}
        elif isinstance(entities, dict):
            entity_dict = entities
        else:
            raise Exception('must pass list or dictionary (multiple strings per entity) for "entities" input')

        if isinstance(relations, list):
            relation_dict = {relation: relation for relation in relations}
        elif isinstance(relations, dict):
            relation_dict = relations
        else:
            raise Exception('must pass list or dictionary (multiple strings per relation) for "relations" input ')

        filtered_tuples = []
        for tuple_x in tuples:
            # Try to see if entity or relation of tuple exists in our entity or
            # relation lists. If not, will result in key error and we will move
            # on to next tuple
            try:
                print(tuple_x)
                tuple_filt = [entity_dict[tuple_x[0]],
                              relation_dict[tuple_x[1]],
                              entity_dict[tuple_x[2]]]
                filtered_tuples.append(tuple_filt)
            except KeyError:
                continue
        return filtered_tuples

    @staticmethod
    def get_cluster_representatives(cluster_members, char_limit=80):
        cluster_key = {}
        for cluster in cluster_members:
            # if the cluster contains more than one member
            if len(''.join(cluster)) > char_limit:
                # Just take the first cluster member
                cluster_key.update({entity: cluster[0]
                                    for entity in cluster})
            else:
                cluster_key.update({entity: ' | '.join(cluster)
                                    for entity in cluster})
        return cluster_key

    @staticmethod
    def get_unique_entities(tuples, top_n_ent):
        tuples_subj_obj = [[tuple_x[0], tuple_x[2]]
                           for tuple_x in tuples if len(tuple_x) > 1]
        tuples_subj_obj_flat = [tuple_x for tuple_pair in tuples_subj_obj
                                for tuple_x in tuple_pair]

        unique_entities = list(set(tuples_subj_obj_flat))
        # unique_entities = Counter(tuples_subj_obj_flat).most_common()
        # if top_n_ent is not None and isinstance(top_n_ent, int):
        #     if top_n_ent > len(unique_entities):
        #         max_entities = len(unique_entities)
        #     else:
        #         max_entities = top_n_ent
        #     unique_entities = [unique_entities[i]
        #                        for i in range(max_entities)]
        return unique_entities

    @staticmethod
    def get_unique_relations(tuples, top_n_rel):
        relations = [tuple_x[1] for tuple_x in tuples if len(tuple_x) > 1]

        unique_relations = list(set(relations))

        # unique_relations = Counter(relations).most_common()
        # if top_n_rel is not None and isinstance(top_n_rel, int):
        #     if top_n_rel > len(unique_relations):
        #         max_entities = len(unique_relations)
        #     else:
        #         max_entities = top_n_rel
        #     unique_relations = [unique_relations[i]
        #                         for i in range(max_entities)]
        return unique_relations

    def load_texts(self, src=None):
        '''
        Loads data into object from a JSON source file or existing dict.
        Result should be formatted as:
            {article_id: {"meta": dict, "text": str}}
        '''
        if isinstance(src, dict):
            self.all_texts = src
            self.source_path = None
        elif isinstance(src, str):
            with open(src, "r") as f:
                self.all_texts = json.load(f)
            self.source_path = src
        else:
            with open(self.source_path, "r") as f:
                self.all_texts = json.load(f)

        if isinstance(self.all_texts, list):
            self.all_texts = {next(iter(x.keys())): next(iter(x.values())) 
                              for x 
                              in self.all_texts}

    @staticmethod
    def load_embeddings(embeddings_path, entity_list):
        embeddings = pickle.load(open(embeddings_path, 'rb'))
        embeddings_dict = {token: vector for token, vector in
                           zip(embeddings['vocab'], embeddings['embeddings'])}
        # Compute avg embeddings for each entity
        entity_embeddings = {}
        for entity in entity_list:
            entity_split = entity.split()
            entity_split_embeddings = [embeddings_dict[token] for token in entity_split if token in embeddings_dict]
            if len(entity_split_embeddings) == 1:
                entity_embeddings.update({
                    entity: np.squeeze(np.array(entity_split_embeddings))
                })
            elif len(entity_split_embeddings) > 1:
                entity_embeddings.update({
                    entity: np.mean(np.array(entity_split_embeddings), axis=0)
                })
        return entity_embeddings

    def output_to_openke(self, timestamp=datetime.now()):
        relation_list = [relation[0] for relation in self.unique_relations]
        final_tuples = self.filter_tuples(self.tuples_preprocessed,
                                          self.entity_cluster_results['cluster_rep'],
                                          relation_list)
        self.final_tuples = final_tuples
        relation_temp = list(set([tuple_x[1] for tuple_x in
                                  self.final_tuples]))
        entity_temp = list(set([entity for tuple_x in self.final_tuples for
                                indx, entity in enumerate(tuple_x) if indx != 1]))
        relation_2_id = {relation: num_indx for num_indx, relation in
                         enumerate(relation_temp)}
        entity_2_id = {entity: num_indx for num_indx, entity in
                       enumerate(entity_temp)}
        tuples_2_id = [[entity_2_id[tuple_x[0]], relation_2_id[tuple_x[1]],
                        entity_2_id[tuple_x[2]]] for tuple_x in
                       self.final_tuples]
        self.final_tuple_ids = {'relation_2_id': relation_2_id,
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

    def preprocess_raw_tuples(self):
        # Temp - Remove tuples missing subject, predicate or object
        tuples = [self.all_tuples[key] for key in self.all_tuples.keys()]
        tuples_cleared = [tuple_x
                          for art in tuples
                          for sent in art
                          for tuple_x in sent
                          if all([token != '' for token in tuple_x])]
        self.all_tuples = tuples_cleared

        # Initialize NLTK Lemmatizer
        lemmatizer = WordNetLemmatizer()
        # 1. Get tf-idf scores for each tuple
        preprocessed_tuples = []

        for num_tuple, tuple_x in enumerate(self.all_tuples):
            tuple_x_clean = []

            for num, element in enumerate(tuple_x):
                # ensure the tuple_x is not empty
                if element is None:
                    continue
                word_tokens = word_tokenize(element.lower())
                # If relationship/verb, lemmatize and clean
                if num == 1:
                    word_tokens_clean = [lemmatizer.lemmatize(word, 'v') for
                                         word in word_tokens if word not in
                                         string.punctuation if word is not None]
                    # Don't allow relations longer than 4 tokens, if longer, break for-loop
                    if len(word_tokens_clean) > 4:
                        break
                    if len(word_tokens_clean) > 1:
                        # convert 'be running' to 'running' - i.e. remove 'is' verb from multi-word relations
                        word_tokens_clean = [word for word in word_tokens_clean
                                             if word != 'be']
                else:
                    word_tokens_temp = [word for word in word_tokens
                                        if word not in string.punctuation
                                        if word is not None]
                    # If the entity is a single word token, only allow nouns and proper nouns
                    if len(word_tokens_temp) == 1:
                        word_token_pos = pos_tag(word_tokens_temp)
                        word_tokens_clean = [pos_tuple[0] for pos_tuple in word_token_pos
                                             if 'NN' in pos_tuple[1]]
                    else:
                        word_tokens_clean = word_tokens_temp

                # If preprocessed element is empty, skip tuple
                if not word_tokens_clean:
                    break
                else:
                    tuple_x_clean.append(' '.join(word_tokens_clean))
            # If subject, relation and object fields exist, append to final
            # output
            if len(tuple_x_clean) == 3:
                preprocessed_tuples.append(tuple_x_clean)

        self.tuples_preprocessed = preprocessed_tuples
        self.unique_entities = self.get_unique_entities(preprocessed_tuples,
                                                        self.top_n_ent)
        self.unique_relations = self.get_unique_relations(preprocessed_tuples,
                                                          self.top_n_rel)

    def set_source(self, src):
        '''
        Accepts a path to a JSON source file and saves it in the object.
        Result should be formatted as:
            {article_id: {"meta": dict, "text": str}}
        '''

        if os.path.exists(src) and src.lower().endswith(".json"):
            self.source_path = src
        else:
            print("Invalid source file, ignored.")


