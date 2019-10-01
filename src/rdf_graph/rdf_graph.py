import fastcluster
import numpy as np
import os
import re
import string
import xmltodict
import yake

from Bio import Entrez
from collections import Counter
from corenlp import CoreNLPClient
from functools import partial
from Levenshtein import ratio
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = CoreNLPClient()

class rdf_graph:
  def __init__(self, max_entail = 1, top_tfidf =  20000, top_n_rel = 50,
                 top_n_ent = 20000, clust_dist_thres = 0.2, coref_opt = False,
                 openke_output_folder = os.curdir):
    '''Inputs:
    a) max_entail = max entailments per sentence clause, this tells CoreNLP
           how many relation triples are extracted from one sentence. CoreNLP
           tends to extract A LOT of redundant triples from one sentence, so
           keep this number low.
    b) top_tfidf = number of top tf-idf triples to use. To extract novel
           knowledge statements, we sort tuples by their mean tf-idf scores and
           only extract the top tfidf tuples. This parameter controls how many
           of those you extract in decreasing order.
    c) top_n_rel = CoreNLP can sometimes pull out some relation verbs.
           Thus, it's a good idea to only use 'top' relations, in terms of the
           counts of that relation across all relation tuples. If you want to
           extract all, replace with: None.
    d) top_n_ent = the same interpretation as 'top_n_rel', excet for
           entities (i.e. subjects & objects)
    e) clust_dist_clust = the minimum 'correlation' distance in the
           agglomerative, hiearchical clustering algorithm for two clusters to
           be joined. This is intentionally set at a very low distance
           threshold, which provides a more conservative clustering approach.
    f) coref_opt = this option implements 'coreference resolution: "Taylor
           is smart, and he is kind' - "he" would be replaced with "Taylor."
           This option is set to false, as it significantly slows down the
           CoreNLP extraction algorihm (over two times).
    g) openke_output_folder = the output folder to output text files for
           OpenKE input
     '''

    # Define Inputs
    self.max_entail = max_entail
    self.top_tfidf = top_tfidf
    self.top_n_rel = top_n_rel
    self.top_n_ent = top_n_ent
    self.clust_dist_thres = clust_dist_thres
    self.coref_opt = coref_opt
    self.openke_output_folder = openke_output_folder

  @staticmethod
  def clean_out_html_tags(abstracts):
    tag_re = re.compile(r'<[^>]+>')
    if isinstance(abstracts,list):
        result = [
          tag_re.sub('', abstract).replace('\n',' ') for abstract in
          abstracts
        ]
    else:
      result = tag_re.sub('', abstracts)

    return result

  def cluster_entities(self, link_method = 'average'):
      entity_list = np.array([entity[0] for entity in self.unique_entities])
      #Create distance matrix (vector) using Levenshtein distance between all entity strings
      dist_vec = pdist(entity_list.reshape(-1,1),
                      lambda u,v: 1 - ratio(u[0],v[0]))
      # Cluster distance matrix to find co-referring entities
      Z = fastcluster.linkage(dist_vec, method=link_method)
      cluster_labels = fcluster(Z, t = self.clust_dist_thres, 
                                criterion='distance')
      cluster_members_all = []
      for clus_label in np.unique(cluster_labels):
          clus_indx = cluster_labels==clus_label
          cluster_members = list(entity_list[clus_indx])
          cluster_members_all.append(cluster_members)
      output = {'cluster_members': cluster_members_all, 
      'cluster_labels': cluster_labels,
      'cluster_rep': self.get_cluster_representatives(cluster_members_all)}
      self.entity_cluster_results = output

  @staticmethod
  def compute_tuple_tfidf_scores(abstracts,tuples):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(abstracts)
    token_2_indx = {token:  np.int(num_indx) for num_indx,token 
                    in enumerate(vectorizer.get_feature_names())}
    tfidf_tuple_scores = []
    for indx, abstract_tuples in enumerate(tuples):
        if abstract_tuples is not None:
          abstract_tuples_flat = [tuple_x for sent in abstract_tuples 
                                for tuple_x in sent]
          tfidf_vector = tfidf_matrix[indx,:]
          for tuple_x in abstract_tuples_flat:
              subj_indx = [token_2_indx[token.lower()] 
                    for token in tuple_x[0].split(' ') 
                    if token.lower() in token_2_indx.keys()]
              subj_score = np.mean(tfidf_vector[:,subj_indx])
              obj_indx = [token_2_indx[token.lower()] 
                    for token in tuple_x[2].split(' ') 
                    if token.lower() in token_2_indx.keys()]
              obj_score = np.mean(tfidf_vector[:,obj_indx])
              tfidf_score = np.mean(np.array([subj_score,obj_score]))
              tfidf_tuple_scores.append([tuple_x, tfidf_score])

    tfidf_tuple_scores_sorted = sorted(tfidf_tuple_scores, 
                                      key = lambda x: x[1], reverse=True)
    return tfidf_tuple_scores_sorted

    def extract_raw_tuples(self, confidence_thres=1, coref_opt=False):
    all_abstracts = self.all_abstracts
    max_entail = self.max_entail
    all_tuples = []

    # Loop through articles or article batches:
    for num, abstract in enumerate(all_abstracts):
        if num%100==0:
            print('{} articles out of {} processed'
                          .format(num, len(all_abstracts)))
        abstract_tuples = []

        if coref_opt==True:
            annots = "tokenize ssplit pos depparse natlog openie dcoref".split()
            properties={"openie.triple.strict": "true",
                        "output_format": "xml",
                        "openie.max_entailments_per_clause": str(max_entail),
                        "openie.openie.resolve_coref": "true"}

            nlp.annotators = annots
            nlp.properties = properties
            xml_parsed_doc = nlp.annotate(abstract)
        else:
            annots = "tokenize ssplit pos depparse natlog openie".split()
            properties={"outputFormat": "xml",
                        "openie.triple.strict": "true",
                        "openie.max_entailments_per_clause": str(max_entail)}

            nlp.annotators = annots
            nlp.properties = properties
            xml_parsed_doc = nlp.annotate(abstract)

        if 'Request is too long to be handled' in xml_parsed_doc:
            raise Exception('Article batch is too large for parser, need to decrease batch size')
        if 'CoreNLP request timed out' in xml_parsed_doc:
            print('article parse failed')
            all_tuples.append(None)
            continue
        json_parsed_doc = xmltodict.parse(xml_parsed_doc)

        try:
            sent_level_doc = json_parsed_doc['root']['document']['sentences']['sentence']
        except TypeError:
            print('article parse failed')
            all_tuples.append(None)
            continue

        for indx, sent in enumerate(sent_level_doc):
            if not isinstance(sent,str):
              if not isinstance(sent['openie'],str) and sent['openie'] is not None:
                if isinstance(sent['openie']['triple'],list):
                  abstract_tuples.append(
                                    [(rel['subject']['text'],
                                      rel['relation']['lemma'],
                                      rel['object']['text']) for rel in
                                     sent['openie']['triple'] if rel is not None if
                                     np.float(rel['@confidence'])>=confidence_thres]
                                )
                elif isinstance(sent['openie']['triple'],dict):
                  single_triple = sent['openie']['triple']
                  if single_triple is not None and np.float(single_triple['@confidence'])>=confidence_thres:
                    abstract_tuples.append(
                                        [(single_triple['subject']['text'],
                                          single_triple['relation']['lemma'],
                                          single_triple['object']['text'])]
                                    )
        all_tuples.append(abstract_tuples)

    self.all_tuples = all_tuples

    return all_tuples

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

      if isinstance(relations,list):
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
              tuple_filt = [entity_dict[tuple_x[0]],
                            relation_dict[tuple_x[1]],
                            entity_dict[tuple_x[2]]]
              filtered_tuples.append(tuple_filt)
          except KeyError:
              continue
      return filtered_tuples

  @staticmethod
  def get_cluster_representatives(cluster_members,char_limit=80):
    cluster_key = {}
    for cluster in cluster_members:
        # if the cluster contains more than one member
        if len(''.join(cluster))>char_limit:
          # Just take the first cluster member
            cluster_key.update({entity: cluster[0] 
              for entity in cluster})
        else:
            cluster_key.update({entity: ' | '.join(cluster) 
              for entity in cluster})
    return cluster_key

  @staticmethod
  def get_unique_entities(tuples, top_n_ent):
    tuples_subj_obj = [[tuple_x[0],tuple_x[2]] 
                      for tuple_x in tuples if len(tuple_x)>1]
    tuples_subj_obj_flat = [tuple_x for tuple_pair in tuples_subj_obj 
                          for tuple_x in tuple_pair]
    unique_entities = Counter(tuples_subj_obj_flat).most_common()
    if top_n_ent is not None and isinstance(top_n_ent,int):
      if top_n_ent > len(unique_entities):
        max_entities = len(unique_entities)
      else:
        max_entities = top_n_ent
      unique_entities = [unique_entities[i] 
                        for i in range(max_entities)]
    return unique_entities

  @staticmethod
  def get_unique_relations(tuples, top_n_rel):
    relations = [tuple_x[1] for tuple_x in tuples if len(tuple_x)>1]
    unique_relations = Counter(relations).most_common()
    if top_n_rel is not None and isinstance(top_n_rel,int):
      if top_n_rel > len(unique_relations):
        max_entities = len(unique_relations)
      else:
        max_entities = top_n_rel
      unique_relations = [unique_relations[i] 
                        for i in range(max_entities)]
    return unique_relations

  def output_to_openke(self, timestamp):
      relation_list = [relation[0] for relation in self.unique_relations]
      final_tuples = self.filter_tuples(self.tuples_preprocessed,
                    self.entity_cluster_results['cluster_rep'],
                    relation_list)
      self.final_tuples = final_tuples
      relation_temp = list(set([tuple_x[1] for tuple_x in
                               self.final_tuples]))
      entity_temp = list(set([entity for tuple_x in self.final_tuples for
                             indx,entity in enumerate(tuple_x) if indx!=1]))
      relation_2_id = {relation: num_indx for num_indx,relation in
                       enumerate(relation_temp)}
      entity_2_id = {entity: num_indx for num_indx,entity in
                     enumerate(entity_temp) }
      tuples_2_id = [[entity_2_id[tuple_x[0]], relation_2_id[tuple_x[1]],
                     entity_2_id[tuple_x[2]]] for tuple_x in
                     self.final_tuples]
      self.final_tuple_ids = {'relation_2_id' : relation_2_id,
              'entity_2_id' : entity_2_id,
              'tuples_2_id' : tuples_2_id}

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
          for ls in [[ent,indx] for ent,indx in entity_2_id.items()]:
              line = "{}\t{} \n".format(ls[0], ls[1])
              f.write(line)

      with open('{}/relation2id.{}.txt'.format(self.openke_output_folder,
                                               timestamp), 'w') as f:
          line = '{}\n'.format(str(len(relation_2_id)))
          f.write(line)
          for ls in [[rel,indx] for rel,indx in relation_2_id.items()]:
              line = "{}\t{} \n".format(ls[0], ls[1])
              f.write(line)

  def preprocess_raw_tuples(self):
      # Initialize NLTK Lemmatizer
      lemmatizer = WordNetLemmatizer()
      # 1. Get tf-idf scores for each tuple
      self.tuple_tfidf_scores = self.compute_tuple_tfidf_scores(self.all_abstracts,
                                                      self.all_tuples)
      preprocessed_tuples = []
      for num_tuple, tuple_x in enumerate(self.tuple_tfidf_scores):
          tuple_x_clean = []
          # Only pull the top # tuples as set by 'tfidf'
          if num_tuple >= self.top_tfidf:
              break

          for num, element in enumerate(tuple_x[0]):
            # ensure the tuple_x is not empty
              if element is None:
                  continue
              word_tokens = word_tokenize(element.lower())
              # If relationship/verb, lemmatize and clean
              if num==1:
                  word_tokens_clean = [lemmatizer.lemmatize(word,'v') for
                                      word in word_tokens if word not in
                                      string.punctuation if word is not None]
                  # Don't allow relations longer than 4 tokens, if longer, break for-loop
                  if len(word_tokens_clean)> 4:
                    break
                  if len(word_tokens_clean)>1:
                    #convert 'be running' to 'running' - i.e. remove 'is' verb from multi-word relations
                    word_tokens_clean = [word for word in word_tokens_clean 
                                        if word!='be']
              else:
                  word_tokens_temp = [word for word in word_tokens 
                                    if word not in string.punctuation 
                                    if word is not None]
                  # If the entity is a single word token, only allow nouns and proper nouns
                  if len(word_tokens_temp)==1:
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
          if len(tuple_x_clean)==3:
              preprocessed_tuples.append(tuple_x_clean)

      self.tuples_preprocessed = preprocessed_tuples
      self.unique_entities = self.get_unique_entities(preprocessed_tuples,
                                                      self.top_n_ent)
      self.unique_relations = self.get_unique_relations(preprocessed_tuples,
                                                        self.top_n_rel)

