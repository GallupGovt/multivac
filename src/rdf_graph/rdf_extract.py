"""
Extract RDF triples from documents.
"""
import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from string import ascii_lowercase

import fastcluster
import numpy as np
import spacy
from bs4 import UnicodeDammit
from corenlp import CoreNLPClient
from nltk.tokenize.treebank import TreebankWordDetokenizer
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from textacy.extract import subject_verb_object_triples
from tqdm import tqdm

OBJECTS_TO_REPLACE = ['that', 'which']

NORM_REGEX_CHARS1 = re.compile(r'[\(\)\"\‘\,\.\%\{\}\`\\\:\[\]\“\•]+')
NORM_REGEX_CHARS2 = re.compile(r'^([\-\—\–]|(’s)|(’))\s?')

REGEX_BREAK = re.compile(r'\n+')
REGEX_SPACE = re.compile(r'\s+')

ABS_REGEX_LATEX = re.compile(r'\$.+\$')
ABS_REGEX_VARIABLE = re.compile(r'\\\\\w')
ABS_REGEX_PRECEDING = re.compile((r'^(conclusions|conclusion|methods|results'
                                  r'|background|abstract|objective|discussion)+'),
                                 flags=re.IGNORECASE)

DOC_REGEX_PARENTS = re.compile(r'\(\)[\s,]*')
DOC_REGEX_BRACKET = re.compile(r'\[\][\s,]*')
DOC_REGEX_ELIPSES = re.compile(r'\.\s\.\s\.')


def preprocess_abstract(abstract, nlp):
    """
    Helper function to clean abstract.

    Parameters
    ----------
    abstract : str
        The text of the abstract
    nlp : spacy
        The SpaCy object.
    """
    abstract = re.sub(REGEX_BREAK, ' ', abstract)
    abstract = re.sub(ABS_REGEX_LATEX, '', abstract)
    abstract = re.sub(ABS_REGEX_VARIABLE, 'variable', abstract)
    abstract = re.sub(REGEX_SPACE, ' ', abstract)
    abstract = ' '.join([re.sub(ABS_REGEX_PRECEDING, '', sent.text.strip()).strip()
                         for sent in nlp(abstract).sents])
    return abstract.strip()


def preprocess_full_document(doc, nlp=None):
    """
    Helper function to clean document.

    Parameters
    ----------
    doc : str
        The text of the document
    nlp : spacy
        The SpaCy object.
    """
    doc = UnicodeDammit(doc, smart_quotes_to='ascii').unicode_markup

    doc = re.sub(DOC_REGEX_PARENTS, '', doc)
    doc = re.sub(DOC_REGEX_BRACKET, '', doc)
    doc = re.sub(DOC_REGEX_ELIPSES, '...', doc)

    doc = re.sub(REGEX_BREAK, ' ', doc)
    doc = re.sub(REGEX_SPACE, ' ', doc)

    return doc


def postprocess_triples(extractor,
                        embeddings_path=None,
                        cluster_entities=False,
                        cluster_relations=False):
    """
    Post-process the triples.

    Parameters
    ----------
    extractor : RDFExtractor
        A fitted RDF extractor object.
    embeddings_path : str
        The path to the embeddings.
        Defaults to None.
    cluster_entities : bool
        Whether to cluster the entities.
        Defaults to False.
    cluster_relations : bool
        Whether to cluster the relations.
        Defaults to False.
    """

    # get the triples, relations, and entities
    triples = extractor.triples
    relations = extractor.relations
    entities = extractor.entities

    # should we cluster the entities?
    entities_dict = {}
    if cluster_entities and embeddings_path:
        clusterer = Clusterer()
        entities_dict = clusterer.cluster(embeddings_path, entities)

    # should we cluster the relations?
    relations_dict = {}
    if cluster_relations and embeddings_path:
        clusterer = Clusterer()
        relations_dict = clusterer.cluster(embeddings_path, relations)

    # make sure entities and relations start with an actual character
    good_rels = [r for r in relations
                 if any(r.lower().startswith(l) for l in ascii_lowercase)]
    good_ents = [e for e in entities
                 if any(e.lower().startswith(l) for l in ascii_lowercase)]

    # remove anything that isn't in the lists above
    final_triples = []
    for t in tqdm(triples):
        if (t[0] in good_ents and t[2] in good_ents and t[1] in good_rels):
            t_new = (entities_dict.get(t[0], t[0]),
                     entities_dict.get(t[2], t[2]),
                     relations_dict.get(t[1], t[1]))
            if t_new not in final_triples:
                final_triples.append(t_new)

    extractor.triples = final_triples
    return extractor


class Clusterer:

    def __init__(self,  clust_dist_thres=0.2, verbose=False):

        self.clust_dist_thres = clust_dist_thres
        self.verbose = verbose

    def load_embeddings(self, embedding_path, entities):
        """
        Load the embeddings (`DA_glove_embeddings_300.pkl`)
        from a pickle file and compute the average embedding
        for each entity.

        Parameters
        ----------
        embedding_path : str
            The path to the embeddings file.
        entities : list
            The list of unique entities
        """
        with open(embedding_path, 'rb') as fb:
            embed = pickle.load(fb)

        # create the embeddings dictionary
        embed_dict = {token: vector for token, vector in zip(embed['vocab'],
                                                             embed['embeddings'])}

        # compute the average embedding for each entity
        entity_embed = {}
        for entity in tqdm(entities):

            entity_split = entity.split()
            entity_split_embed = [embed_dict[token] for token in entity_split
                                  if token in embed_dict]

            if len(entity_split_embed) == 1:
                entity_embed.update({entity: np.squeeze(np.array(entity_split_embed))})
            elif len(entity_split_embed) > 1:
                entity_embed.update({entity: np.mean(np.array(entity_split_embed), axis=0)})

        return entity_embed

    @staticmethod
    def get_representatives(cluster_members, char_limit=100):
        """
        Get the representative mentions for each cluster.

        Parameters
        ----------
        cluster_members : list
            The cluster members
        char_limit : int, optional
            The maximum number of characters for any
            entity to be included.
            Defaults to 120.
        """
        cluster_key = {}
        for cluster in cluster_members:

            if len(''.join(cluster)) > char_limit:
                cluster_key.update({entity: cluster[0]
                                    for entity in cluster})
            else:
                cluster_key.update({entity: ' | '.join(cluster)
                                    for entity in cluster})
        return cluster_key

    def cluster(self,
                embedding_path,
                entities,
                link_method='average'):
        """
        Cluster the entities using agglomerative clustering.

        Parameters
        ----------
        embedding_path : str
            The path to the embeddings file.
        entities : list
            The list of unique entities
        link_method : str, optional
            The link method used by `fastcluster`
            Defaults to 'average'
        """
        embedding_dict = self.load_embeddings(embedding_path, entities)
        embeddings = np.array([embedding for embedding in embedding_dict.values()])
        entities_list = np.array([entity for entity in embedding_dict.keys()])

        # create distance matrix using cosine similarity between all entity strings
        dist_vec = pdist(embeddings, 'cosine')

        # cluster distance matrix to find co-referring entities
        linkage_matrix = fastcluster.linkage(dist_vec, method=link_method)
        cluster_labels = fcluster(linkage_matrix, t=self.clust_dist_thres, criterion='distance')
        cluster_members_all = []
        for clus_label in tqdm(np.unique(cluster_labels)):
            clus_indx = cluster_labels == clus_label
            cluster_members = list(entities_list[clus_indx])
            cluster_members_all.append(cluster_members)

        output = self.get_representatives(cluster_members_all)

        return output


class StanfordCoreferenceResolution:
    """
    Stanford CoreNLP co-reference.

    Parameters
    ----------
    timeout : int
        The timeout for the parser
        Defaults to 30000
    memory : str
        The memory allocation.
        Defaults to '6G'

    """

    def __init__(self, timeout=30000, memory='6G'):

        self.detok = TreebankWordDetokenizer()

        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'dcoref'],
                                    output_format='json',
                                    timeout=timeout,
                                    memory=memory)

    def resolve(self, doc, raise_errors=True):
        """
        Resolve the co-references for a single document.

        Parameters
        ----------
        doc : str
            A document whose co-references will be resolved.
        raise_errors : bool, optional
            Whether to raise errors.
            Defaults to True.

        Returns
        -------
        resolve_doc : str or None
            A document whose co-references have been resolved.
            If there was a problem and `raise_errors=False`,
            then `None` will be returned.
        """
        try:
            parsed = self.client.annotate(doc)
        except Exception as error:
            if raise_errors:
                raise error
            return
        return self.replace_coreferences(parsed)

    def resolve_all(self, docs, raise_errors=True):
        """
        Resolve co-references for all the documents.

        Parameters
        ----------
        docs : list of str
            A list of documents
        raise_errors : bool, optional
            Whether to raise errors.
            Defaults to False.

        Returns
        -------
        resolved_docs : list of str
            A list of documents, with co-references resolved.
        """
        resolved_docs = []
        for doc in tqdm(docs):
            resolved_docs.append(self.resolve(doc, raise_errors))
        return resolved_docs

    @staticmethod
    def restructure_coreference_dict(corefs_dict):
        """
        Given a dictionary of co-references, restructure it into
        a new dictionary where the keys are sentence numbers
        and the values are lists of references that need to
        be resolved.

        Parameters
        ----------
        corefs_dict : dict
            A co-reference dictionary, output from Stanford.
        """
        corefs_list = [corefs_dict[key] for key in corefs_dict
                       if len(corefs_dict[key]) > 1 and
                       any(not co['isRepresentativeMention']
                           for co in corefs_dict[key])]

        corefs_dict = defaultdict(list)
        for i, coref in enumerate(corefs_list):

            # get the first representative mention from the list;
            # if there are no representative mentions, continue
            represent = [co['text'] for co in coref
                         if co['isRepresentativeMention']]
            if len(represent) >= 1:
                represent = represent[0]
            else:
                continue

            # loop through the (non-representative) mentions,
            # add to the dictionary list for that sentence
            for co in coref:
                if not co['isRepresentativeMention']:
                    mention = {'represent': represent,
                               'text': co['text'],
                               'startIndex': co['startIndex'],
                               'endIndex': co['endIndex'],
                               'sentNum': co['sentNum']}
                    corefs_dict[co['sentNum']].append(mention)

        return corefs_dict

    def replace_coreferences(self, parsed):
        """
        We want to replace all the references with their
        representative mention.

        Parameters
        ----------
        parsed : dict
            The full output from Stanford, with co-references and sentences.
        """
        corefs = parsed['corefs']
        sents = parsed['sentences']
        corefs_dict = self.restructure_coreference_dict(corefs)

        sents = [[s['word'] for s in sent['tokens']] for sent in sents]
        sents_new = []

        # we do this on a sentence-by-sentence basis
        for sent_i, sent in enumerate(sents, start=1):

            sent_new = []
            # we check to see if the sentence is in the co-reference dictionary;
            # if it's not we won't need to do anything.
            if sent_i in corefs_dict:

                last_end = 0
                # we loop through the (sorted) references and add them
                # to our new sentence list one-by-one, being careful to
                # capture any preceding or ending text
                sorted_sent = sorted(corefs_dict[sent_i],
                                     key=lambda x: x['startIndex'])
                for co_i, co in enumerate(sorted_sent):

                    start = co['startIndex'] - 1
                    end = co['endIndex'] - 1
                    represent = co['represent']

                    # here we want to check whether this is the first co-reference;
                    # if it is, then we need to get any text *before* it
                    if co_i == 0:
                        sent_new.extend(sent[:start])
                        sent_new.append(represent[0].upper() + represent[1:]
                                        if start == 0
                                        else represent)

                    # otherwise, we just get the co-reference and anything
                    # between it and the preceding end from the previous co-reference
                    else:
                        sent_new.extend(sent[last_end:start])
                        sent_new.append(represent)

                    last_end = end

                sent_new.extend(sent[last_end:])

            else:
                sent_new = sent

            sents_new.append(sent_new)

        # we need to detokenize the sentence; basically this handles
        # putting punctuation and weird symbols for parentheses back together
        sents = ' '.join([self.detok.detokenize(sent, convert_parentheses=True)
                          for sent in sents_new])
        return sents


class RDFExtractor:
    """
    A class to extract RDF triples from documents.

    Parameters
    ----------
    min_sub_char_len : int, optional
        The minimum number of characters
        for a subject. If subject is less than
        `min_obj_char_len`, the entire triple
        will be excluded.
        Defaults to 3.
    min_obj_char_len : int, optional
        The minimum number of characters
        for an object. If object is less than
        `min_obj_char_len`, the entire triple
        will be excluded.
        Defaults to 3.
    lowercase : bool, optional
        Whether to make the RDF text lowercase.
        Defaults to True.
    lemmatize : bool or str, optional
        Whether to lemmatize the triple. Options are ::
        - True = lemmatize all parts of the triple
        - False = do not lemmatize any parts of the triple
        - 'pred' = only lemmatize the predicate
        Defaults to 'pred'.
    remove_numeric : bool, optional
        Whether to remove triples with subjects and objects.
        Options are ::
        - True = remove triples with all numeric characters (sub or obj)
        - False = do not remove triples with all numeric characters
        - 'any' = remove triples with *any* numeric characters
    """

    def __init__(self,
                 nlp,
                 min_sub_char_len=3,
                 min_obj_char_len=3,
                 lowercase=True,
                 lemmatize=False,
                 remove_numeric='any'):

        if not (isinstance(remove_numeric, bool) or remove_numeric == 'any'):
            raise ValueError("The `remove_numeric` parameter must be boolean "
                             "or 'any', not {}.".format(remove_numeric))

        if not (isinstance(lemmatize, bool) or lemmatize == 'pred'):
            raise ValueError("The `lemmatize` parameter must be boolean or "
                             "'pred', not {}.".format(lemmatize))

        self.nlp = nlp
        self.num_ignore = False if remove_numeric else True
        self.num_check_any = True if remove_numeric == 'any' else False

        self.min_sub_char_len = min_sub_char_len
        self.min_obj_char_len = min_obj_char_len

        self.lowercase = lowercase
        self.lemmatize = lemmatize

        self.attr_pred = ('lemma_' if lemmatize in [True, 'pred']
                          else ('lower_' if lowercase else 'text'))
        self.attr = ('lemma_' if lemmatize is True
                     else ('lower_' if lowercase else 'text'))

        self._results = []
        self._triples = set()
        self._is_extracted = False
        self._entity_to_id = None
        self._relation_to_id = None

    @property
    def triples(self):
        """
        A list of the unique RDF triples.
        """
        if not self._is_extracted:
            return
        return list(self._triples)

    @triples.setter
    def triples(self, new_triples):
        """
        A list of the unique RDF triples.
        """
        self._triples = new_triples

    @property
    def sentences_and_triples(self):
        """
        A list of sentences and RDF triples (not unique)
        """
        if not self._is_extracted:
            return
        return self._results

    @sentences_and_triples.setter
    def sentences_and_triples(self, new_results):
        """
        A list of sentences and RDF triples (not unique)
        """
        if not self._is_extracted:
            return
        self._results = new_results

    @property
    def n_triples(self):
        """
        The number of unique RDF triples.
        """
        if not self._is_extracted:
            return
        return len(self._triples)

    @property
    def entities(self):
        """
        A list of all unique entities.
        """
        if not self._is_extracted:
            return
        entities = ([i[0] for i in self._triples] +
                    [i[2] for i in self._triples])
        entities = sorted(list(set(entities)))
        return entities

    @property
    def relations(self):
        """
        A list of all unique relations (predicates).
        """
        if not self._is_extracted:
            return
        relations = [i[1] for i in self._triples]
        relations = sorted(list(set(relations)))
        return relations

    @property
    def entity_ids(self):
        """
        A dictionary mapping IDs to entities.
        """
        if not self._is_extracted:
            return
        return self.to_id(self.entities)

    @property
    def relation_ids(self):
        """
        A dictionary mapping IDs to relations (predicates).
        """
        if not self._is_extracted:
            return
        return self.to_id(self.relations)

    @staticmethod
    def to_id(a):
        """
        A helper method to convert an array of elements
        to a dictionary mapping IDs to elements.

        Parameters
        ----------
        a : array-like
            An array of unique elements.

        Returns
        -------
        dict
            A dictionary mapping unique ID to element
            in the original array.
        """
        return {i: elem for i, elem in enumerate(a)}

    @staticmethod
    def check_numeric(string, check_any=False, ignore=False):
        """
        A helper method to check whether is numeric.
        If `check_any=True`, then the method will
        check if *any* digits exist in the string.

        Parameters
        ----------
        string : str
            The string to check
        check_any : bool, optional
            Check if any element of the string is
            a number.
            Defaults to False
        ignore : bool, optional
            Ignore the check entirely.
            Defaults to False.

        Returns
        -------
        bool
            Whether the string is numeric.
        """
        if ignore:
            return False
        if check_any:
            return any(char.isdigit() for char in string)
        try:
            float(string)
            return True
        except ValueError:
            return False

    def _triple_checks_out(self, triple):
        """
        Check to see if the triple conforms to our expectations ::
        - No numeric values in subjects, verbs, or objects
        - No verbs in subject or object
        - No subject of object smaller than the minimum character length

        Parameters
        ----------
        triple : tuple of SpaCy spans
            A tuple of spacy object triples

        Returns
        -------
        bool
            Whether the triple checks out
        """
        sub, pred, obj = triple
        if (self.check_numeric(sub.text.strip(),
                               self.num_check_any,
                               self.num_ignore) or
                self.check_numeric(obj.text.strip(),
                                   self.num_check_any,
                                   self.num_ignore) or
                self.check_numeric(pred.text.strip(),
                                   self.num_check_any,
                                   self.num_ignore) or
                any(tok.pos_ == 'VERB' for tok in obj) or
                any(tok.pos_ == 'VERB' for tok in sub) or
                len(sub.text) < self.min_sub_char_len or
                len(obj.text) < self.min_obj_char_len):
            return False
        return True

    def _get_preceding_chunk(self, span, chunks, reverse=True):
        """
        Given a 'subject', get the preceding chunk. This is necessary
        in cases where the subject is (likely) incorrect.

        Parameters
        ----------
        span : SpaCy span object
            The span to check and get the chunk for.
        chunks : list of SpaCy span objects
            A list of noun phrases (chunks).
        """
        sent = span.sent
        for chunk in reversed(chunks) if reverse else chunks:
            if ((chunk.text.lower() in
                    sent.text[:span.start_char - sent.start_char].lower()) and
                    len(chunk.text) > self.min_sub_char_len):
                return chunk
        return span

    def _get_chunk(self, span, chunks):
        """
        Given a SpaCy span object and a list of noun phrases (chunks),
        check to see if the span is within any of the chunks. If so,
        return the first chunk that contains the span. Otherwise,
        just return the span.

        Parameters
        ----------
        span : SpaCy span object
            The span to check and get the chunk for.
        chunks : list of SpaCy span objects
            A list of noun phrases (chunks).

        Returns
        -------
        span or chunk : SpaCy span object
            The noun phrase chunk or original span.
        """
        span_start, span_end = span.start_char, span.end_char
        for chunk in chunks:
            chunk_start, chunk_end = chunk.start_char, chunk.end_char
            if span_start >= chunk_start and span_end <= chunk_end:
                return chunk
        return span

    def _postprocess(self, span, chunks):
        """
        This is a bit of a hack. Basically, we are applying post-processing
        rules to fix subjects that SpaCy screwed up.

        Right now, this only covers the following subjects: ['that', 'which'].
        If these are the subjects, we check to see if there is a preceding
        chunk. Otherwise, we check to see if there is a chunk that contains
        the subject. If neither of these are true, then we stick with the
        subject.

        Parameters
        ----------
        span : SpaCy span object
            The span object to perform post-processing on.
        chunks : list of SpaCy span objects
            A list of noun phrases (chunks).

        Returns
        -------
        span or chunk : SpaCy span object
            The noun phrase chunk or original span.
        """
        if span.text.lower().strip() in OBJECTS_TO_REPLACE:
            try:
                span = self._get_preceding_chunk(span, chunks)
            except Exception:
                span = self._get_chunk(span, chunks)
        else:
            span = self._get_chunk(span, chunks)

        return span

    def _normalize(self, string):
        """
        Normalize the string with some regular expressions

        Parameters
        ----------
        string : str
            The string to check
        """
        string = re.sub(NORM_REGEX_CHARS1, '', string)
        string = re.sub(NORM_REGEX_CHARS2, '', string)
        return string.strip()

    def extract(self, doc, raw=False):
        """
        Extract all triples from a given document.

        Parameters
        ----------
        doc : SpaCy Document
            A SpaCy document, which will be passed to
            `subject_verb_object_triples()`.
        raw : bool, optional
            Whether to take the raw output from
            `subject_verb_object_triples()` or
            do some additional cleaning.

        Returns
        -------
        results : list of dict
            A list of dictionaries, which include:
            - 'RDF'   = The RDF triple (tuple of str)
            - 'NOUNS' = All noun phrases extracted from the sentence
            - 'VERBS' = All verbs extracted from the sentence
        """
        results = []
        for triple in subject_verb_object_triples(doc):

            # make sure that the triple passes basic checks
            if self._triple_checks_out(triple):

                # unpack the subject-verb-object triples
                sub, pred, obj = triple

                # get the sentence from the subject span
                sentence = sub.sent

                # get the noun chunks and verbs from the sentence
                nouns = [cnk for cnk in sentence.noun_chunks]
                verbs = [tok for tok in sentence if tok.pos_ == 'VERB']

                # if `raw=True`, then we simply append the raw
                # triple text; no post-processing
                if raw:
                    results.append({'SENT': sentence,
                                    'RDF': (sub.text, pred.text, obj.text),
                                    'NOUNS': nouns,
                                    'VERBS': verbs})
                    continue

                # otherwise, we do some post-processing
                sub = self._postprocess(sub, nouns)
                obj = self._get_chunk(obj, nouns)

                if self._triple_checks_out((sub, pred, obj)):

                    # we get the final strings
                    sub = getattr(sub, self.attr).strip()
                    obj = getattr(obj, self.attr).strip()
                    pred = getattr(pred, self.attr_pred).strip()

                    # finally, we apply some regular expressions to the strings
                    sub = self._normalize(sub)
                    obj = self._normalize(obj)

                    results.append({'RDF': (sub, pred, obj),
                                    'NOUNS': nouns,
                                    'VERBS': verbs})

        return results

    def extract_all(self, docs, raw=False, verbose=True):
        """
        Extract all triples from a list of documents.

        Parameters
        ----------
        docs : list of str
            A list of documents, converted to SpaCy documents
            and passed to `extract()`.
        raw : bool, optional
            Whether to take the raw output from
            `subject_verb_object_triples()` or
            do some additional cleaning.
            Defaults to False.
        verbose : bool, optional
            Whether to print progress.
            Defaults to True.
        """
        self._results = []
        self._triples = set()
        self._entity_to_id, self._relation_to_id = None, None

        # make sure the `max_length` is greater than the max doc
        self.nlp.max_length = max([len(doc) for doc in docs]) + 1

        failed = 0
        for doc in tqdm(docs) if verbose else docs:
            try:
                doc = self.nlp(doc)
                results = self.extract(doc, raw=raw)
                self._results.extend(results)
                self._triples.update([r['RDF'] for r in results])
            except Exception:
                failed += 1
                continue

        if verbose:
            print('Number failed: ', failed)

        self._is_extracted = True

    @staticmethod
    def _create_type_constraint(train2id):
        """
        This is modified from here ::
          "https://github.com/thunlp/OpenKE/blob/"
          "OpenKE-PyTorch/benchmarks/FB15K/n-n.py"

        Parameters
        ----------
        train2id list of lists
            A list of triple lists

        Returns
        -------
        rel_left : dict of dicts
            The subject-relation types
        rel_right : dict of dicts
            The object-relation types
        """

        rel_left = {}
        rel_right = {}

        for triple in train2id[1:]:
            if not triple:
                continue

            sub, obj, rel = triple

            if rel not in rel_left:
                rel_left[rel] = {}
            if rel not in rel_right:
                rel_right[rel] = {}

            rel_left[rel][sub] = 1
            rel_right[rel][obj] = 1

        return rel_left, rel_right

    def package_entities_and_relations(self):
        """
        Package the entities and relations into a format
        usable by OpenKE.
        """
        entity_dict = {v: k for k, v in self.entity_ids.items()}
        relation_dict = {v: k for k, v in self.relation_ids.items()}

        entity2relation = [[len(self.entities)]]
        for triple in self.triples:
            entity2relation.append([entity_dict[triple[0]],
                                    entity_dict[triple[2]],
                                    relation_dict[triple[1]]])
        entity2id = [[rel, i] for rel, i in entity_dict.items()]
        relation2id = [[rel, i] for rel, i in relation_dict.items()]

        rel_left, rel_right = self._create_type_constraint(entity2relation)

        return (entity2id,
                relation2id,
                entity2relation,
                rel_left,
                rel_right)


def main():

    def bool_or_str(value):
        if value.lower().strip() in ['t', 'true']:
            return True
        elif value.lower().strip() in ['f', 'false']:
            return False
        return value

    parser = argparse.ArgumentParser(prog='create_triples')

    parser.add_argument('json_input', help="The JSON input file.")

    parser.add_argument('-o', '--json_output',
                        required=True,
                        help="The output file directory.")

    parser.add_argument('-ms', '--min_sub_char_len', default=3, type=int,
                        help="The minimum character length of subjects.")

    parser.add_argument('-mo', '--min_obj_char_len', default=3, type=int,
                        help="The minimum character length of objects.")

    parser.add_argument('-lm', '--lemmatize', default=False,
                        type=bool_or_str,
                        help="Whether to lemmatize text.")

    parser.add_argument('-rn', '--remove_numeric', default='any',
                        type=bool_or_str,
                        help="Whether to remove numeric subjects and objects.")

    parser.add_argument('-ep', '--embeddings_path',
                        help="The path to the embeddings.")

    parser.add_argument('-sd', '--stanford_dir',
                        help="The directory with Stanford JAR files.",
                        default=None)

    parser.add_argument('-rc', '--resolve_coreferences',
                        action='store_true',
                        help="Resolve the co-references.")

    parser.add_argument('-pp', '--preprocess',
                        default='abstract',
                        choices=['abstract', 'document', 'none'],
                        help="Preprocess the document or abstract.")

    parser.add_argument('-pe', '--package_entities',
                        action='store_true',
                        help="Package the entities.")

    parser.add_argument('-ce', '--cluster_entities',
                        action='store_true',
                        help="Cluster the entities.")

    parser.add_argument('-cr', '--cluster_relations',
                        action='store_true',
                        help="Cluster the relations.")

    args = parser.parse_args()

    # initialize the spaCy object
    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        raise OSError('The model `en_core_web_lg` must be downloaded.')

    with open(args.json_input) as fb:
        docs = json.load(fb)

    if args.preprocess == 'abstract':
        docs = [preprocess_abstract(doc, nlp) for doc in docs]
    elif args.preprocess == 'document':
        docs = [preprocess_full_document(doc, nlp) for doc in docs]

    if args.stanford_dir is not None and args.resolve_coreferences:
        print("Resolving co-references...")
        os.environ['CORENLP_HOME'] = args.stanford_dir
        resolver = StanfordCoreferenceResolution()
        docs = resolver.resolve_all(docs)

    print("Extracting triples...")
    extractor = RDFExtractor(nlp,
                             min_sub_char_len=args.min_sub_char_len,
                             min_obj_char_len=args.min_obj_char_len,
                             lemmatize=args.lemmatize,
                             remove_numeric=args.remove_numeric)
    extractor.extract_all(docs)

    if args.stanford_dir is not None and args.resolve_coreferences:
        resolver.client.stop()

    extractor = postprocess_triples(extractor,
                                    args.embeddings_path,
                                    args.cluster_entities,
                                    args.cluster_relations)

    with open(args.json_output, 'w') as fb:
        triples = extractor.triples
        json.dump(triples, fb)

    if args.package_entities:

        (entity2id,
         relation2id,
         train2id,
         rel_left,
         rel_right) = extractor.package_entities_and_relations()

        directory = os.path.dirname(args.json_output)
        with open(os.path.join(directory, 'entity2id.txt'), 'w') as fb:
            len_entity = len(entity2id)
            for i, row in enumerate(entity2id, start=1):
                sep = '\n' if i < len_entity else ''
                fb.write('\t'.join([str(v) for v in row]) + sep)

        with open(os.path.join(directory, 'relation2id.txt'), 'w') as fb:
            len_relation = len(relation2id)
            for i, row in enumerate(relation2id):
                sep = '\n' if i < len_relation else ''
                fb.write('\t'.join([str(v) for v in row]) + sep)

        with open(os.path.join(directory, 'train2id.txt'), 'w') as fb:
            len_train = len(train2id)
            for row in train2id:
                sep = '\n' if i < len_train else ''
                fb.write('\t'.join([str(v) for v in row]) + sep)

        with open(os.path.join(directory, 'type_constraint.txt'), 'w') as fb:
            fb.write('{}\n'.format(len(rel_left)))
            for i in rel_left:

                fb.write('{}\t{}'.format(i, len(rel_left[i])))
                for j in rel_left[i]:
                    fb.write('\t{}'.format(j))

                fb.write('\n')
                fb.write('{}\t{}'.format(i, len(rel_right[i])))
                for j in rel_right[i]:
                    fb.write('\t{}'.format(j))
                fb.write('\n')


if __name__ == '__main__':

    main()
