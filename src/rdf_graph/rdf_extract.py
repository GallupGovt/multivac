"""
Extract RDF triples from documents.
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
from collections import defaultdict

import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm

from stanfordcorenlp import StanfordCoreNLP
from textacy.extract import subject_verb_object_triples

REGEX_PCT = re.compile(r'\s%')
REGEX_SPACE = re.compile(r'\s+')
CHARS_TO_REMOVE = ['(', ')', '"', '‘', ',', '.']

REGEX_ABSTRACT = re.compile(r'^conclusions|methods|results'
                            r'abstract|conclusion|objective'
                            r'([a-z\s]+)',
                            re.IGNORECASE)


def clean_abstract_rdf(triple):
    """
    Helper function to clean abstract.
    """
    sub, pred, obj = triple
    sub = re.sub(REGEX_ABSTRACT, r'\1', sub)
    return (sub, pred, obj)


def clean_full_document_rdf(triple):
    """
    TODO :: Add some post-processing logic
    for full documents. Will do this once
    the initial PR is reviewed.
    """
    return triple


class StanfordServer:
    """
    A simple class to launch the Stanford server from a JAR file.
    This is just for convenience, since Stanford CoreNLP wrappers
    do not always handle stopping and starting the server very well.

    Obviously, we'll also have the server running if we run this from
    within our docker environment, but that means we always have to
    use docker, use a specific version of the CoreNLP library, etc.

    Parameters
    ----------
    path : str or None, optional
        The path to the Stanford CoreNLP directory.
        Optionally, you can set the 'CORENLP_HOME'
        environment variable. If None, assume that
        you are currently in the proper directory.
        Defaults to None.
    port : int, optional
        The port to use for the server.
        Defaults to 9000.
    timeout : int, optional
        The max timeout for processing a document.
        Defaults to 15000.
    memory : str, optional
        The maximum amount of memory to allocate, in gigabytes.
        The default value is six gigabytes.
        Defaults to 'mx6g'
    """

    def __init__(self, path=None, port=9000, timeout=15000, memory='mx6g'):
        self.path = path
        self.path = os.environ.get('CORENLP_HOME', path)
        self.port = port
        self.timeout = timeout
        self.memory = memory
        self.server = None

    def start(self):
        """
        Start the server.
        """
        path = '*' if self.path is None else os.path.join(self.path, '*')
        jar = 'edu.stanford.nlp.pipeline.StanfordCoreNLPServer'
        command = (f'java -{self.memory} -cp "{path}" {jar} '
                   f'-port {self.port} -timeout {self.timeout}')
        self.server = subprocess.Popen(shlex.split(command),
                                       stdout=subprocess.PIPE)

    def stop(self):
        """
        Stop the server.
        """
        os.kill(self.server.pid, signal.SIGKILL)


class StanfordCorefernceResolution:
    """
    Stanford CoreNLP co-reference.

    Parameters
    ----------
    host : str
        The Stanford CoreNLP Server URL.
        Defaults to 'http://localhost'
    port : int, optional
        The Stanford CoreNLP Server port.
        Defaults to 9000.
    """

    def __init__(self, host='http://localhost', port=9000):

        self.detok = TreebankWordDetokenizer()
        self.client = StanfordCoreNLP(host, port=port)
        self.properties = {'annotators': 'tokenize,ssplit,dcoref',
                           'pipelineLanguage': 'en',
                           'outputFormat': 'json'}

    def resolve(self, doc, raise_errors=False):
        """
        Resolve the co-references for a single document.

        Parameters
        ----------
        doc : str
            A document whose co-references will be resolved.
        raise_errors : bool, optional
            Whether to raise errors.
            Defaults to False.

        Returns
        -------
        resolve_doc : str or None
            A document whose co-references have been resolved.
            If there was a problem and `raise_errors=False`,
            then `None` will be returned.
        """
        try:
            parsed = self.client.annotate(doc, properties=self.properties)
            parsed = json.loads(parsed)
        except Exception as error:
            if raise_errors:
                raise error
            return
        return self.replace_coreferences(parsed)

    def resolve_all(self, docs, raise_errors=False):
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

    try:
        nlp = spacy.load('en_core_web_lg')
    except OSError:
        raise OSError('The model `en_core_web_lg` must be downloaded.')

    def __init__(self,
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

    @property
    def n_triples(self):
        """
        The number of unique RDF triples.
        """
        if not self._is_extracted:
            return
        return len(self._triples)

    @property
    def sentences_and_triples(self):
        """
        A list of sentences and RDF triples (not unique)
        """
        if not self._is_extracted:
            return
        return [(res['SENT'], res['RDF']) for res in self._results]

    @property
    def entities(self):
        """
        A list of all unique entities.
        """
        if not self._is_extracted:
            return
        entities = ([' '.join(i[0]) for i in self._triples] +
                    [' '.join(i[2]) for i in self._triples])
        entities = sorted(list(set(entities)))
        return entities

    @property
    def relations(self):
        """
        A list of all unique relations (predicates).
        """
        if not self._is_extracted:
            return
        relations = [' '.join(i[1]) for i in self._triples]
        relations = sorted(list(set(relations)))
        return relations

    @property
    def entity_ids(self):
        """
        A dictionary mapping IDs to entities.
        """
        if not self._is_extracted:
            return
        return self._entity_to_id

    @property
    def relation_ids(self):
        """
        A dictionary mapping IDs to relations (predicates).
        """
        if not self._is_extracted:
            return
        return self._relation_to_id

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

        TODO :: This really needs to be improved.

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
        if span.text.lower().strip() in ['that', 'which']:
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
        string = re.sub(REGEX_PCT, "%", string)
        string = ''.join([char for char in string
                          if char not in CHARS_TO_REMOVE])
        string = re.sub(REGEX_SPACE, ' ', string)
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
            - 'SENT'  = Full text of the sentence (SpaCy object)
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

                    results.append({'SENT': sentence,
                                    'RDF': (sub, pred, obj),
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
        self._entity_to_id = self.to_id(self.entities)
        self._relation_to_id = self.to_id(self.relations)


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

    parser.add_argument('-rw', '--raw', action='store_true',
                        help="Whether to get raw tokens.")

    parser.add_argument('-sd', '--stanford_dir',
                        help="The directory with Stanford JAR files.",
                        default=None)

    parser.add_argument('-sh', '--stanford_host',
                        help="The Stanford host (e.g. 'http://localhost').",
                        default=None)

    parser.add_argument('-sp', '--stanford_port',
                        type=int,
                        help="The Stanford host port (e.g. 9000).",
                        default=9000)

    parser.add_argument('-sm', '--stanford_memory',
                        help="The Stanford host memory (e.g. 'mx6g').",
                        default='mx6g')

    parser.add_argument('-st', '--stanford_timeout',
                        type=int,
                        help="The Stanford host timeout (e.g. 15000).",
                        default=15000)
    parser.add_argument('-pp', '--postprocessing',
                        help="Post-processing to perform.",
                        default='abstract',
                        choices=['none', 'abstract', 'full_document'])

    parser.add_argument('-rc', '--resolve_coreferences',
                        action='store_true',
                        help="Resolve the co-references.")

    args = parser.parse_args()

    with open(args.json_input) as fb:
        docs = json.load(fb)

    if args.stanford_dir is not None and args.resolve_coreferences:
        server = StanfordServer(args.stanford_dir,
                                args.stanford_port,
                                args.stanford_timeout,
                                args.stanford_memory)
        server.start()

    if args.resolve_coreferences:
        print("Resolving co-references...")
        host = ('http://localhost'
                if args.stanford_host is None or args.stanford_dir
                else args.stanford_host)
        resolver = StanfordCorefernceResolution(host,
                                                args.stanford_port)
        docs = resolver.resolve_all(docs)

    print("Extracting triples...")
    extractor = RDFExtractor(min_sub_char_len=args.min_sub_char_len,
                             min_obj_char_len=args.min_obj_char_len,
                             lemmatize=args.lemmatize,
                             remove_numeric=args.remove_numeric)
    extractor.extract_all(docs, raw=args.raw)

    with open(args.json_output, 'w') as fb:
        triples = extractor.triples
        if args.postprocessing == 'abstract':
            triples = [clean_abstract_rdf(triple) for triple in triples]
        elif args.postprocessing == 'full_document':
            triples = [clean_full_document_rdf(triple) for triple in triples]
        json.dump(triples, fb)

    if args.stanford_dir is not None and args.resolve_coreferences:
        server.stop()


if __name__ == '__main__':

    main()
