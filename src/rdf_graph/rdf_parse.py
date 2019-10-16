
import argparse
from corenlp import CoreNLPClient
import pandas as pd
import re

def tokenize_text(text, parser=None):
    if parser is None:
        parser = StanfordParser()

    p = stanford_parse(parser, text)

    return [x.text for x in p.tokens]


def clean_queries(queries, verbose=False):
    clean = list()

    if verbose:
        print(("Performing basic clean on {} queries.".format(len(queries))))

    for query in queries:
        # strip whitespace, and quotes
        # Remove any sentence fragments preceding question
        # Remove non-alphabetic characters at the start of the string
        query = query.strip()
        query = re.sub(r"“|”", "\"", query)
        query = re.sub(r"‘|’", "\'", query)
        query = re.sub(r"`", "\'", query)
        query = query.strip("\"")
        query = query.strip("\'")
        query = query[query.index(re.split(r"\"", query)[-1]):]
        query = query[query.index(re.split(r"NumericCitation", query, re.IGNORECASE)[-1]):]
        query = query[query.index(re.split(r"[\.\!\?]\s+", query)[-1]):]
        query = re.sub(r"^(?!\()[^a-zA-Z]+","", query)
        query = re.sub(r"^(\(.*\))?\W+","", query)
        query = re.sub(r"(\s+)([\)\]\}\.\,\?\!])", r"\2", query)
        query = re.sub(r"([\(\[\{])(\s+)", r"\1", query)

        if len(query) > 0:
            tok_chk = [len(x) for x in query.split()]

            if sum(tok_chk)/len(tok_chk) < 2:
                continue

            query = query[0].upper() + query[1:]
            clean.append(query)

    if verbose:
        print(("{} cleaned queries remaining.".format(len(queries))))

    return clean

class StanfordParser(object):
    def __init__(self, nlp=None, annots=None, props=None):
        if annots is None:
            annots = "tokenize pos lemma ner depparse"

        if nlp is None:       
            self.nlp_client = CoreNLPClient(annotators = annots.split(), 
                                            output_format='json')
        else:
            self.nlp_client = nlp

        if props is not None:
            self.nlp_client.default_properties.update(props)

        _ = self.nlp_client.annotate("Let's get this party started!")
        del(_)

    def get_parse(self, sentence):
        return self.nlp_client.annotate(sentence)

    def get_deps(self, sentence, deptype='basicDependencies', ret='asis'):
        if isinstance(sentence, str):
            sentence = self.get_parse(sentence, deptype)

        deps = sentence[deptype]

        if ret == 'asis':
            retval = deps
        else:
            retval = {}
            retval['deps'] = {x['dep']: x['dependent'] for x in deps}
            retval['heads'] = {x['dependentGloss']: x['governorGloss'] for x in deps}
            retval['governors'] = {x['dependent']: x['governorGloss'] for x in deps}
            retval['dependents'] = {x['dependent']: x['dependentGloss'] for x in deps}
            retval['text'] = ["{}({}-{}, {}-{})".format(x['dep'],
                                                x['governorGloss'],
                                                x['governor'],
                                                x['dependentGloss'],
                                                x['dependent']) for x in deps]
        return retval


class stanford_token(object):
    def __init__(self, text='', index=None, lemma_='', pos_='',
                 ner='', dep_='', head=None):
        self.i = index
        self.text = text
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.ner = ner
        self.dep_ = dep_
        self.head = head
        self.has_children = False

    def __repr__(self):
        return "{}:{}=>{}:{}".format(self.i,
                                     self.text,
                                     self.dep_,
                                     self.head)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.compareTo(other) == 0

    def __lt__(self, other):
        return self.compareTo(other) < 0

    def compareTo(self, other):
        result = 0

        if self.__repr__() != other.__repr__():
            if self.__repr__() < other.__repr__():
                result -= 1
            else:
                result += 1

        return result


class stanford_parse(object):
    def __init__(self, parser, sentence, deptype='basicDependencies'):

        if isinstance(sentence, str):
            self.text = sentence
            self.parse = parser.get_parse(sentence)['sentences'][0]
        else:
            self.text = ' '.join([x['originalText'] for x in sentence['tokens']])
            self.parse = sentence
        self.tokens = []
        self.root = 0
        self.rdfs = {}

        if 'parse' in self.parse:
            self.parse_string = re.sub(r"\s+", " ", self.parse['parse'])
        else:
            self.parse_string = ''

        self.deps = sorted(self.parse[deptype], key=lambda k: k['dependent'])

        for i, w in enumerate(self.parse['tokens']):
            tok = stanford_token(text=w['originalText'],
                                 index=w['index'],
                                 lemma_=w['lemma'],
                                 pos_=w['pos'],
                                 ner=w['ner'],
                                 dep_=self.deps[i]['dep'].replace(":",""),
                                 head=self.deps[i]['governor']-1)
            self.tokens.append(tok)
            if tok.dep_ == 'ROOT':
                self.root = len(self.tokens)-1

        self._tokens = {t.text: t.i for t in self.tokens}

        for tok in self.tokens:
            self.tokens[tok.head].has_children = True

        self.dep_tree = parser.get_deps(self.parse, deptype, "tree")

        # self.parse_tree = get_eng_tree(self.parse_string)

        # self.store_rdfs()
        self.substitute_rdfs()
        self.expand_rdfs()

    def __repr__(self):
        return ' '.join(["{}".format(t) for t in self.tokens])

    def expand_rdfs(self):
        if len(self.parse['entitymentions']) == 0:
            return

        for rdf in self.rdfs:
            for node, node_toks in self.rdfs[rdf].items():
                for entity in self.parse['entitymentions']:
                    toks = range(entity['tokenBegin'], entity['tokenEnd'])
                    overlap = list(set(node_toks).intersection(toks))
                    
                    if len(overlap) > 0:
                        extra_toks = list(set(toks).difference(node_toks))
                        self.rdfs[rdf][node] = sorted(list(set(node_toks).union(toks)))

    def get_children(self, tok):
        return set([t for t in self.tokens if t.head+1 == tok.i])

    def get_rdfs(self, use_tokens=True, how='longest'):
        # Return the RDF triples of a sentence parse in various formats. By
        # default this will return a list of tokens comprising the subj-pred-obj
        # triple with the most tokens (if there are multiple triples found).
        if how == 'asis':
            result = self.rdfs
        elif how == 'list':
            result = self.rdfs.copy().values()
            
            if not use_tokens:
                for rdf in result:
                    for part in ['subject', 'relation', 'object']:
                        rdf[part] = ' '.join([self.tokens[int(t)].text 
                                          for t 
                                          in rdf[part]])
                        # result[rdf] = [x for x in result[rdf].values()]

            result = [list(x.values()) for x in result]
            result = [x for sl in result for x in sl]
        else:
            result = []
            longest = 0
            which_long = 0

            for idx, rdf in self.rdfs.items():
                toks = []

                for part in ['subject', 'relation', 'object']:
                    toks += [self.tokens[int(t)] for t in rdf[part]]

                if how == 'longest' and len(toks) > longest:
                    longest = len(toks)
                    which_long = idx
                    result = toks
                else:
                    result += toks

            if not use_tokens:
                result = ' ::: '.join([x.text for x in result])

        return result

    def get_root(self):
        return self.tokens[self.root]

    def in_children(self, parent_list, child_list):
        # Return True or False if a child_list of tokens is entirely contained 
        # in within the children of a parent_list of tokens.

        result = False

        if len(parent_list) > 0 and len(child_list) > 0:
            for parent in parent_list:
                for child in child_list:
                    if child in self.pos_tree(parent):
                        result = True
                        break

                if result:
                    break

        return result

    def pos_tree(self, t, tree_tokens=None, pos=[""]):
        # Recursively discover all tokens matching the 'pos' start(s) in a 
        # dependency parse tree or sub-tree. 
        if not tree_tokens:
            tree_tokens = set()

        if t not in tree_tokens and any([t.pos_.startswith(x) for x in pos]):
            tree_tokens.add(t)

        if t.has_children:
            for child in self.get_children(t):
                if any([child.pos_.startswith(x) for x in pos]):
                    tree_tokens.add(child)

                    # TBD: is it better to stop recursing when we encounter a
                    #      different type of POS (i.e., keep going if it's a 
                    #      compound noun/adjective phrase, but stop at 
                    #      prepositions/adverbs/verbs/etc. if pos=['N','J'])?
                    #
                    # if child.has_children:
                    #     grandkids = self.pos_tree(child, tree_tokens, pos=pos)
                    #     tree_tokens = tree_tokens.union(grandkids)

                if child.has_children:
                    grandkids = self.pos_tree(child, tree_tokens, pos=pos)
                    tree_tokens = tree_tokens.union(grandkids)

        return sorted(list(tree_tokens))

    def store_rdfs(self):
        # Legacy code, to store 'official' RDF triples as extracted by OpenIE
        # if we want to do that again.
        for idx, rdf in enumerate(self.parse['openie']):
            self.rdfs[idx] = {"subject": [],
                              "relation": [],
                              "object": []}

            for node in self.rdfs[idx]:
                tok_range = range(*rdf[node+"Span"])
                self.rdfs[idx][node] = [x for x in tok_range]

    def substitute_rdfs(self):
        # Find all nouns (and adjectives) and verbs (and adverbs)
        # Create groups by subtree 
        # Remove groups that are fully contained by others

        self.rdfs = {0: {'subject': [], "relation": [], "object": []}}

        verbs = [t for t in self.tokens if t.pos_.startswith('V')]
        nouns = [t for t in self.tokens if t.pos_.startswith('N')]

        for i in range(len(nouns)):
            nouns[i] = self.pos_tree(nouns[i], pos=['N','J'])

        for i in range(len(verbs)):
            verbs[i] = self.pos_tree(verbs[i], pos=['V','R'])

        for i in range(len(nouns)): 
            for nounlist in [x for j,x in enumerate(nouns) if j!=i]: 
                if(all(x in nounlist for x in nouns[i])): 
                    nouns[i] = [] 

        for i in range(len(verbs)): 
            for verblist in [x for j,x in enumerate(verbs) if j!=i]: 
                if(all(x in verblist for x in verbs[i])): 
                    verbs[i] = [] 

        nouns = [x for x in nouns if len(x) > 0]
        verbs = [x for x in verbs if len(x) > 0]

        if len(nouns) == 1:
            self.rdfs[0]['subject'] = [int(x.i)-1 for x in nouns[0]]
        else:
            rdfs_idx = 0

            for nounlist in nouns:
                if len([x for x in nounlist if 'subj' in x.dep_]) > 0:
                    if len(self.rdfs[rdfs_idx]['subject']) > 0:
                        rdfs_idx += 1
                        if rdfs_idx not in self.rdfs:
                            self.rdfs[rdfs_idx] = {"subject": [], 
                                                   "relation": [], 
                                                   "object": []}

                    self.rdfs[rdfs_idx]['subject'] = sorted([int(x.i)-1 
                                                             for x 
                                                             in nounlist])
                else:
                    if len(self.rdfs[rdfs_idx]['object']) > 0:
                        rdfs_idx += 1
                        if rdfs_idx not in self.rdfs:
                            self.rdfs[rdfs_idx] = {"subject": [], 
                                                   "relation": [], 
                                                   "object": []}

                    self.rdfs[rdfs_idx]['object'] = sorted([int(x.i)-1 
                                                            for x 
                                                            in nounlist])

        if len(verbs) == 1:
            self.rdfs[0]['relation'] = [int(x.i)-1 for x in verbs[0]]
        else:
            rdfs_idx = 0

            for verblist in verbs:
                subj = self.rdfs[rdfs_idx]['subject']
                obj = self.rdfs[rdfs_idx]['object']
                if not (self.in_children(verblist, subj) and \
                   self.in_children(verblist, obj)):
                    rdfs_idx += 1

                    if rdfs_idx not in self.rdfs:
                        self.rdfs[rdfs_idx] = {'subject': [], 
                                               "relation": [], 
                                               "object": []}

                self.rdfs[rdfs_idx]['relation'] = sorted([int(x.i)-1 
                                                          for x 
                                                          in verblist])

    def toString(self):
        return ' '.join([t.text for t in self.tokens])


def process_texts(parser, texts, clean=False, verbose=False, form='longest'):
    # Perform basic parsing and extraction on an iterable of sentences
    texts = clean_queries(texts, verbose)
    sentences = [stanford_parse(parser, text) for text in texts]
    processed = []

    for sentence in sentences:
        if len(sentence.rdfs) > 0:
            processed.append(sentence.get_rdfs(use_tokens=False, how=form))
        else:
            processed.append([])

    return processed, texts

def run(args_dict):
    parser = StanfordParser()

    if args_dict['text_file'].upper().endswith(".CSV"):
        texts = pd.read_csv(args_dict['text_file'])
        texts = texts['text']
    else:
        with open(args_dict['text_file'], "r") as f:
            texts = pd.Series(f.readlines())

    if args_dict['just_clean']:
        texts = clean_queries(texts, args_dict['verbose'])
    else:
        contents, texts = process_texts(parser, 
                                        texts, 
                                        args_dict['clean'],
                                        args_dict['verbose'],
                                        args_dict['form'])

        if args_dict['out_file'].upper().endswith(".CSV"):
            result = pd.concat([texts, pd.Series(contents)], axis=1)
            result.columns = ['text', 'annotations']
            result.to_csv(args_dict['out_file'], index=False)
        else:
            with open(args_dict['out_file'], "w") as f:
                try:
                    f.write('\n'.join([','.join(x) for x in contents]))
                except:
                    print(contents[0])

    with open(args_dict['text_file']+"_cleaned.txt", "w") as f:
        f.write('\n'.join(texts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process queries into key semantic components.')
    parser.add_argument('-t', '--text_file', required=True,
                        help='Path to sentences to parse.')
    parser.add_argument('-o', '--out_file',
                        help='Filename for output.')
    parser.add_argument('-f', '--form', choices=['asis', 'list',
                        'longest', 'all'],
                        help='Method for returning RDF components of queries.')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Clean queries before processing.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output on progress.')
    parser.add_argument('--just_clean', action='store_true',
                        help='Just clean and return the texts.')

    args_dict = vars(parser.parse_args())
    run(args_dict)
