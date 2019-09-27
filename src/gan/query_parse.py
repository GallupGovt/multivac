
import argparse
from corenlp import CoreNLPClient
import pandas as pd
import re

class StanfordParser(object):
    def __init__(self, nlp=None, max_entail=1000):
        if nlp is None:
            annots = "tokenize ssplit pos lemma ner depparse natlog openie dcoref".split()
            properties={"openie.triple.strict": "true",
                        "openie.max_entailments_per_clause": str(max_entail),
                        "openie.openie.resolve_coref": "true"}

            self.nlp_client = CoreNLPClient(annotators = annots, 
                                            output_format='json')
        else:
            self.nlp_client = nlp

        self.nlp_client.properties = properties

        _ = self.nlp_client.annotate("Let's get this party started!")
        del(_)

    def get_parse(self, sentence):
        return self.nlp_client.annotate(sentence)['sentences'][0]

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
        self.text = sentence
        self.parse = parser.get_parse(sentence)
        self.tokens = []
        self.root = 0
        self.rdfs = {}
        self.parse_string = re.sub(r"\s+", " ", self.parse['parse'])

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

        self.store_rdfs()
        self.expand_rdfs()
        self.substitute_rdfs()

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

    def get_root(self):
        return self.tokens[self.root]

    def get_children(self, tok):
        return set([t for t in self.tokens if t.head+1 == tok.i])

    def get_rdfs(self, use_tokens=True, how='longest'):
        result = []
        longest = 0
        which_long = 0

        for idx, rdf in self.rdfs.items():
            toks = [self.tokens[int(t)] for n in rdf.values() for t in n]

            if how == 'longest' and len(toks) > longest:
                longest = len(toks)
                which_long = idx
                result = toks
            else:
                result += toks

        if not use_tokens:
            result = ' '.join([x.text for x in result])

        return result

    def store_rdfs(self):
        for idx, rdf in enumerate(self.parse['openie']):
            self.rdfs[idx] = {"subject": [],
                              "relation": [],
                              "object": []}

            for node in self.rdfs[idx]:
                tok_range = range(*rdf[node+"Span"])
                self.rdfs[idx][node] = [x for x in tok_range]

    def substitute_rdfs(self):
        # if len(self.rdfs) == 0:
        #     subj = []
        #     rel = []
        #     obj = []

        #     for token in self.tokens:
        #         pos = token.pos_
        #         dep = token.dep_
        pass

    def toString(self):
        return ' '.join([t.text for t in self.tokens])

    def pos_tree(self, t, tree_tokens=None, pos='N'):
        if not tree_tokens:
            tree_tokens = set()

        if t not in tree_tokens and t.pos_.startswith(pos):
            tree_tokens.add(t)

        if t.has_children:
            for child in self.get_children(t):
                if child.pos_.startswith(pos):
                    tree_tokens.add(child)

                if child.has_children:
                    grandkids = self.pos_tree(child, tree_tokens, pos=pos)
                    tree_tokens = tree_tokens.union(grandkids)

        return sorted(list(tree_tokens))


def process_queries(parser, queries, clean=False, verbose=False, use_rdfs=False):

    if clean:
        cl = lambda text: re.sub(r"^\W+|[^\w{W}(?<!?)]+$", "", text)
        queries = queries.apply(cl)

    questions = [stanford_parse(parser, query) for query in queries]
    processed = []

    for question in questions:
        if len(question.rdfs) > 0 and use_rdfs:
            processed.append(question.get_rdfs(use_tokens=False))
            continue

        verbs = [t for t in question.tokens if t.pos_.startswith('V')]
        nouns = [t for t in question.tokens if t.pos_.startswith('N')]

        for i in range(len(nouns)):
            nouns[i] = question.pos_tree(nouns[i])

        for i in range(len(verbs)):
            verbs[i] = question.pos_tree(verbs[i], pos='V')

        for i in range(len(nouns)): 
            for nounlist in [x for j,x in enumerate(nouns) if j!=i]: 
                if nounlist is None or nouns[i] is None: 
                    continue 

                if(all(x in nounlist for x in nouns[i])): 
                    nouns[i] = [] 
                    continue

        nouns = ' '.join([x.text for nl in nouns for x in nl])

        for i in range(len(verbs)): 
            for verblist in [x for j,x in enumerate(verbs) if j!=i]: 
                if verblist is None or verbs[i] is None: 
                    continue 

                if(all(x in verblist for x in verbs[i])): 
                    verbs[i] = [] 
                    continue

        verbs = ' '.join([x.text for vl in verbs for x in vl])

        processed.append(' '.join([nouns, verbs]))

    return processed

def run(args_dict):
    parser = StanfordParser()
    queries = pd.read_csv(args_dict['query_file'])
    contents = process_queries(parser, 
                               queries['Query'], 
                               args_dict['clean'], 
                               args_dict['verbose'],
                               args_dict['use_rdfs'])
    queries['Annotations'] = pd.Series(contents)
    queries.to_csv(args_dict['out_file'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process queries into key semantic components.')
    parser.add_argument('-q', '--query_file', required=True,
                        help='Path to queries.')
    parser.add_argument('-o', '--out_file',
                        help='Filename for output.')
    parser.add_argument('-r', '--use_rdfs', action='store_true',
                        help='Prefer RDFs of queries when processing.')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Pre-clean queries before populating.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output on progress.')

    args_dict = vars(parser.parse_args())
    run(args_dict)
