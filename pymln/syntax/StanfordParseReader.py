
import os

from syntax.Nodes import Article, Sentence, Token


class StanfordParseReader(object):
    ''' 
    Replicates StanfordParseReader.java from USP implementation found at 
    http://alchemy.cs.washington.edu/usp/ 

    Given a set of dependency, POS, and morphology files parsed from source
    documents, this compiles lists of tokens and dictionary mappings defining
    the dependency relationships in the sentences in a given document.
    '''
    def __init__(self): 
        self._isDebug=False
        self._ignored_deps = set()
        self._ignored_deps.add("aux")
        self._ignored_deps.add("auxpass")
        self._ignored_deps.add("det")
        self._ignored_deps.add("cop")
        self._ignored_deps.add("complm")
#       self._ignored_deps.add("num")
#       self._ignored_deps.add("number")
        self._ignored_deps.add("preconj")
        self._ignored_deps.add("predet")
        self._ignored_deps.add("punct")
#       self._ignored_deps.add("quantmod")
        
        self._ignored_deps.add("expl")
        self._ignored_deps.add("mark")
#       self._ignored_deps.add("parataxis")


    def readParse(self, fileName, data_dir, ignoreDep=True):
        '''
        Given a filename of the type "$FILENAME.dep" gets the file and 
        corresponding *.morph and *.input files and reads the Tokens and 
        Dependency relationships by sentence in those files. Each file in such
        a trio should contain the same number of sentences represented as blocks 
        of text with each dependency/token on its own line, separated by blank 
        lines. 
        '''
        file = os.path.splitext(fileName)[0]
        morph_file = os.path.join(data_dir, file + '.morph')
        input_file = os.path.join(data_dir, file + '.input')
        dep_file = os.path.join(data_dir, fileName)

        doc = Article(file)
        doc = self.readTokens(doc, morph_file, input_file)
        doc = self.readDeps(doc, dep_file, ignoreDep)

        return doc


    def readTokens(self, doc, morph_file, input_file):
        '''
        Reads a morphology and input (POS tagged lemmas) file simultaneously, 
        parsing single tokens from each line into a Token() object and 
        appending each Token to its respective Sentence() object, which 
        are collected in an Article() object "doc" and returned.
        '''
        isNew=True

        with open(morph_file, 'r') as mor, open(input_file, 'r') as inp:
            for mline in mor.readlines():
                mline = mline.strip()
                iline = inp.readline().strip()

                if iline == '':
                    isNew = True
                    continue

                ts = iline.split('_')

                if isNew:
                    sent = Sentence()
                    sent.add_token(Token('ROOT','ROOT'))
                    doc.sentences.append(sent)
                    isNew = False

                pos = ts[1]
                lemma = mline.replace(':','.').lower()
                form = iline[0]

                doc.sentences[-1].add_token(Token(pos,lemma,form))

        return doc


    def readDeps(self, doc, deps_file, ignoreDep):
        '''
        Reads a dependency relationships file and adds these relationships to 
        their respective Sentence() objects in an Article() in the form of 
        reciprocal python dictionaries. The updated Article() "doc" is then
        returned.
        '''
        blank = False
        senId = 0

        currSent = doc.sentences[senId]
        currNonRoots = set()
        currRoots = set()

        with open(deps_file, 'r') as d:
            for line in d.readlines():
                line = line.strip()

                if len(line) == 0:
                    if not blank:
                        senId += 1

                    blank = True

                    if currRoots is not None:
                        dep_chds = currSent.get_children(0)
                        for i in currRoots:
                            dep_chds.add((i, 'ROOT'))
                            currSent.set_parent(i, ('ROOT', 0))
                        currSent.set_children(0, dep_chds)
                        doc.sentences[senId] = currSent

                        currSent = None
                        currNonRoots = None
                        currRoots = None

                    continue
                else:
                    if blank:
                        blank = False
                        currSent = doc.sentences[senId]
                        currNonRoots = set()
                        currRoots = set()

                    rel = line[:line.index("(")]
                    items = line[line.index('('):].replace('(','').replace(')','')
                    items = items.split(', ')
                    gov, dep = items[0], items[1]
                    gov = (int(gov[gov.rfind('-')+1:]), gov[:gov.rfind('-')])
                    dep = (int(dep[dep.rfind('-')+1:]), dep[:dep.rfind('-')])

                    if ('conj' not in rel) & (gov[0] == dep[0]):
                        continue

                    currNonRoots.add(dep[0])
                    if dep[0] in currRoots:
                        currRoots.remove(dep[0])
                    if gov[0] not in currNonRoots:
                        currRoots.add(gov[0])

                    if ignoreDep & (rel in self._ignored_deps):
                        continue

                    currSent.set_parent(dep[0], (rel, gov[0]))

                    if gov[0] in currSent.get_children():
                        currSent.add_child(gov[0], (dep[0], rel))
                    else:
                        currSent.set_children(gov[0], set())
                        currSent.add_child(gov[0], (dep[0], rel))

            if currRoots is not None:
                dep_chds = currSent.get_children(0)
                for i in currRoots:
                    dep_chds.add((i, 'ROOT'))
                    currSent.set_parent(i, ('ROOT', 0))
                currSent.set_children(0, dep_chds)
                doc.sentences[senId] = currSent

                currSent = None
                currNonRoots = None
                currRoots = None

        return doc








