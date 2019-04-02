
import os
import re
from sortedcontainers import SortedSet
from multivac.pymln.syntax.Nodes import Article, Sentence, Token


class StanfordParseReader(object):
    '''
    Replicates StanfordParseReader.java from USP implementation found at
    http://alchemy.cs.washington.edu/usp/

    Given a set of dependency, POS, and morphology files parsed from source
    documents, this compiles lists of tokens and dictionary mappings defining
    the dependency relationships in the sentences in a given document.
    '''
    ignored_deps = set()
    ignored_deps.add("aux")
    ignored_deps.add("auxpass")
    ignored_deps.add("det")
    ignored_deps.add("cop")
    ignored_deps.add("complm")
#    ignored_deps.add("num")
#    ignored_deps.add("number")
    ignored_deps.add("preconj")
    ignored_deps.add("predet")
    ignored_deps.add("punct")
    ignored_deps.add("quantmod")

    ignored_deps.add("expl")
    ignored_deps.add("mark")
#    ignored_deps.add("parataxis")

    def __init__(self):
        return None


    def readParse(fileName, data_dir, ignoreDep=True):
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
        doc = StanfordParseReader.readTokens(doc, morph_file, input_file)
        doc = StanfordParseReader.readDeps(doc, dep_file, ignoreDep)

        return doc


    def readTokens(this_doc, morph_file, input_file):
        '''
        Reads a morphology (lemmas) and input (POS tagged words) file
        simultaneously, parsing single tokens from each line into a Token()
        object and appending each Token to its respective Sentence() object,
        which are collected in an Article() object "this_doc" and returned.
        '''
        isNew = True
        sents = []
        currSent = Sentence()

        with open(morph_file, 'r') as mor, open(input_file, 'r') as inp:
            for mline in mor.readlines():
                mline = mline.strip()
                iline = inp.readline().strip()

                if len(iline) == 0:
                    isNew = True
                    continue

                ts = iline.rsplit('_', 1)

                if isNew:
                    if len(currSent.get_tokens()) > 0:
                        sents.append(currSent)
                        currSent = Sentence()

                    currSent.add_token(Token('ROOT','ROOT'))
                    isNew = False

                pos, form = ts[1], ts[0]
                lemma = mline.replace(':','.').lower()
                if "_nn" in lemma:
                    lemma = lemma[:lemma.index("_nn")]

                currSent.add_token(Token(pos,lemma,form))

        if len(currSent.get_tokens()) > 0:
            sents.append(currSent)

        this_doc.sentences = sents

        return this_doc


    def readDeps(this_doc, dep_file, ignoreDep):
        '''
        Reads a dependency relationships file and adds these relationships to
        their respective Sentence() objects in an Article() in the form of
        reciprocal python dictionaries. The updated Article() "doc" is then
        returned.
        '''
        blank = False
        skip_sentence = False
        senId = 0

        currSent = this_doc.sentences[senId]
        currNonRoots = set()
        currRoots = set()

        with open(dep_file, 'r') as d:
            for line in d.readlines():
                line = line.strip()

                if len(line) == 0:
                    if not blank:
                        skip_sentence = False
                        senId += 1

                    blank = True

                    if currRoots is not None:
                        dep_chds = currSent.get_children(0)
                        for i in currRoots:
                            dep_chds.add(('ROOT', i))
                            currSent.set_parent(i, ('ROOT', 0))
                        currSent.set_children(0, dep_chds)
                        this_doc.sentences[senId-1] = currSent

                        currSent = None
                        currNonRoots = None
                        currRoots = None

                    continue
                else:
                    if skip_sentence:
                        continue

                    if blank:
                        blank = False
                        currSent = this_doc.sentences[senId]
                        currNonRoots = set()
                        currRoots = set()

                    rel = line[:line.index("(")]

                    if rel.lower() == 'root':
                        continue

                    #items = re.sub(r"\'","",line)
                    items = line[line.index('(')+1:-1]

                    try:
                        item_split = re.search(r"-\d+\'*, ", items).end(0)
                    except:
                        print("ERR: " + line)
                        print("Malformed dependency in " + dep_file)
                        continue

                    gov = items[:item_split-2]
                    dep = items[item_split:]

                    gov = (int(gov[gov.rfind('-')+1:].replace("'","")), gov[:gov.rfind('-')])
                    dep = (int(dep[dep.rfind('-')+1:].replace("'","")), dep[:dep.rfind('-')])

                    # If we are on the wrong sentence:
                    if not StanfordParseReader.tokens_in_sent(currSent, gov, dep):
                        badId = senId
                        # increment sentences until we find the right one.
                        while True:
                            senId += 1

                            # If we reach the end of the document and no
                            # sentence contains our token(s) then this sentence
                            # is unparsable, so we skip it.
                            try:
                                nextSent = this_doc.sentences[senId]
                            except IndexError:
                                print('Unparsable sentence {} in article {}. Missing tokens {} and/or {}.'.format(badId,dep_file,gov,dep))
                                senId = badId
                                del this_doc.sentences[senId]
                                senId -= 1
                                skip_sentence = True
                                break

                            if StanfordParseReader.tokens_in_sent(nextSent, gov, dep):
                                # Then transfer any work we've done for the
                                # wrong one to the right one,
                                nextSent._tkn_children = currSent._tkn_children
                                nextSent._tkn_par = currSent._tkn_par

                                # and clear that work from the wrong one.
                                currSent._tkn_children = {0: SortedSet()}
                                currSent._tkn_par = {}
                                this_doc.sentences[badId] = currSent
                                currSent = None

                                # Finally, pick up with the right one where
                                # we left off.
                                currSent = nextSent
                                break

                    if skip_sentence:
                        continue

                    if (rel.startswith('conj')) & (gov[0] == dep[0]):
                        continue

                    currNonRoots.add(dep[0])
                    if dep[0] in currRoots:
                        currRoots.remove(dep[0])
                    if gov[0] not in currNonRoots:
                        currRoots.add(gov[0])

                    if ignoreDep & (rel in StanfordParseReader.ignored_deps):
                        continue

                    currSent.set_parent(dep[0], (rel, gov[0]))

                    if gov[0] not in currSent.get_children():
                        currSent.set_children(gov[0], SortedSet())

                    currSent.add_child(gov[0], (rel, dep[0]))

            if currRoots is not None:
                dep_chds = currSent.get_children(0)
                for i in currRoots:
                    dep_chds.add(('ROOT', i))
                    currSent.set_parent(i, ('ROOT', 0))
                currSent.set_children(0, dep_chds)
                this_doc.sentences[senId-1] = currSent

                currSent = None
                currNonRoots = None
                currRoots = None

        return this_doc

    def tokens_in_sent(sent, tok1, tok2):
        # tok1 and tok2 are tuples of type (int, str) where int is the index
        # and str is the _form.
        result = False

        try:
            if tok1[0] < len(sent._tokens) and tok2[0] < len(sent._tokens):
                if sent._tokens[tok1[0]]._form == tok1[1] \
                    and sent._tokens[tok2[0]]._form == tok2[1]:
                    result = True
        except:
            print("Error in {} with tokens {} and {}".format(sent, tok1, tok2))
            raise Exception

        return result






