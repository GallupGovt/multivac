#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import json
import pickle
import re as reg
import spacy
import stanfordnlp

import multivac.src.data.equationparsing as eq

from interruptingcow import timeout

from multivac import settings
from multivac.src.data.textparsing import clean_doc


def create_parse_files(doc, docNum, writeFile = True, pathToFolders=''):
    """ Creates parse files and stores them in the folder passed when
        writeFile=True and pathToFolders is provided
        The following file types are created
            * dep -- for dependencies
            * input -- for POS tagging
            * morph -- lemmatized words
    """

    d_documentData = {
        'depData' : [],
        'posData' : [],
        'morData' : []
    }

    l_depSentences = [] # for dependencies
    l_posSentences = [] # for POS tagging
    l_morSentences = [] # for morphology/lemmatization

    # Loop over every sentence
    for sent in list(doc.sentences)[0:]:

        l_depTokens_tuples=[]
        l_depTokens=[]
        l_posTokens=[]
        l_morTokens=[]
        l_depTokens_latex_tuples=[]
        l_depTokens_latex=[]
        l_posTokens_latex=[]
        l_morTokens_latex=[]

        adjustedPosition = 0
        adjustmentDictionary = {0:0}
        latexInSentenceMap = {}

        # Loop over every word in the sentence
        for token in sent.words:

            tokenHeadText = get_token_governor(token, sent)

            if 'Ltxqtn' in tokenHeadText:
                pass

            if  (token.text==' ') or token.text is None:
                adjustedPosition= adjustedPosition-1
                adjustmentDictionary[int(token.index)]=adjustedPosition
                continue


            elif 'Ltxqtn' in token.text :
                if len(token.text) < 14:
                    print('This token is problematic: ', token.text)

                latexEquationId = token.text

                (l_depTokens_latex_sub_tuples, l_posTokens_latex_sub,
                 l_morTokens_latex_sub =
                 eq.latexParsing(latexEquationId, int(token.index) +
                                 adjustedPosition)
                )

                # Need to adjust position so that it we add all the new tokens,
                # then subtract 1 for LateXEquation##
                if len(l_morTokens_latex_sub)>0:
                    lastAdjustedPosition = adjustedPosition
                    adjustedPosition = (adjustedPosition +
                                        (len(l_posTokens_latex_sub) -1)
                    )
                    adjustmentDictionary[(int(token.index) )]=adjustedPosition


                    # Go backwards and make sure all the previous ones are good
                    new_l_depTokens_tuples = []

                    for depSet in l_depTokens_tuples:
                        t1, t2, t3 = depSet
                        t2=list(t2)
                        t3=list(t3)

                        #current position is the threshold for change
                        if (t2[1]-lastAdjustedPosition)>int(token.index) and
                            t1 not in ['combine', 'compare', 'function', 'transform']:
                            t2[1] = t2[1]+(len(l_posTokens_latex_sub)-1)
                        if (t3[1]-lastAdjustedPosition)>int(token.index) and
                            t1 not in ['combine', 'compare', 'function', 'transform']:
                            t3[1] = t3[1]+(len(l_posTokens_latex_sub)-1)

                        adjustedTuple = (t1,tuple(t2),tuple(t3))
                        new_l_depTokens_tuples.append(adjustedTuple)

                    l_depTokens_tuples = new_l_depTokens_tuples


                    # Now add the original dependency
                    headTokenPosition =  token.governor
                    childTokenPosition = int(token.index)
                    l_depTokens_tuples.append(
                        (
                            token.dependency_relation.replace(":",""),
                            (tokenHeadText,
                             headTokenPosition +
                             get_adjustment_position(headTokenPosition,
                                                     adjustmentDictionary)
                            ),
                            (token.text,
                             childTokenPosition +
                             get_adjustment_position(childTokenPosition,
                                                     adjustmentDictionary)
                            )
                        )
                    )

                    # Now add to the master list
                    l_depTokens_tuples = (
                        l_depTokens_tuples + l_depTokens_latex_sub_tuples
                    )
                    l_posTokens = l_posTokens + l_posTokens_latex_sub
                    l_morTokens = l_morTokens + l_morTokens_latex_sub

                    # For keeping track of latex tokens and their tag IDs
                    eq.LATEXMAPTOKENS[latexEquationId]= ' '.join(l_morTokens_latex_sub)

                    # Use this to replace the Ltxqtn tag when it's a head

                    latexInSentenceMap[token.text]= {
                        'tokenIndex': token.index ,
                        'tokenHead' : l_morTokens_latex_sub[0]
                    }
                # This is for when the the latex parser errors out and can't
                # find it. It adds the actual latexequation marker
                else:

                    headTokenPosition =  token.governor
                    childTokenPosition = int(token.index)
                    l_depTokens_tuples.append(
                        (token.dependency_relation.replace(":",""),
                            (tokenHeadText,
                             headTokenPosition +
                             get_adjustment_position(
                                headTokenPosition,
                                (token.text,
                                 childTokenPosition +
                                 get_adjustment_position(
                                    childTokenPosition, adjustmentDictionary
                                 )
                                )
                             )
                            )
                        )
                    )
                    l_posTokens.append("{0}_{1}".format(token.text, token.upos))
                    l_morTokens.append(token.text)
                    adjustedPosition = adjustedPosition+1

            else:
                ## For all other types of words

                ## Create dependency trees
                childTokenPosition = int(token.index)
                headTokenPosition =  token.governor
                headAdjustment = get_adjustment_position(headTokenPosition,
                                                         adjustmentDictionary)
                childAdjustment = get_adjustment_position(childTokenPosition,
                                                          adjustmentDictionary)

                if token.dependency_relation not in ['ROOT']:

                    l_depTokens_tuples.append(
                        (
                            token.dependency_relation.replace(":",""),
                            (
                                tokenHeadText,
                                headTokenPosition + headAdjustment
                            ),
                            (
                                token.text,
                                childTokenPosition + childAdjustment
                            )
                        )
                    )

                ## Create POS (input)
                l_posTokens.append("{0}_{1}".format(token.text, token.upos))

                ## Create Morphologies
                if token.lemma is None:
                    l_morTokens.append(token.text)
                else:
                    l_morTokens.append(token.lemma)


        ## Need to Parse out DEPTokens from tuples out to text
        for depSet in l_depTokens_tuples:
            t1, t2, t3 = depSet
            headToken = t2[0]
            childToken = t3[0]
            if 'Ltxqtn' in headToken:
                try:
                    headToken = latexInSentenceMap[headToken]['tokenHead']
                except:
                    pass

            if 'Ltxqtn' in childToken:
                try:
                    childToken = latexInSentenceMap[childToken]['tokenHead']
                except:
                    pass

            l_depTokens.append(
                "{0}({1}-{2}, {3}-{4})".format(t1,
                                               headToken,t2[1],
                                               childToken,
                                               t3[1])
            )

        for depSet in l_depTokens_latex_tuples:
            t1, t2, t3 = depSet
            l_depTokens_latex.append(
                "{0}({1}-{2}, {3}-{4})".format(t1,
                                               t2[0],
                                               t2[1],
                                               t3[0],
                                               t3[1])
            )


        l_depSentences.append("\n".join(l_depTokens + l_depTokens_latex))
        l_posSentences.append("\n".join(l_posTokens))
        l_morSentences.append("\n".join(l_morTokens))


    d_documentData['depData'].append(l_depSentences)
    d_documentData['posData'].append(l_posSentences)
    d_documentData['morData'].append(l_morSentences)


    if writeFile:
        with open(pathToFolders+'/dep/{0:04d}.dep'.format(docNum),
                  "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_depSentences))
        with open(pathToFolders+'/input/{0:04d}.input'.format(docNum),
                  "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_posSentences))
        with open(pathToFolders+'/morph/{0:04d}.morph'.format(docNum),
                  "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_morSentences))

        print('Files written to folder:', pathToFolders)
    return d_documentData


def get_adjustment_position(tokenPosition, adjustmentDictionary):
    '''This determines the adjustment position for DEP files, because things
       get reordered when there are equations
    '''
    if len(adjustmentDictionary)>1:
        for key, val in sorted(list(adjustmentDictionary.items()),
                               key=lambda x:x, reverse=True):
            if tokenPosition>key:
                return val
    return 0


def get_token_governor(token, sent):
    if token.governor==0:
        govWord = 'ROOT'
    else:
        govWord = sent.words[token.governor-1].text
    return govWord


def load_data(jsonPath, picklePath = None):
    """Load data - if picklePath is specified, load the pickle. Else, try
       json file.
       This returns the JSON file as well as a list of document texts
    """
    if picklePath is not None:
        l_docs = pickle.load(open(picklePath, "rb" ))
    else:

        ## Read JSON data into the datastore variable - this comes from
        ## other team members effort.
        with open(jsonPath, 'r') as f:
            datastore = json.load(f)

        ## These were some bad files - nothing substantive in them, or they
        ## were retrieved in bad format
        for e in ['1805.10677v1', '0911.5378v1']:
            datastore.pop(e)

        ## Extract texts
        l_docs = [value['text'] for key,value in
                  list(datastore.items())[0:] if value['text']]

    print('# of documents: ', len(l_docs))

    return datastore, l_docs


def nlp_parse_main(args_dict):
    ''' Main run file that orchestrates everything
    '''

    ## Load NLP engines
    spacynlp = spacy.load('en_core_web_sm')
    nlp = stanfordnlp.Pipeline(models_dir=settings.stanf_nlp_dir,
                               treebank='en_ewt', use_gpu=False,
                               pos_batch_size=3000)

    ## Load documents
    jsonObj, allDocs = load_data(settings.processed_dir / 'data')

    ## Process and Clean documents
    try:
        allDocsClean = pickle.load(open('allDocsClean.pkl', "rb" ))
        print('Loaded pickle!')
    except FileNotFoundError:
        print('Starting from scratch')
        allDocsClean= []
        for i, doc in enumerate(allDocs):
            if i%10==0:
                print(i)
            allDocsClean.append(clean_doc(doc, spacynlp))

        with open('allDocsClean.pkl', 'wb') as f:
            pickle.dump(allDocsClean, f)


    allDocs2 = [eq.extract_and_replace_latex(doc) for docNum, doc in
                enumerate(allDocsClean)]
    print('Number of LateX Equations parsed: {}'.format(len(eq.LATEXMAP)))


    ## Put equations back into text - this will be fed to glove embedding
    if args_dict['nlp_newjson']:
        print('***************\nBuilding JSON file for glove embedding...')
        allDocs3 = []
        percentCompletedMultiple = int(len(allDocs2)/10)
        for i, doc in enumerate(allDocs2[0:]):
            if i%percentCompletedMultiple == 0:
                print('{}% completed'.format(round(i/(len(allDocs2))*100, 0)))
            newDoc = reg.sub(r'Ltxqtn[a-z]{8}', eq.put_equation_tokens_in_text,
                             doc)
            allDocs3.append(newDoc)

        jsonObj2 = copy.deepcopy(jsonObj)
        allDocs3Counter = 0

        for key, value in list(jsonObj2.items()):
            if value['text']:
                jsonObj2[key]['text']=allDocs3[allDocs3Counter]
                allDocs3Counter = allDocs3Counter+1

        with open('articles-with-equations.json', 'w', encoding='utf8') as fp:
            json.dump(jsonObj2, fp)


    ## Parse files into DIM
    startPoint=-1
    if args_dict['nlp_bp'] is not None:
        startPoint = args_dict['nlp_bp']

    for i, doc in enumerate(allDocs2[0:]):
        print('Processing document #{}'.format(i))
        if i > startPoint:

            # Use exception handling so that the process doesn't get stuck and
            # time out because of memory errors
            try:
                with timeout(300, exception=RuntimeError):
                    nlpifiedDoc = nlp(doc)
                    thisDocumentData = create_parse_files(
                        nlpifiedDoc, i, True, settings.data_dir]
                    )
            except RuntimeError:
                print("Didn't finish document #{} within five minutes. Moving to next one.".format(i))
