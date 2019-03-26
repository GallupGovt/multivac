####################################################################
## Load packages
####################################################################
import argparse
import equationparsing as eq
import textparsing
import spacy
import stanfordnlp
import pickle
import json
import gc

from interruptingcow import timeout

####################################################################
## Functions 
####################################################################

def getAdjustmentPosition(tokenPosition, adjustmentDictionary):
    '''This determines the adjustment position for DEP files, because things get reordered when there are equations
    '''
    if len(adjustmentDictionary)>1:
        for key, val in sorted(list(adjustmentDictionary.items()), key=lambda x:x, reverse=True):
            if tokenPosition>key:
                return val
    return 0



def get_token_governor(token, sent):
    if token.governor==0:
        govWord = 'ROOT'
    else:
        govWord = sent.words[token.governor-1].text
    return govWord
        

def create_parse_files(doc, docNum, writeFile = True, pathToFolders=''):
    """ Creates parse files and stores them in the folder passed when writeFile=True and pathToFolders is provided
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
                
                l_depTokens_latex_sub_tuples, l_posTokens_latex_sub, l_morTokens_latex_sub = eq.latexParsing(
                    latexEquationId, int(token.index) + adjustedPosition)
                
                # Need to adjust position so that it we add all the new tokens, then subtract 1 for LateXEquation##
                if len(l_morTokens_latex_sub)>0:
                    lastAdjustedPosition = adjustedPosition
                    adjustedPosition = adjustedPosition + (len(l_posTokens_latex_sub) -1)
                    adjustmentDictionary[(int(token.index) )]=adjustedPosition


                    ## Go backwards and make sure all the previous ones are good
                    new_l_depTokens_tuples = []

                    for depSet in l_depTokens_tuples:
                        t1, t2, t3 = depSet
                        t2=list(t2)
                        t3=list(t3)

                        #current position is the threshold for change
                        if (t2[1] - lastAdjustedPosition) > int(token.index) and t1 not in ['combine', 'compare', 'function', 'transform']:
                            t2[1] = t2[1]+(len(l_posTokens_latex_sub)-1)
                        if (t3[1] - lastAdjustedPosition) > int(token.index) and t1 not in ['combine', 'compare', 'function', 'transform']:
                            t3[1] = t3[1]+(len(l_posTokens_latex_sub)-1)

                        adjustedTuple = (t1,tuple(t2),tuple(t3))
                        new_l_depTokens_tuples.append(adjustedTuple)

                    l_depTokens_tuples = new_l_depTokens_tuples

            
                    # Now add the original dependency
                    headTokenPosition =  token.governor
                    childTokenPosition = int(token.index)
                    l_depTokens_tuples.append( ( token.dependency_relation.replace(":","") , 
                                                (tokenHeadText, 
                                                 headTokenPosition + getAdjustmentPosition(headTokenPosition,
                                                                                           adjustmentDictionary)), 
                                                (token.text, childTokenPosition + getAdjustmentPosition(
                                                    childTokenPosition, adjustmentDictionary)) ) )
                    
                    # Now add to the master list
                    l_depTokens_tuples = l_depTokens_tuples + l_depTokens_latex_sub_tuples
                    l_posTokens = l_posTokens + l_posTokens_latex_sub
                    l_morTokens = l_morTokens + l_morTokens_latex_sub

                    # For keeping track of latex tokens and their tag IDs
                    eq.latexMapTokens[latexEquationId]= ' '.join(l_morTokens_latex_sub)
                    
                    # Use this to replace the Ltxqtn tag when it's a head

                    latexInSentenceMap[token.text]= { 'tokenIndex': token.index , 'tokenHead' : l_morTokens_latex_sub[0] }
                
                else: #This is for when the the latex parser errors out and can't find it. It adds the actual latexequation marker

                    headTokenPosition =  token.governor
                    childTokenPosition = int(token.index)
                    l_depTokens_tuples.append( ( token.dependency_relation.replace(":","") , 
                                                (tokenHeadText, 
                                                 headTokenPosition + getAdjustmentPosition(headTokenPosition,
                                                                                           adjustmentDictionary)), 
                                                (token.text, childTokenPosition + getAdjustmentPosition(
                                                    childTokenPosition, adjustmentDictionary)) ) )
                    l_posTokens.append("{0}_{1}".format(token.text, token.upos))  
                    l_morTokens.append(token.text)
                    adjustedPosition = adjustedPosition+1

            else:
                ## For all other types of words 
                
                ## Create dependency trees
                childTokenPosition = int(token.index)
                headTokenPosition =  token.governor 
                headAdjustment = getAdjustmentPosition(headTokenPosition, adjustmentDictionary)
                childAdjustment = getAdjustmentPosition(childTokenPosition, adjustmentDictionary)
                
                if token.dependency_relation not in ['ROOT']:
                    
                    l_depTokens_tuples.append(( token.dependency_relation.replace(":","") , 
                                               (tokenHeadText, 
                                                headTokenPosition + headAdjustment), 
                                               (token.text,
                                                childTokenPosition + childAdjustment) ) )
                            
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
            
            l_depTokens.append("{0}({1}-{2}, {3}-{4})".format(t1, headToken,t2[1],childToken, t3[1]))
 
        for depSet in l_depTokens_latex_tuples:
            t1, t2, t3 = depSet
            l_depTokens_latex.append("{0}({1}-{2}, {3}-{4})".format(t1, t2[0],t2[1],t3[0], t3[1]))
 

        l_depSentences.append("\n".join(l_depTokens + l_depTokens_latex))
        l_posSentences.append("\n".join(l_posTokens))
        l_morSentences.append("\n".join(l_morTokens))
        

    d_documentData['depData'].append(l_depSentences)
    d_documentData['posData'].append(l_posSentences)
    d_documentData['morData'].append(l_morSentences)


    if writeFile:
        with open(pathToFolders+'/dep/{0:04d}.dep'.format(docNum), "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_depSentences))
        with open(pathToFolders+'/input/{0:04d}.input'.format(docNum), "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_posSentences))
        with open(pathToFolders+'/morph/{0:04d}.morph'.format(docNum), "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_morSentences))
                
        print('Files written to folder:', pathToFolders)
    return d_documentData



def load_data(jsonPath, picklePath = None):
    """Load data - if picklePath is specified, load the pickle. Else, try json file.
    This returns the JSON file as well as a list of document texts 
    """
    if picklePath is not None:
        l_docs = pickle.load(open(picklePath, "rb" ))
    else:

        ## Read JSON data into the datastore variable - this comes from Peter and Domonique's effort. Don
        with open(jsonPath, 'r') as f:
            datastore = json.load(f)
                  
        ## These were some bad files - nothing substantive in them, or they were retrieved in bad format
        for e in ['1805.10677v1', '0911.5378v1']: 
            datastore.pop(e)
                
        ## Extract texts
        l_docs = [value['text'] for key,value in list(datastore.items())[0:] if value['text'] ]
        
    print('# of documents: ', len(l_docs))
    
    return datastore, l_docs


def run(args_dict):
    ''' Main run file that orchestrates everything  
    '''
    
    ## Load NLPs
    spacynlp = spacy.load('en_core_web_sm')
    nlp = stanfordnlp.Pipeline(models_dir=args_dict['stanfordnlp'], 
                            treebank='en_ewt', use_gpu=False, pos_batch_size=3000)
    
    
    ## Load documents 
    jsonObj, allDocs = load_data(args_dict['data'])

    
    ## Process and Clean documents 
    try: 
        allDocsClean = pickle.load(open('resources/allDocsClean.pkl', "rb" ))
        print('Loaded pickle!')
    except:
        print('Starting from scratch')
        allDocsClean= []
        for i, doc in enumerate(allDocs):
            if i%10==0:
                print(i)
            allDocsClean.append(textparsing.clean_doc(doc, spacynlp))
            
        with open('resources/allDocsClean.pkl', 'wb') as f:
            pickle.dump(allDocsClean, f)


    allDocs2 = [eq.extract_and_replace_latex(doc, docNum) for docNum, doc in enumerate(allDocsClean)]
    print('Number of LateX Equations parsed: {}'.format(len(eq.latexMap)))
    

    ## Parse files into DIM 
    startPoint=0 
    if args_dict['beginpoint'] is not None:
        startPoint = args_dict['beginpoint']
    
    for i, doc in enumerate(allDocs2[0:]):
        print('Processing document #{}'.format(i))
        if i > startPoint:
            
            # Use exception handling so that the process doesn't get stuck and time out because of memory errors
            try:
                with timeout(300, exception=RuntimeError):
                    nlpifiedDoc = nlp(doc)
                    thisDocumentData = create_parse_files(nlpifiedDoc, i, True, args_dict['output'])
            except RuntimeError:
                print("Didn't finish document #{} within five minutes. Moving to next one.".format(i))



if __name__ == '__main__':
    #-s '../../../stanfordnlp_resources/'
    #-d '../../../data/20181212.json'
    #-o resources

    parser = argparse.ArgumentParser(description='Parse texts for natural language and equations.')
    parser.add_argument('-b', '--beginpoint', required=False, help='Which document to start with', type=int)
    parser.add_argument('-s', '--stanfordnlp', required=True, help='Path to Stanford NLP Model')
    parser.add_argument('-d', '--data', required=True, help='Path to JSON data file')
    parser.add_argument('-o', '--output', required=True, help='Where DIM files are to be written')
    args_dict = vars(parser.parse_args())
    run(args_dict)
