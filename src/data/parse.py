# from multivac.src.data import get
# from multivac.src.data import process

from multivac import settings

import json
import pandas as pd
import pickle
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy


def load_data(picklePath = None):
    """
    Load data - if picklePath is specified, load the pickle. 
    Else, try json file.
    """
    if picklePath is not None:
        l_docs = pickle.load(open(picklePath, "rb" ))
    else:
        nlp = spacy.load('en')
        #if above doesn't work, load english model from local 

        #Read JSON data into the datastore variable
        with open(settings.processed_dir / 'data.json', 'r') as f:
            datastore = json.load(f)

        ## Create nlpified object
        l_docs = [nlp(v['text']) for v in datastore.values() if v['text']]

        ## Save pickle of nlpified 
        with open(settings.processed_dir / 'NLPifiedDocs.pkl', 'wb') as f:
            pickle.dump(l_docs, f)

    return l_docs


def create_tf_idf(l_docs, writeFile=True, pathToFolders=''):
    """ Creates a TF-IDF matrix of terms in the corpus and saves this to disk
        as a sparse matrix in the data/processed folder when writeFile=True and 
        pathToFolders is provided.
    """

    tfidf = TfidfVectorizer(sublinear_tf=True, 
                            min_df=10, 
                            norm=None, 
                            ngram_range=(1, 3), 
                            stop_words='english', 
                            use_idf=True,
                            smooth_idf=True)

    features = tfidf.fit_transform(l_docs)

    if writeFile:
        if len(pathToFolders) == 0:
            pathToFolders = settings.processed_dir

        with open(pathToFolders / 'multivac_tfidf.pkl', 'wb') as f:
            pickle.dump(features, f)

        return True
    else:
        return features


def train_glove(l_docs, writeFile = True, pathToFolders=''):
    """ Trains aa 367-dimensional domain-adapted GloVe word-embeddings model 
        (300 base dimensions, with 67 additional domain-specific dimensions) 
        on the corpus and saves this file in the same folder. 

        IN DEVELOPMENT
    """

    return None
    

def create_parse_files(l_docs, writeFile = True, pathToFolders=''):
    """ Creates parse files and stores them in the data/proecssed folder
        when writeFile=True and pathToFolders is provided

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
    
    for di, doc in enumerate(l_docs[0:]):

        l_depSentences = [] # for dependencies
        l_posSentences = [] # for POS tagging
        l_morSentences = [] # for morphology/lemmatization 

        #
        # EQUATION PARSING INSERTION POINT
        # 
        
        for sent in list(doc.sents)[0:]:
            
            l_depTokens=[]
            l_posTokens=[]
            l_morTokens=[]
            
            for token in sent:
                
                ## For dependency trees
                childTokenPosition = token.i - sent.start  + 1
                headTokenPosition =  token.head.i - sent.start +1 

                if token.dep_ not in ['ROOT','punct']:
                    l_depTokens.append("{0}({1}-{2}, {3}-{4})".format(token.dep_,
                                                                      token.head.text, 
                                                                      headTokenPosition, 
                                                                      token.text, 
                                                                      childTokenPosition ))

                ## For POS
                l_posTokens.append("{0}_{1}".format(token, token.tag_))
                #print(token.tag_)

                ## For Morphologies
                l_morTokens.append(token.lemma_)


            l_depSentences.append("\n".join(l_depTokens))
            l_posSentences.append("\n".join(l_posTokens))
            l_morSentences.append("\n".join(l_morTokens))
    
        d_documentData['depData'].append(l_depSentences)
        d_documentData['posData'].append(l_posSentences)
        d_documentData['morData'].append(l_morSentences)

    if writeFile:
        if len(pathToFolders) == 0:
            pathToFolders = settings.processed_dir

        with open(pathToFolders+'\\dep\\{0:04d}.dep'.format(di), "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_depSentences))
        with open(pathToFolders+'\\input\\{0:04d}.input'.format(di), "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_posSentences))
        with open(pathToFolders+'\\morph\\{0:04d}.morph'.format(di), "w", encoding='utf8') as text_file:
            text_file.write('\n\n'.join(l_morSentences))
        
        print('Files written to folder:', pathToFolders)

        return True
    else:
        return d_documentData


def main():
    # query apis to obtain articles
    allDocs = load_data()
    create_tf_idf(allDocs)
    create_parse_files(allDocs, writeFile=True)
    train_glove(allDocs, writeFile=True)


if __name__ == '__main__':
    main()