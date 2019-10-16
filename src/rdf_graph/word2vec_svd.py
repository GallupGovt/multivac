import numpy as np
from nltk.util import skipgrams
from collections import Counter
from scipy import sparse
from scipy.sparse.linalg import svds


class Word2VecSVD:
    def __init__(self, text, win_size=10, n_comp=100,
                 smooth_alpha=0.75, positive_thres=True,
                 min_freq=0, min_sg=0):
        '''Inputs:
        * text = a nested list containing the TOKENIZED review text: a
        list containing text for each review. The text must be tokenized and
        preprocessed before input to this function.
        '''
        self.text = text
        self.win_size = win_size
        self.n_comp = n_comp
        self.smooth_alpha = smooth_alpha
        self.positive_thres = positive_thres

        self.compute_skipgrams(win_size, min_freq, min_sg)
        self.calc_word_count_matrix()
        self.word_count_to_PMI()
        self.PMI_matrix_svd()

    def calc_word_count_matrix(self):
        print('Compute word count matrix \n')
        row_indxs = []
        col_indxs = []
        dat_values = []
        for (tok1, tok2), sg_count in self.skipgram_counts.items():
            tok1_indx = self.tok2indx[tok1]
            tok2_indx = self.tok2indx[tok2]

            row_indxs.append(tok1_indx)
            col_indxs.append(tok2_indx)
            dat_values.append(sg_count)

        self.word_count_mat = sparse.csr_matrix(
            (dat_values, (row_indxs, col_indxs))
        )

    def compute_skipgrams(self, win_size, min_freq=0, min_sg=0):
        print('Counting skipgrams and building vocab \n')
        # Initialize variables
        tok2indx = dict()
        unigram_counts = Counter()
        doc_all_clean = []
        skipgrams_list = []
        win_size_ext = win_size // 2
        # Loop through text and preprocess
        for doc_indx, doc in enumerate(self.text):
            if doc_indx % 10000 == 0:
                print('{} reviews processed'.format(doc_indx))
            # Update unigram counts
            for token in doc:
                unigram_counts[token] += 1
                if token not in tok2indx:
                    tok2indx[token] = len(tok2indx)
            # Update Skipgram counts
            skipgrams_list.append(list(skipgrams(doc, 2, win_size_ext)))

        # Flatten skip grams into flat list
        skipgram_counts = Counter([skipgram for skip_list in skipgrams_list
                                   for skipgram in skip_list])
        # Given the bottom-up process used to count unigrams and skipgrams, we
        # have to threshold out words that don't meet the user-specified min_freq
        # threshold after the fact. Need a way to optimize this section of code
        if min_freq > 0:
            # Threshold unigram counts based off 'min_freq'
            unigram_counts = Counter({token: unigram_counts[token] for token in unigram_counts
                                      if unigram_counts[token] >= min_freq})
            # Create new tok2indx
            tok2indx = {token: count for count, token
                        in enumerate(unigram_counts.keys())}
            # Remove skipgrams that contain removed unigram tokens
            # This is poorly optimized, need a better solution
            skipgram_counts = Counter({sg_tuple: count for sg_tuple, count
                                       in skipgram_counts.items()
                                       if all(token in unigram_counts.keys()
                                              for token in sg_tuple)
                                       if count >= min_sg})
        # Set necessary objects to class object
        self.tok2indx = tok2indx
        self.skipgram_counts = skipgram_counts
        # Print unigram counts
        print('vocabulary size: {}'.format(len(unigram_counts)))
        print('most common: {} \n'.format(unigram_counts.most_common(10)))
        # Print skipgram counts
        print('number of skipgrams: {}'.format(len(skipgram_counts)))
        print('most common: {} \n'.format(skipgram_counts.most_common(10)))

    def PMI_matrix_svd(self):
        print('Compute word embeddings/vectors \n')
        uu, _, _ = svds(self.pmi_mat, k=self.n_comp, tol=1E-2)
        norms = np.sqrt(np.sum(np.square(uu), axis=1, keepdims=True))
        uu /= np.maximum(norms, 1e-7)
        self.word_vecs_norm = uu

    def word_count_to_PMI(self):
        print('Convert word count to ppmi mat \n')
        num_skipgrams = self.word_count_mat.sum()
        # set smoothing parameters
        nca_denom = np.sum(np.array(self.word_count_mat.sum(axis=0))
                           .flatten() ** self.smooth_alpha)
        sum_over_words = np.array(self.word_count_mat.sum(axis=0)).flatten()
        sum_over_words_alpha = sum_over_words ** self.smooth_alpha
        sum_over_contexts = np.array(self.word_count_mat.sum(axis=1)).flatten()
        # set up vars for sparse matrix
        row_indxs = []
        col_indxs = []
        pmi_dat_values = []
        ii = 0
        for (tok1, tok2), sg_count in self.skipgram_counts.items():
            ## Get Indx of each token
            tok1_indx = self.tok2indx[tok1]
            tok2_indx = self.tok2indx[tok2]
            # Get Terms for pair-wise PMI calc
            nwc = sg_count
            Pwc = nwc / num_skipgrams
            nw = sum_over_contexts[tok1_indx]
            Pw = nw / num_skipgrams
            nc = sum_over_words[tok2_indx]
            Pc = nc / num_skipgrams
            # Calculate PMI (type based on input parameters)
            if self.smooth_alpha > 0:
                nca = sum_over_words_alpha[tok2_indx]
                Pca = nca / nca_denom
                if self.positive_thres:
                    pmi = max(np.log2(Pwc / (Pw * Pca)), 0)
                else:
                    pmi = np.log2(Pwc / (Pw * Pca))
            else:
                if self.positive_thres:
                    pmi = max(np.log2(Pwc / (Pw * Pc)), 0)
                else:
                    pmi = np.log2(Pwc / (Pw * Pc))

            # Assign values for Sparse Matrix
            row_indxs.append(tok1_indx)
            col_indxs.append(tok2_indx)
            pmi_dat_values.append(pmi)

        # Create Sparse Positive Mutual Information Matrix
        self.pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
