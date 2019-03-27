getPackages <- function (list.of.packages) {
# Takes a list or vector of package names and loads them, installing first if they
# are not already installed.

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages)
   lapply(list.of.packages,require,character.only=T)
}


pks = c('data.table','dplyr','text2vec','Rtsne','quanteda','doParallel','foreach')
getPackages(pks)


trainEmbeddings <- function(docs, 
                            term_count_min=5L, 
                            skip_grams_window=10L, 
                            word_vectors_size=300, 
                            x_max=100, 
                            n_iter=100, 
                            convergence_tol=0.01, 
                            learning_rate=0.05,
                            verbose=FALSE) {
  toks <- tokens(tolower(docs))
  feats <- dfm(toks, verbose=verbose) %>% 
    dfm_trim(min_termfreq=term_count_min) %>%
    featnames()
  toks <- tokens_select(toks, feats, 
                        selection='keep',
                        valuetype='fixed',
                        padding=TRUE,
                        case_insensitive=FALSE,
                        verbose=TRUE)
  my_fcm <- fcm(toks, 
                context="window",
                window=skip_grams_window,
                count="weighted",
                weights=1/(1:skip_grams_window),
                tri=TRUE)
  
  glove <- GlobalVectors$new(word_vectors_size=word_vectors_size,
                             vocabulary=featnames(my_fcm),
                             x_max=x_max,
                             learning_rate=learning_rate)
  
  if(verbose) print('Fitting GloVe model...')
  
  wv_main = glove$fit_transform(my_fcm,
                                n_iter=n_iter,
                                convergence_tol=convergence_tol)
  
  if(verbose) print('Done.')
  
  # Combine context and target word vectors in the same manner as
  # original GloVe research 
  word_vectors = wv_main + t(glove$components)
  
  results = list(word_vectors,feats)
  return(results)
}

## Run Glove Model
setwd("~/Desktop/ASKE_MULTIVAC")

#text_data = read.csv('text_for_glove.csv')
#glove_model = trainEmbeddings(text_data$text,verbose=TRUE)
#write.csv(glove_model,'Domain_Adapted_GloVe.csv')