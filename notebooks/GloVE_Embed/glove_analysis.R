#
# R functions and packages for featurizing text with GloVe word embeddings
# 

getPackages <- function (list.of.packages) {
  #
  # Takes a list or vector of package names and loads them, installing first if they 
  # are not already installed.
  # 
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  
  if(length(new.packages)) install.packages(new.packages)
  lapply(list.of.packages,require,character.only=T)
}


pks = c('data.table','dplyr','text2vec','Rtsne','quanteda','doParallel','foreach')
getPackages(pks)


resetCluster <- function(pks) {
  if(dim(showConnections())[1]>0) {
    stopImplicitCluster()
    gc()
  }
  
  registerDoParallel(detectCores()-1)
  x <- foreach(i=1:detectCores()-1, .packages=pks) %dopar% { return(1) }
}


loadGloveModel <- function(path) {
  wv = fread(path, header=FALSE)
  terms <- wv$V1
  wv[,V1:=NULL]
  wv <- data.matrix(wv)
  rownames(wv) <- terms
  
  return(wv)
}


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
                        verbose=verbose)
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
  
  return(word_vectors)
}


gloved <- function(tokens, wv) {
  #
  # Average word embeddings for a text; ignore 
  # out-of-vocabulary words 
  # 
  
  tokens <- tokens[tokens %in% rownames(wv)]
  embeds <- wv[tokens, , drop=FALSE]
  
  vector <- colMeans(embeds)
  
  return(vector)
}


getTokens <- function(txts){
  tokens <- foreach(i = 1:length(txts), .inorder=TRUE, .multicombine = TRUE, 
                    .options.multicore = list(preschedule = FALSE)) %dopar% {
                      tolower(unlist(quanteda::tokens(txts[i])))
                    }
  return(tokens)
}


deriveBias <- function(pairs, wv, method='pca', diag=FALSE) {
  #
  # Give a set of word pairs, calculate the average of their
  # differences as a "bias" vector.
  # 
  if(class(pairs) == 'list') {
    diffs <- lapply(pairs, function(x) wv[x[1],,drop=FALSE]-wv[x[2],,drop=FALSE])
    diffs <- do.call(rbind, diffs)
  } else {
    pairs <- pairs[tolower(pairs) %in% rownames(wv)]
    diffs <- wv[tolower(pairs),,drop=FALSE]
    method <- 'pca'
  }
  
  if(method=='pca'){
    pr.out <- prcomp(diffs, center=F, scale=F)
    b <- matrix(pr.out$rotation[,1],1,300)
    
    if(diag) {
      pr.var <- pr.out$sdev**2
      print(pr.var/sum(pr.var))
    }
  } else if(method=='mean') {
    b <- matrix(colMeans(diffs), 1, 300)
  } else {
    print('No aggregation method supplied.')
    b = NULL
  }
  
  return(b)
}


calcBiasThemes <- function(tokens, bias, wv) {
  b <- lapply(tokens, function(x) avgBias(x, bias, wv))
  b <- unlist(b)
  
  return(b)
}


avgBias <- function(tokens, bias, wv) {
  tokens <- tokens[tokens %in% rownames(wv)]
  embeddings <- wv[tokens, , drop=FALSE]
  bias <- sim2(embeddings, bias, method='cosine', norm='l2')
  bias <- mean(bias)
  return(bias)
}