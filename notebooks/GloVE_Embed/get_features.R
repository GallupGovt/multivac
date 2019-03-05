#!/usr/bin/env Rscript

#
# Featurization script
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

pks = c('optparse')

getPackages(pks)

option_list = list(
  make_option(c("-a", "--articles"), type="character", default="articles.csv", 
              help="file with articles data [default= %default]", metavar="character"),
  make_option(c("-c", "--comments"), type="character", default="comments.csv", 
              help="file with comments data [default= %default]", metavar="character"),
  make_option(c("-d", "--directory"), type="character", default=NULL, 
              help="working directory", metavar="character"),
  make_option(c("-s", "--source"), type="character", default=NULL, 
              help="name of the articles/comments source", metavar="character"),
  make_option(c("-t", "--texts"), type="character", default="texts.csv", 
              help="file with combined text data [default= %default]", metavar="character"),
  make_option(c("-w", "--word_embeddings"), type="character", default="texts.csv", 
              help="file with combined text data [default= %default]", metavar="character"),
  make_option(c("-v", "--verbose"), action="store_true", default=TRUE,
              help="Print extra output [default]")
)

opt = parse_args(OptionParser(option_list=option_list))

source("clustering.R")
source('glove_analysis.R')
source('bias_themes_analysis.R')

setwd(opt$directory)

#
# Export packages to parallel nodes
# 

resetCluster(pks)

if (opt$verbose) {
  print('Environment instantiated.')
}

corpus <- fread(opt$texts, header=FALSE)
arts <- fread(opt$articles)
comments <- fread(opt$comments)
SOURCE <- opt$source


if (opt$verbose) {
  print('Articles and comments datasets loaded.')
}

#
# Load word embeddings, or if they don't exist 
# get the raw text data for embedding training
# 

if(!is.null(opt$word_embeddings)){
  wv = loadGloveModel(opt$word_embeddings)
} else {
  wv = trainEmbeddings(corpus$V2, verbose=opt$verbose)
  write.table(wv,
              paste0(SOURCE,'_corpus_embeddings.txt'),
              col.names=FALSE,
              quote=FALSE,
              fileEncoding='utf-8')
}

if (opt$verbose) {
  if (exists(opt$word_embeddings)) {
    print(paste0('Word embeddings loaded from ', opt$word_embeddings))
  } else {
    print(paste0('Word embeddings trained, and saved to ',
                 getwd(),'/',SOURCE,'_corpus_embeddings.txt'))
  }
}

#
# Begin feature extraction from articles and comments datasets:
# 
# 
# 1. Derive bias component vectors
# 

gender_bias <- deriveBias(genderPairs, wv=wv, method='pca', diag=TRUE)

#
# Unfortunately many "black" names do not occur in some corpora, 
# so we have to reduce the overall number of pairs. 
# 

adj_black_names <- black_names[tolower(black_names) %in% rownames(wv)]
adj_white_names <- sample(white_names, length(adj_black_names))
adj_racePairs <- lapply(1:length(adj_black_names), 
                        function(x) tolower(c(adj_black_names[x], adj_white_names[x])))

adj_race_names <- unlist(adj_racePairs)

race_bias <- deriveBias(adj_race_names, wv=wv, diag=TRUE)
power_bias <- deriveBias(powerPairs, wv=wv, method='pca', diag=TRUE)

if (opt$verbose) {
  print('Bias component vectors derived.')
}

#
# Derive embeddings for articles and article titles, and merge into one 
# data.table. 
# 

arts_text_tokens = getTokens(arts$art_text)

if (opt$verbose) {
  print('Articles tokenized.')
}

gloved_arts <- lapply(arts_text_tokens, gloved, wv)
gloved_arts <- data.table(do.call(rbind, gloved_arts))

if (opt$verbose) {
  print('Article word embeddings calculated.')
}

# 
# Propagate bias components
#

gloved_arts$gender <- calcBiasThemes(arts_text_tokens, gender_bias, wv)
gloved_arts$race <- calcBiasThemes(arts_text_tokens, race_bias, wv)
gloved_arts$power <- calcBiasThemes(arts_text_tokens, power_bias, wv)

if (opt$verbose) {
  print('Bias components calculated for each article.')
}

# 
# Set column names and add article id
#

names(gloved_arts) <- paste('art_text_embeddings_',names(gloved_arts), sep='')
gloved_arts$art_id <- arts$art_id

#
# Add in the article metadata (sentiment scores and number of comments per article)
# 

sent_cols <- c(names(arts)[names(arts) %like% 'sent_'], 'art_comments', 'art_id')
gloved_arts <- merge(gloved_arts, arts[,sent_cols,with=FALSE], by='art_id')

if (opt$verbose) {
  print('Article text features complete, beginning with article titles.')
}

#
# Okay, now article titles, same deal
# 

arts_title_tokens = getTokens(arts$art_title)
arts_title_tokens <- lapply(arts_title_tokens, tolower)

if (opt$verbose) {
  print('Article titles tokenized.')
}

gloved_titles <- lapply(arts_title_tokens, gloved, wv)
gloved_titles <- data.table(do.call(rbind, gloved_titles))

if (opt$verbose) {
  print('Title word embeddings calculated.')
}

gloved_titles$gender <- calcBiasThemes(arts_title_tokens, gender_bias, wv)
gloved_titles$race <- calcBiasThemes(arts_title_tokens, race_bias, wv)
gloved_titles$power <- calcBiasThemes(arts_title_tokens, power_bias, wv)

if (opt$verbose) {
  print('Bias components calculated for each article title.')
}

# 
# Set column names and add article id
#

names(gloved_titles) <- paste('title_embeddings_',names(gloved_titles), sep='')
gloved_titles$art_id <- arts$art_id

#
# Goody! Now merge the two together. 
# 

gloved_arts <- merge(gloved_titles, gloved_arts, by='art_id')

if (opt$verbose) {
  print('Article and article title features complete, saving progress...')
}

# 
# Save the dataset created so far. 
# 

fwrite(gloved_arts, paste0(SOURCE,'_articles_features.csv'))

if (opt$verbose) {
  print('Done.')
}

#
# Now, for the comments!
# 

comment_txt_tokens = getTokens(comments$comment_txt)

if (opt$verbose) {
  print('Comments tokenized.')
}

gloved_comments <- lapply(comment_txt_tokens, gloved, wv=wv)
gloved_comments <- data.table(do.call(rbind, gloved_comments))

if (opt$verbose) {
  print('Comment word embeddings calculated.')
}

gloved_comments$gender <- calcBiasThemes(comment_txt_tokens, gender_bias, wv)
gloved_comments$race <- calcBiasThemes(comment_txt_tokens, race_bias, wv)
gloved_comments$power <- calcBiasThemes(comment_txt_tokens, power_bias, wv)

if (opt$verbose) {
  print('Bias components calculated for each comment.')
}

# 
# Set column names and add article id for merging
#

names(gloved_comments) <- paste0('comment_embeddings_', names(gloved_comments))
gloved_comments$art_id <- comments$art_id

# 
# Perform clustering of comments based on word usage
# 

text_embed_cols <- names(gloved_comments)[names(gloved_comments) %like% '_V']

if (opt$verbose) {
  print('Removing incomplete cases...')
}

mask <- complete.cases(gloved_comments)
gloved_comments <- gloved_comments[mask,]

if (opt$verbose) {
  print('Clustering comments on word usage...')
}

ptm = proc.time()
group_clusters <- pickBestCluster(gloved_comments[,text_embed_cols,with=F],reproducable=TRUE)
proc.time() - ptm

gloved_comments$CLUSTER <- group_clusters

if (opt$verbose) {
  print(paste0('Clustering complete. Found ', 
               length(unique(group_clusters)),' clusters.'))
}

#
# Add in the comments metadata (sentiment scores, author and upvotes)
# 

cols <- c(names(comments)[names(comments) %like% 'sent_'], 'commenter', 'upvotes')

for(i in 1:length(cols)) {
  set(gloved_comments, j=cols[i], value=comments[mask,cols[i],with=F])
}

if (opt$verbose) {
  print('Comments features complete, saving progress...')
}

fwrite(gloved_comments, paste0(SOURCE,'_comments_features.csv'))

if (opt$verbose) {
  print('Done.')
}


#
# Finally, merge these all together, and save out to file. 
#

gloved_comb <- merge(gloved_arts, gloved_comments, by='art_id')

if (opt$verbose) {
  print('Combined dataset created, saving to disk...')
}

fwrite(gloved_comb, paste0(SOURCE,'_combined_features.csv'))

if (opt$verbose) {
  print('Done.')
}

if (opt$verbose) {
  print(paste0('Final dataset at ',getwd(),'/',SOURCE,'_combined_features.csv'))
}

stopImplicitCluster()
gc()