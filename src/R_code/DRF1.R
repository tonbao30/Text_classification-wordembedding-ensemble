# START LIBRARY ----
library('dplyr')
library(h2o)
h2o.init( nthreads = -1,max_mem_size = '5G')

# READ TRAIN FILE ----
path <- paste0(getwd(), "/corpus2.csv")

train.data <- h2o.importFile(path = path ,destination_frame = "corpus",header = TRUE)

# TOKENISE -----
tokenize <- function(sentences) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  
  # convert to lower case
  tokenized.lower <- h2o.tolower(tokenized)
  # remove short words (less than 2 characters)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]}

words <- tokenize(train.data$nsw_token)

# WORD EMBEDDING------

w2v.model <- h2o.word2vec(model_id = 'w2v_e20_v200_w15_f10',words, sent_sample_rate = 0, epochs = 20,vec_size = 200, window_size = 15, min_word_freq =10) #vec_size:min_word_freq:word_model:
model_path <- h2o.saveModel(object=w2v.model, path=getwd(), force=TRUE)
# w2v.model <- h2o.loadModel('C:/Users/Bi PC/1.Sem2-2018/1.FIT 5149/Group Project/text_processing/data/w2v_e30_v200_w30_f0')

# Embedding word for train dataset ----
label.vecs <- h2o.transform(w2v.model, words, aggregate_method = "AVERAGE")
label.vecs$C1
valid.labels <- ! is.na(label.vecs$C1)

data <- h2o.cbind(train.data[valid.labels, "label"], label.vecs[valid.labels, ])


### SET FOLD -----


data.split <- h2o.splitFrame(
  data,           ##  splitting the H2O frame we read above
  0.95,   ##  create splits of 60% and 20%; 
  ##  H2O will create one more split of 1-(sum of these parameters)
  ##  so we will get 0.6 / 0.2 / 1 - (0.6+0.2) = 0.6/0.2/0.2
  seed=1234)    ##  setting a seed will ensure reproducible results (not R's seed)

train <- h2o.assign(data.split[[1]], "train.hex")   
## assign the first result the R variable train
## and the H2O name train.hex
test <- h2o.assign(data.split[[2]], "valid.hex")   ## R valid, H2O valid.hex


### Running AUTOML ----
# 
rf2 <- h2o.randomForest(        ##
  training_frame = train,       ##
  # validation_frame = valid,     ##
  x=names(label.vecs),                       ##
  y="label",                         ##
  model_id = "rf_final",     ## 
  ntrees = 500,                 ##
  max_depth = 30,               ## Increase depth, from 20
  stopping_rounds = 2,          ##
  stopping_tolerance = 1e-2,    ##
  score_each_iteration = T,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE,##
  seed=123)


# save model
leader <- h2o.saveModel(object=rf2, path=getwd(), force=TRUE)


## LOAD TEST ----
test.data <- h2o.importFile(path = paste0(getwd(), "/test2.csv") ,destination_frame = "corpus", header = TRUE)

#tokenise word
test.words <- tokenize(test.data$nsw_token)
# embeding
test.label.vecs <- h2o.transform(w2v.model, test.words, aggregate_method = "AVERAGE")

!is.na(test.label.vecs$C1)
test.valid.labels <- ! is.na(test.label.vecs$C1)

predict.data <- h2o.cbind(test.data[test.valid.labels, "id"], test.label.vecs[test.valid.labels, ])


predict <- function(test.data, w2v, model) {
  words <- tokenize(as.character(as.h2o(test.data$nsw_token)))
  test.vec <- h2o.transform(w2v, words, aggregate_method = "AVERAGE")
  h2o.predict(model, test.vec)
}


a<-predict(test.data,w2v.model,rf2)

result <- h2o.cbind(test.data$id,a$predict)

final_df <- as.data.frame(result)




write.table(final_df,"DRF_test_label.txt",sep = " ", row.names = F,col.names = F, quote = F)
