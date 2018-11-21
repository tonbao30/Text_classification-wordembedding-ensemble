# START LIBRARY ----
library('dplyr')
library(h2o)
h2o.init( nthreads = -1,max_mem_size = '10G')

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
# 
w2v.model2 <- h2o.word2vec(model_id = 'w2v_e20_v200_w20_f5',words, sent_sample_rate = 0, epochs = 20,vec_size = 200, window_size = 20, min_word_freq =5) #vec_size:min_word_freq:word_model
model_path <- h2o.saveModel(object=w2v.model2, path=getwd(), force=TRUE)
# saved_model2 <- h2o.loadModel('w2v_e20_v200_w20_f5')

# Embedding word for train dataset ----
label.vecs <- h2o.transform(w2v.model2, words, aggregate_method = "AVERAGE")
label.vecs$C1
valid.labels <- ! is.na(label.vecs$C1)

data <- h2o.cbind(train.data[valid.labels, "label"], label.vecs[valid.labels, ])

### SET FOLD -----
# fold_numbers <- h2o.kfold_column(data, nfolds=5)
# names(fold_numbers) <- "fold_numbers"
# data <- h2o.cbind(data,fold_numbers)

### Running AUTOML ----
# 
# aml <- h2o.automl(x = names(label.vecs), y = 'label',
#                   training_frame = data,  max_models=5,fold_column="fold_numbers", seed = 1234)


aml3 <- h2o.automl(x = names(label.vecs), y = 'label',
                   training_frame = data,  max_models=5,nfolds=5, seed = 1234)
# save model
leader <- h2o.saveModel(object=aml3@leader, path=getwd(), force=TRUE)

# loadleader <- h2o.loadModel('StackedEnsemble_BestOfFamily_0_AutoML_20181020_195834')


## LOAD TEST ----
test.data <- h2o.importFile(path = paste0(getwd(), "/test2.csv") ,destination_frame = "corpus", header = TRUE)

# #tokenise word
# test.words <- tokenize(test.data$nsw_token)
# # embeding
# test.label.vecs <- h2o.transform(w2v.model2, test.words, aggregate_method = "AVERAGE")
# 
# !is.na(test.label.vecs$C1)
# test.valid.labels <- ! is.na(test.label.vecs$C1)
# 
# predict.data <- h2o.cbind(test.data[test.valid.labels, "id"], test.label.vecs[test.valid.labels, ])

predict <- function(test.data, w2v, model) {
  words <- tokenize(as.character(as.h2o(test.data$nsw_token)))
  test.vec <- h2o.transform(w2v, words, aggregate_method = "AVERAGE")
  h2o.predict(model, test.vec)
}


a<-predict(test.data,w2v.model2,aml3@leader)

result <- h2o.cbind(test.data$id,a$predict)
result
final_df <- as.data.frame(result)




write.table(final_df,"ensemble_20182010_w2v_e20_v200_w20_f5.txt",sep = " ", row.names = F,col.names = F, quote = F)


