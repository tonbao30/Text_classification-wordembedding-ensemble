# WORD EMBEDDING _ TRAIN SET (H2O)
library(caret)
library('dplyr')
library(h2o)
h2o.init( nthreads = -1,max_mem_size = '10G')
setwd("C:/Users/Bi PC/1.Sem2-2018/1.FIT 5149/Group Project/text_processing/data")
path_local <- getwd()
path <- paste0(path_local, "/corpus2.csv")
train.data <- h2o.importFile(path = path ,destination_frame = "corpus",col.names = c("index","id", "label","nsw_token")
                             ,header = TRUE)

tokenize <- function(sentences, stop.words = STOP_WORDS) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  
  # convert to lower case
  tokenized.lower <- h2o.tolower(tokenized)
  # remove short words (less than 2 characters)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]
  
}


words <- tokenize(train.data$nsw_token)


# train Word2vec model
# 
new.w2v.model <- h2o.word2vec(model_id = 'w2v_e20_v150_w30_f0',words, sent_sample_rate = 0, epochs = 20,vec_size = 150,window_size = 30) #vec_size:min_word_freq:word_model:
model_path <- h2o.saveModel(object=new.w2v.model, path=getwd(), force=TRUE)
# new.w2v.model <- h2o.loadModel("C:/Users/Bi PC/1.Sem2-2018/1.FIT 5149/Group Project/text_processing/data/W2V_20181911")

# create SVM feature dataframe

label.vecs <- h2o.transform(new.w2v.model, words, aggregate_method = "AVERAGE")


is.na(label.vecs$C1) # remove all documents having no features
valid.labels <- ! is.na(label.vecs$C1)

label.vecs[valid.labels, ]
data <- h2o.cbind(train.data[valid.labels, "label"], label.vecs[valid.labels, ]) # data frame (first column is label, other columns is feature)


## TRAIN SVM (USE WORKD EMBEDDING FEATURE)------
library(liquidSVM)
library(caret)

round(memory.limit(10000000000)/2^20, 2) # set memory limit for large vector size

data <- as.data.frame(data)

smp_size <- floor (0.75*nrow(data)) # split train/set

set.seed(12114)
train_data <- data[sample(nrow(data)),]


model <- mcSVM(label~.,train_data, mc_type="OvA_ls",gpus=1, threads = 8,display=1)

### ESTIMATE ACCURACY -----

train_pred <- predict(model, train_data)

train_conf_trrix<-confusionMatrix(train_pred,train_data$label)



## PREDICT RESULT-----------------

# process test.csv
path <- paste0(path_local, "/test2.csv")
test.data <- h2o.importFile(path = path ,destination_frame = "corpus"
                            ,header = TRUE)


test.words <- tokenize(test.data$nsw_token)

# new.w2v.model<- h2o.loadModel("C:/Users/Bi PC/1.Sem2-2018/1.FIT 5149/Group Project/text_processing/data/W2V_20181911")

test.label.vecs <- h2o.transform(new.w2v.model, test.words, aggregate_method = "AVERAGE")

# remove all empty docs
!is.na(test.label.vecs$C1)
test.valid.labels <- ! is.na(test.label.vecs$C1)

predict.data <- h2o.cbind(test.data[test.valid.labels, "id"], test.label.vecs[test.valid.labels, ])

final_test <- as.data.frame(predict.data)
nrow(test.data$id)
## PREDICT ----
test_pred <- predict(model, final_test)
result <- as.data.frame(final_test$id)
result$label <- test_pred


## COMBINE RESULT ------
df<- as.data.frame(test.data$id)
ids <- as.vector(df$id)
unique_id <- unique(ids)
rm(df)
final_label<- vector()

result$`final_test$id` <- as.character(result$`final_test$id`)
for (i in seq(1,26610)){
  if (unique_id[i] %in% as.character(result$`final_test$id`)){
    final_label[i] = as.character(result[ result$`final_test$id`==unique_id[i],]$label)
  }
  else{
    final_label[i] = 'c1'
  }
}


final_df <- data.frame(cbind(unique_id,final_label))

write.table(final_df,"77svm_full_train_test_label.txt",sep = " ", row.names = F,col.names = F, quote = F)


