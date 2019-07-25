# Text_classification-wordembedding-ensemble project

This project aims to build document classification models  based one labeled dataset "testing.docs.txt" to predict the targeted documents in "testing_label_pred.txt"

# Dataset Description
The data
## Data preparation 
Data preparation is the first step in the process of model development. This step was implemented in
the Python programming language. Both “training_docs.txt” and “testing_docs.txt” are preprocessed
using this technique.
Python library used: re, nltk, pandas, multiprocessing, sklearn.feature_extraction.text
Python code file: “Document preprocessing.ipynb”
Output file: “corpus2.csv” and “test 2.csv”
Data preparation steps are recorded below.
- Remove stop words
- Remove character “TEXT” at the beginning of the content
- Tokenise words by using regular expression with pattern r"\w+(?:[-.@']\w+)*"
- Lemmatise words
- Remove a word if the length of this word is less than 3
- Concatenate all remaining tokens into “nsw_token”, which will be used for feature selection
in next stages
- Produce clean data input files under the names “corpus2.csv” and “test 2.csv”
## Feature Extraction

## Feature Selection
Library used: h2o (version 3.20.0.10) Example output: w2v_e30_v200_w30_f0 In the beginning, we considered two directions for feature selection using TF-IDF and using word embedding. After testing several models, we realised the embedding feature outperforms TF-IDF in this prediction task. The library h2o with the function h2o.word2vec supported our feature selection in all models.

From the H20 documentation, we try the following h2o.word2vec setting to produce features.

    - Epochs: Specifies the number of training iterations to run.
    - vec_size: Specifies the size of word vectors.
    - window_size: This specifies the size of the context window around a specific word.
    - min_word_freq: Specifies an integer for the minimum word frequency. Word2vec will discard words that appear less than this number of times. 
    
    Reference: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/word2vec.html

We extracted the features used for one of our best models as an example of features selection as an H2O object named “w2v_e30_v200_w30_f0”. This object has the following setting:

   - epochs = 30
   - Vector size = 200
   - Window size = 30
   - min_word_freq = 0

# Model Selection

# Ensemble

# Result
