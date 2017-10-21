# Document-Classifier-LSTM
A bidirectional LSTM that reads the abstract of a paper and classifies it into 165 different categories from arxiv.

The architecture is still far from optimal. The trained model currently achieves 79% validation accuracy.

I am using 500k paper abstracts from arxiv. In order to download your own data refer to the arxiv OAI api https://arxiv.org/help/bulk_data.

The embedding space uses the trained word vectors from GoogleNews-vectors-negative300. You can download the .bin file here https://code.google.com/archive/p/word2vec/. 

The model is a toy project that illustrates how to build a simple LSTM for sequence classification using TensorFlow and Keras.


## Usage:

1) In order to train your own model you must prepare your data set using the data_prep.py script. The preprocessing converts to lower case, tokenizes and removes very short words. The preprocessed files and label files should be saved in a /data folder.

2) You can now run the classifier.py script that will build and train the model.

3) The trained model is exported to json and the weights to h5 for later use.


## List of dependencies:

1) numpy

2) pandas

3) nltk

4) sklearn

5) gensim

6) keras

7) tensorfow

8) csv
