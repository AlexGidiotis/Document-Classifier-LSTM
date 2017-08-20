# Document-Classifier-LSTM
A bidirectional LSTM that reads the abstract of a paper and classifies it into 165 different categories from arxiv.

The architecture is still far from optimal. The trained model currently achieves 58% validation accuracy.

I am using 500k paper abstracts from arxiv. In order to download your own data refer to the arxiv OAI api https://arxiv.org/help/bulk_data.

The embedding space uses the trained word vectors from GoogleNews-vectors-negative300. You can download the .bin file here https://code.google.com/archive/p/word2vec/. 

The model is built using TensorFlow and Keras.

List of dependencies:
numpy
pandas
nltk
sklearn
gensim
keras
tensorfow
csv