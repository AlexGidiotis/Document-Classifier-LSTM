# Document-Classifier-LSTM
Recurrent Neural Networks for multilclass, multilabel classification of texts. The models that learn to tag samll texts with 169 different tags from arxiv. 

In classifier.py is implemented a standard BLSTM network with attention.

In hatt_classifier.py you can find the implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).

The neural networks were built using Keras and Tensorflow.

The best performing model is the attention BLSTM that achieves a micro f-score of 0.67 on the test set.

The Hierarchical Attention Network achieves only 0.65 micro f-score.

I am using 500k paper abstracts from arxiv. In order to download your own data refer to the [arxiv OAI api](https://arxiv.org/help/bulk_data).

Pretrained word embeddings can be used. The embeddings can either be GloVe or Word2Vec. You can download the   [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec) or the [GloVe embeddings](https://nlp.stanford.edu/projects/glove). 


## Usage:

1) In order to train your own model you must prepare your data set using the data_prep.py script. The preprocessing converts to lower case, tokenizes and removes very short words. The preprocessed files and label files should be saved in a /data folder.

2) You can now run classifier.py or hatt_classifier.py to build and train the models.

3) The trained models are exported to json and the weights to h5 for later use.

4) You can use utils.visualize_attention to visualize the attention weights.


## Requirements

- Python
- NLTK
- NumPy
- Pandas
- SciPy
- OpenCV
- scikit-learn
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Keras](https://github.com/fchollet/keras)

Run `pip install -r requirements.txt` to install the requirements.