# Document-Classifier-LSTM
A multilclass, multilabel classification model for texts. The model is a bidirectional LSTM with a max pooling layer on top that learns to
tag samll texts with 169 different tags from arxiv.

This neural network was built using Keras and Tensorflow.

The trained model achieves a micro f-score of 0.65 on the test set.

I am using 500k paper abstracts from arxiv. In order to download your own data refer to the arxiv OAI api https://arxiv.org/help/bulk_data.

Pretrained word e,beddings can be used. The embeddings can either be GloVe or Word2Vec. You can download the   GoogleNews-vectors-negative300.bin file here https://code.google.com/archive/p/word2vec/. 


## Usage:

1) In order to train your own model you must prepare your data set using the data_prep.py script. The preprocessing converts to lower case, tokenizes and removes very short words. The preprocessed files and label files should be saved in a /data folder.

2) You can now run the classifier.py script that will build and train the model.

3) The trained model is exported to json and the weights to h5 for later use.

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