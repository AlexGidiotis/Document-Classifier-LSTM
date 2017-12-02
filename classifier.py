import json

import numpy as np
import pandas as pd

import gensim
from gensim import corpora
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# Modify this paths as well
DATA_DIR = '/home/alex/Documents/git_projects/Document-Classifier-LSTM/data/'
TRAIN_FILE = 'train_set.csv'
TRAIN_LABS = 'train_set_labels.csv'
EMBEDDING_FILE = '/home/alex/Documents/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 80000
# Max number of words in each abstract.
MAX_SEQUENCE_LENGTH = 100 # MAYBE BIGGER
# This is fixed.
EMBEDDING_DIM = 300
# The name of the model.
STAMP = 'doc_blstm'


class Corpus(object):
	# Data generator.
	# INitialize the input files.
	def __init__(self,in_file,target_file=None):
		self.in_file = in_file
		self.target_file = target_file
		self.__iter__()

	# Yield one row,target pair per iteration. Each row is an abstract (preprocessed)
	def __iter__(self):
		for i,(line,target_list) in enumerate(zip(open(self.in_file),open(self.target_file))):
			# We train using only the first of possibly multiple classes.
			yield ' '.join(line.strip().replace('-',' ').split(',')),target_list.strip().split(',')


#============================================================== Prepare word embeddings ===========================================================
def prepare_embeddings(word_index): 
	# Returns the embedding matrix for the words in our corpus.
	print('Preparing embedding matrix')
	print('Indexing word vectors')
	# Read the pre-trained embeddings.
	word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,
		binary=True)
	print('Found %s word vectors of word2vec' % len(word2vec.vocab))


	# Fill the embedding matrix.
	nb_words = min(MAX_NB_WORDS, len(word_index))
	embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
	for word, i in word_index.items():
		if i >= MAX_NB_WORDS: continue
		if word in word2vec.vocab:
			embedding_matrix[i] = word2vec.word_vec(word)
	# WOrds without embeddings are left with zeros.
	print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix,
		axis=1) == 0))


	return embedding_matrix, nb_words


#============================================================= Main function =====================================================================
train_set = Corpus(DATA_DIR+TRAIN_FILE,DATA_DIR+TRAIN_LABS)

# LOad the data.
X_data = []
y_data = []
for c,(vector,target) in enumerate(train_set):  # load one vector into memory at a time
	X_data.append(vector)
	y_data.append(target)
	if c % 10000 == 0: 
		print c

print len(X_data), 'training examples'

# Dictionary of classes.
class_list = list(set([y for y_seq in y_data for y in y_seq]))

nb_classes = len(class_list)
print nb_classes,'classes'
class_dict = dict(zip(class_list, np.arange(len(class_list))))
with open('class_dict.json', 'w') as fp:
	json.dump(class_dict, fp)
print 'Exported class dictionary'


# Prepare the labels for training.
y_data_int = []
for y_seq in y_data:
	y_data_int.append([class_dict[y] for y in y_seq])

y_data = np.array(y_data_int)

mlb = MultiLabelBinarizer()
y_data = mlb.fit_transform(y_data)
'''
y_data = to_categorical(y_data,
	num_classes=nb_classes)
'''

# Tokenize and pad text.
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_data)
X_data = tokenizer.texts_to_sequences(X_data)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
with open('word_index.json', 'w') as fp:
	json.dump(word_index, fp)
print 'Exported word dictionary'
X_data = pad_sequences(X_data,
	maxlen=MAX_SEQUENCE_LENGTH,
	padding='post',
	truncating='post',
	dtype='float32')
print('Shape of data tensor:', X_data.shape)
print('Shape of label tensor:', y_data.shape)


# LOad the embeddings. (Requires a lot of memory.) 
embedding_matrix, nb_words = prepare_embeddings(word_index)
#======================================================= Split the data into train/val sets =======================================================
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
	test_size=0.1,
	random_state=42)

#============================================================ Build the model =====================================================================
embedding_layer = Embedding(nb_words,
		EMBEDDING_DIM,
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
		trainable=False)

blstm_layer = Bidirectional(LSTM(300,
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	recurrent_dropout=0.0,
	dropout=0.1,
	kernel_initializer='glorot_uniform'),
	merge_mode='concat')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,),
	dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
blstm_1 = blstm_layer(embedded_sequences)

dense_1 = Dropout(0.1)(blstm_1)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(200,
	activation='relu',
	kernel_initializer='glorot_uniform')(dense_1)

dense_2 = Dropout(0.1)(dense_1)
dense_2 = BatchNormalization()(dense_2)

preds = Dense(nb_classes,
	activation='sigmoid')(dense_2)

#=========================================================== Train the model ===================================================================
model = Model(inputs=[sequence_input],
	outputs=preds)
model.compile(loss='binary_crossentropy',
	optimizer='adam',
	metrics=['categorical_accuracy'])

model.summary()
print(STAMP)


# Save the model.
model_json = model.to_json()
with open(STAMP + ".json", "w") as json_file:
	json_file.write(model_json)


early_stopping =EarlyStopping(monitor='val_loss',
	patience=10)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path,
	monitor='val_categorical_accuracy',
	verbose=1,
	save_best_only=True,
	save_weights_only=True)


hist = model.fit(X_train, y_train,
	validation_data=(X_val, y_val),
	epochs=200,
	batch_size=64,
	shuffle=True,
	callbacks=[early_stopping, model_checkpoint])