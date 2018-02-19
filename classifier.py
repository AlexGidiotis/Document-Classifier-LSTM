import json
import re

import numpy as np
import pandas as pd

import gensim
from gensim import corpora
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Convolution1D, MaxPooling1D, Flatten, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


# Modify this paths as well
DATA_DIR = '/home/alex/Documents/git_projects/Document-Classifier-LSTM/data/'
TRAIN_FILE = 'train_set.csv'
TRAIN_LABS = 'train_set_labels.csv'
EMBEDDING_FILE = '/home/alex/Documents/Python/glove.6B/glove.6B.200d.txt'

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 80000
# Max number of words in each abstract.
MAX_SEQUENCE_LENGTH = 100 # MAYBE BIGGER
# This is fixed.
EMBEDDING_DIM = 200
# The name of the model.
STAMP = 'doc_blstm'


class Corpus(object):
	# Data generator.
	# INitialize the input files.
	def __init__(self,in_file,
		target_file=None):
		self.in_file = in_file
		self.target_file = target_file
		self.__iter__()

	# Yield one row,target pair per iteration. Each row is an abstract (preprocessed)
	def __iter__(self):
		for i,(line,target_list) in enumerate(zip(open(self.in_file),open(self.target_file))):
			# We train using only the first of possibly multiple classes.
			labels = target_list.strip().split(',')

			yield ' '.join(line.strip().replace('-',' ').split(',')),labels


def load_data(train_set,
	multilabel=True):
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

	y_data_int = []
	for y_seq in y_data:
		y_data_int.append([class_dict[ y_seq[0]]])

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

	if multilabel:
		mlb = MultiLabelBinarizer()
		mlb.fit([class_dict.values()])
		y_data = mlb.transform(y_data_int)
	else:
		y_data = to_categorical(y_data_int)
		y_h_data = to_categorical(y_h_data_int)

	print('Shape of label tensor:', y_data.shape)

	X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
		test_size=0.1,
		random_state=42)

	return X_train, X_val, y_train, y_val, nb_classes, word_index


def prepare_embeddings(wrd2id): 
	vocab_size = len(wrd2id)
	print "Found %s words in the vocabulary." % vocab_size

	embedding_idx = {}
	glove_f = open(EMBEDDING_FILE)
	for line in glove_f:
		values = line.split()
		wrd = values[0]
		coefs = np.asarray(values[1:],
			dtype='float32')
		embedding_idx[wrd] = coefs
	glove_f.close()
	print "Found %s word vectors." % len(embedding_idx)

	embedding_mat = np.zeros((vocab_size+1,EMBEDDING_DIM))
	for wrd, i in wrd2id.items():
		embedding_vec = embedding_idx.get(wrd)
		# words without embeddings will be left with zeros.
		if embedding_vec is not None:
			embedding_mat[i] = embedding_vec

	print embedding_mat.shape
	return embedding_mat, vocab_size


def build_model(nb_classes,
	word_index,
	embedding_dim,
	seq_length,
	stamp,
	multilabel=True):

	embedding_matrix, nb_words = prepare_embeddings(word_index)

	input_layer = Input(shape=(seq_length,),
		dtype='int32')

	embedding_layer = Embedding(input_dim=nb_words+1,
		output_dim=embedding_dim,
		input_length=seq_length,
		weights=[embedding_matrix],
		trainable=True)(input_layer)

	lstm_1 = Bidirectional(LSTM(100, name='blstm_1',
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	recurrent_dropout=0.0,
	dropout=0.5, 
	kernel_initializer='glorot_uniform',
	return_sequences=True),
	merge_mode='concat')(embedding_layer)

	lstm_2 = Bidirectional(LSTM(100, name='blstm_2',
	activation='tanh',
	recurrent_activation='hard_sigmoid',
	recurrent_dropout=0.0,
	dropout=0.5, 
	kernel_initializer='glorot_uniform',
	return_sequences=False),
	merge_mode='concat')(lstm_1)

	drop2 = Dropout(0.5)(lstm_2)

	dense1 = Dense(512,
		activation='relu',
		kernel_initializer='lecun_uniform')(drop2)
	dense1 = BatchNormalization()(dense1)

	drop3 = Dropout(0.5)(dense1)

	if multilabel:
		predictions = Dense(nb_classes, activation='sigmoid')(drop3)

		model = Model(inputs=input_layer, outputs=predictions)

		adam = Adam(lr=0.001,
			decay=0.0)

		model.compile(loss='binary_crossentropy',
			optimizer=adam,
			metrics=[])


	else:
		predictions = Dense(nb_classes, activation='softmax')(drop3)

		model = Model(inputs=input_layer, outputs=predictions)

		adam = Adam(lr=0.001,
			decay=0.0)

		model.compile(loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])

	model.summary()
	print(stamp)

	# Save the model.
	model_json = model.to_json()
	with open(stamp + ".json", "w") as json_file:
		json_file.write(model_json)


	return model


def load_model(stamp,
	multilabel=True):
	json_file = open(stamp+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights(stamp+'.h5')
	print("Loaded model from disk")

	model.summary()

	adam = Adam(lr=0.001)
	if multilabel:
		model.compile(loss='binary_crossentropy',
			optimizer=adam,
			metrics=[])
	else:
		model.compile(loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])

	return model

def fine_tune_model(stamp,
	nb_classes):

	json_file = open(stamp+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights(stamp+'.h5')
	print("Loaded model from disk")

	last_dense = model.layers[-2].output

	predictions = Dense(nb_classes,
		activation='sigmoid',
		name='prediction')(last_dense)

	new_model = Model(inputs=model.layers[0].input, outputs=predictions)

	new_model.summary()

	adam = Adam(lr=1e-5)
	
	new_model.compile(loss='binary_crossentropy',
		optimizer=adam,
		metrics=[])

	return new_model


if __name__ == '__main__':
	multilabel = True

	load_previous = raw_input('Type yes/no/fine-tune if you want to load previous model: ')

	train_set = Corpus(DATA_DIR+TRAIN_FILE,DATA_DIR+TRAIN_LABS)

	X_train, X_val, y_train, y_val, nb_classes, word_index = load_data(train_set,
		multilabel)

	if load_previous == 'yes':
		model = load_model(STAMP,
			multilabel)
	elif load_previous == 'fine-tune':
		model = fine_tune_model(STAMP,
			nb_classes)
	else:
		model = build_model(nb_classes,
			word_index,
			EMBEDDING_DIM,
			MAX_SEQUENCE_LENGTH,
			STAMP,
			multilabel)

	early_stopping =EarlyStopping(monitor='val_loss',
		patience=5)
	bst_model_path = STAMP + '.h5'
	model_checkpoint = ModelCheckpoint(bst_model_path,
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=True)

	hist = model.fit(X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=200,
		batch_size=128,
		shuffle=True,
		callbacks=[early_stopping, model_checkpoint])

	print hist.history