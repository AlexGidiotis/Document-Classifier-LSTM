
import json
import re
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Convolution1D, MaxPooling1D, Flatten, concatenate, GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D ,TimeDistributed
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import regularizers


import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from attention import AttentionWithContext
from data_gen import hierarchicalCorpus as Corpus


# Modify this paths as well
DATA_DIR = '/home/alex/Documents/git_projects/Document-Classifier-LSTM/data/'
TRAIN_FILE = 'train_set.csv'
TRAIN_LABS = 'train_set_labels.csv'
EMBEDDING_FILE = '/home/alex/Documents/Python/glove.6B/glove.6B.200d.txt'

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 80000
# Max number of words in each abstract.
MAX_SEQUENCE_LENGTH = 100 # MAYBE BIGGER
MAX_SENT_LEN = 25
MAX_SEQ_LEN = 5
# This is fixed.
EMBEDDING_DIM = 200
# The name of the model.
STAMP = 'doc_hatt_blstm'


def f1_score(y_true, y_pred):
	"""
	Compute the micro f(b) score with b=1.
	"""
	y_true = tf.cast(y_true, "float32")
	y_pred = tf.cast(tf.round(y_pred), "float32") # implicit 0.5 threshold via tf.round
	y_correct = y_true * y_pred


	sum_true = tf.reduce_sum(y_true, axis=1)
	sum_pred = tf.reduce_sum(y_pred, axis=1)
	sum_correct = tf.reduce_sum(y_correct, axis=1)


	precision = sum_correct / sum_pred
	recall = sum_correct / sum_true
	f_score = 2 * precision * recall / (precision + recall)
	f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)


	return tf.reduce_mean(f_score)


def load_data(train_set,
	multilabel=True):
	"""
	"""

	X_data = []
	y_data = []
	for c,(vector,target) in enumerate(train_set):
		X_data.append(vector)
		y_data.append(target)
		if c % 10000 == 0: 
			print(c)

	num_texts = len(X_data)
	print(num_texts, 'training examples')

	
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
		oov_token=1)

	X_data_flat = []
	for raw_txt in X_data:
		flat_txt = ''
		for sent in raw_txt:
			flat_txt += sent
		X_data_flat.append(flat_txt)


	tokenizer.fit_on_texts(X_data_flat)
	X_data_int = np.zeros((num_texts,MAX_SEQ_LEN,MAX_SENT_LEN))
	for idx,raw_txt in enumerate(X_data):
		sentences_batch = np.zeros((MAX_SEQ_LEN,MAX_SENT_LEN))
		tokens = tokenizer.texts_to_sequences(raw_txt)
		sentences =  pad_sequences(tokens,
			maxlen=MAX_SENT_LEN,
			padding='post',
			truncating='post',
			dtype='int32')
		for j,sent in enumerate(sentences):
			if j >= MAX_SEQ_LEN:
				break
			sentences_batch[j,:] = sent
		X_data_int[idx,:,:] = sentences_batch

	X_data = X_data_int
	print(('Shape of data tensor:', X_data.shape))

	
	word_index = tokenizer.word_index
	print(('Found %s unique tokens' % len(word_index)))
	with open('word_index.json', 'w') as fp:
		json.dump(word_index, fp)
	print('Exported word dictionary')


	class_freqs = Counter([y for y_seq in y_data for y in y_seq]).most_common()

	class_list = [y[0] for y in class_freqs]
	nb_classes = len(class_list)
	print(nb_classes,'classes')
	class_dict = dict(zip(class_list, np.arange(len(class_list))))

	with open('class_dict.json', 'w') as fp:
		json.dump(class_dict, fp)
	print('Exported class dictionary')


	y_data_int = []
	for y_seq in y_data:
		y_data_int.append([class_dict[y] for y in y_seq])


	if multilabel:
		mlb = MultiLabelBinarizer()
		mlb.fit([list(class_dict.values())])
		y_data = mlb.transform(y_data_int)
	else:
		y_data = to_categorical(y_data_int)
		y_h_data = to_categorical(y_h_data_int)
	print(('Shape of label tensor:', y_data.shape))

	X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
		train_size=0.8,
		test_size=0.2,
		random_state=42)

	return X_train, X_val, y_train, y_val, nb_classes, word_index


def prepare_embeddings(wrd2id): 
	"""
	"""

	vocab_size = MAX_NB_WORDS
	print("Found %s words in the vocabulary." % vocab_size)


	embedding_idx = {}
	glove_f = open(EMBEDDING_FILE)
	for line in glove_f:
		values = line.split()
		wrd = values[0]
		coefs = np.asarray(values[1:],
			dtype='float32')
		embedding_idx[wrd] = coefs
	glove_f.close()
	print("Found %s word vectors." % len(embedding_idx))


	embedding_mat = np.random.rand(vocab_size+1,EMBEDDING_DIM)

	wrds_with_embeddings = 0
	# Keep the MAX_NB_WORDS most frequent tokens.
	for wrd, i in wrd2id.items():
		if i > vocab_size:
			continue

		embedding_vec = embedding_idx.get(wrd)
		# words without embeddings will be left with random values.
		if embedding_vec is not None:
			wrds_with_embeddings += 1
			embedding_mat[i] = embedding_vec


	print(embedding_mat.shape)
	print('Words with embeddings:',wrds_with_embeddings)

	return embedding_mat, vocab_size


def build_model(nb_classes,
	word_index,
	embedding_dim,
	seq_length,
	stamp,
	multilabel=True):
	"""
	"""

	embedding_matrix, nb_words = prepare_embeddings(word_index)

	input_layer = Input(shape=(MAX_SEQ_LEN,MAX_SENT_LEN),
		dtype='int32')


	sentence_input = Input(shape=(MAX_SENT_LEN,),
		dtype='int32')
	embedding_layer = Embedding(input_dim=nb_words+1,
		output_dim=embedding_dim,
		input_length=MAX_SENT_LEN,
		weights=[embedding_matrix],
		embeddings_regularizer=regularizers.l2(0.00),
		trainable=True)(sentence_input)

	drop1 = SpatialDropout1D(0.3)(embedding_layer)

	sent_lstm = Bidirectional(LSTM(100, name='blstm_1',
		activation='tanh',
		recurrent_activation='hard_sigmoid',
		recurrent_dropout=0.0,
		dropout=0.4, 
		kernel_initializer='glorot_uniform',
		return_sequences=True),
		merge_mode='concat')(drop1)


	sent_att_layer = AttentionWithContext()(sent_lstm)
	sentEncoder = Model(sentence_input, sent_att_layer)
	sentEncoder.summary()
	
	textEncoder = TimeDistributed(sentEncoder)(input_layer)

	drop2 = Dropout(0.4)(textEncoder)

	lstm_1 = Bidirectional(LSTM(100, name='blstm_2',
		activation='tanh',
		recurrent_activation='hard_sigmoid',
		recurrent_dropout=0.0,
		dropout=0.4, 
		kernel_initializer='glorot_uniform',
		return_sequences=True),
		merge_mode='concat')(drop2)
	lstm_1 = BatchNormalization()(lstm_1)

	att_layer = AttentionWithContext()(lstm_1)


	drop3 = Dropout(0.5)(att_layer)
	

	if multilabel:
		predictions = Dense(nb_classes, activation='sigmoid')(drop3)

		model = Model(inputs=input_layer, outputs=predictions)

		adam = Adam(lr=0.001,
			decay=0.0)

		model.compile(loss='binary_crossentropy',
			optimizer=adam,
			metrics=[f1_score])


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
	"""
	"""

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
			metrics=[f1_score])
	else:
		model.compile(loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])

	return model


if __name__ == '__main__':

	multilabel,load_previous = sys.argv[1:]

	print(multilabel,load_previous)

	if multilabel == 'multi':
		multilabel = True
	else:
		multilabel = False


	if load_previous == 'load':
		load_previous = True
	else:
		load_previous = False


	train_set = Corpus(DATA_DIR+TRAIN_FILE,DATA_DIR+TRAIN_LABS)

	X_train, X_val, y_train, y_val, nb_classes, word_index = load_data(train_set,
		multilabel)

	if load_previous:
		model = load_model(STAMP,
			multilabel)
	else:
		model = build_model(nb_classes,
			word_index,
			EMBEDDING_DIM,
			MAX_SEQUENCE_LENGTH,
			STAMP,
			multilabel)

	if multilabel:
		monitor_metric = 'val_f1_score'
	else:
		monitor_metric = 'val_loss'

	early_stopping =EarlyStopping(monitor=monitor_metric,
		patience=5)
	bst_model_path = STAMP + '.h5'
	model_checkpoint = ModelCheckpoint(bst_model_path,
		monitor=monitor_metric,
		verbose=1,
		save_best_only=True,
		mode='max',
		save_weights_only=True)

	hist = model.fit(X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=100,
		batch_size=128,
		shuffle=True,
		callbacks=[model_checkpoint])

	print(hist.history)