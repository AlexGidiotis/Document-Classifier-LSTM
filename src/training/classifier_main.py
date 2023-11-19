"""Module for running simple classifier"""
import logging
import os
import json
import re
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import SpatialDropout1D
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

from src.model.attention import AttentionWithContext
from src.model.classifier import f1_score, load_data, prepare_embeddings, build_model, load_model
from src.data_io.data_gen import Corpus


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Modify this paths as well
DATA_DIR = '<PATH_TO_DATA>/data/'
TRAIN_FILE = 'train_set.csv'
TRAIN_LABS = 'train_set_labels.csv'
EMBEDDING_FILE = '<PATH_TO_EMBEDDINGS>/glove.6B/glove.6B.200d.txt'
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 80000
# Max number of words in each abstract.
MAX_SEQUENCE_LENGTH = 100 # MAYBE BIGGER
# This is fixed.
EMBEDDING_DIM = 200
# The name of the model.
STAMP = 'doc_blstm'


def main():
    load_previous = sys.argv[1]

	logging.info(load_previous)

	if load_previous == 'load':
		load_previous = True
	else:
		load_previous = False


	train_set = Corpus(DATA_DIR+TRAIN_FILE,DATA_DIR+TRAIN_LABS)

	X_train, X_val, y_train, y_val, nb_classes, word_index = load_data(train_set)

	if load_previous:
		model = load_model(STAMP)
	else:
		model = build_model(nb_classes,
			word_index,
			EMBEDDING_DIM,
			MAX_SEQUENCE_LENGTH,
			STAMP)


	monitor_metric = 'val_f1_score'

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

	logging.info((hist.history))


if __name__ == "__main__":
    main()
