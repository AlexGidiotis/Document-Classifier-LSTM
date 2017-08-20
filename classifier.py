import gensim
from gensim import corpora
from gensim.models import KeyedVectors

import numpy as np
import pandas as pd

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

# Modify this paths as well
DATA_DIR = '/home/alex/Documents/Python/paper_tagger/data/'
TRAIN_FILE = 'train_set.csv'
TRAIN_LABS = 'train_set_labels.csv'
EMBEDDING_FILE = '/home/alex/Documents/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'


MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 100 # MAYBE BIGGER
EMBEDDING_DIM = 300
STAMP = 'doc_blstm'


class Corpus(object):

	def __init__(self,train,in_file,target_file=None):
		self.train = train
		self.in_file = in_file
		self.target_file = target_file
		self.__iter__()


	def __iter__(self):
		if self.train == True: 
			for i,(line,target_list) in enumerate(zip(open(self.in_file),open(self.target_file))):
				yield ' '.join(line.strip().replace('-',' ').split(',')),target_list.strip().split(',')[0]
		else:
			for i,(line,target_list) in enumerate(zip(open(self.in_file),open(self.target_file))):
				yield ' '.join(line.strip().replace('-',' ').split(',')),target_list.strip().split(',')[0]


#============================================================== Prepare word embeddings ===========================================================
def prepare_embeddings(word_index): 
    print('Preparing embedding matrix')

    print('Indexing word vectors')

    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    nb_words = min(MAX_NB_WORDS, len(word_index))

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS: continue
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return embedding_matrix, nb_words


train_set = Corpus(True,DATA_DIR+TRAIN_FILE,DATA_DIR+TRAIN_LABS)

X_data = []
y_data = []
for c,(vector,target) in enumerate(train_set):  # load one vector into memory at a time
	X_data.append(vector)
	y_data.append(target)
	if c % 10000 == 0: 
		print c

print len(X_data), 'training examples'
class_list = list(set(y_data))
nb_classes = len(class_list)
print nb_classes,'classes'
class_dict = dict(zip(class_list, np.arange(len(class_list))))
y_data = np.array([class_dict[y] for y in y_data])
y_data = to_categorical(y_data, num_classes=nb_classes)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_data)

X_data = tokenizer.texts_to_sequences(X_data)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

X_data = pad_sequences(X_data, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post', dtype='float32')
print('Shape of data tensor:', X_data.shape)
print('Shape of label tensor:', y_data.shape)

embedding_matrix, nb_words = prepare_embeddings(word_index)
#======================================================= Split the data into train/val sets =======================================================
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.25, random_state=42)

#============================================================ Build the model ====================================================================
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

blstm_layer = Bidirectional(LSTM(250, activation='tanh', recurrent_activation='hard_sigmoid', recurrent_dropout=0.0, dropout=0.5, 
            kernel_constraint=maxnorm(10), kernel_initializer='glorot_uniform'), merge_mode='concat')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
blstm_1 = blstm_layer(embedded_sequences)

dense_1 = Dropout(0.5)(blstm_1)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(dense_1)

dense_2 = Dropout(0.5)(dense_1)
dense_2 = BatchNormalization()(dense_2)

preds = Dense(nb_classes, activation='softmax')(dense_2)

#=========================================================== Train the model ===================================================================
model = Model(inputs=[sequence_input],
        outputs=preds)
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc'])

model.summary()
print(STAMP)

model_json = model.to_json()
with open(STAMP + ".json", "w") as json_file:
    json_file.write(model_json)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
        epochs=200, batch_size=256, shuffle=True, callbacks=[early_stopping, model_checkpoint])