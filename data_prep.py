import os
import re
import csv

import pandas as pd
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')

# Simple preprocessing for texts.
def preprocess(text):
	min_length = 3
	text = re.sub('\d+','#',text)
	text = re.sub('\.',' eos ',text)
	# Tokenize
	words = map(lambda word: word.lower(), word_tokenize(text))
	tokens = words
	# Remove non characters
	p = re.compile('[a-zA-Z#]+')
	# Filter tokens (we do not remove stopwords)
	filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length and (token not in english_stopwords), tokens))
	# Encode to ascii
	filtered_tokens = [token.encode('ascii','ignore') for token in filtered_tokens]

	return filtered_tokens


# Modify this path
root_path = '/home/alex/Documents/Data/arxiv_data/'
test_split = 0.1


# Read all the data.
df = pd.DataFrame()
for doc in sorted(os.listdir(root_path)):
	if doc.split('_')[1] != 'dump': continue
	df_temp = pd.read_csv(root_path+doc,
		usecols=['abstract', 'categories'])
	df = df.append(df_temp,
		ignore_index=True)
# Shuffle the dataset.
df = df.sample(frac=1).reset_index(drop=True)


# Split to train and test set.
train_df = df[:int((1-test_split)*len(df))].reset_index(drop=True)
test_df = df[int((1-test_split)*len(df)):].reset_index(drop=True)
print train_df.shape[0],'training examples'
print test_df.shape[0],'test examples'


# Preprocess the data and labels for the train and test set.
X_train = []
y_train = []
for c,(abstr,labs) in enumerate(zip(train_df['abstract'].tolist(),train_df['categories'].tolist())):
	X_train.append(preprocess(abstr))
	labs = labs.strip('[').strip(']').split(',')
	labs = [lab.strip() for lab in labs]
	y_train.append(labs)
	if c % 10000 == 0: print c
X_test = []
y_test = []
for c,(abstr,labs) in enumerate(zip(test_df['abstract'].tolist(),test_df['categories'].tolist())):
	X_test.append(preprocess(abstr))
	labs = labs.strip('[').strip(']').split(',')
	labs = [lab.strip() for lab in labs]
	y_test.append(labs)
	if c % 10000 == 0: print c


# Write the outputs to .csv
print 'Writting...'
with open("data/train_set.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X_train)
with open("data/test_set.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X_test)
with open("data/train_set_labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(y_train)
with open("data/test_set_labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(y_test)