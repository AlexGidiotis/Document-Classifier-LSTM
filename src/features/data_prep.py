"""Module for data preprocessing functions"""
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
	words = [word.lower() for word in word_tokenize(text)]
	tokens = words
	# Remove non characters
	p = re.compile('[a-zA-Z#]+')
	# Filter tokens (we do not remove stopwords)
	filtered_tokens = list([token for token in tokens if p.match(token) and len(token)>=min_length and (token not in english_stopwords)])
	# Encode to ascii
	filtered_tokens = [token.encode('ascii','ignore') for token in filtered_tokens]

	return filtered_tokens
