import string

import fasttext
import nltk
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from spacy.lang.es import Spanish
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np


nltk.download("stopwords")

STOPLIST = stopwords.words("spanish")
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

parser = Spanish()

train = pd.read_csv("train.csv", sep='	', names=['text', 'hate'])
test = pd.read_csv("test.csv", sep='	', names=['text', 'hate'])

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(train['text'], train['hate'])

predicted = text_clf.predict(test['text'])
print(np.mean(predicted == test['hate']))
