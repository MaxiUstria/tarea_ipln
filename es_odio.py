import string
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from spacy.lang.es import Spanish
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np


nltk.download("stopwords")

STOPLIST = stopwords.words("spanish")
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

path = Path(sys.argv[1])
test_list = sys.argv[2:]

train = pd.read_csv(path / "train.csv", sep='	', names=['text', 'hate'])


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(train['text'], train['hate'])


for test_file in test_list:
    test = pd.read_csv(test_file, sep='	', usecols=[0], names=['text'])
    predicted = text_clf.predict(test['text'])
    with open(f"{test_file[:-4]}.out", "w+") as output:
        output.write("\n".join(map(str, predicted)))
