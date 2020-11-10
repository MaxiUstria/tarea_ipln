import string

import fasttext
import nltk
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from spacy.lang.es import Spanish

nltk.download("stopwords")

STOPLIST = stopwords.words("spanish")
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

parser = Spanish()

model = fasttext.train_unsupervised('fasttext.es.300.txt', model='skipgram')


def spacy_tokenizer(sentence):
    tokens = parser(sentence)

    tokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens
    ]

    tokens = [
        word for word in tokens if word not in STOPLIST and word not in SYMBOLS]

    return tokens


for token in spacy_tokenizer("prueba"):
    print(model[token])
