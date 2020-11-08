import string

import nltk
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from spacy.lang.es import Spanish

nltk.download("stopwords")

STOPLIST = stopwords.words("spanish")
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

parser = Spanish()


def spacy_tokenizer(sentence):
    tokens = parser(sentence)

    tokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens
    ]

    tokens = [word for word in tokens if word not in STOPLIST and word not in SYMBOLS]

    return tokens


class Predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def clean_text(text):
    return text.strip().lower()
