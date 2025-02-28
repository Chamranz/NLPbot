import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from custom_pipeline import TfidfEmbeddingVectorizer
import cloudpickle
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np

model = joblib.load('word2vecmodel.pkl')
segmenter = Segmenter()
morh_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stop_words = stopwords.words('russian')
nltk.download("stopwords")

def text_prep(text) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morh_vocab)

    lemmas = [_.lemma for _ in doc.tokens]
    words = [lemma for lemma in lemmas if lemma.isalpha() and len(lemma) > 2]
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

class TfidfEmbeddingVectorizer(object):
    """Get tfidf weighted vectors"""
    def __init__(self, model):
        self.word2vec = model.wv
        self.word2weight = None
        self.dim = model.vector_size

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec.get_vector(w) * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
def get_propability(data, themes):
    data = [text_prep(text) for text in data]
    themes = [text_prep(text) for text in data]
    for text in data:
        vector = TfidfEmbeddingVectorizer(model)
        vector.transform(text)




