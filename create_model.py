from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import cufflinks as cf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
# Предварительные настройки, переменные и данные для кода
nltk.download("stopwords")

sns.set(style="darkgrid")

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

df = pd.read_csv('dataset2.csv')
RND_STATE = 73

# вежливо просим Наталью все же приступить к работе
segmenter = Segmenter()
morh_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stop_words = stopwords.words('russian')

# Самый тривиальный препроцессинг
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

df['content'] = df.content.apply(text_prep)

df2 = pd.read_csv('dataset2.csv')
df2['content'] = df2.content.apply(text_prep)
test = df2['content'].tolist()


model = Word2Vec(sentences=test,
                 vector_size=300,
                 min_count=10,
                 window=2,
                 seed=RND_STATE)
import joblib

# Сохранение модели
#joblib.dump(model, 'word2vecmodel.pkl')
# Это класс для создания эмбедингов и вырвавнивания весов по матрицу TF-IDF
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

#Делаем простенький пайплайн

vects = TfidfEmbeddingVectorizer(model)
vects.fit(test)
print(vects.transform((test)).shape)



