from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfEmbeddingVectorizer(object):

  def __init__(self, model):
    self.word2vec = model.wv
    self.word2weight = None
    self.dim = model.vector_size

  def fit(self, X, y):
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(X)

    max_idf = max(tfidf.idf_)
    self.word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return self

  def transform(self, X):
    return np.array([
            np.mean([self.word2vec.get_vector(w) * self.word2weight[w]
                    for w in words if w in self.word2vec] or
                  [np.zeros(self.dim)], axis=0)
          for words in X
    ])