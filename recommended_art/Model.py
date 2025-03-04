import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np

# Загружаем модель и готовимся к препроцессингу
model = joblib.load('word2vecmodel.pkl')  # Модель уже обучена на большом массиве спаршенных статей
segmenter = Segmenter()
morph_vocab = MorphVocab() 
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
stop_words = stopwords.words('russian')
nltk.download("stopwords")

# Самый тревиальный препроцессинг
def text_prep(text) -> str:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = [_.lemma for _ in doc.tokens]
    words = [lemma for lemma in lemmas if lemma.isalpha() and len(lemma) > 2]
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


# Делаем эмбеддинг + учитываем частоту встречаемости слов (взвешиваем) => единый вектор для сравнения по косинусному расстоянию
class TfidfEmbeddingVectorizer(object):

    def __init__(self, model):
        self.word2vec = model.wv
        self.word2weight = None
        self.dim = model.vector_size

    def fit(self, X):
        """Обучение TF-IDF на загруженном датасете (модуль parse.py)."""
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )
        return self

    def transform(self, X):
        """Трансформация списка токенов в векторы с учетом весов TF-IDF."""
        return np.array([
            np.mean([
                self.word2vec.get_vector(w) * self.word2weight[w]
                for w in words if w in self.word2vec
            ] or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def get_probability(data, themes):
    """
    Рассчитывает вероятность соответствия темам для набора текстов.

    data: Список текстов (статей).
    themes: Список тем.
    Возвращает словарь с проранжированными статьями для каждой темы.
    """
    # Препроцессинг данных и тем
    preprocessed_data = [text_prep(text) for text in data]
    preprocessed_themes = [text_prep(theme) for theme in themes]

    # Создаем словарь для хранения результатов
    themes_articles = {}  # Здесь будут храниться проранжированные списки статей для каждой темы

    # Для каждой темы вычисляем косинусное сходство со всеми текстами
    for theme in preprocessed_themes:
        # Преобразуем тему в вектор
        vectorizer_theme = TfidfEmbeddingVectorizer(model)
        vectorizer_theme.fit([theme.split()])  # Обучаем на токенах темы
        vector_theme = vectorizer_theme.transform([theme.split()])[0]  # Получаем вектор темы

        # Создаем список для хранения сходства между темой и текстами
        theme_recommendations = []

        for text in preprocessed_data:
            # Преобразуем текст в вектор
            vectorizer_text = TfidfEmbeddingVectorizer(model)
            vectorizer_text.fit([text.split()])  # Обучаем на токенах текста
            vector_text = vectorizer_text.transform([text.split()])[0]  # Получаем вектор текста

            # Вычисляем косинусное сходство (прям вручную)
            similarity = (
                np.dot(vector_text, vector_theme) /
                (np.linalg.norm(vector_text) * np.linalg.norm(vector_theme))
            )

            # Добавляем кортеж (сходство, исходный текст) в список рекомендаций
            theme_recommendations.append((similarity, text))

        # Сортируем рекомендации по убыванию сходства и сохраняем только тексты
        sorted_recommendations = [
            text for _, text in sorted(theme_recommendations, key=lambda x: x[0], reverse=True)
        ]

        # Сохраняем результаты для текущей темы
        themes_articles[theme] = sorted_recommendations

    return themes_articles # С этим словарем будет работать уже в модуле отправке обратного сообщения (в разработке)


