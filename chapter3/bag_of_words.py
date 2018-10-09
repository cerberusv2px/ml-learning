from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]

vectorizer = CountVectorizer(stop_words='english', binary=True)
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
