from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk import pos_tag

# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('gathering', 'v'))
# print(lemmatizer.lemmatize('gathering', 'n'))
#
# stemmer = PorterStemmer()
# print(stemmer.stem('gathering'))

wordnet_tags = ['n', 'v']
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

stemmer = PorterStemmer()
print('Stemmed:' % [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])


def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())


lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]

print('Lemmatized:' % [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
