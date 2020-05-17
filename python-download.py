import stanfordnlp
import nltk
import gensim.downloader as api

nltk.download('cmudict')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stanfordnlp.download('en', force=True)

GLOVE_MODEL = api.load('glove-wiki-gigaword-50')
WORD2VEC_MODEL = api.load('word2vec-google-news-300')
