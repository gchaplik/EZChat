import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemm = PorterStemmer()
def token(input):
    return nltk.word_tokenize(input)

def LS(word):
    return stemm.stem(word.lower())

def BOW(TSent,low):
    pass


