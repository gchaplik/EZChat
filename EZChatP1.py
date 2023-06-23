import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemm = PorterStemmer()
def token(input):
    return nltk.word_tokenize(input)

def LS(word):
    return stemm.stem(word.lower())

def BOW(TSent,low):
    TSent=[LS(w) for w in TSent]
    bag=np.zeros(len(low),dtype=np.float32)
    for index,w in enumerate(low):
        if w in TSent:
            bag[index]=1.0
    return bag


