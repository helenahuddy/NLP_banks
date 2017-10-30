from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
import pymorphy2
import tensorflow as tf
import numpy as np
word_vectors = Word2Vec.load_word2vec_format('ruscorpora.model.bin', binary=True) 
morph = pymorphy2.MorphAnalyzer()
rus = stopwords.words('russian')
def get_word_variants(word, word_vectors):
               
        try:
            return word_vectors[word]   #if 'word' in word_vectors.vocab
        except KeyError:
             try:
                p = morph.parse(word)[0]
                return word_vectors[p.normal_form]
             except KeyError:
                return []

def clean_str_vec(string):
        """
    Tokenization/string cleaning for all datasets except for SST.
    """
        string = re.sub(r"[^а-яА-Я0-9(),:!?\'\`]", " ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

def makeVec(text):  
        res=[]
        text=clean_str_vec(text).split()
        for word in text:
            if word in rus:
                pass
            else:
                res+=list(get_word_variants(word, word_vectors))
        return (res)


class w2w:
    """
    converting a string to array of numbers
    """
    def __init__(
      self,texts):
        self.texts=texts
        
     
    def vectors(self):
        return np.array([makeVec(i) for i in self.texts])
        