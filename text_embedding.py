import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import re, string

path = r"glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)

def create_text_embedding(text):
    
    text = text.lower()
    text = strip_punc(text)
    text_array = text.split()
    
    captions = COCO.get_all_captions()
    
    embedding = np.zeros((1, 50))
    
    for item in text_array:
        count = 0
        for cap in captions:
            if item in cap:
                count += 1
        IDF = np.log10(len(captions) / count)
        embedding += glove[item] * IDF
        
    embedding /= np.linalg.norm(embedding)
    
    return embedding