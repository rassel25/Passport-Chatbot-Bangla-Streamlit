import numpy as np
from bltk.langtools import Tokenizer
from bangla_stemmer.stemmer.stemmer import BanglaStemmer
tokenizer = Tokenizer()

def tokenise(sentence):
    return tokenizer.word_tokenizer(sentence)

def stemming(words):
    return BanglaStemmer().stem(words)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemming(wd) for wd in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype = np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
           bag[index] = 1.0
    return bag