# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:14:19 2019

@author: Vithya
"""

from nltk.stem import PorterStemmer
from nltk import sent_tokenize, word_tokenize
import re
import nltk
from nltk.corpus import stopwords
#from autocorrect import spell
import traceback
nltk.download('punkt')   
import string
ps = PorterStemmer()
lemmer = nltk.stem.WordNetLemmatizer()
from nltk.tokenize import WordPunctTokenizer
text_tokenizer = WordPunctTokenizer()

#set a threshold value and remove words with len<2 and len>15
def filter_low_high_len_words(tokens):
    fn_filter = lambda x: x if (len(x)>=2 and len(x)<15) else ''
    filter_low_high_len_words = list(filter(fn_filter, tokens)) if tokens else []
    return filter_low_high_len_words

#generate bi grams from the tokens
def gen_bigrams(tokens):
    bigrm = nltk.bigrams(tokens)
    return [' '.join(i) for i in bigrm]

#tokenizing the words
def tokenize_only(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Tokenize the words
    3. decoding using utf-8
    4. Filter words of length>15 or length<2
    5. Remove all stopwords
    6. Lemmatize the words
    """
    #filter punctuation and replace it by ' '
    regex = re.compile('[^a-zA-Z]')
    #First parameter is the replacement, second parameter is your input string
    text = regex.sub(' ', text)
    text = ''.join([i for i in text.replace('\d+','')])
    tokens = [word.lower() for word in text_tokenizer.tokenize(text)]
    #filter words having length<2 or length>15
    #tokens = filter_low_high_len_words(tokens)
    #lemmatize the tokens
    tokens = [lemmer.lemmatize(token) for token in tokens]
    tokens = [ps.stem(token) for token in tokens]
    #generate bigrams from the tokens
    bigrams = gen_bigrams(tokens)
    #filter stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    #join all tokens
    return tokens+bigrams