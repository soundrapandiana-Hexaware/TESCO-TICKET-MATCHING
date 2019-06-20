# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:05:38 2019

@author: Vithya
"""

import re
import nltk
import string
import pickle 
import warnings
import traceback
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import WordPunctTokenizer

nltk.download('punkt')   
text_tokenizer = WordPunctTokenizer()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
nltk.download('wordnet')
nltk.download('stopwords') 
warnings.filterwarnings('ignore')
from nltk.stem import PorterStemmer,WordNetLemmatizer
ps = PorterStemmer()
lemmer = nltk.stem.WordNetLemmatizer()


df = pd.read_excel("D://TesoTicketMatching//Data//preprocessed_training_dataset.xlsx")
#df.replace(" ", np.nan, inplace=True)


def is_empty_string(test_str1):
    if not test_str1.strip():
        values =  np.nan
        return values
    else: 
        return test_str1

df['Ticket_Subject_v1'] = df['Ticket_Subject_v1'].apply(is_empty_string)    


df=df.dropna()


top_accuracy_df = df[(df.Sub_Category == 'CASH - DWC - MATCHING BETWEEN RESA & SL') | (df.Sub_Category == 'STOCK - IBT - IBT MISSING IN DESTINATION')| (df.Sub_Category == 'PRICE - WRONG PRICE IN SL')
                     | (df.Sub_Category == 'DI - INV ADJ - DIFF BTWN SL & GO')| (df.Sub_Category == 'STOCK - PO - ORDER LOKCED')| (df.Sub_Category == 'DI - IBT RECEIPT - MISSING IN RMS')
                     | (df.Sub_Category == 'DI - PO - RECEIPTS - MISSING IN RMS')| (df.Sub_Category == 'DI - RTV - MISSING IN RMS')| (df.Sub_Category == 'STOCK - GAP SCAN - ORS REPORT DIFFERENCES')
                     | (df.Sub_Category == 'DI - BOL - TICB BOLS MISSING IN SL & IMOF')| (df.Sub_Category == 'DI - STOCK COUNT - MISSING IN RMS')
                     | (df.Sub_Category == 'DI - BOL - RECEIPTS - MISSING IN RMS')| (df.Sub_Category == 'DI - IBT OUT - MISSING IN RMS')]


top_accuracy_df.reset_index(drop=True,inplace=True)


top_accuracy_df

#top_accuracy_df.to_excel("D://TesoTicketMatching//Data//tesco_top_accuracy_subcategories.xlsx",index=False)



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
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for word in text_tokenizer.tokenize(text)]
    #filter words havinh length<2 or length>15
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



top_accuracy_df['Ticket_Subject_v1']= top_accuracy_df['Ticket_Subject_v1'].str.replace('\d+', '')

#porter_stemmer = PorterStemmer()
lem = WordNetLemmatizer()

def lemmatize_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [lem.lemmatize(token) for token in tokens]
    return ' '.join(stemmed_tokens)

top_accuracy_df['Ticket_Subject_v1'] = top_accuracy_df['Ticket_Subject_v1'].apply(lemmatize_sentences)



porter_stemmer = PorterStemmer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

top_accuracy_df['Ticket_Subject_v1'] = top_accuracy_df['Ticket_Subject_v1'].apply(stem_sentences)



#new_df.reset_index(drop=True,inplace=True)



X_train, X_test, y_train, y_test = train_test_split(top_accuracy_df['Ticket_Subject_v1'], top_accuracy_df['Sub_Category'], test_size=0.2, random_state=42)

classifier_data = Pipeline([('vectorizer', TfidfVectorizer(tokenizer=tokenize_only)), ('clf', OneVsRestClassifier(SVC(kernel='linear')))])
classifier_data.fit(X_train, y_train)
pred_y = classifier_data.predict(X_test)
pred_x = classifier_data.predict(top_accuracy_df['Ticket_Subject_v1'])
#(pred_x)
#pred_y


# get the accuracy
print(accuracy_score(y_test, pred_y))


predicted_res= pd.DataFrame(pred_x)
predicted_res.rename(columns={0:'predicted_result'},inplace=True)


df_out = pd.merge(top_accuracy_df,predicted_res[['predicted_result']],how = 'left',left_index = True, right_index = True)
df_out.to_excel("predicted_output_top_accuracy.xlsx",index=False)

pred_data = classifier_data.predict(["DI monitoring - report 28HU - Inventory Adjustments difference between SL/GO"])


'''
import pickle

pkl_filename = "tesco_trained_top_13categories.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(classifier_data, file)  
'''    

