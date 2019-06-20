# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:47:09 2019

@author: Vithya
"""

import sys
import pickle
import warnings
import traceback
from tokenization import tokenize_only
warnings.filterwarnings("ignore")

class ExtractTypes():
   
    def __init__(self):
        # Getting model:
        with open("D://TesoTicketMatching//Data//tesco_trained_top_13categories.pkl", 'rb') as m:
            self.textModel = pickle.load(m)
        
    def get_text_category(self, docText):
        
        text_category = 'NA'
        try:
            text_category = self.textModel.predict([docText]) 
            #result= {"Predicted_Sub_Category":text_category[0]}
            
        except:
            print(traceback.format_exc())
        #return result            
        return text_category[0]
        
    

    