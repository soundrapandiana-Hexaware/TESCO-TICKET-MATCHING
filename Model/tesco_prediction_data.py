# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:09:56 2019

@author: Vithya
"""
import pickle
import warnings
from tokenization import tokenize_only
warnings.filterwarnings("ignore")


pkl_filename = "D://TesoTicketMatching//Data//tesco_trained_top_13categories.pkl"

# Load from file
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)
    
Ticket_Subject ="Discrepancies on sales Stock excel and ReSa "    
Ypredict = pickle_model.predict([Ticket_Subject])
print("Predicted_SubCategory :",Ypredict[0]) 