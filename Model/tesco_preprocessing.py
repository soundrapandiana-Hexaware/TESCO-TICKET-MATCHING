# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:58:26 2019

@author: Vithya
"""


import pandas as pd
from dateutil.parser import parse
import re

old_data =  pd.read_excel("D://TesoTicketMatching//Data//sub_category_stats.xlsx")
new_data =  pd.read_excel("D://TesoTicketMatching//Data//training_dataset_latest.xlsx")

old_data.rename(columns={"IM Number":"IM_Number"},inplace=True)

new_data.shape

tesco_data = pd.concat([old_data,new_data])
tesco_data=tesco_data.reset_index(drop=True)
tesco_data.shape

final_data = tesco_data.sort_values('Completion time').groupby('IM_Number').last()
final_data.reset_index(inplace=True)
final_data.columns
final_data.shape

check  = tesco_data.sort_values('Completion time').groupby('IM_Number').last()
check.reset_index(inplace=True)

#final_data.to_excel('D://TesoTicketMatching//Data//training_dataset_V1.xlsx', index = False)

len(check['Category'].unique().tolist())
len(check['Sub_Category'].unique().tolist())

len(check['IM_Number'].unique().tolist())

def is_string(string):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    
    try: 
        a,b = parse(string, fuzzy_with_tokens=True)
        #print(a)
        #print(b)
        values = ''.join(b)
        #
        return values

    except :
        return string

#val = parse('PO Receipts : TH RMS SL DI Report - Thu Apr 18 11:20:02 ICT 2019',fuzzy_with_tokens = True )
#print(val)
check['Ticket_Subject_v1'] = check['Ticket_Subject'].apply(is_string) 
check['Ticket_Subject_v1'] = check['Ticket_Subject_v1'] .map(lambda x: re.sub(r'\W+', ' ', x))

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)
 
check['Ticket_Subject_v1'] = check['Ticket_Subject_v1'].apply(remove_non_ascii)

#check.to_excel('D://TesoTicketMatching//Data//preprocessed_training_dataset.xlsx', index = False)











    
    
   