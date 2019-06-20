# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:48:38 2019

@author: Vithya
"""

import pandas as pd

storeline_data =  pd.read_excel("D://TescoTicketMatching//Data//DIMappedData.xlsx", sheet_name='MappedData')
print(storeline_data.head())


storeline_mapping_data =storeline_data [["Start time","Completion time","Category","New IM or Re-Work","Sub Category","IM Number"]]
storeline_mapping_data.rename(columns={"Sub Category":"Sub_Category"},inplace=True)

raw_data= pd.read_csv("D://TescoTicketMatching//Data//P3P4RawData_v1.csv")
raw_mapping_data = raw_data[["Ticket Id","Ticket Subject","Ticket Status","Date (Ticket Solved)","Date (Ticket Created)"]]
raw_mapping_data.rename(columns={"Ticket Id":"IM Number","Ticket Subject":"Ticket_Subject"},inplace=True)
raw_mapping_data['IM Number'] = raw_mapping_data['IM Number'].astype(str).str.strip()
storeline_mapping_data['IM Number'].dtypes


Columns = ["IM Number","Ticket_Subject","New IM or Re-Work","Category","Sub_Category","Start time",
           "Completion time","Ticket Status","Date (Ticket Solved)","Date (Ticket Created)"] 
#data.to_excel('D://TesoTicketMatching//Data//tesco_training_data.xlsx',columns = Columns,index=False)


df = pd.merge(storeline_mapping_data, raw_mapping_data, on='IM Number', how='outer')
#df
#df.to_excel('D://TesoTicketMatching//Data//tesco_data_outer.xlsx',columns = Columns,index=False)
data_category =  df[(df.Ticket_Subject.notnull())&(df.Category.notnull())&(df.Sub_Category.notnull())]
data_category.reset_index()
data_category.rename(columns={"IM Number":"IM_Number"},inplace=True)

len(data_category['IM Number'].unique().tolist())
#data_category.to_excel('D://TesoTicketMatching//Data//tesco_test_data.xlsx',columns = Columns,index=False)

unique_id_counts = data_category.IM_Number.value_counts()
unique_id_counts = unique_id_counts.to_frame().reset_index()
unique_id_counts.rename(columns={"index":"IM_Number","IM_Number":"Counts"},inplace=True)
unique_id_counts

count_mapping_data = pd.merge(data_category, unique_id_counts, on='IM_Number', how='outer')
count_mapping_data

(data_category['Completion time']).dtypes

Actual_Data = data_category.sort_values('Completion time').groupby('IM_Number').last()
Actual_Data.to_excel('D://TescoTicketMatching//Data//training_dataset_latest.xlsx')


