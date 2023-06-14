# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:45:47 2023

@author: Zohair
"""

# Importing all the necessary libraries
import tensorflow as tf
from tensorflow import keras 
import h5py
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Bidirectional

from Multimodal_baseline_Functions import *
from tensorflow.keras.layers import Reshape, Dropout
from tensorflow.keras.utils import plot_model


# import keras_metrics
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling3D
from tensorflow.keras import regularizers
import seaborn as sns
import matplotlib.pyplot as plt   
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers  
from tensorflow.keras.applications.inception_v3 import InceptionV3
###

import os
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from tqdm import tqdm

GLOVE_DIR = "data/glove.6B"
EMBEDDING_DIM = 50


datapath='data/MVSA_Single/'
datainfo='data/MVSA_Single/DataSameSentiment.xlsx'

df= pd.read_csv(datapath+'labelResultAll.txt', delim_whitespace=True)
df[['text,image','image']] = df['text,image'].str.split(',',expand=True)
df.rename(columns = {'text,image':'text'}, inplace = True)

#df=pd.read_excel(datainfo)  

df['status']=''

#update status if both are neutral
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
     if str(row['text'])=='neutral' and str(row['image'])=='neutral':
         df.at[index,'status']='both_neutral'
     else:
         df.at[index,'status']='0'



df['sentiment']=''

#update sentiment if alteast one neutral 
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
     if str(row['text'])=='neutral' or str(row['image'])=='neutral':
         if str(row['text']) !='neutral':
             df.at[index,'sentiment']= str(row['text'])
         else:
              df.at[index,'sentiment']= str(row['image'])
             
#update sentiment if both are Pos or Neg
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
     if str(row['text'])=='positive' and str(row['image'])=='positive':
         df.at[index,'sentiment']= str(row['text'])
         
     elif str(row['text'])=='negative' and str(row['image'])=='negative':
         df.at[index,'sentiment']= str(row['image'])
                      



#update sentiment if alteast one Pos or Neg
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
     if str(row['text'])=='positive' and str(row['image'])=='negative':
         df.at[index,'sentiment']= str(row['text'])
         
     elif str(row['text'])=='negative' and str(row['image'])=='positive':
         df.at[index,'sentiment']= str(row['image'])
                      

#update status if both are neutral


#delete rows where sentiment is not same
#df = df[df["matches"] != 0]
#df.reset_index(drop=True, inplace=True)

df['textinput']=''
df['imageinput']=''
for ind in df.index:
    print(df['ID'][ind])
    
    with open(datapath+'data/'+str(df['ID'][ind])+'.txt',encoding= 'unicode_escape') as f:
        contents = f.read()
        
        contents = re.sub('[!@#RT$]', '', contents)
        contents = re.sub(r'http\S+', '', contents)
        
        df.at[ind, 'textinput'] = contents.strip()
        df.at[ind, 'imageinput'] = datapath+str(df['ID'][ind])+'.jpg'
        print(contents)

    print('Done. Image and Text Path Frames')
    
    
    
    
df.to_excel(r'datacreated_large.xlsx', index=False)    
df.to_pickle("datacreated_large.pkl")      
  
df = pd.read_pickle("datacreated_large.pkl")      






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    