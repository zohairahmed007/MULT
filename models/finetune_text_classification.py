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
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords



GLOVE_DIR = "data/glove.6B"
EMBEDDING_DIM = 50


datapath='data/MVSA_Single/data/'
datainfo='data/MVSA_Single/DataSameSentiment.xlsx'
  
df = pd.read_pickle("datacreated.pkl")      


''' #Seperate images of pos,neg
df_pos= df[df["text"] == 'positive']
df_neg= df[df["text"] == 'negative']
df_neu= df[df["text"] == 'neutral']

import shutil

for ind in df_neu.index:
    print(df['imageinput'][ind])
    original ='C:\\Users\\Zohair\\Desktop\\3rd_paper_experiments\\code\\multimodel\\data\\MVSA_Single\\data\\'+  str(df['ID'][ind])+'.jpg'
    target = 'C:\\Users\\Zohair\\Desktop\\3rd_paper_experiments\\code\\multimodel\\data\\MVSA_Single\\neu\\'+ str(df['ID'][ind])+'.jpg'
    
    shutil.copyfile(original, target)
    #break


    print('Done')'''
    
    
#pretrain encoder emb   
import tensorflow_hub as hub
import tensorflow_text
#use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
model_path ='data/tf/universal-sentence-encoder-multilingual-large_3' # in my case
# one thing the path you have to provide is for folder which contain assets, variable and model
# not of the model.pb itself

pretrained_model_load = hub.KerasLayer(model_path, trainable=True)



sent_1 = ["the location is great"]
sent_2 = ["amazing location"]

emb_1 = pretrained_model_load(sent_1)
emb_2 = pretrained_model_load(sent_2)

print(emb_1.shape)
print(" similarity between two embeddings",np.inner(emb_1, emb_2).flatten()[0])

len(pretrained_model_load.weights)
len(pretrained_model_load.variables)

#delete rows where sentiment is == neutral
df = df[df["text"] != 'neutral']

#convert pos, neg into 0 into 1
df["label"] = df["text"].apply( lambda x: 1 if x =='positive'  else 0 )

df_new = df[["textinput", "label"]]

#label distribution 
print(df.groupby(['label']).size())


#imbalance deal
good_reviews=df_new[df_new["label"] != 0]
bad_reviews=df_new[df_new["label"] != 1]

good_df = good_reviews.sample(n=len(bad_reviews), random_state=42)
bad_df = bad_reviews
review_df = good_df.append(bad_df).reset_index(drop=True)
print(review_df.shape)

#Balance label distribution 
print(review_df.groupby(['label']).size())


from sklearn.preprocessing import OneHotEncoder
type_one_hot = OneHotEncoder(sparse=False).fit_transform(
  review_df.label.to_numpy().reshape(-1, 1)
)

from sklearn.model_selection import train_test_split
train_reviews, test_reviews, y_train, y_test =  train_test_split(
    review_df.textinput,
    type_one_hot,
    test_size=.1,
    random_state=42
  )



#create emb from pretrained_model_load
from tqdm import tqdm

X_train = []
for r in tqdm(train_reviews):
  emb = pretrained_model_load(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_train.append(review_emb)
X_train = np.array(X_train)
X_test = []
for r in tqdm(test_reviews):
  emb = pretrained_model_load(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_test.append(review_emb)
X_test = np.array(X_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#classfication seqential


model = keras.Sequential()
model.add(
  keras.layers.Dense(
    units=256,
    input_shape=(X_train.shape[1], ),
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)
model.add(
  keras.layers.Dense(
    units=128,
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.summary()



from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
TRAINED_MODEL_PATH='./train_model/'+'sa'+'.{epoch:02d}--{val_accuracy:.2f}.hdf5'
checkpoint=ModelCheckpoint(TRAINED_MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=True,
    callbacks = [checkpoint]
)

model.evaluate(X_test, y_test)

#testing

print(test_reviews.iloc[0])
print("Bad" if y_test[0][0] == 1 else "Good")

y_pred = model.predict(X_test[:1])
print(y_pred)
"Bad" if np.argmax(y_pred) == 0 else "Good"


print(test_reviews.iloc[1])
print("Bad" if y_test[1][0] == 1 else "Good")


y_pred = model.predict(X_test[1:2])
print(y_pred)
"Bad" if np.argmax(y_pred) == 0 else "Good"


path_text="C:\\Users\\Zohair\\Desktop\\3rd_paper_experiments\\code\\multimodel\\text\\"

## get layer from places_mg_model (change directory to load best places_mg_model from folder 'training_models')
text_model=load_model(path_text+'train_model\\sa.06--0.86.hdf5')
predictions = text_model.predict(X_test)


"Bad" if np.argmax(predictions[129]) == 0 else "Good"
