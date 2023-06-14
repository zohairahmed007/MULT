#!/usr/bin/env python
# coding: utf-8



import h5py
import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K,optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Activation, Dense,Flatten,Dropout,MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D,concatenate,average
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications.mobilenet  import preprocess_input
from tensorflow.keras.utils import get_file,get_source_inputs
#from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
###################################################


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.regularizers import l2, l1

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import bert

import numpy as np 
import pandas as pd
import re
import glob
import os
import cv2
import sys
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

path_text="../../text/"
df = pd.read_pickle(path_text+"datacreated_large.pkl")   
df["label"] = df["text"].apply( lambda x: 1 if x =='positive'  else 0 )

df = df[df.sentiment != 'neutral']


df_orignal=df

df=df.drop(['text', 'image', 'status','sentiment'], axis=1)
df['image_path']=''

for i, row in df.iterrows():
    
    df.at[i,'image_path'] =str(row['ID'])+'.jpg'

df=df.drop(['ID', 'imageinput'], axis=1)

df=df.set_index('image_path')
df.rename(columns = {'textinput':'text','label':'label'}, inplace = True)

df.groupby('label').size()








from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

#train.to_pickle("train_large.pkl")      
#test.to_pickle("test_large.pkl")  

train=pd.read_pickle("train_large.pkl")    
test=pd.read_pickle("test_large.pkl")     


 #Seperate images of pos,neg

train_pos=train[train["label"] == 1]
train_neg=train[train["label"] == 0]



test = test.sort_values('image_path')
train = train.sort_values('image_path')

print(train.shape)
train.head()

batch_size =  80
img_width = 299
img_height = 299
depth = 3
max_length = 20 #Setup according to the text

nClasses = train.label.nunique()
Classes = train.label.unique()
input_shape = (img_width, img_height, depth)


def get_missing(file, df):
  parts = file.split(os.sep)
  idx = parts[-1]
  cls = parts[-2]
  indexes = df[:,0]
  classes = df[:,2]

  if idx in indexes:
    text = df[idx == indexes][0,1]
    return pd.NA, pd.NA, pd.NA
  else:
    text = df[cls == classes][0,1]
    
  return idx, text, cls   

vec_get_missing = np.vectorize(get_missing, signature='(),(m,n)->(),(),()')  

print("Number of training images:",train.shape[0])
print("Number of test images:",test.shape[0])

# Import the BERT BASE model from Tensorflow HUB (layer, vocab_file and tokenizer)

# Import the BERT BASE model from Tensorflow HUB (layer, vocab_file and tokenizer)


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# Preprocessing of texts according to BERT +
# Cleaning of the texts
 
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

def remove_tags(text):
    return TAG_RE.sub('', text)

TAG_RE = re.compile(r'<[^>]+>')
vec_preprocess_text = np.vectorize(preprocess_text)

def get_tokens(text, tokenizer):
  tokens = tokenizer.tokenize(text)
  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  length = len(tokens)
  if length > max_length:
      tokens = tokens[:max_length]
  return tokens, length  

def get_masks(text, tokenizer, max_length):
    """Mask for padding"""
    tokens, length = get_tokens(text, tokenizer)
    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))
vec_get_masks = np.vectorize(get_masks, signature = '(),(),()->(n)')

def get_segments(text, tokenizer, max_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens, length = get_tokens(text, tokenizer)
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))
vec_get_segments = np.vectorize(get_segments, signature = '(),(),()->(n)')

def get_ids(text, tokenizer, max_length):
    """Token ids from Tokenizer vocab"""
    tokens, length = get_tokens(text, tokenizer)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids
vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')

def get_texts(path):
    path = path.decode('utf-8')
    parts = path.split(os.sep)
    image_name = parts[-1]
    is_train = parts[-3] == 'train'
    if is_train:
      df = train
    else:
      df = test

    text = df['text'][image_name]
    return text





vec_get_text = np.vectorize(get_texts)
def prepare_text(paths):
    #Preparing texts
    
    texts = vec_get_text(paths)
    
    text_array = vec_preprocess_text(texts)
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze().astype(np.int32)
    masks = vec_get_masks(text_array,
                          tokenizer,
                          max_length).squeeze().astype(np.int32)
    segments = vec_get_segments(text_array,
                                tokenizer,
                                max_length).squeeze().astype(np.int32)
    
    return ids, segments, masks

def clean(i, tokens):
  try:
    this_token = tokens[i]
    next_token = tokens[i+1]
  except:
    return tokens
  if '##' in next_token:
      tokens.remove(next_token)
      tokens[i] = this_token + next_token[2:]
      tokens = clean(i, tokens)
      return tokens
  else:
    i = i+1
    tokens = clean(i, tokens)
    return tokens

def clean_text(array):
  array = array[(array!=0) & (array != 101) & (array != 102)]
  tokens = tokenizer.convert_ids_to_tokens(array)
  tokens = clean(0, tokens)
  text = ' '.join(tokens)
  return text


 

# Images preprocessing
# Images preprocessing
def load_image(path):
    path = path.decode('utf-8')
    image = cv2.imread(path)
    image = cv2.resize(image, (img_width, img_height))
    image = image/255
    image = image.astype(np.float32)
    parts = path.split(os.sep)
    labels =  Classes 
    labels = labels.astype(np.int32)
    
    return image, labels
    
vec_load_image = np.vectorize(load_image, signature = '()->(r,c,d),(s)')
    



def prepare_data(paths):
    #Images and labels
    images, labels = tf.numpy_function(vec_load_image, 
                                      [paths], 
                                      [tf.float32, 
                                        tf.int32])
    
    
    [ids, segments, masks, ] = tf.numpy_function(prepare_text, 
                                              [paths], 
                                              [tf.int32, 
                                               tf.int32,
                                               tf.int32])
    images.set_shape([None, img_width, img_height, depth])
    labels.set_shape([None, nClasses])
    ids.set_shape([None, max_length])
    masks.set_shape([None, max_length])
    segments.set_shape([None, max_length])
    return ({"input_word_ids": ids, 
             "input_mask": masks,  
             "segment_ids": segments, 
             "image": images},
            {"class": labels})
    

    #return dataset


# Images loading using tf.data
def tf_data(path, batch_size):
    paths = tf.data.Dataset.list_files(path)
    paths = paths.batch(64)
    dataset = paths.map(prepare_data, tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset   


data_train = tf_data('images/train/*/*.jpg', batch_size)
data_test = tf_data('images/test/*/*.jpg', batch_size)



ip, op = next(iter(data_train))
images = ip['image'][:16]
input_word_ids = ip['input_word_ids'][:16]
true_labels =  op['class'][:16]



# Images Model
 


 
model1 = tf.keras.models.load_model('V3/weights-improvement-27-0.65.hdf5')
layer_name = 'flatten'
model2= tf.keras.models.Model(inputs=model1.input, outputs=model1.get_layer(layer_name).output)

model_cnn = models.Sequential()
model_cnn.add(model2)
model_cnn.add(layers.Dense(128, name='Dense_128'))

model_cnn.summary()

# Keep model layers trainable
for layer in model_cnn.layers:
    layer.trainable = True



from tensorflow.keras.utils import plot_model
plot_model(model_cnn, to_file='model_cnn.png')

# Bert + LSTM text model
input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
input_masks = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_masks")
input_segments = layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
_, seq_out = bert_layer([input_ids, input_masks, input_segments])
out = layers.LSTM(128, name='LSTM')(seq_out)
model_lstm = models.Model([input_ids, input_masks, input_segments], out)

# Keep the Bert + LSTM layers trainable
for layer in model_lstm.layers:
    layer.trainable = True

model_lstm.summary()

plot_model(model_lstm, to_file='bert_lstm.png')



# Stochastic Gradient Descent optimizer
sgd = optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=False)

# Stacking early-fusion multimodal model

input_word_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = layers.Input(shape=(max_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                    name="segment_ids")
image_input = layers.Input(shape = input_shape, dtype=tf.float32,
                           name = "image")

image_side = model_cnn(image_input)
text_side = model_lstm([input_word_ids, input_mask, segment_ids])
# Concatenate features from images and texts
merged = layers.Concatenate()([image_side, text_side])
merged = layers.Dense(256, activation = 'relu')(merged)
output = layers.Dense(nClasses, activation='softmax', name = "class")(merged)

model = models.Model([input_word_ids, input_mask, segment_ids, image_input], output)

model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='multimodal.png')

# Stochastic Gradient Descent optimizer
sgd = optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=False)

# Compile model
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              metrics=['accuracy'])


# Setup callbacks, logs and early stopping condition
checkpoint_path = "stacking_early_fusion/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy',save_best_only=True,verbose=1, mode='max')
csv_logger = callbacks.CSVLogger('stacking_early_fusion/stacking_early.log')
es = callbacks.EarlyStopping(patience = 3, restore_best_weights=True)

# Reduce learning rate if no improvement is observed
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=1, min_lr=0.00001)

#Training

# Training
history = model.fit(data_train,
                    epochs=15,
                    steps_per_epoch = 6,
                    validation_data = data_test,
                    validation_steps = 6,
                    callbacks=[cp, csv_logger, reduce_lr])





from sklearn.metrics import classification_report

target_names = ['0', '1']
print(classification_report(ground_truth1, predicted_classes, target_names=target_names))


from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt     
cm = confusion_matrix(ground_truth1, predicted_classes)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);



