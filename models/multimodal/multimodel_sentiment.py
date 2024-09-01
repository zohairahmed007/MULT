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
    
## Settings
BATCH_SIZE=8
IMAGE_SIZE=224
EPOCHS=50
SEED=1

##------------------------------------------------------------------------------------ 

object_mg_model=load_model('./training_models/visual_sg.hdf5')

for layer in object_mg_model.layers:
  layer._name = 'object_'+layer.name

object_output = object_mg_model.get_layer(name='object_dense_1').output




path_text="./training_models/"

## get layer from places_mg_model (change directory to load best places_mg_model from folder 'training_models')
text_model=load_model(path_text+'textual_use_finetune.hdf5')

#print(places_mg_model.summary() )
for layer in text_model.layers:
  layer._name = 'text_'+layer.name

text_output = text_model.get_layer(name='text_dense_2').output

## Average both models output  

average_layer  = average([object_output,text_output])

## Create multimodel_fusion_model 
multimodel_fusion_model = Model(inputs=[object_mg_model.input,  text_model.input], outputs=average_layer)


df = pd.read_pickle(r"testdata/testdata.pkl")

new_df=df

# Start--> Use for Large Dataset
'''df["label"] = df["text"].apply( lambda x: 1 if x =='positive'  else 0 )   
df = df.sample(frac = 1)
df=df.head(350)
df.groupby('label').size()


df.drop(df.loc[df['label']==0].index, inplace=True)

df1=pd.read_pickle("t4sa_no_data.pkl")

df1['label']=0

df1.rename(columns = {'id':'ID', 'text':'textinput','image_name':'imageinput'}, inplace = True)

df=df.drop(['text', 'image', 'matches'], axis=1)

result = pd.concat([df,df1])
result = result.sample(frac = 1)'''
# End--> Use for Large Dataset



# Start--> Use for Getting Data direct from folder
'''import os
pos_image_name = os.listdir("./data/val/pos")
pos_image_name=[x.split('.')[0] for x in pos_image_name] #remove extentions

neg_image_name = os.listdir("customdata/")
neg_image_name=[x.split('.')[0] for x in neg_image_name] #remove extentions

merge_image_name=pos_image_name+neg_image_name

for i in range(0, len(merge_image_name)):
    merge_image_name[i] = int(merge_image_name[i]) 


new_df = df[df['ID'].isin(merge_image_name)]


count=0
for index, row in new_df.iterrows():
    
    if str(row['label']) == '0' and count < 100:
        count=count+1
        #print (count)
        new_df.drop(index, inplace=True)'''
    
# End--> Use for Getting Data direct from folder



# Text Modality 
import tensorflow_hub as hub
import tensorflow_text


model_path =path_text+'./training_models/textual' 

pretrained_model_load = hub.KerasLayer(model_path, trainable=True)

#new_df["label"] = new_df["text"].apply( lambda x: 1 if x =='positive'  else 0 )

from sklearn.preprocessing import OneHotEncoder
type_one_hot = OneHotEncoder(sparse=False).fit_transform(
  new_df.label.to_numpy().reshape(-1, 1)
)

X_test=new_df.textinput
y_test=type_one_hot

#create emb from pretrained_model_load
from tqdm import tqdm

X_test_new = []
for r in tqdm(X_test):
  emb = pretrained_model_load(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_test_new.append(review_emb)
X_test_new = np.array(X_test_new)

X_test=X_test_new

print(X_test.shape, y_test.shape)

ground_truth1 = np.argmax(y_test,axis=1)


# Image Modality 
import os.path
from tensorflow.keras.preprocessing import image
img_array = []
for index, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):

    image_input='customdata\\'+str(row['ID'])+ ".jpg"
    
    
    if (os.path.exists(image_input)==True):
        img = image.load_img(image_input, target_size=(224, 224))
        img = np.expand_dims(img, axis=0)
        img_array.append(img)


img_array = np.concatenate(img_array, axis=0)

print(img_array.shape)


predictions_visual = object_mg_model.predict([img_array],BATCH_SIZE,verbose=1)

predictions_textual = text_model.predict([X_test],BATCH_SIZE,verbose=1)

predictions = multimodel_fusion_model.predict([img_array,X_test],BATCH_SIZE,verbose=1)
#print(predictions)


##  Predictions
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth1)[0]
accuracy= round((100-((len(errors)/len(img_array))*100)), 2)
print("No of errors = {}/{}".format(len(errors),len(img_array)))
print('Accuracy : ',accuracy , '%')



from sklearn.metrics import classification_report

target_names = ['0', '1']

print(classification_report(ground_truth1, predicted_classes, target_names=target_names))

classification_report_Dataframe =  pd.DataFrame(classification_report(ground_truth1, predicted_classes, target_names=target_names,output_dict=True)).transpose()

new_df = new_df.reset_index()

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
