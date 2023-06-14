#!/usr/bin/env python
# coding: utf-8


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, MaxPooling1D, GlobalAveragePooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau 
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.regularizers import l2, l1
import cv2
from tensorflow.keras.callbacks import CSVLogger
import sys
import time
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_v3 import InceptionV3
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

# Loading images from Google Drive


train_data_dir = './images/train'
validation_data_dir = './images/test'
nb_train_samples = 3519 
nb_validation_samples = 880
n_classes = 2
epochs = 10
batch_size = 75



# Checking image format: if RGB channel is coming first or last so, model will check first and then input shape will be feeded accordingly.
img_width = 299
img_height = 299

if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 


#Model that enable the freezing of the resnet layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
x = base_model.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.4)(x)
x = Flatten()(x)

predictions = Dense(2,
                    kernel_regularizer=l2(0.005),
                    activity_regularizer=l1(0.005), 
                    activation='softmax')(x)

model = Model(base_model.input, predictions)

model.summary()



# Train & Test Data Generators with image augmentation 

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    zoom_range=[.8, 1],
    channel_shift_range=30,
    fill_mode='reflect')

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = 11,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = 11,
    class_mode='categorical')
     

from tensorflow.keras import callbacks
# Setup callbacks and logs 
checkpoint_path = "InceptionV3/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy',save_best_only=True,verbose=1, mode='max')
csv_logger = callbacks.CSVLogger('InceptionV3/InceptionV3.log')
     

# Reduce LR if no improvement on the test accuracy is observed
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                              patience=2, min_lr=0.00001)


# Compile the model using Stochastic Gradiend Descent (SGD) optimizer
model.compile(
    optimizer=SGD(lr=.01, momentum=.9), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

#Fitting
model.fit(train_generator,
          steps_per_epoch = nb_train_samples // batch_size,
          validation_data=validation_generator,
          validation_steps=nb_validation_samples // batch_size,
          epochs=10,
          verbose=1,
          callbacks=[cp])
     