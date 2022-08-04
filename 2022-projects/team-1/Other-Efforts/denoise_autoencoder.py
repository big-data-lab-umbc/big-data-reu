# connects colab to your google drive
# skip if your dataset is not on google drive or you're not using colab
from google.colab import drive
drive.mount('/content/drive')

!pip install preprocess

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
import glob
import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import pickle, datetime
import preprocess as pp
import cv2
from pathlib import Path

from tensorflow import keras
from keras.datasets import cifar10
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, MaxPool2D
from keras.layers.convolutional import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
from tensorflow.keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator



import  PIL.Image

data_path = 'put your dataset path here'
img_path= data_path


os.chdir(img_path) # changes the current working directory to the file path specified. This directory should be the directory of data you plan on using for the model'
print(os.path.abspath(os.getcwd()))

batch_size = 32


# # this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        #fill_mode = "nearest",
        #validation_split = 0.2
        )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
        rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        './train',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 256x256
        batch_size=batch_size,
        shuffle=False,
        class_mode='input') # this needs to be changed
        # The inputs should be noisy images, and labels should be clean versions of those images

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        './validation',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='input',
        shuffle=False)

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
        './test',
        target_size=(256, 256),
        batch_size=1,
        class_mode=None,
        shuffle=False)


for i in range(9):
    plt.subplot(330 + 1 + i)
    images, labels = train_generator.next()
    image = (images[0]*255).astype('uint8')
    plt.imshow(image)

import tensorflow as tf

# creating the denoising autoencoder
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape = (256,256,3)),
      #layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      #layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(12, (3, 3), activation='relu', padding='same', strides=2),
      ])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(12, kernel_size=3, strides=2, activation='relu', padding='same'),                                  
      #layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      #layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

from tensorflow.keras import layers, losses

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(),  metrics=['accuracy'])

h2 = autoencoder.fit(            #use fit instead of fit_generator
        train_generator,
        steps_per_epoch= 120 // batch_size,
        epochs = 100,
        validation_data=validation_generator,
        validation_steps= 64 // batch_size)

# denoised images from test dataset
decoded_imgs = autoencoder.predict(test_generator)

# create a visual that shows the original and denoised images next to each other, 
# then saves the figure so the differences can be observed
from google.colab import files
n = 100
w = max(1.2*n, 600)
plt.figure(figsize=(w, 22))
for i in range(n):
  # Original images
  ax = plt.subplot(2, n, i+1)
  x = test_generator.next()
  plt.imshow(x.reshape(256, 256, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)


  # Reconstructed images
  ax = plt.subplot(2, n, i+n+1)
  plt.imshow(decoded_imgs[i].reshape(256, 256, 3))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.savefig('/content/drive/MyDrive/newly_organized_pngs/denoise6.png')
# files.download('denoise.png')