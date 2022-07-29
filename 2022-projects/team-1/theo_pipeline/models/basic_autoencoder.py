import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Conv2DTranspose
from tensorflow.keras.utils import  plot_model
from keras.preprocessing.image import ImageDataGenerator
import h5py
import io
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import norm
from keras.constraints import UnitNorm, Constraint


# from https://blog.keras.io/building-autoencoders-in-keras.html

# build and compile the model
def getModel():
    autoencoder = models.Sequential()
    autoencoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape = (256, 256, 1)))
    autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    # autoencoder.add(layers.Dropout(0.25))

    autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding = 'same'))
    autoencoder.add(layers.UpSampling2D((2, 2)))
    autoencoder.add(layers.Conv2D(1, (3, 3), activation='linear', padding='same'))

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# anything you want to do after training the model
def evaluateModel(autoencoder, val, log_dir):
    decoded_imgs = autoencoder.predict(val)
    n = len(val)
    w = min(1.2*n, 600)
    plt.figure(figsize=(w, 2*2))

    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i+1)
        plt.imshow(np.squeeze( val[i][0].reshape(256, 256, 1) ))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstructed images
        ax = plt.subplot(2, n, i+n+1)
        plt.imshow(np.squeeze( decoded_imgs[i].reshape(256, 256, 1) ))#, cmap="gray") 
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig(log_dir + 'original_reconstructed.png')

def getVersion():
    return "0.01"


def getDataType():
    return "autoencoder"