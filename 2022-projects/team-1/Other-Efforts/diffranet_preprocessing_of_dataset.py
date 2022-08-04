
# !pip install preprocess

# %load_ext tensorboard
import glob
import os
from os import listdir
from random import shuffle
import pickle, datetime
import preprocess as pp
import cv2
from pathlib import Path

from tensorflow import keras
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator


def rawTrain(LABELED_GW):
    train_datagen = ImageDataGenerator(
        # rescale=1./255
    )

    train = train_datagen.flow_from_directory(
        LABELED_GW + "/train/",
        target_size=(256, 256),
        batch_size=1,
        # color_mode = "grayscale",
        class_mode='binary',
        shuffle=False
    )
    return train

# each train, val, test generator should return tuples of inputs and labels
def getData(LABELED_GW, batch_size=32): #, max_samples=1000
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        featurewise_center=True,
        # featurewise_std_normalization=True,
        horizontal_flip=True
    )
    rtg = rawTrain(LABELED_GW)
    rt = []
    print(len(rtg))
    # for _ in range(min(len(rtg), max_samples)):
    for _ in range(len(rtg)):
        rt.append( rtg.next()[0].reshape( (256,256,3) ) )
    train_datagen.fit(rt)

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        # featurewise_std_normalization=True,
    )
    test_datagen.fit(rt)

    train = train_datagen.flow_from_directory(
        LABELED_GW+'/train',
        target_size=(256, 256),
        batch_size=batch_size,
        # color_mode = "grayscale",
        class_mode='binary'
    )

    val = test_datagen.flow_from_directory(
        LABELED_GW+'/validation',
        target_size=(256, 256),
        batch_size=batch_size,
        # color_mode = "grayscale",
        class_mode='binary'
    )

    test = test_datagen.flow_from_directory(
        LABELED_GW+'/test',
        target_size=(256, 256),
        batch_size=1,
        # color_mode = "grayscale",
        class_mode=None,
        shuffle=False
    )
    return train, val, test