##### CREATE DATALOADERS
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

from data_loaders.utils import _custom_get_batches, _getLabels, _getImageLoaders
from utils.paths import getDataDirectory

import types

def _getData(dataset_name, test_dataset_name, label_set, normalize, augment, rgb, rescale, batch_size, shuffle, rt=None, dim=256):
    train_data = _getLabels(label_set, "train")
    validation_data = _getLabels(label_set, "validation")
    test_data = _getLabels(label_set, "test")

    # training_data = train_data.iloc[train_index]
    # validation_data = train_data.iloc[lambda i: i not in train_index]

    data_path = getDataDirectory(dataset_name)
    test_data_path = getDataDirectory(test_dataset_name)
    
    if not shuffle:
        augment = False

    train_datagen, test_datagen = _getImageLoaders(normalize, augment, rgb, rescale, data_path, rt)


    color_mode = "rgb" if rgb else "grayscale"
    train = train_datagen.flow_from_dataframe(
        train_data, 
        directory=data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=batch_size,
        color_mode = color_mode,
        class_mode="raw",
        shuffle=shuffle,
    )
    val = test_datagen.flow_from_dataframe(
        validation_data, 
        directory=test_data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=1,
        color_mode = color_mode,
        class_mode="raw",
        shuffle=False
    )
    test = test_datagen.flow_from_dataframe(
        test_data, 
        directory=test_data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=1,
        color_mode = color_mode,
        class_mode="raw",
        shuffle=False
    )
    train._get_batches_of_transformed_samples = types.MethodType( _custom_get_batches, train ) # replace method for loading train batches

    return train, val, test