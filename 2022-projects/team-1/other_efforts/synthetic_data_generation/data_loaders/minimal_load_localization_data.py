from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

from data_loaders.utils import _getLabels
from utils.paths import getDataDirectory

import types

def getData(dataset_name, test_dataset_name, label_set, get_batches_with_fake_waves_fn, dim=256):
    train_data = _getLabels(label_set, "train")
    validation_data = _getLabels(label_set, "validation")
    test_data = _getLabels(label_set, "test")

    data_path = getDataDirectory(dataset_name)
    test_data_path = getDataDirectory(test_dataset_name)
    
    
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train = train_datagen.flow_from_dataframe(
        train_data, 
        directory=data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=32,
        color_mode = "rgb",
        class_mode="raw",
        shuffle=True,
    )
    val = test_datagen.flow_from_dataframe(
        validation_data, 
        directory=test_data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=1,
        color_mode = "rgb",
        class_mode="raw",
        shuffle=False
    )
    test = test_datagen.flow_from_dataframe(
        test_data, 
        directory=test_data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=1,
        color_mode = "rgb",
        class_mode="raw",
        shuffle=False
    )
    train._get_batches_of_transformed_samples = types.MethodType( get_batches_with_fake_waves_fn, train ) # replace method for loading train batches
    return train, val, test