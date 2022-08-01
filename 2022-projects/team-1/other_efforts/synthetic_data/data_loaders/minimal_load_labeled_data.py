from utils.paths import getDataDirectory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

from data_loaders.utils import _getLabels

def getData(dataset_name, label_set, dim=256):
    train_data = _getLabels(label_set, "train")
    validation_data = _getLabels(label_set, "validation")
    test_data = _getLabels(label_set, "test")

    data_path = getDataDirectory(dataset_name)
    
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train = train_datagen.flow_from_dataframe(
        train_data, 
        directory=data_path,
        x_col="filename", y_col="label",
        target_size=(dim, dim),
        batch_size=32,
        color_mode = "rgb",
        class_mode='binary',
        shuffle=True,
    )
    val = test_datagen.flow_from_dataframe(
        validation_data, 
        directory=data_path,
        x_col="filename", y_col="label",
        target_size=(dim, dim),
        batch_size=1,
        color_mode = "rgb",
        class_mode='binary',
        shuffle=False
    )
    test = test_datagen.flow_from_dataframe(
        test_data, 
        directory=data_path,
        x_col="filename", y_col="label",
        target_size=(dim, dim),
        batch_size=1,
        color_mode = "rgb",
        class_mode='binary',
        shuffle=False
    )

    return train, val, test