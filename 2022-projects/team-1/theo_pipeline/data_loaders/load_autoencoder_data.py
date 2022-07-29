from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.paths import getDataDirectory

from data_loaders.utils import _getLabels, _getImageLoaders

import pandas as pd

def _getData(dataset_name, label_set, normalize=True, augment=True, rgb=False, rescale=True, batch_size=32, shuffle=True, rt=None, dim=256):
    train_data = _getLabels(label_set, "unlabeled")
    validation_data = _getLabels(label_set, "validation")
    test_data = _getLabels(label_set, "test")

    # training_data = train_data.iloc[train_index]
    # validation_data = train_data.iloc[lambda i: i not in train_index]

    data_path = getDataDirectory(dataset_name)
    
    train_datagen, test_datagen = _getImageLoaders(normalize, augment, rgb, rescale, data_path, rt)


    color_mode = "rgb" if rgb else "grayscale"
    train = train_datagen.flow_from_dataframe(
        train_data, 
        directory=data_path,
        x_col="filename",
        target_size=(dim, dim),
        batch_size=batch_size,
        color_mode = color_mode,
        class_mode='input',
        shuffle=shuffle,
    )
    val = test_datagen.flow_from_dataframe(
        validation_data, 
        directory=data_path,
        x_col="filename",
        target_size=(dim, dim),
        batch_size=1,
        color_mode = color_mode,
        class_mode='input',
        shuffle=False
    )
    test = test_datagen.flow_from_dataframe(
        test_data, 
        directory=data_path,
        x_col="filename",
        target_size=(dim, dim),
        batch_size=1,
        color_mode = color_mode,
        class_mode='input',
        shuffle=False
    )

    return train, val, test