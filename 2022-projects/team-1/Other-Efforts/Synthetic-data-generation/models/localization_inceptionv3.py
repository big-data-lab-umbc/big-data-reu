from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout, Conv2D

from models.utils import IOU_metric
MODE = "simple_coords"
VERSION = "0.03"

def getModel():
    base_model = InceptionV3(input_shape=(256,256,3), include_top=False, weights="imagenet")


    i = Input([256, 256, 3], dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    x = preprocess_input(x)
    x = base_model(x)

    x = Conv2D(40, (1, 1), strides = (1, 1), use_bias = False, activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    # x = layers.Dense(4, activation='linear')(x) # v2   
    x = layers.Dense(4, activation='sigmoid')(x) # v3   

    model = Model(i, x) 

    model.compile(
        optimizer = Adam(lr=0.000005),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = [IOU_metric]
    )
    return model

def evaluateModel(model, val, log_dir):
    pass

def getVersion():
    return MODE+"_"+VERSION

def getDataType():
    return MODE
    # return "coords_small"