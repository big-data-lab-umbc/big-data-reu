from tensorflow import keras
import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout, Conv2D

from models.localization_inceptionv3 import getModel as _getModel
MODE = "simple_coords"
VERSION = "0.03"

# build and compile the model
def getModel(n=200):
    from utils.paths import getTrainingLogRoot
    
    # i refer to layers by name, but the automatically generated layer names differ depending on what models you've already built
    # so reset the count
    keras.backend.clear_session()

    model = _getModel()
    
    x = model.get_layer( "tf_op_layer_Sub" ).output
    submodel = Model( model.get_layer("inception_v3").input, model.get_layer("inception_v3").get_layer("mixed7").output )
    x = submodel(x)

    x = Conv2D(10, (1, 1), strides = (1, 1), use_bias = False, activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    # x = model.layers[-1].output
    x = keras.layers.Dense(1, activation='sigmoid', name="uniquenamerighthere")(x)

    model2 = Model( model.input, x )
    for i in range(len(model2.layers)-5):
        model2.layers[i].trainable = False
    # model2.summary()
    model2.compile(
        # optimizer = Adam(lr=0.000005), # no lrplateau
        optimizer = Adam(lr=1e-5), # lrplateau
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )
    return model2

# anything you want to do after training the model
def evaluateModel(model, val, log_dir):
    pass

def getVersion():
    return MODE+"_"+VERSION+"_lrplateau"

def getDataType():
    return "labeled"