from tensorflow import keras
import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout, Conv2D

from models.localization_inceptionv3 import getModel as _getModel, getVersion as _getVersion

# build and compile the model
def getModel(n_epochs=200, dataset_name="fft_data_256", label_set="simple_synthetic_2018"):
    from utils.paths import getTrainingLogRoot
    # i refer to layers by name, but the automatically generated layer names differ depending on what models you've already built
    # so reset the count
    keras.backend.clear_session()

    run_id = "localization_inceptionv3-{}/{}/{}/{}".format(_getVersion(), dataset_name, label_set, str(n_epochs))
    root = getTrainingLogRoot(run_id)
    target_dir = root + sorted(os.listdir(root), key = lambda x: os.stat(root+x).st_mtime, reverse = True)[0] + "/"

    model = _getModel()
    model.load_weights(target_dir + "best_model/")
    # model = keras.models.load_model( target_dir + "best_model/", custom_objects={"IOU_metric":IOU_metric} )
    print( model.summary() )
    
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
    return _getVersion()+"_lrplateau"

def getDataType():
    return "labeled"