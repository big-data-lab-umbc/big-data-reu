from tensorflow import keras
import os

MODEL = "basic_autoencoder"
LABELS = "2018"
PRE_EPOCHS = 300

# build and compile the model
def getModel(n=100, N=5):
    from utils,paths import getTrainingLogRoot
    run_id = "{}-{}/theo_fft-rgb/{}/{}".format(MODEL, getVersion(), LABELS, str(PRE_EPOCHS))
    root = getTrainingLogRoot(run_id)
    target_dir = root + sorted(os.listdir(root), key = lambda x: os.stat(root+x).st_mtime, reverse = True)[0] + "/"

    model = keras.models.load_model( target_dir + "best_model/" )
    # print( model.summary() )
    model2 = keras.models.Sequential()
    for l in model.layers[:N]:
        model2.add(l)
    for i in range(N-1):
        model2.layers[i].trainable = False
    model2.add(keras.layers.Flatten())
    model2.add(keras.layers.Dense(64,activation='relu'))
    model2.add(keras.layers.Dense(1,activation='sigmoid'))
    # model2.summary()
    model2.compile(
        optimizer='adam', 
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )
    return model2

# anything you want to do after training the model
def evaluateModel(model, val, log_dir):
    pass

def getVersion():
    return "0.01"

def getDataType():
    return "labeled"