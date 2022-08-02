from tensorflow import keras
import os

# build and compile the model
def getModel(n=100, N=5):

    model = keras.models.load_model("/home/kchen/reu2022_team1/research/autoencoder/best_model")
    # print( model.summary() )
    model2 = keras.models.Sequential()
    for l in model.layers[:N]:
        model2.add(l)
    for i in range(N-1):
        model.layers[i].trainable = False
        if i == 2 or i == 4:
            model.layers[i].kernel_regularizer = keras.regularizers.l2(l = 0.01)
            model.layers[i].kernel_constraint = keras.constraints.UnitNorm(axis=[0, 1, 2])
    model2.add(keras.layers.Flatten())
    model2.add(keras.layers.Dense(256, activation='relu'))
    model2.add(keras.layers.Dropout(0.5))
    model2.add(keras.layers.Dense(64, activation='relu'))
    model2.add(keras.layers.Dropout(0.5))
    model2.add(keras.layers.Dense(1, activation='sigmoid'))
    # model2.summary()
    model2.compile(
        optimizer=keras.optimizers.Adam(clipvalue = 1, clipnorm = 1), 
        metrics=['accuracy'],
        loss='binary_crossentropy'
    )
    return model2

# model = getModel()
# model.summary()

# anything you want to do after training the model
def evaluateModel(model, val, log_dir):
    pass
