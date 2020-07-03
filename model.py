from tensorflow.keras.layers import Dense, Activation, Conv2D, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1
import tensorflow as tf 
# tf.distribute.OneDeviceStrategy

# Starts out as 3x4
# Layer -> normalization -> Act -> Dropout -> Layer
# Possibly skip activation
def create_model(**kwargs):
    # return conv_model(**kwargs)
    # return deep_dense_model(**kwargs)
    return single_dense_model(**kwargs)

# The heart of the matter
def single_dense_model(learning_rate=0.001, dropout=0, inter_activation='tanh',
        num_layers=8, neurons=100, scale=False, 
        skip=0, batch_normalization=False, regularization=False, 
        **kwargs):
    # num_layers = num
    # main_input = Input(shape=(17,1,1))
    if regularization:
        if regularization == "l2":
            reg = lambda : l2(0.01)
        elif regularization == "l1":
            reg = lambda : l1(0.01)
    else:
        reg = None
    # main_input = Input(shape=(17))
    main_input = Input(shape=(18))
    # layers = Flatten()(main_input)
    # First dense layer
    layers = main_input
    prev = None
    for i in range(num_layers):
        # Prepare activation
        if inter_activation == 'leakyrelu':
            activator = lambda l: LeakyReLU()(l)
        elif inter_activation == 'prelu':
            activator = lambda l: PReLU()(l)
        else:
            activator = lambda l: Activation(inter_activation)(l)
        # Scale neurons down
        # Generate a layer with all the bells
        layers = makeFullLayer(layers, 
                neurons, 
                batchNorm=batch_normalization, 
                activator=activator, 
                reg=reg())
        # Add a skip connection feature with no downscaling
        if not scale:
            if i == 0:
                prev = layers
            if (i+1) % skip == 0:
                layers = Add()([prev, layers])
                prev = layers
        layers = Dropout(dropout)(layers)
    # Output Layer
    # layers = Dense(25)(layers)
    layers = Dense(15)(layers)
    # layers = Dense(4)(layers)
    layers = Activation('softmax')(layers)
    model = Model(inputs=main_input, 
            outputs=layers)
    decay = learning_rate /  kwargs['epochs']
    model.compile(Adam(lr=learning_rate, decay=decay), "categorical_crossentropy", 
            metrics=["accuracy"])
    # model.compile(Adam(lr=learning_rate,clipnorm=1.0, clipvalue=0.5), "categorical_crossentropy", 
            # metrics=["accuracy"])
    return model

def makeFullLayer(layer, nNeurons, batchNorm=False, activator=None, reg=None):
    # Assemble a full NN layer from pieces

    # Generate the meat of the layer
    if reg is None:
        l = Dense(nNeurons)(layer) 
    else:
        l = Dense(nNeurons,
            kernel_regularizer=reg,
            activity_regularizer=reg)(layers)
    # Put a batch norm on    
    if batchNorm:
        l = BatchNormalization()(l)
    # Use an activator is needed
    if activator is not None:
        l = activator(l)

    return l


def generateResidualPGMLBlock(layer, nLayers, nNeurons, batchNorm, 
        activator=None, reg=None):
    pass
