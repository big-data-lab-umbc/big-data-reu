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
def single_dense_model(learning_rate=0.001, dropout=0, activator='tanh',
        num_layers=8, neurons=100, scale=False, 
        skip=0, batch_normalization=False, regularization=False, 
        **kwargs):
    # num_layers = num
    # main_input = Input(shape=(17,1,1))
    # main_input = Input(shape=(17))
    main_input = Input(shape=(18))
    layers = main_input
    prev = None
    # Scale neurons down
    for i in range(num_layers):
        # Generate a layer with all the bells
        layers = makeFullLayer(layers, 
                neurons, 
                dropout=dropout,
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

def makeFullLayer(layer, nNeurons, dropout=0.0, batchNorm=False, activator=None, reg=None):
    # Assemble a full NN layer from pieces
    # Get the regularizer of we need one
    if reg is not None:
        if reg == "l2":
            reg = lambda : l2(0.01)
        elif reg == "l1":
            reg = lambda : l1(0.01)

    # Generate the meat of the layer
    if reg is None:
        l = Dense(nNeurons)(layer) 
    else:
        l = Dense(nNeurons,
            kernel_regularizer=reg,
            activity_regularizer=reg)(layers)

    # Use an activator is needed
    if activator is not None:
        # Get activator returns a lambda which call like a function so this is
        # getActivator(activator)(l) -> activator(l), g o f style
        l = getActivator(activator)(l)
    # Rule of thumb is now to put normalization after activation    
    # Put a batch norm on    
    if batchNorm:
        l = BatchNormalization()(l)
    l = Dropout(dropout)(l)
    return l


def generateResidualPGMLBlock(layer, nLayers, nNeurons, batchNorm=False, 
        activator=None, reg=None):
    l = layer
    prev = layer
    for i in range(nLayers):
        l = makeFullLayer(l, nNeurons, batchNorm, activator, reg)
    
    l = Add()([prev, l])
    return l

def getActivator(name, **kwargs):
    if kwargs is not None:
        print("ERROR: More than name provided?")
        exit()
    name = name.lower()
    # Prepare activation for future use with helper functions
    if name == 'leakyrelu':
        activator = lambda l: LeakyReLU()(l)
    elif name == 'prelu':
        activator = lambda l: PReLU()(l)
    else:
        activator = lambda l: Activation(name)(l)
    return activator
