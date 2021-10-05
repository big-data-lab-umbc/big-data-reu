from tensorflow.keras.layers import Dense, Activation, Conv2D, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Nadam
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
def single_dense_model(learning_rate=1e-3, dropout=0, inter_activation='tanh',
        num_layers=8, neurons=100, scale=False, skip=0, 
        batch_normalization=False, regularization=None, indim=15, outdim=13,
        optimizer="adam", momentum=0.0, withmomentum=False,
        **kwargs):
    activator = inter_activation
    # Determine number of residual blocks
    if skip !=0:
        blocks = num_layers // skip
        nLayers = skip
        skip = True
    else:
        blocks = 1
        nLayers = num_layers
        skip = False

    main_input = Input(shape=(indim))
    layers = main_input
    if scale:
        # Is "scale_block_repeat" set?
        if "scale_block_repeat" in kwargs.keys():
            repeat = kwargs["scale_block_repeat"]
        else:
            repeat = 1
    rep_block = 0
    scaleBlock = 0
    for block in range(blocks):

        if not scale:
            nNeurons = neurons
        else:
            if rep_block < repeat:
                rep_block += 1
                # print(1)
            elif rep_block == repeat:
                # print(2)
                rep_block = 1
                scaleBlock += 1
            nNeurons = neurons // (2**scaleBlock)
            if nNeurons <= 16:
                nNeurons = 16

        layers = generateResidualPGMLBlock(layers, nLayers, nNeurons, 
                dropout=dropout, batchNorm=False, activator=activator, 
                reg=regularization, skip=skip, block=block)
    # Output Layer
    # With doubles
    # layers = Dense(15)(layers)
    # Without doubles
    layers = Dense(outdim)(layers)
    layers = Activation('softmax')(layers)
    model = Model(inputs=main_input, 
            outputs=layers)
    ##############
    opt = getOptimizer(optimizer=optimizer, momentum=momentum,
            nesterov=withmomentum, learning_rate=learning_rate)
    # decay = learning_rate /  kwargs['epochs']
    # model.compile(Adam(lr=learning_rate, decay=decay), "categorical_crossentropy", 
            # metrics=["accuracy"])
    model.compile(opt, "categorical_crossentropy", 
            metrics=["accuracy"])
    # model.compile(Adam(lr=learning_rate,clipnorm=1.0, clipvalue=0.5), "categorical_crossentropy", 
            # metrics=["accuracy"])
    return model

def makeFullLayer(layers, nNeurons, dropout=0.0, batchNorm=False, activator=None, reg=None):
    # Assemble a full NN layer from pieces
    # Get the regularizer of we need one
    if reg is not None:
        if reg == "l2":
            reg = lambda : l2(0.01)
        elif reg == "l1":
            reg = lambda : l1(0.01)

    # Generate the meat of the layer
    if reg is None:
        l = Dense(nNeurons)(layers) 
    else:
        l = Dense(nNeurons,
            kernel_regularizer=reg(),
            activity_regularizer=reg())(layers)

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


def generateResidualPGMLBlock(layer, nLayers, nNeurons, dropout=0.0, 
        batchNorm=False, activator=None, reg=None, skip=True, block=None):
    l = layer
    prev = layer
    for i in range(nLayers):
        l = makeFullLayer(l, nNeurons, dropout, batchNorm, activator, reg)
        # makeFullLayer(layer, nNeurons, dropout=0.0, batchNorm=False, activator=None, reg=None)
    if skip:
        if prev.shape[1] == l.shape[1]:
            l = Add()([prev, l])
        else:
            # In the case that the block before has different output dimensions
            # than this block we have to push the data through a single layer to
            # get it shaped right
            # Typically we do this in convoluational networks by convolving the
            # data into the new shape. Here we will just push it through a dense
            # layer. No activation...
            # There are a lot of ways to do this even using an upsampling block
            # instead but we'll try this for now. Something to look into...
            a = Dense(l.shape[1], name="skip_adjuster_{}".format(block))(prev)
            l = Add()([a, l])
    return l

def getActivator(name, **kwargs):
    if kwargs:
        print("ERROR: More than name provided?")
        print(kwargs)
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

def getOptimizer(optimizer='adam', nesterov=False, momentum=None, learning_rate=0.1):
    optimizer = optimizer.lower()
    if optimizer == 'adam':
        optimizer = Adam(lr=learning_rate)
    elif optimizer == 'nadam':
        optmizer = Nadam(lr=learning_rate)
    elif optimizer == 'sgd':
        if momentum is None:
            momentum = 0.0
        optimizer = SGD(lr=learning_rate, nesterov=nesterov, momentum=momentum)
    else:
        print("{} optimizer not an acceptable variant.".format(optimizer))
        print("Try: Adam, Nadam, or SGD.")
        return None
    return optimizer

if __name__ == "__main__":
    # Load params.json
    from json import load as loadf
    with open("params.json", "rb") as infile:
        params = loadf(infile)

    m = create_model(**params)
    m.summary()
