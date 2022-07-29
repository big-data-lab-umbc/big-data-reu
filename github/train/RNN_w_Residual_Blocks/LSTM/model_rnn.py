from operator import le
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Flatten,Dropout,Add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, Activation
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow.keras.models import Model

def create_model(**kwargs):
    # return conv_model(**kwargs)
    # return deep_dense_model(**kwargs)
    return rnn_model_create(**kwargs)

def rnn_model_create(learning_rate=1e-3, dropout=0,
        num_layers_rnn=1,num_layers = 1,neurons_rnn = 100, neurons=100, indim=15, outdim=13, activation='tanh',
        optimizer="adam", momentum=0.0, withmomentum=False, layer_type="LSTM",inter_activation = 'leakyrelu',
        skip = 0,scale=False,regularization = None,**kwargs):
    main_input = Input(shape=(3, 5))
    l = main_input
    for layer in range(num_layers_rnn - 1):
        l = getLayer(name=layer_type, neurons=neurons_rnn, return_sequences=True, **kwargs)(l)

    l = getLayer(name=layer_type, neurons=neurons_rnn, **kwargs)(l)
    l = Flatten()(l)
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

        l = generateResidualPGMLBlock(l, nLayers, nNeurons, 
                dropout=dropout, batchNorm=False, activator=activator, 
                reg=regularization, skip=skip, block=block)
    l = Dense(outdim)(l)
    out = getActivation('softmax')(l)


    model = Model(inputs=main_input, outputs=out)
    opt = getOptimizer(optimizer=optimizer, momentum=momentum,
            nesterov=withmomentum, learning_rate=learning_rate)
    model.compile(opt, "categorical_crossentropy", 
            metrics=["accuracy"])
    model.summary()
    return model

def getLayer(name="LSTM", neurons=None, return_sequences=False, **kwargs):
    if name == "GRU":
        # https://keras.io/api/layers/recurrent_layers/gru/
        # The requirements to use the cuDNN implementation are:
        # activation == tanh
        # recurrent_activation == sigmoid
        # recurrent_dropout == 0
        # unroll is False
        # use_bias is True
        # reset_after is True
        # Inputs, if use masking, are strictly
        # right-padded.
        # Eager execution is enabled in the outermost context.
        return GRU(neurons, return_sequences=return_sequences)#, **kwargs)
    elif name == "LSTM":
        # https://keras.io/api/layers/recurrent_layers/lstm/
        # The requirements to use the cuDNN implementation are:
        # activation == tanh
        # recurrent_activation == sigmoid
        # recurrent_dropout == 0
        # unroll is False
        # use_bias is True
        # Inputs, if use masking, are strictly right-padded.
        # Eager execution is enabled in the outermost context
        return LSTM(neurons, return_sequences=return_sequences)#, **kwargs)

def getActivation(name, **kwargs):
    if kwargs:
        print("ERROR: More than name provided?")
        print(kwargs)
        exit()
    name = name.lower()
    # Prepare activation for future use with helper functions
    if name == 'leakyrelu':
        activator = LeakyReLU()
    elif name == 'prelu':
        activator = PReLU()
    else:
        activator =  Activation(name)
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
        l = getActivation(activator)(l)
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
if __name__ == "__main__":
    # Load params.json
    from json import load as loadf
    with open("params.json", "rb") as infile:
        params = loadf(infile)

    m = create_model(**params)
    m.summary()
