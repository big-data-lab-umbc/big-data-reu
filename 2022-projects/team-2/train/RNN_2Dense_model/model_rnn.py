from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Flatten
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, Activation
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow.keras.models import Model

def create_model(**kwargs):
    # return conv_model(**kwargs)
    # return deep_dense_model(**kwargs)
    return rnn_model_create(**kwargs)

def rnn_model_create(learning_rate=1e-3, dropout=0,
        num_layers=1, neurons=100, indim=15, outdim=13, activation='tanh',
        optimizer="adam", momentum=0.0, withmomentum=False, layer_type="LSTM",
        **kwargs):
    main_input = Input(shape=(3, 5))
    l = main_input
    for layer in range(num_layers - 1):
        l = getLayer(name=layer_type, neurons=neurons, return_sequences=True, **kwargs)(l)

    l = getLayer(name=layer_type, neurons=neurons, **kwargs)(l)
    l = Flatten()(l)
    l = Dense(128, activation = 'relu')(l)
    l = Dense(64, activation = 'relu')(l)
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

if __name__ == "__main__":
     # Load params.json
     from json import load as loadf
     with open("params.json", "rb") as infile:
         params = loadf(infile)

     m = create_model(**params)
     m.summary()
