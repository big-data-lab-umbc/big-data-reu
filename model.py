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
        num_layers=8, neurons=100,lstm=False,nlstm=0, scale=False, 
        skip=0, batch_normalization=False, regularization=False, branch=None,
        **kwargs):
    # num_layers = num
    # main_input = Input(shape=(17,1,1))
    if regularization:
        if regularization == "l2":
            reg = l2(0.01)
        elif regularization == "l1":
            reg = l1(0.01)
    # main_input = Input(shape=(17))
    main_input = Input(shape=(18))
    # layers = Flatten()(main_input)
    # First dense layer
    layers = main_input
    prev = None
    for i in range(num_layers):
        # Scale neurons down
        if scale == True:
            if neurons/(2**i) > 4:
                if regularization:
                    layers = Dense(neurons/(2**i),
                        kernel_regularizer=reg,
                        activity_regularizer=reg)(layers)
                else:
                    layers = Dense(neurons/(2**i))(layers)
            else:
                # layers = Dense(4)(layers)
                layers = Dense(25)(layers)
        else:
            if regularization:
                layers = Dense(neurons,
                    kernel_regularizer=reg,
                    activity_regularizer=reg)(layers)
            else:
                layers = Dense(neurons)(layers)
        # Adding batch normalization
        if batch_normalization == True:
            BatchNormalization()(layers)
        # Adding activation
        if inter_activation == 'leakyrelu':
            layers = LeakyReLU()(layers)
        elif inter_activation == 'prelu':
            layers = PReLU()(layers)
        else:
            layers = Activation(inter_activation)(layers)
        # Add a skip connection feature with no downscaling
        if not scale:
            if i == 0:
                prev = layers
            if (i+1) % skip == 0:
                layers = Add()([prev, layers])
                prev = layers
        # Slide in an lstm
        if lstm and nlstm > 0:
            layers = LSTM(neurons)(layers)
            nlstm -= 1
        layers = Dropout(dropout)(layers)
    # Output Layer
    # layers = Dense(25)(layers)
    layers = Dense(15)(layers)
    # layers = Dense(4)(layers)
    if branch is not None:
        layers = Activation('softmax', name="output_{}".format(branch))(layers)
    else:
        layers = Activation('softmax')(layers)
    model = Model(inputs=main_input, 
            outputs=layers)
    decay = learning_rate /  kwargs['epochs']
    model.compile(Adam(lr=learning_rate, decay=decay), "categorical_crossentropy", 
            metrics=["accuracy"])
    # model.compile(Adam(lr=learning_rate,clipnorm=1.0, clipvalue=0.5), "categorical_crossentropy", 
            # metrics=["accuracy"])
    return model

class DenseIsolationTrainer:
    def __init__(self, **kwargs):
        self.model = single_dense_model(**kwargs)
        self.kwargs = kwargs
        # Expose parts of the underlying model
        self.optimizer = self.model.optimizer

    def fit(self, **kwargs):
        # Use train on batch
        pass

    def compiler(self, **kwargs):
        self.model.compile(**kwargs)
