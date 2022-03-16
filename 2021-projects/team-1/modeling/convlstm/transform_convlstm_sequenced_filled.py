import math
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'''
This document contains code for a ConvLSTM neural network predicting SIC per pixel and per month for spatio-temporal image data.
'''

# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_train_rolling_filled_final.npy", "rb") as f:
        X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_train_rolling_filled_final.npy", "rb") as f:
        y_train = np.load(f)
# with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_test_rolling_filled_final.npy", "rb") as f:
#         X_test = np.load(f)
# with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_test_rolling_filled_final.npy", "rb") as f:
#         y_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_actual.npy", "rb") as f:
        y_land_mask = np.load(f)
# with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_extent_train_rolling_final.npy", "rb") as f:
#         y_extent_train = np.load(f)
# with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_extent_test_rolling_final.npy", "rb") as f:
#         y_extent_test = np.load(f)
# with open("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy", "rb") as f:
#         y_extent = np.load(f)

def custom_activation(inp):
        return tf.clip_by_value(tf.multiply(inp, y_land_mask), clip_value_min = 0.0, clip_value_max = 100.0)

'''
# standardize data
X_train_norm = (X_train - X_train.mean())/X_train.std()
y_train_norm = (y_train - y_train.mean())/y_train.std()
X_test_norm = (X_test - X_test.mean())/X_test.std()
y_test_norm = (y_test - y_test.mean())/y_test.std()
'''

# reshape y_land_mask
y_land_mask = y_land_mask.reshape(448, 304, 1)

# define custom mse loss, which applies land mask to each output of the network.
def custom_mse(y_true, y_pred):
	print("y_true:", y_true)
	print("y_pred:", y_pred)
	#print("Max of y_pred: %.4f" % tf.reduce_max(y_pred))
	#print("Min of y_pred: %.4f" % tf.reduce_min(y_pred))
	#print("Mean of y_pred: %.4f" % tf.reduce_mean(y_pred))
	#y_pred=tf.Print(y_pred, [y_pred], "print_yP-red")
	y_pred_masked = tf.math.multiply(y_pred, y_land_mask)
	y_true_masked = tf.math.multiply(y_true, y_land_mask)
	squared_resids = tf.square(y_true_masked - y_pred_masked)
	loss = tf.reduce_mean(squared_resids)
	return loss

#load per-pixel area file
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/area_size.npy", "rb") as f:
        areas = np.load(f)

#calculate ice extent
def calc_ice_extent(array):
        array = array / 100.0
        # use simple extent calculation
        tot_ice_extent = np.sum(np.multiply(np.squeeze(array) > 15.0, areas), axis=(1,2)) / 1e6
        return tot_ice_extent


# DATA
#DATASET_NAME = "organmnist3d"
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
# INPUT_SHAPE = (28, 28, 28, 1)
INPUT_SHAPE = (12, 448, 304, 10)
NUM_CLASSES = 11

# OPTIMIZER
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4

# TRAINING
EPOCHS = 10

# TUBELET EMBEDDING
PATCH_SIZE = (1, 56, 38)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
print(INPUT_SHAPE[0] // PATCH_SIZE[0])
print("NUM_PATCHES:", NUM_PATCHES)

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
# PROJECTION_DIM = 128
# NUM_HEADS = 8
# NUM_LAYERS = 8

PROJECTION_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def create_branch(inputs,
	tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES):
    # start of image branch
	# Create image patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)
    
    # transformer encoder
    dropout = 0.1
    for _ in range(transformer_layers):
        x = transformer_encoder(encoded_patches, 256, 4, 64, 0.1)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    return x

    ## Create multiple layers of the Transformer block.
    #for _ in range(transformer_layers):
    #    # Layer normalization and MHSA
    #    x1 = layers.LayerNormalization(epsilon=1e-3)(encoded_patches)
    #    attention_output = layers.MultiHeadAttention(
    #        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
    #    )(x1, x1)

    #    # Skip connection
    #    x2 = layers.Add()([attention_output, encoded_patches])

    #    # Layer Normalization and MLP
    #    x3 = layers.LayerNormalization(epsilon=1e-3)(x2)
    #    x3 = keras.Sequential(
    #        [
    #            layers.Dense(units=embed_dim * 4, activation=tf.nn.relu),
    #            layers.Dense(units=embed_dim, activation=tf.nn.relu),
    #        ]
    #    )(x3)

    #   # Skip connection
    #    encoded_patches = layers.Add()([x3, x2])

    ## Layer normalization and Global average pooling.
    ##representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    #representation = layers.LayerNormalization(epsilon=1e-3)(encoded_patches)
    #representation = layers.GlobalAvgPool1D()(representation)
    #return representation

def create_transformer(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    x = create_branch(inputs, tubelet_embedder, positional_encoder, input_shape, transformer_layers, num_heads, embed_dim, layer_norm_eps, num_classes)
    # Classify outputs.
    # outputs = layers.Dense(units=num_classes, activation="softmax")(representation)
    
	# Regression outputs.
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    # x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(448*304, activation="linear")(x)
    image_output = layers.Reshape((448, 304, 1), input_shape = (448*304,), name="image_output")(x)

	# end of image branch

    # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=image_output)
    model = keras.models.Model(inputs=inputs,
				outputs=image_output,
				name="sea_ice_net")		
	# compile model
	# optimized with Adam, image output uses custom loss, and extent output uses mse loss
	# RMSE for both outputs is measured
    #optimizer = keras.optimizers.Adam(lr=0.0001)
    #model.run_eagerly = True
    model.compile(optimizer="adamax", 
		loss= custom_mse,
		metrics= [keras.metrics.RootMeanSquaredError()])		
	# add custom loss function to the model
    return model

transformer_model = create_transformer(
			tubelet_embedder=TubeletEmbedding(
				embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
			),
			positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM)
		)


# define ConvLSTM model
# def create_convLSTM_image():
# 	#add ConvLSTM layers
# 	inputs = keras.layers.Input(shape=X_train.shape[1:])
# 	x = keras.layers.ConvLSTM2D(16, (5,5), padding="same", return_sequences=False,
# 		activation="relu", data_format = 'channels_last')(inputs)
# 	#x = keras.layers.MaxPooling3D((2,2,2), padding='same')(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	#x = keras.layers.ConvLSTM2D(8, (5,5), padding="same", return_sequences=True, activation="relu", data_format = 'channels_last')(x)
# 	#x = keras.layers.MaxPooling3D((2,2,2), padding='same')(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	#x = keras.layers.ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, activation="relu", data_format = 'channels_last')(x)
# 	x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	x = keras.layers.Conv2D(128, (5,5), padding="same", activation="relu")(x)
# 	#x = keras.layers.MaxPooling2D((4,4), padding='same')(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	#x = keras.layers.Conv2D(128, (5,5), padding="same", activation="relu")(x)
# 	x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	x = keras.layers.Conv2D(32, (5,5), padding="same", activation="relu")(x)
# 	#x = keras.layers.MaxPooling2D((4,4), padding='same')(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	#x = keras.layers.Conv2D(32, (5,5), padding="same", activation="relu")(x)
# 	x = keras.layers.Flatten()(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	x = keras.layers.Dense(256, activation="relu")(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	x = keras.layers.Dense(512, activation="relu")(x)
# 	#x = keras.layers.BatchNormalization()(x)
# 	x = keras.layers.Dense(448*304, activation="linear")(x)
# 	sic_output = keras.layers.Reshape((448, 304, 1), input_shape = (448*304,))(x)

#         #input_mask = keras.layers.Input(shape=y_train_mask.shape[1:])
#         #loss_inp = keras.layers.Input(shape=y_train.shape[1:])
# 	model = keras.models.Model(inputs=inputs,
# 		outputs=sic_output,
# 		name="SIC_net")
#         #compile model
# 	model.compile(optimizer="adamax", loss=custom_mse, metrics=[keras.metrics.RootMeanSquaredError()])
# 	return model

sample_weight = np.ones(shape=(len(y_train),))
train_extent = calc_ice_extent(y_train)

#for i in range(len(sample_weight)):
#	sample_weight[i] = np.abs((train_extent[i] - np.mean(train_extent)) / np.mean(train_extent))

sample_weight[9::12]=1.2

print("first 5 sample weights: {}".format(sample_weight[0:5]))

'''
def exp_decay(epoch, lr):
	#init_rate = 0.1
	k = 0.1
	learning_rate = lr * math.exp(-k*epoch)
	return learning_rate
learning_rate = keras.callbacks.LearningRateScheduler(exp_decay)
'''

early_stopping = keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)

# convLSTM_image = create_convLSTM_image()
convLSTM_image = transformer_model
print(convLSTM_image.summary())
history2 = convLSTM_image.fit(x=X_train, y=y_train,
	batch_size=8,
	epochs=500,
	validation_split = .2,
	#sample_weight=sample_weight,
	callbacks=[early_stopping])
convLSTM_image.save("transform_convLSTM_image")

# image output
image_train_preds = convLSTM_image.predict(X_train, batch_size=8)
print("image_train_preds:", image_train_preds)
print("y_train:", y_train)
#image_train_mse, image_train_rmse = convLSTM_image.evaluate(X_train, y_train)

#image_train_preds = (image_train_preds_norm * y_train.std()) + y_train.mean()
image_train_rmse = math.sqrt(mean_squared_error(y_train.flatten(), image_train_preds.flatten()))

print("Image Concentration Train RMSE: {}".format(image_train_rmse))
print("Image Concentration Train NRMSE: {}".format(image_train_rmse / np.mean(y_train)))
print("Image Concentration Train NRMSE (std. dev.): {}".format(image_train_rmse / np.std(y_train)))
print("Train Prediction Shape: {}".format(image_train_preds.shape))

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_test_rolling_filled_final.npy", "rb") as f:
        X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_test_rolling_filled_final.npy", "rb") as f:
        y_test = np.load(f)

image_test_preds = convLSTM_image.predict(X_test, batch_size=4)
#image_test_mse, image_test_rmse = convLSTM_image.evaluate(X_test, y_test)

#image_test_preds = (image_test_preds_norm * y_test.std()) + y_test.mean()
image_test_rmse = math.sqrt(mean_squared_error(y_test.flatten(), image_test_preds.flatten()))

print("Image Concentration Test RMSE: {}".format(image_test_rmse))
print("Image Concentration Test NRMSE: {}".format(image_test_rmse / np.mean(y_test)))
print("Image Concentration Test NRMSE: {}".format(image_test_rmse / np.std(y_test)))
print("Test Prediction Shape: {}".format(image_test_preds.shape))

# save image outputs:
print(image_test_preds.shape)
with open("convlstm_image_rolling_filled_preds.npy", "wb") as f:
  np.save(f, image_test_preds)
with open("convlstm_image_rolling_filled_actual.npy", "wb") as f:
  np.save(f, y_test)

# Post-Processing
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
        land_mask = np.squeeze(np.load(f))

land_mask = land_mask[420:504]
pred_ice = np.squeeze(image_test_preds)
real_ice = np.squeeze(y_test)
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

# save post-processed output
with open("transform_postproc_convlstm_image_rolling_preds.npy", "wb") as f:
        np.save(f, pred_ice)
with open("transform_postproc_convlstm_image_rolling_actual.npy", "wb") as f:
        np.save(f, real_ice)

#calculate predicted extent
train_pred_extent = calc_ice_extent(image_train_preds)
train_actual_extent = calc_ice_extent(y_train)
train_extent_rmse = math.sqrt(mean_squared_error(train_actual_extent, train_pred_extent))
print("Last Month Predicted Extent(Train): {}".format(train_pred_extent[-1]))
print("Last Month Actual Extent (Train): {}".format(train_actual_extent[-1]))
print("Train Extent RMSE: {}".format(train_extent_rmse))
print("Train Extent NRMSE: {}".format(train_extent_rmse / np.mean(train_actual_extent)))
print("Train Extent NRMSE (std. dev.): {}".format(train_extent_rmse / np.std(train_actual_extent)))

test_pred_extent = calc_ice_extent(image_test_preds)
test_actual_extent = calc_ice_extent(y_test)
test_extent_rmse = math.sqrt(mean_squared_error(test_actual_extent, test_pred_extent))
print("Last Month Predicted Extent(Test): {}".format(test_pred_extent[-1]))
print("Last Month Actual Extent (Test): {}".format(test_actual_extent[-1]))
print("Test Extent RMSE: {}".format(test_extent_rmse))
print("Test Extent NRMSE: {}".format(test_extent_rmse / np.mean(test_actual_extent)))
print("Test Extent NRMSE (std. dev.): {}".format(test_extent_rmse / np.std(test_actual_extent)))

# save extent outputs:
print(image_test_preds.shape)
with open("transform_convlstm_extent_rolling_filled_preds.npy", "wb") as f:
  np.save(f, test_pred_extent)
with open("transform_convlstm_extent_rolling_filled_actual.npy", "wb") as f:
  np.save(f, test_actual_extent)


# Plot Loss (Sea Ice Extent)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss (Sea Ice Extent)')
plt.xlabel('Epoch')
plt.ylabel('Masked MSE')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('transform_Rolling_Filled_Extent_ConvLSTM_Loss_Plot.png')


# Plot Predicted vs. Actual Values (Sea Ice Extent)
fig = plt.figure(figsize = (24, 6))
ax = fig.add_subplot(111)
ax.plot([i+1 for i in range(train_actual_extent.shape[0])], train_pred_extent, c='b', label='Predicted')
ax.plot([i+1 for i in range(train_actual_extent.shape[0])], train_actual_extent, c='r', label='Actual')
ax.set_title('Sea Ice Extent by Month (Training Data)')
ax.set_xlabel('Month')
ax.set_ylabel('Sea Ice Extent (in $km^2$)')
ax.legend()
ax.grid(True)
fig.savefig('transform_Rolling_Filled_Extent_ConvLSTM_train_pred_vs_actual.png')

fig = plt.figure(figsize = (24, 6))
ax = fig.add_subplot(111)
ax.plot([i+1 for i in range(test_actual_extent.shape[0])], test_pred_extent, c='b', label='Predicted Extent')
ax.plot([i+1 for i in range(test_actual_extent.shape[0])], test_actual_extent, c='r', label='Actual Extent')
ax.set_title('Sea Ice Extent by Month (Testing Data)')
ax.set_xlabel('Month')
ax.set_ylabel('Sea Ice Extent (in $km^2$)')
ax.legend()
ax.grid(True)
fig.savefig('transform_Rolling_Filled_Extent_ConvLSTM_test_pred_vs_actual.png')
