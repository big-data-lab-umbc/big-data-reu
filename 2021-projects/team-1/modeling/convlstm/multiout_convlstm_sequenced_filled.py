import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.compat.v1.keras import backend as K
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers

'''
Author: Peter I. Kruse
This model is a multi-task regression ConvLSTM, taking spatio-temporal input and producing both SIC map and sea ice extent output for each month in the data.
'''

# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/X_train_rolling_filled_final.npy", "rb") as f:
	X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/y_train_rolling_filled_final.npy", "rb") as f:
	y_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/X_test_rolling_filled_final.npy", "rb") as f:
	X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/y_test_rolling_filled_final.npy", "rb") as f:
	y_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/modeling/y_land_mask_actual.npy", "rb") as f:
        y_land_mask = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/y_extent_train_rolling_final.npy", "rb") as f:
        y_extent_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/y_extent_test_rolling_final.npy", "rb") as f:
        y_extent_test = np.load(f)

#reshape y_land_mask to 3 dimensions
y_land_mask = y_land_mask.reshape(448, 304, 1)

from sklearn.metrics import mean_squared_error
# define a custom loss function, which apply a mask turning land values to 0 during the optimization process
def custom_mse(y_true, y_pred):	
	#apply the mask
	y_pred_masked = tf.math.multiply(y_pred, y_land_mask)
	y_true_masked = tf.math.multiply(y_true, y_land_mask)
	#calculate MSE
	squared_resids = tf.square(y_true_masked - y_pred_masked)
	mse = tf.reduce_mean(squared_resids)
	return mse

# DATA
#DATASET_NAME = "organmnist3d"
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
# INPUT_SHAPE = (28, 28, 28, 1)
INPUT_SHAPE = (12, 448, 304, 11)
NUM_CLASSES = 11

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 10

# TUBELET EMBEDDING
PATCH_SIZE = (2, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
print(INPUT_SHAPE[0] // PATCH_SIZE[0])
print("NUM_PATCHES:", NUM_PATCHES)

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

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
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    # outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=outputs)
    return representation

transformer = create_transformer(
			tubelet_embedder=TubeletEmbedding(
				embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
			),
			positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM)
		)

# make multi output model
# define model class
class MultiOutputConvLSTM():
	# this convlstm contains two branches, one predicting SIC with images, and one predicting sea ice extent.
	def make_default_hidden_layers(self, inputs):
	# this method makes the default hidden layers, which both branches of the network will utilize
	# ConvLSTM2d -> MaxPooling3D -> ConvLSTM2D -> Conv2D -> Flatten -> Dense
		# x = keras.layers.ConvLSTM2D(8, (5,5), padding="same", return_sequences=False)(inputs)
		# x = keras.layers.MaxPooling2D((4, 4))(x)
		# x = keras.layers.Conv2D(128, (5,5), activation="relu")(x)
		# x = keras.layers.MaxPooling2D((4,4))(x)
		# x = keras.layers.Conv2D(32, (5,5), activation="relu")(x)
		# x = keras.layers.Flatten()(x)
		x = transformer(inputs)
		x = keras.layers.Dense(256, activation="relu")(x)
		return x

	def build_image_branch(self, inputs):
	# build the branch for image output
	# Dense Layer -> Reshape into image
		x = self.make_default_hidden_layers(inputs)
		x = keras.layers.Dense(512, activation="relu")(x)
		x = keras.layers.Dense(1024, activation="relu")(x)
		x = keras.layers.Dense(448*304, activation = "linear")(x)
		image_output = keras.layers.Reshape((448, 304, 1), input_shape=(448*304,), name="image_output")(x)
		return image_output

	def build_extent_branch(self, inputs):
	# build branch for univariate sea ice extent prediction
	# Dense -> Dense -> Dense -> Output
		x = self.make_default_hidden_layers(inputs)
		x = keras.layers.Dense(128, activation="relu")(x)
		x = keras.layers.Dense(32, activation="relu")(x)
		x = keras.layers.Dense(8, activation="relu")(x)
		extent_output = keras.layers.Dense(1, activation="linear", name="extent_output")(x)
		return extent_output

	def assemble_full_model(self, time_steps, width, height, features):
	# put it all together
		input_shape = (time_steps, width, height, features)
		#Spatio-temporal model input
		inputs = keras.layers.Input(shape=input_shape)

		# build image and extent output branches
		image_branch = self.build_image_branch(inputs)
		extent_branch = self.build_extent_branch(inputs)
		
		# initialize model: accepts input data, land mask, and actual values as input.
		# only input data is fed to the model - other data is used to calculate masked loss
		# outputs are an image of SIC for all pixels and a single sea ice extent value per month
		model = keras.models.Model(inputs=inputs,
					outputs=[image_branch, extent_branch],
					name="sea_ice_net")		
		# compile model
		# optimized with Adam, image output uses custom loss, and extent output uses mse loss
		# RMSE for both outputs is measured
		model.compile(optimizer="adamax", 
			loss={
			"image_output": custom_mse,
			"extent_output": "mse"},
			metrics={
			"image_output": keras.metrics.RootMeanSquaredError(),
			"extent_output": keras.metrics.RootMeanSquaredError()})		
		# add custom loss function to the model
		return model

# contstruct model from class
convLSTM_multiout = MultiOutputConvLSTM().assemble_full_model(12, 448, 304, 10)
print(convLSTM_multiout.summary())

# define loss weights for both branches
extent_sample_weights = np.ones(len(y_extent_train))
extent_sample_weights[9::12] = 1.2
image_sample_weights = np.ones(len(y_train))
image_sample_weights[9::12] = 1.2

# define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

# compile model
# optimized with Adam, image output uses custom loss, and extent output uses mse loss
# RMSE for both outputs is measured
# fit model
print(X_train.shape, y_train.shape, y_extent_train.shape)
history = convLSTM_multiout.fit(x=X_train, y=[y_train, y_extent_train],
				epochs=1000,
				batch_size=4,
				validation_split=.2,
				sample_weight=[image_sample_weights, extent_sample_weights],
				callbacks=[early_stopping])

# save fitted model
convLSTM_multiout.save("multiout_convLSTM")

# image/exent output
# predict training values
image_train_preds, extent_train_preds = convLSTM_multiout.predict(X_train, batch_size=4)

# compare to actual training values
image_train_rmse = math.sqrt(mean_squared_error(y_train.flatten(), image_train_preds.flatten()))
extent_train_rmse = math.sqrt(mean_squared_error(y_extent_train, extent_train_preds))

#print RMSE and NRMSE
print("Image Concentration Train RMSE: {} \nExtent Train RMSE: {}".format(image_train_rmse, extent_train_rmse))
print("Image Concentration Train NRMSE: {} \nExtent Train NRMSE: {}".format(image_train_rmse / np.mean(y_train), extent_train_rmse / np.mean(y_extent_train)))
print("Image Concentration Train NRMSE (std. dev): {} \nExtent Train NRMSE (std. dev): {}".format(image_train_rmse / np.std(y_train), extent_train_rmse / np.std(y_extent_train)))
print("Image Train Prediction Shape: {} \nExtent Train Predictions Shape: {}".format(image_train_preds.shape, extent_train_preds.shape))

#predict test values
image_test_preds, extent_test_preds = convLSTM_multiout.predict(X_test, batch_size=4)

#compare to actual test values
image_test_rmse = math.sqrt(mean_squared_error(y_test.flatten(), image_test_preds.flatten()))
extent_test_rmse = math.sqrt(mean_squared_error(y_extent_test, extent_test_preds))

# print RMSE and NRMSE
print("Image Concentration Test RMSE: {} \nExtent Test RMSE: {}".format(image_test_rmse, extent_test_rmse))
print("Image Concentration Test NRMSE: {} \nExtent Test NRMSE: {}".format(image_test_rmse / np.mean(y_test), extent_test_rmse / np.mean(y_extent_test)))
print("Image Concentration Test NRMSE (std. dev): {} \nExtent Test NRMSE (std. dev): {}".format(image_test_rmse / np.std(y_test), extent_test_rmse / np.std(y_extent_test)))
print("Image Test Prediction Shape: {} \nExtent Test Predictions Shape: {}".format(image_test_preds.shape, extent_test_preds.shape))

# save image/extent outputs:
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/multiout_filled_convlstm_image_rolling_preds.npy", "wb") as f:
	np.save(f, image_test_preds)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/multiout_filled_convlstm_image_rolling_actual.npy", "wb") as f:
	np.save(f, y_test)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/multiout_filled_convlstm_extent_rolling_preds.npy", "wb") as f:
	np.save(f, extent_test_preds)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/multiout_filled_convlstm_extent_rolling_actual.npy", "wb") as f:
	np.save(f, y_extent_test)


import matplotlib.pyplot as plt
# Plot Loss (Image)
plt.plot(history.history['image_output_loss'])
plt.plot(history.history['val_image_output_loss'])
plt.title('Multi Output Model Loss (Image)')
plt.xlabel('Epoch')
plt.ylabel('Masked MSE')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Multiout_Filled_Rolling_Image_ConvLSTM_Loss_Plot.png')

# Plot Loss (Extent)
plt.clf()
plt.plot(history.history['extent_output_loss'])
plt.plot(history.history['val_extent_output_loss'])
plt.title('Multi Output Model Loss (Extent)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Multiout_Filled_Rolling_Extent_ConvLSTM_Loss_Plot.png')


# Plot Predicted vs. Actual Values (Sea Ice Extent)
# train
fig = plt.figure(figsize = (24, 6))
ax = fig.add_subplot(111)
ax.plot([i+1 for i in range(y_extent_train.shape[0])], extent_train_preds, c='b', label='Predicted')
ax.plot([i+1 for i in range(y_extent_train.shape[0])], y_extent_train, c='r', label='Actual')
ax.set_title('Sea Ice Extent by Month (Training Data)')
ax.set_xlabel('Month')
ax.set_ylabel('Sea Ice Extent (in $km^2$)')
ax.legend()
ax.grid(True)
fig.savefig('Multiout_Filled_Rolling_ExtentConvLSTM_train_pred_vs_actual.png')

# test
fig = plt.figure(figsize = (24, 6))
ax = fig.add_subplot(111)
ax.plot([i+1 for i in range(y_extent_test.shape[0])], extent_test_preds, c='b', label='Predicted Extent')
ax.plot([i+1 for i in range(y_extent_test.shape[0])], y_extent_test, c='r', label='Actual Extent')
ax.set_title('Sea Ice Extent by Month (Testing Data)')
ax.set_xlabel('Month')
ax.set_ylabel('Sea Ice Extent (in $km^2$)')
ax.legend()
ax.grid(True)
fig.savefig('Multiout_Filled_Rolling_ExtentConvLSTM_test_pred_vs_actual.png')

