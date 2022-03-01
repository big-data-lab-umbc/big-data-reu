import math
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.compat.v1.keras import backend as K
from sklearn.metrics import mean_squared_error
from numpy.random import seed

'''
This model is a multi-task regression ConvLSTM, taking spatio-temporal input and producing both SIC map and sea ice extent output for each month in the data.
'''

# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_train_rolling_filled_final.npy", "rb") as f:
	X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_train_rolling_filled_final.npy", "rb") as f:
	y_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_test_rolling_filled_final.npy", "rb") as f:
	X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_test_rolling_filled_final.npy", "rb") as f:
	y_test = np.load(f)
#with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_train_rolling.npy", "rb") as f:
#	y_train_mask = np.load(f)
#with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_test_rolling.npy", "rb") as f:
#	y_test_mask = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_actual.npy", "rb") as f:
        y_land_mask = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_extent_train_rolling_final.npy", "rb") as f:
        y_extent_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_extent_test_rolling_final.npy", "rb") as f:
        y_extent_test = np.load(f)
#with open("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy", "rb") as f:
#	y_extent = np.load(f)

'''
X_train_norm = (X_train - X_train.mean()) / X_train.std()
y_train_norm = (y_train - y_train.mean()) / y_train.std()
X_test_norm = (X_test - X_test.mean()) / X_test.std()
y_test_norm = (y_test - y_test.mean()) / y_test.std()
'''

#reshape y_land_mask to 3 dimensions
y_land_mask = y_land_mask.reshape(448, 304, 1)

#y_extent_train_norm = (y_extent_train - y_extent_train.mean()) / y_extent_train.std()
#y_extent_test_norm = (y_extent_test - y_extent_test.mean()) / y_extent_test.std()

from sklearn.metrics import mean_squared_error
# define a custom loss function, which apply a mask turning land values to 0 during the optimization process
def custom_mse(y_true, y_pred):	
	#apply the mask
	y_pred_masked = tf.math.multiply(y_pred, y_land_mask)
	y_true_masked = tf.math.multiply(y_true, y_land_mask)
	#calculate MSE
	return K.mean(K.square(y_pred_masked - y_true_masked))

#define rmse loss
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))

def custom_activation(inp):
	return tf.clip_by_value(tf.multiply(inp, y_land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy() 
'''
# define a data generator, which loads data one batch at a time in order to save memory when running the model
def _wrap_in_dictionary(data, mask, image_targets, extent_targets, batch_size):
	# calculate number of batches
	batches = (len(data) + batch_size - 1) // batch_size
	while(True):
		for i in range(batches):
			# partition data into samples of size batch_size
			X1 = data[i*batch_size : (i+1)*batch_size]
			X2 = mask[i*batch_size : (i+1)*batch_size]
			Y1 = image_targets[i*batch_size : (i+1)*batch_size]
			Y2 = extent_targets[i*batch_size : (i+1)*batch_size]
			X = {"train_input": X1, "mask": X2} 
			y = {"image_output": Y1, "extent_output": Y2} 
			yield (X, y)
'''
print(X_train.shape)


# make multi output model
# define model class
class MultiOutputConvLSTM():
	# this convlstm contains two branches, one predicting SIC with images, and one predicting sea ice extent.
	def make_default_hidden_layers(self, inputs):
	# this method makes the default hidden layers, which both branches of the network will utilize
	# ConvLSTM2d -> MaxPooling3D -> ConvLSTM2D -> Conv2D -> Flatten -> Dense
		x = keras.layers.ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, data_format="channels_last")(inputs)
		x = keras.layers.MaxPooling2D((4, 4))(x)
		x = keras.layers.Conv2D(128, (5,5), activation="relu", padding="same")(x)
		x = keras.layers.MaxPooling2D((4,4))(x)
		x = keras.layers.Conv2D(32, (5,5), activation="relu", padding="same")(x)
		x = keras.layers.Flatten()(x)
		x = keras.layers.Dense(256, activation="relu")(x)
		return x

	def build_image_branch(self, inputs):
	# build the branch for image output
	# Dense Layer -> Reshape into image
		x = self.make_default_hidden_layers(inputs)
		x = keras.layers.Dense(512, activation="relu")(x)
		x = keras.layers.Dense(448*304, activation="linear")(x)
		image_output = keras.layers.Reshape((448, 304, 1), input_shape = (448*304,), name="image_output")(x)
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

# sample weighting
extent_sample_weights = np.ones(len(y_extent_train))
extent_sample_weights[9::12] = 1.2
image_sample_weights = np.ones(len(y_train))
image_sample_weights[9::12] = 1.2

'''
def exp_decay(epoch, lr):
        k = 0.1
       	learning_rate = lr * math.exp(-k*epoch)       
        return learning_rate
learning_rate = keras.callbacks.LearningRateScheduler(exp_decay)
'''

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

#convLSTM_multiout.save("multiout_convLSTM")

# image/exent output
# predict training values
image_train_preds, extent_train_preds = convLSTM_multiout.predict(X_train, batch_size=4)

#image_train_preds = (image_train_preds_norm * y_train.std()) + y_train.mean()
#extent_train_preds = (extent_train_preds_norm * y_extent_train.std()) + y_extent_train.mean()

# compare to actual training values
#print(convLSTM_multiout.evaluate(x=X_train, y={"image_output": y_train, "extent_output": y_extent_train}))
#total_mse, image_train_mse, image_train_rmse, extent_train_mse, extent_train_rmse = convLSTM_multiout.evaluate(x=X_train, y={"image_output": y_train, "extent_output": y_extent_train})

image_train_rmse = math.sqrt(mean_squared_error(y_train.flatten(), image_train_preds.flatten()))
extent_train_rmse = math.sqrt(mean_squared_error(y_extent_train, extent_train_preds))

#print RMSE
print("Image Concentration Train RMSE: {} \nExtent Train RMSE: {}".format(image_train_rmse, extent_train_rmse))
print("Image Concentration Train NRMSE: {} \nExtent Train NRMSE: {}".format(image_train_rmse / np.mean(y_train), extent_train_rmse / np.mean(y_extent_train)))
print("Image Concentration Train NRMSE (std. dev): {} \nExtent Train NRMSE (std. dev): {}".format(image_train_rmse / np.std(y_train), extent_train_rmse / np.std(y_extent_train)))
print("Image Train Prediction Shape: {} \nExtent Train Predictions Shape: {}".format(image_train_preds.shape, extent_train_preds.shape))

#predict test values
image_test_preds, extent_test_preds = convLSTM_multiout.predict(X_test, batch_size=4)

#image_test_preds = (image_test_preds_norm * y_test.std()) + y_test.mean()
#extent_test_preds = (extent_test_preds_norm * y_extent_test.std()) + y_extent_test.mean()

#compare to actual test values
#print(convLSTM_multiout.evaluate(x=X_test, y={"image_output": y_test, "extent_output": y_extent_test}))
#total_loss, image_test_mse, image_test_rmse, extent_test_mse, extent_test_rmse = convLSTM_multiout.evaluate(x=X_test, y={"image_output": y_test, "extent_output": y_extent_test})

image_test_rmse = math.sqrt(mean_squared_error(y_test.flatten(), image_test_preds.flatten()))
extent_test_rmse = math.sqrt(mean_squared_error(y_extent_test, extent_test_preds))

# print RMSE
print("Image Concentration Test RMSE: {} \nExtent Test RMSE: {}".format(image_test_rmse, extent_test_rmse))
print("Image Concentration Test NRMSE: {} \nExtent Test NRMSE: {}".format(image_test_rmse / np.mean(y_test), extent_test_rmse / np.mean(y_extent_test)))
print("Image Concentration Test NRMSE (std. dev): {} \nExtent Test NRMSE (std. dev): {}".format(image_test_rmse / np.std(y_test), extent_test_rmse / np.std(y_extent_test)))
print("Image Test Prediction Shape: {} \nExtent Test Predictions Shape: {}".format(image_test_preds.shape, extent_test_preds.shape))

# save image/extent outputs:
with open("multiout_filled_convlstm_image_rolling_preds.npy", "wb") as f:
	np.save(f, image_test_preds)
with open("multiout_filled_convlstm_image_rolling_actual.npy", "wb") as f:
	np.save(f, y_test)
with open("multiout_filled_convlstm_extent_rolling_preds.npy", "wb") as f:
	np.save(f, extent_test_preds)
with open("multiout_filled_convlstm_extent_rolling_actual.npy", "wb") as f:
	np.save(f, y_extent_test)

# Post-Processing
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
        land_mask = np.squeeze(np.load(f))

land_mask = land_mask[420:504]
pred_ice = np.squeeze(image_test_preds)
real_ice = np.squeeze(y_test)
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

# save post-processed output
with open("postproc_multiout_filled_convlstm_image_rolling_preds.npy", "wb") as f:
        np.save(f, pred_ice)
with open("postproc_multiout_filled_convlstm_image_rolling_actual.npy", "wb") as f:
        np.save(f, y_test)


mse = mean_squared_error(real_ice.flatten(), pred_ice.flatten())
rmse = math.sqrt(mse)
nrmse = rmse / np.mean(y_test)
nrmse_std = rmse / np.std(y_test)
print("Post-Processed MSE: ", mse, "\n", "RMSE: ", rmse, "\n", "NRMSE: ", nrmse, "\n", "NRMSE (std. dev)", nrmse_std)

import matplotlib.pyplot as plt
# Plot Loss (Image)
plt.plot(history.history['image_output_loss'])
plt.plot(history.history['val_image_output_loss'])
plt.title('Multi Output Model Loss (Image)')
plt.xlabel('Epoch')
plt.ylabel('Masked RMSE')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Multiout_Filled_Rolling_Image_ConvLSTM_Loss_Plot.png')

# Plot Loss (Extent)
plt.clf()
plt.plot(history.history['extent_output_loss'])
plt.plot(history.history['val_extent_output_loss'])
plt.title('Multi Output Model Loss (Extent)')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
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

