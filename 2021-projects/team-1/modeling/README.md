# Modeling
This directory contains the models used in our research

`y_land_mask_actual.npy` is the land mask that is used in the custom loss funtion in both the CNN and ConvLSTM to mask out land values.

## CNN
------------
The following are the train and test sets that were used as inputs in each CNN model: 
- `x_train_whole_lag_two.npy`
- `x_test_whole_lag_two.npy`
- `y_train_whole_lag_two.npy`
- `y_test_whole_lag_two.npy`

These files can be found in the preprocessing directory in the **CNN** folder.

#### Initial Model
`cnn_image_custom_loss.py`
- Inputs: Monthly averaged 25km dataset containing images with 10 input channels, corresponding to the 10 predictors 
- Outputs: Outputs monthly per-pixel sea ice concentration predictions and true values in two separate .npy files. Predictions are post-processed to remove SIC values over land and induce a range of [0, 100]
- Model: Convolutional neural network with custom loss function which multiplies each predicted image by the land.

Predicted and real values for this model will be saved as numpy files labeled `pred_ice_comparison_base_cnn_maskthresh_lag_one_small_batch.npy` and `real_ice_comparison_base_cnn_maskthresh_lag_one_small_batch.npy` respectively.

#### Extent Loss Model
`cnn_image_custom_ice_extent.py`
- Inputs: Monthly averaged 25km dataset containing images with 10 input channels, corresponding to the 10 predictors 
- Outputs: Outputs monthly per-pixel sea ice concentration predictions and true values in two separate .npy files. Predictions are post-processed to remove SIC values over land and induce a range of [0, 100]
- Model: Convolutional neural network with custom loss function which calculates and combines SIC and SIE loss. 

Predicted and real values for this model will be saved as numpy files labeled `pred_ice_extent_maskthresh_lag_one_small_batch.npy` and `real_ice_extent_maskthresh_lag_one_small_batch.npy` respectively. 



## Multi-task CNN
------------
`multiout_cnn.py`
- Input: Monthly averaged 25km dataset containing images with 10 input channels, corresponding to the 10 predictors 
- Outputs: Monthly averaged per-pixel sea ice concentration and monthly averaged total sea ice extent
- Modeling process: A multi-layer, multi-task CNN model that trains on the first 33 years of data and tests on the last 8 years. The model learns     the per-pixel values of all attributes from the previous month and predicts both per-pixel SIC and total sea ice extent for the next month.

The actual and predicted monthy averaged per-pixel sea ice concentratiosn will be saved as numpy files labeled `real_ice_multiout_cnn_post_1_32.npy` and `pred_ice_multiout_cnn_post_1_32.npy` respectively.

The actual and predicted monthly averaged per-pixel sea ice extents will be saved as numpy files labeled `pred_extent_multiout_cnn_lag_one.npy` and `real_extent_multiout_cnn_lag_one.npy respectively` 


## ConvLSTM
------------
`convlstm_sequenced_filled.py`
- Input: Monthly averaged 25km dataset with a rolling, stateless window, containing samples of 12 months and 10 atmospheric variables.
- Output: Monthly averaged per-pixel sea ice concentration
- Model: A multi-layer ConvLSTM model that trains on the first 33 years of data and tests on the last 7 years (the data can be found in the preprocessing directory: **/preprocessing/convlstm/Filled Sequence Data**). The model learns the per-pixel values of all attributes from the previous month and predicts per-pixel SIC for the next month.

The actual and predicted averaged per-pixel sea ice concentration will be saved as numpy files labeled `convlstm_image_rolling_filled_preds.npy` and `convlstm_image_rolling_filled_actual`.npy respectively.

 `convlstm_plots.py` can be used to plot results from this model.

## Multi-task ConvLSTM
-------------
`multiout_convlstm_sequenced_filled.py`
- Input: Monthly averaged 25km dataset with a rolling, stateless window, containing samples of 12 months and 10 atmospheric variables including all atmospheric variables.
- Outputs: Monthly averaged per-pixel sea ice concentration and monthly averaged total sea ice extent
- Model: A multi-layer, multi-task ConvLSTM model that trains on the first 33 years of data and tests on the last 7 years  (the data can be found in the preprocessing directory: **/preprocessing/convlstm/Filled Rolling Data**. The model learns the per-pixel values of all attributes from the previous month and predicts both per-pixel SIC and total sea ice extent for the next month.

The actual and predicted monthly averaged per-pixel *sea ice concentration* will be saved as numpy files labeled `multiout_filled_convlstm_image_rolling_actual.npy` and `multiout_filled_convlstm_image_rolling_preds.npy` respectively.

The actual and predicted monthly averaged total *sea ice extent* will be saved as numpy files labeled `multiout_filled_convlstm_extent_rolling_actual.npy` and `multiout_filled_convlstm_extent_rolling_preds.npy` respectively.

`multiout_plots.py` can be used to plot all results from this model. 