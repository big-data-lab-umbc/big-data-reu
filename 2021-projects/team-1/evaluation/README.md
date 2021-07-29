#Evaluation
This directory contains the files needed for the post-processed results from each respective model.

*note that for all models, SIC values below 0 are set to 0, and all values above 100 are set to 100 before RMSE is calculated. 

### ConvLSTM
`maskthresh_rolling_convlstm.py` 

The following code can be used to calculate the RMSE of predicted sea ice concentration values from the convLSTM model. 
The results of this code will be saved as a numpy file labeled `convlstm_rolling_pred_ice.npy`.

### Multi-task ConvLSTM
`maskthresh_multiout_convlstm.py`

This code is used to calculate the RMSE of predicted sea ice concentration values from the Multi-task ConvLSTM model. 
The results of this code will be saved as a numpy file labeled `multiout_convlstm_rolling_pred_ice.npy`.

### CNN
`maskthresh.py`

This code is used to calculate the MSE and RMSE of predicted sea ice concentration values from the CNN model.  It also removes values over land pixels with the same land masking process as the ConvLSTM. The results of this code will be saved as a numpy file labeled `pred_ice_comparison_base_cnn_maskthresh.npy` for predicted values and `real_ice_comparison_base_cnn_maskthresh.npy` for actual values. 

`cnn_ice_extent_calc.py` 

This code is used to calculate sea ice extent based on sea ice concentration images using per-pixel area calculation and a simple SIE algorithm. 
The resulting SIE values are saved as numpy files labeled `area_extents_extent_cnn_lag_one.npy` for values calculated from the model's predictions and `real_area_extents_extent_cnn_lag_one.npy` for the real SIE values. 