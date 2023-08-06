# **Dense and LSTM based Model**
## Dense layer based model

The model is created based on standardizing the features and scales the target variable. It uses dense layers to predict ice bed topography.The model is trained with Nearest Neighbor interpolated data with physics based infusion of velocity. RMSE, MAE, R^2 are used to evaluate the model. Finally, the topography is visualized using a 2D plot. 


## LSTM+Dense layer based Model

The LSTM+Dense based model is trained with the same dataset as Dense based model.RMSE, MAE, R^2 metrics are also used here to evaluate the model.The model creates a prediction with the 1201x1201 dataset and a visualization is produced and saved as an h5 file. 




