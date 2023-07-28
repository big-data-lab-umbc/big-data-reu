*Dense_With_NewPreprocessing.ipynb*
-standardizes the features and scales the target variable
-uses dense layers to predict ice bed topography
-KNN MAG data used with the training
-Evaluates the model using RMSE, MAE, R^2
-Visualizes predictions as a 2D plot


*Dense+LSTM_With_NewPreprocessing.ipynb*
-standardizes the features and scales the target variable
-uses Dense and LSTM layers to predict ice bed topography
-KNN MAG data used with the training
-Evaluates the model using RMSE, MAE, R^2
-Visualizes predictions as a 2D plot

https://colab.research.google.com/drive/1RpweHU6xLkExC9KoBTyBpwAVAwabnWEm 
Dense + LSTM with vMag dataset.ipynb
The full velocity magnitude csv is read in, cleaned, and split into a testing and training section. 
The Dense + LSTM model is defined and trained. 
The model creates predictions for each training and testing data and metrics are calculated. 
The model creates a prediction for the 1201x1201 dataset and a visualization is produced and saved as an h5 file. 

https://colab.research.google.com/drive/10pE5ayaUZJfz00SjFfSWW2RKjLOmbhg 
LSTM with vMag dataset.ipynb
The full velocity magnitude csv is read in, cleaned, and split into a testing and training section. 
The LSTM model is defined and trained. 
The model creates predictions for each training and testing data and metrics are calculated. 
The model creates a prediction for the 1201x1201 dataset and a visualization is produced and saved as an h5 file. 


