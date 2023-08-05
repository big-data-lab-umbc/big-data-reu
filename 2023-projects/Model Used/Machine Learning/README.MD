# **XGBoost Model with velocity magnitude**
## Variational Auto Encoder (VAE) with Keras Layers

  The VAE creates a sampling layer and build encoder and decoder using Keras.
  The encoder and decoder are defined with custom parameters for our dataset including number of independent variables. 
  After data is preprocessed using the Nearest Neighbors interpolation, the model is trained.  
  The model training averages about 40 minutes to train with a 60-40 train-test split.
  Finally the trained model is applied to the test set to predict the test and validation data. Metrics are collected and the predictions are   plotted using matplotlib.

## Variational Auto Encoder (VAE) with XGBoost

  Building on the VAE previously defined, this project aimed to capitalize on the XGB model with the VAE predictions.
  The encoded data from the VAE latent space would be used as an input to the XGB model which would then train on the compressed data. 
  After training, the model would then predict the testing and validation data. Metrics were collected. 
  Next, 1201x1201 data was predicted and metrics collected and plotted. 

## Validation on the XGBOOST Model
  XGBoost is the top performing model. Data was preprocessed according to the shared preprocessing steps. 
  In ablation study , the XGB model was run on nearest neighbors, bilinear, and kriging interpolated dataets. 

  The model was training in ~4 minutes for all training points. The model was then used to predict testing and validation data. RMSE, MAE, and R^2 were collected for these predctions. The predictions were then plotted against the known data. These predictions demonstrated the top metrics collected by any model in this experiment. Additionally, the metrics were improved from the previous physic-based model. KFold validation was applied to confirm results of the model.
  Next, the model was applied to the 1201x1201 dataset were it would predict the entire 1.4m points. Metrics were collected here in comparison to the physics based 2014 model by Morlinghem. The predictions were also graphed as topography maps and density of predictions.Because of the impressive metrics of the XGBoost model, the terrain ruggedness index (TRI) was also computed for the XGBoost model to identify the amount of detail and roughness of the topography captured by the model. 

## Velocity Derivation

  Purpose: calculate the velocity magnitude of iceflow at (x,y) using basic magnitude equation. 
  Input: 
        data_full.csv
        df_1201_validation_data.csv 
  Output: 
        data_full_vMag.csv
        d1201_vMag.csv
