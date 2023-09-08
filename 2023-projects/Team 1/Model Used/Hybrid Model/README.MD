# **Hybrid Model Approaches**
## Variational Auto Encoder (VAE) with XGBoost

  Building on the VAE previously defined, this project aimed to capitalize on the XGB model with the VAE predictions.
  The encoded data from the VAE latent space would be used as an input to the XGB model which would then train on the compressed data. 
  After training, the model would then predict the testing and validation data. Metrics were collected. 
  Next, 1201x1201 data was predicted and metrics collected and plotted. 

## LSTM+Dense layer based Model

The LSTM+Dense based model is trained with the same dataset as Dense based model.RMSE, MAE, R^2 metrics are also used here to evaluate the model.The model creates a prediction with the 1201x1201 dataset and a visualization is produced and saved as an h5 file. 

