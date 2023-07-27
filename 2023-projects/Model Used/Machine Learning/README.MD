Keras VAE.ipynb
  it creates a sampling layer and build encoder and decoder.
  Then VAE is defined as model with custom train_step.
  Data is obtained from CSV file and model is trained.
  Model VAE with Nearest Neighbors Dataset is plotted.


STGP.ipynb
  Spatio Temporal Gaussian Processing.
  Data is cleaned by dropping duplicates.
  Velocity magnitude is computed to match final process for result comparison.
  Modeling was created and loss vs. Epoch was plotted.
  Then RMSE stat was obtained.
  1201 Runs Out of Space with the Resources Exhausted Error.


VAE-XGBoost.ipynb
  Similar data processing as STGP.
  VAE model was defined and trained.
  XGB model was trained with combined xtrain and vae encoder feats.
  1201x1201 data was validated.
  Model VAE & XGB with Nearest Neighbors Dataset was plotted.


XGBoost_Validation.ipynb
  Data was obtained and cleaned like previous ipynb files.
  XGB Regression model was defined and trained.
  Model performance was evaluated by calculating RMSE stat.
  Validation data (Predicted vs Actual) was plotted.
  1201x1201 data was validated.
  Plot density of 1201 vs physics model predictions was obtained.
  1201 Validation data (Prediction vs Actual) was plotted.


derive_velocity.ipynb
  Purpose: calculate the velocity magnitude of iceflow at (x,y)
  Input: 
        data_full.csv
        df_1201_validation_data.csv 
  Output: 
        data_full_vMag.csv
        d1201_vMag.csv
  Data was modified using np.sqrt.


merging data.ipynb
  train.csv, y_test.csv, and test.csv files were merged into data_full.csv.


gpr.py
  python file to run Gaussian Processing on Taki.

gpr.slurm
  To run the modeling on Taki environment.
