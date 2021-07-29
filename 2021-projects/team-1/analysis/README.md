# Analysis
This directory contains all of the files used for analysis on sea ice data as well as the vector autoregression model that the deep learning models were initially compared to.

*Climatology* contains `climatology.py`, which is the main climatology file for anomaly detection in the atmospheric variables, as well as three .ipynb files which are also used for anomaly detection. 

**In order to run any .ipynb files, you will need jupyter notebook**

*var* contains the python files corresponding to the vector autoregression models that we compared our deep learning models to. 

`var_model.py`

This code fits the VAR model and plots the predictions against the real values.

`var_data_maker.py` 

This code generates the training and testing data used in the VAR model.