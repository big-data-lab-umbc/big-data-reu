#  Data Preproccessing
This directory contains all the files needed to preprocess data for the CNN and ConvLSTM respectively. The `area_data.py` and `msk_file.py` both contain code regarding the reshaping and filtering of land masks to be applied in the custom loss function.

------------
#### Atmospheric Variables
These are the 10 features that are recorded in the dataset
- sea surface temperature
- humidity
- surface pressure
- sea ice conventration
- wind speed
- air temperature
- shortwave radiation
- longwave radiation
- rain rate
- snow rate


The **cnn** directory contains all of the files needed to preprocess data into a form suitable for the convolutional neural network model.

`whole_preprocess.py`

This code calculates the monthly average of each feature and changes all "-999" values across each feature to "NaN", then saves the resulting numpy arrays.

`comp_preprocess.py`

This code fills in the "North pole hole" by  making NaN values in that region to be 100 percent ice, it also splits the data into training and testing saves each set as numpy arrays.


------------


The **convlstm** directory contains all of the files needed to preprocess the data into a form suitable for the convolutional long short term memory model.
- `whole_comparison_preprocess.py` splits the data to be similar to the input data of the CNN, so that they CNN and ConvLSTM can be compared.
- `convlstm_seq_filled_preprocess.py` creates a sliding sequence for every 12 months and adds an extra dimension, time, to the data. 
- `convlstm_rolling_filled.py`loads rolling, sequenced data and removes first 12 months of output observations and last month of training observations.