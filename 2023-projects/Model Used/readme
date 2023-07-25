Steps to Run Code

Add a shortcut to this folder in my drive for the data
https://drive.google.com/drive/folders/1j8qnjxflXDDSgQUwYMRGFRRqt44VRHO2 

The code can be run as “run all” from there. You will need to authorize the code to access your google drive. 

import h5py

new_filename_2 = '/content/drive/MyDrive/REU 2023 Team 1: Ice Bed Topography Prediction/Research/Chen_Work/Old Models/Dense_LSTM_Vmag_Trial2.h5'

with h5py.File(new_filename_2, 'w') as hf:
    hf.create_dataset("track bed prediction",  data=prediction_on_1201_2D)

h5 file save location and name can be changed by modifying the file path string in the code above


Data and code sources

Data preprocessing taken as a copy from 
https://colab.research.google.com/drive/1nTcapkrEzOS99_DeKiYTVwveb6qyEUOR 

Models, metrics, and visualizations taken from 
Dense + LSTM
https://colab.research.google.com/drive/1MSVpN0OIAokC2T3HPOFavkbbhuClC19l 
LSTM
https://colab.research.google.com/drive/1RU_7ZLeKs1T_cQcNATRmBHWcIwYvvbAx 
