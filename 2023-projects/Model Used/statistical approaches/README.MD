
# **Statistical Approaches**

## Universal Kriging First Pass Prediction

The following contains code to perform purely spatial, first pass prediction for Universal Kriging on a particular slice of the total number of ground truth observations.

To get started, please follow these steps:

1. Download interpolated dataset training_test_combined_vmag_ky_230710.csv and place it in the same folder as UK_FirstPass_Final.py.

2. Execute the following command to run the Universal Kriging first pass prediction, replacing `0` and `100` with the desired range of exclusive indices for the slice of ground truth pixels flat array you intend to predict upon:
python UK_FirstPass_Final.py 0 100

3. After running the command, an output file named "pixels (0, 100) residuals.csv" will be generated. Once you have obtained all the required output files, you can merge them into a single dataframe or CSV format.



**Gaussian Process**
STGP.ipynb

  Spatio Temporal Gaussian Processing
  After preprocessing the data, the spatio-temporal model is trained. 
  Training time is greatly affected by the number of epochs trained. The example uses 15 epochs completing training in about 40 seconds. 
  Next, the loss was plotted. 
  Testing and validation datasets were predicted. Metrics were collected. 
  Unfortunately, the 1201 dataset encounters an out-of-space error due to the amount of memory needed to predict the large dataset. 
  Because of this, there is no visual associated with the STGP. Further research with batching and more resources could prove to predict the 1201 dataset could show improved results.

gpr.py

  python file to run Gaussian Processing on Taki.

gpr. slurm

  To run the modeling on the Taki environment.
