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
