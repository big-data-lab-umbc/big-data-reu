# **Benchmarking Existing Symbolic Regression Models**

### **Introduction**
Several symbolic regression models were benchmarked across our data to predict precipitation rate. Our benchmarking procedure mirrors the benchmarking procedure performed at [SRBench](https://cavalab.org/srbench/). Each model performs 10 iterations using scikit-learn's `train_test_split()` function to split the 2730 observations into 75% training (2047 observations) and 25% testing (683 observations) with the random state parameter set to the iteration of the for loop. This ensures that the data split will be the same for each model across each iteration.

### **Symbolic Regression Models Tested**

1.   [gplearn](https://github.com/trevorstephens/gplearn)
2.   [GP-GOMEA](https://github.com/marcovirgolin/GP-GOMEA)
3.   [PySR](https://github.com/MilesCranmer/PySR)
4.   [FFX](https://github.com/natekupp/ffx)
5.   [Feyn](https://docs.abzu.ai/)
6.   [RILS-ROLS](https://github.com/kartelj/rils-rols)
7.   [DSO](https://github.com/dso-org/deep-symbolic-optimization)
8.   [PyOperon](https://github.com/heal-research/pyoperon)

### **Setup and Installation**
It is reccomended to install a private conda environment for each of these packages individually according to their documentation.

### **Benchmarking models**
Each of the symbolic regression models follow a similar template to have comparable results. Modify the following code to match the dataset's path on your local machine:
```
  # import datasets
  df = pd.read_csv("/dataset_your_path_here")
```

Activate the conda environment respective to the model you wish to run. No command line arguments are needed, simply run:
```
python benchmark_<model>.py

```

 After running the model, a CSV named `benchmark_metrics_{model}.csv` is produced containing various metrics such as `train R^2`, `test R^2`, `train_NRMSE`, `test_NRMSE`, `simplicity`, and the `equation` for each iteration.


#### **Parallel Processing**
For each package except PySR, parallel processing is used to run all 10 iterations in parallel. This significantly decreases the wait time to benchmark the model. For PySR, in order to return reproducible results with random_state, we had to turn off parallelism. For the rest of the models, we used the `multiprocessing` tool from Python and used the following code to create a pool of worker processes:
```
  #create multiprocessing Pool object with 10 process workers
  pool = multiprocessing.Pool()
  pool = multiprocessing.Pool(processes=10)
```
By default, we have set `processes = 10`, but you may need to adjust this number depending on your machine's specifications.



