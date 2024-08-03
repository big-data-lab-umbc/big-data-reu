# **Symbolic Regression on Subsets of Data**

### **Introduction**
One significant challenge to quantitative precipitation estimation is ensuring that the methods are applicable to various types and intensities.  To test whether the accuracy and interpretability of the learned equations improves, we applied four methods to subset the data before applying symbolic regression on the separate subsets.

Ways we separated our data:

1.   Precipitation Type
2.   Clustering Algorithms
3.   Threshold of Mean $Z_{DR}$ and $ρ_{hv}$
4.   Decision Tree Leaf Nodes

After separating the data, we benchmarked across these all four methods using Feyn. We ran 10 trials of symbolic regression with each trial splitting the train and test data with a different `random_state` (being the index of the loop). This ensures that the data split will be the same for each sub-dataset across each iteration.

### **Setup and Installation**
Feyn is required to run these benchmarks. Install their package according to their [documentation](https://docs.abzu.ai/docs/guides/getting_started/quick_start).

### **Quickstart**

Modify the following code to match the dataset's path on your local machine:
```
  # import datasets
  df = pd.read_csv("/dataset_your_path_here")
```

#### **Input**
For **2. Clustering Algorithms**, the code takes in a command-line input with the name of the column with the cluster labels
(0, 1, 2). The command to run this code is below:
```
python subset_feyn_clusters.py cluster_column_name
```

For **3. Threshold of Mean $Z_{DR}$ and $ρ_{hv}$**, the code takes in the name of the radar variable as input. The command to run this code is below, where `radar_name` is `Z`, `ZDR`, `KDP`, or `RhoHV`:
```
python subset_feyn_radar_mean.py radar_name
```