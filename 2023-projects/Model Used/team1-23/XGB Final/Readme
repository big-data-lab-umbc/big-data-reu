Final Process


________________________________________
Final Model: Extreme Gradient Boosting (XGB)
________________________________________
Contents

Contents	1
Folder Set Up with Generated Files	2
Final Processing Steps	3
Step 0: Merging	3
Step 1: Derive Velocity Magnitude Feature	3
Step 2: Modeling	3
Final Model	4
Parameters	4
Statistics	4
Visualizations	5
Maps	6





Folder Set Up with Generated Files
 
 
Final Processing Steps
Note: Change the main file path (“mainPath” variable) as needed in the "paths” cell; ensure packages are installed

Raw Starter Files: (currently uses the Nearest Neighbors interpolation) 
-	Adjust data inputs as needed for testing
●	train.csv
●	test.csv
●	y_test.csv
Step 0: Merging
File:  0_merging_ky_230615.ipynb
Input: raw starter files
Output: (to data folder)
●	Test_full.csv
○	This is the merged test only, this file is not actually used in any scripts
●	Data_full.csv	
○	This is the merged train, test, y_test files. 
○	This is the most important file in the scripts.
Step 1: Derive Velocity Magnitude Feature
File:  1_derive_velocity_ky_230711.ipynb
Purpose: calculate the velocity magnitude of iceflow at (x,y)
Input: 
●	data_full.csv
●	df_1201_validation_data.csv 
Output: 
●	data_full_vMag.csv
●	d1201_vMag.csv
Step 2: Modeling
File:  2_XGBT4M_full_ky_230711.ipynb
Input:
●	data_full_vMag.csv
●	d1201_vMag.csv
●	bed_BedMachine.h5 	#1201 physics model
Output: (Maps to 2_Results folder)
●	2_Results/XGBT4M_Results_<datetime>_D<depth of tree>I<number of iterations>E<assigned eta>.png 
Final Model
Parameters
●	Standard Scaling
●	Generated seed = 168
●	60-40-20 train-test-validation split
●	max_depth= 7
●	n_estimators= 350
●	min_child_weight= 0.25
●	subsample= 0.8
●	eta=.25

Statistics
Training time:
CPU times: user 3min 36s, sys: 323 ms, total: 3min 36s
Wall time: 2min 8s

Model Prediction Beginning
Model predicted.
Transform data back to original scale.
CPU times: user 3.84 s, sys: 6.45 ms, total: 3.84 s
Wall time: 2.73 s

Validation stats statements.
RMSE: 32.284464365975296
RMSE Percentage: 12894.338512283366
Mean Absolute Error: 22.13960268173217
Mean Absolute Percentage Error: 1.0216433060319823
R^2 Score: 0.9673897122924211
CPU times: user 1.21 s, sys: 0 ns, total: 1.21 s
Wall time: 659 ms

Predicting 1201
Time taken: 12.964ms

Predicted 1201 compared to the Physics Model
Euclidean Distance: 96433.46875
Cosine Similarity: 97.688%
Pearson Correlation Coefficient: 80.611%

KFold Cross Validation; k = 5
CPU times: user 24min 51s, sys: 1.91 s, total: 24min 52s
Wall time: 15min 11s
Visualizations 
  



RMSE: [30.55, 31.56]
MAE: [20.81, 21.37]
R^2: .97
Jumps in MAPE & RMSE are due to the large range we are working with - this is common.
 
Maps

Solution for dataset 1-2 assignment of prediction and physics model: 
Data set 1 is the predicted data and data set 2 is the physics data. 
 
 

	

