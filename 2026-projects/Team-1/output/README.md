# Outputs / Visualizations

This folder is for the purpose of showing examples of our generated outputs from our models in direct prediction and residual correction. All the python code for generation is under dpm_code and rcm_code with specific models specified in the name of the files. Each models code should be consistent in generating mapped images with the pixel layouts as well as metric summaries for comparison. 


For all models: Include maps and distribution histograms from the test results for t+1 00 UTC in both dpm (direct prediction) and rcm (residual correction), being our first timestep forecast, 12 hours after the initialization date.

This folder will also be used to show all visualizations and images included in our technical report and paper. 

Naming Convention: (modelnames_modelapproach_timestep_date)

Model Names: 
1. nf = Normalizing Flows
2. xgb = XGBoost
3. rf = Random Forest

Model Approachs:
1. dpm = direct prediction
2. rcm = residual correction

Timestep:
1. t1 = 12 hours from initialization
2. t3 = 36 hours from init
3. t5 = 60 hours from init
