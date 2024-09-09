"""
Feyn is not an algorithm that was tested on SR bench. I am benchmarking feyn
and using multiprocessing from python to run all the tests in parallel.
"""

import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import feyn
import sympy as sp

def get_symbolic_model(expr):
    """takes string as input and
    outputs a SymPy equation"""
    feature_names = ['Z', 'ZDR', 'KDP', 'RhoHV']
    local_dict = {f:sp.Symbol(f) for f in feature_names}
    sp_model = sp.parse_expr(expr, local_dict=local_dict)
    return sp_model

def get_simplicity(sp_model):
    """takes SymPy equation as input and
    outputs a score representing the equation's simplicity"""
    # compute numumber of components
    num_components = 0
    for _ in sp.preorder_traversal(sp_model):
        num_components += 1

    # compute simplicity as per srbench criteria by La Cava et al. (2021)
    simplicity = -np.round(np.log(num_components)/np.log(5), 1)
    return simplicity

def model(seed: int):
  print(f"Iteration {seed}")

  #instantiate model
  ql = feyn.QLattice(random_seed =1)

  #split data into train and test, with seed being the iteration of the for loop
  train_df, test_df = train_test_split(df, test_size=0.25, random_state=seed)

  #fit and train model
  models = ql.auto_run(
    data=train_df,
    output_name='gauge_precipitation_matched',
    )

  #get best model
  best_model = models[0]

  # predict on train and test
  y_pred_train = best_model.predict(train_df)
  y_pred_test = best_model.predict(test_df)

  #get prediction on whole dataset
  y_pred = best_model.predict(df)

  # calculate train and test r2
  r2_train = r2_score(train_df['gauge_precipitation_matched'], y_pred_train)
  r2_test = r2_score(test_df['gauge_precipitation_matched'], y_pred_test)

  #calculate r2 overall on all dataset
  all_r2 = r2_score(df['gauge_precipitation_matched'], y_pred)

  #calculate train and test rmse and overall rmse
  rmse_train = (np.sqrt(mean_squared_error(train_df['gauge_precipitation_matched'], y_pred_train)))
  rmse_test = (np.sqrt(mean_squared_error(test_df['gauge_precipitation_matched'], y_pred_test)))
  all_rmse = np.sqrt(mean_squared_error(df['gauge_precipitation_matched'], y_pred))

  #calculate nrmse
  train_nrmse = (rmse_train/train_df['gauge_precipitation_matched'].mean())
  test_nrmse = (rmse_test/test_df['gauge_precipitation_matched'].mean())
  all_nrmse = (all_rmse/df['gauge_precipitation_matched'].mean())

  #convert the best model to sympify string
  expr = str(best_model.sympify())

  # ensure that equation string has consistent variable names: Z, ZDR, KDP, RhoHV
  feature_dict = {"x0":"Z", "x1":"ZDR", "x2":"KDP", "x3":"RhoHV",
                  "x_0":"Z", "x_1":"ZDR", "x_2":"KDP", "x_3":"RhoHV",
                  "Reflectivity_sc":"Z", "Zdr_sc":"ZDR", "Kdp_sc":"KDP", "Rhohv_sc":"RhoHV"}

  for x, var in feature_dict.items():
    expr = expr.replace(x, var)

  # simplify equation and append to list
  simplified_expr = get_symbolic_model(expr)

  # append simplicity score to list
  expr_simple = get_simplicity(simplified_expr)

  print(f"returning for seed {seed}")

  # Package metrics into a dictionary to return
  return {
    'train_r2': r2_train,
    'test_r2': r2_test,
    'all_r2': all_r2,
    'train_rmse': rmse_train,
    'test_rmse': rmse_test,
    'train_nrmse': train_nrmse,
    'test_nrmse': test_nrmse,
    'all_nrmse': all_nrmse,
    'equation': simplified_expr,
    'simplicity': expr_simple,
  }


if __name__ == "__main__":

  #read in main dataset
  main_df = pd.read_csv("dataset.csv")

  #split datasets into regular and scaled df. 
  df = main_df[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv', 'gauge_precipitation_matched']]

  #create multiprocessing Pool object with 10 processes
  pool = multiprocessing.Pool() 
  pool = multiprocessing.Pool(processes=10) 

  #our seeds for train/test split
  seed_inputs = [0,1,2,3,4,5,6,7,8,9]

  #time how long it takes
  start = time.time()
  
  #run all processors in parallel to perform 'model' method, with seed_inputs as arguments
  results = pool.map(model, seed_inputs) 

  #end time
  end = time.time()

  #calculate time it took
  total = end- start
  print(f"time running all SR algorithms: {total}" )

  # Process and print results
  train_r2 = [result['train_r2'] for result in results]
  test_r2 = [result['test_r2'] for result in results]
  all_r2= [result['all_r2'] for result in results]
  train_rmse = [result['train_rmse'] for result in results]
  test_rmse = [result['test_rmse'] for result in results]
  train_nrmse = [result['train_nrmse'].item() for result in results]
  test_nrmse = [result['test_nrmse'].item() for result in results]
  all_nrmse =   [result['all_nrmse'].item() for result in results]
  equations = [result['equation'] for result in results]
  simplicity = [result['simplicity'].item() for result in results]

  #convert output to dataframe
  metrics ={
    'train_r2': train_r2,
    'test_r2': test_r2,
    'all_r2': all_r2,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_nrmse': train_nrmse,
    'test_nrmse': test_nrmse,
    'all_nrmse': all_nrmse,
    'equation': equations,
    'simplicity': simplicity,
  }
  output = pd.DataFrame(metrics)

  output.to_csv(f"benchmark_metrics_feyn.csv")

  ##############################################################################################
  #report the iteration with the approx. median (the 5th highest) test r^2
  med_test_r2 = np.sort(output['test_r2'].values)[4]
  
  #get row high highest med_test_r2, and get each metric within each column
  print("Metrics for Iteration with Median Test R2")
  
  print("Train R2", output.loc[output['test_r2'] == med_test_r2, 'train_r2'].values[0])
  print("Test R2", output.loc[output['test_r2'] == med_test_r2, 'test_r2'].values[0])
  print("Train NRMSE", output.loc[output['test_r2'] == med_test_r2, 'train_nrmse'].values[0])
  print("Test NRMSE", output.loc[output['test_r2'] == med_test_r2, 'test_nrmse'].values[0])
  print("Simplicity", output.loc[output['test_r2'] == med_test_r2, 'simplicity'].values[0])
  print("Equation:", output.loc[output['test_r2'] == med_test_r2, 'equation'].values[0])
  
  
  #report the iteration with best test r^2
  max_test_r2 = output['test_r2'].max()
  print("\nMetrics for Iteration with Best Test R2")
  
  #get row high highest max_test_r2, and get each metric within each column
  print("Overall R2", output.loc[output['test_r2'] == max_test_r2, 'all_r2'].values[0])
  print("Train R2", output.loc[output['test_r2'] == max_test_r2, 'train_r2'].values[0])
  print("Test R2", output.loc[output['test_r2'] == max_test_r2, 'test_r2'].values[0])
  print("Train NRMSE", output.loc[output['test_r2'] == max_test_r2, 'train_nrmse'].values[0])
  print("Test NRMSE", output.loc[output['test_r2'] == max_test_r2, 'test_nrmse'].values[0])
  print("Simplicity", output.loc[output['test_r2'] == max_test_r2, 'simplicity'].values[0])
  print("Equation:", output.loc[output['test_r2'] == max_test_r2, 'equation'].values[0])
  #########################################################################################

  # print out each metric and add to google sheet
  print("\n\nTrain R2:", train_r2)
  print("Test R2:", test_r2)
  print("R2:", all_r2)
  print("Train NRMSE:", train_nrmse)
  print("Test NRMSE:", test_nrmse)
  print("NRMSE:", all_nrmse)
  print("Equations:", equations)
  print("Simplicity:", simplicity)



