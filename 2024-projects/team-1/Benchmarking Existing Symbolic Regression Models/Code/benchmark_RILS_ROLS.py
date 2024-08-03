"""
RILS-ROLS symbolic regression benchmark tests. Uses multiprocessing from 
Python to run all ten runs in parallel, and returns each output in order.
"""

import multiprocessing 
import time 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

# import package

from rils_rols.rils_rols import RILSROLSRegressor, RILSROLSBinaryClassifier



def round_floats(ex1):
    """takes SymPy equation as input and
    outputs a SymPy equation with rounded floats"""
    ex2 = ex1
    for a in sp.preorder_traversal(ex1):
        if isinstance(a, sp.Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, sp.Integer(0))
            else:
                ex2 = ex2.subs(a, round(a, 3))
    return ex2

def get_symbolic_model(expr):
    """takes string as input and
    outputs a simplified SymPy equation"""
    feature_names = ['Z', 'ZDR', 'KDP', 'RhoHV']
    local_dict = {f:sp.Symbol(f) for f in feature_names}
    sp_model = sp.parse_expr(expr, local_dict=local_dict)
    return sp_model

def get_simplicity(simplified_expr):
    """takes simplified SymPy equation as input and
    outputs a score representing the equation's simplicity"""
    # compute numumber of components
    num_components = 0
    for _ in sp.preorder_traversal(simplified_expr):
        num_components += 1

    # compute simplicity as per srbench criteria by La Cava et al. (2021)
    simplicity = -np.round(np.log(num_components)/np.log(5), 1)
    return simplicity



def model(seed: int):
  """
  Runs RILS ROLS symbolic regression model. Given a seed, it splits into train/test based on that seed
  """
  print(f"Iteration {seed}")
  #split data into train and test, with seed being the iteration of the for loop
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

  # RILSROLSRegressor inherit BaseEstimator (sklearn), so we have fit, predict and score methods, where the score method is R2
  model = RILSROLSRegressor(sample_size=1,random_state=1, complexity_penalty=0.001, max_fit_calls=100000)

  # train model and predict on train/test data and on all data
  model.fit(X_train, y_train)
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  y_pred_all = model.predict(X)

  # calculate train and test r2 and r2 on all data
  r2_train = r2_score(y_train, y_pred_train)
  r2_test = r2_score(y_test, y_pred_test)
  all_r2 = r2_score(y, y_pred_all)


  # calculate test and train rmse
  rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
  rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

  # normalized rmse with mean of y values
  train_nrmse= rmse_train / y_train.mean()
  test_nrmse= rmse_test / y_test.mean()
  all_nrmse = (np.sqrt(mean_squared_error(y,y_pred_all))/y.mean())

  # get best equation from model (depends on the package), ensure it is a string
  expr = str(model.model_string())
  # ensure that equation string has consistent variable names: Z, ZDR, KDP, RhoHV
  # double check that this is correct (it's possible that in some packages, x1 = Z, x2 = ZDR and so on)
  feature_dict = {"x0":"Z", "x1":"ZDR", "x2":"KDP", "x3":"RhoHV",
                  "x_0":"Z", "x_1":"ZDR", "x_2":"KDP", "x_3":"RhoHV",
                  "Reflectivity_sc":"Z", "Zdr_sc":"ZDR", "Kdp_sc":"KDP", "Rhohv_sc":"RhoHV"}

  for x, var in feature_dict.items():
    expr = expr.replace(x, var)

  # simplify equation and append to list
  simplified_expr = get_symbolic_model(expr)

  # append simplicity score to list
  expr_simple = get_simplicity(simplified_expr)

  # Package metrics into a dictionary to return
  return {
      'seed': seed,
      'train_r2': r2_train,
      'test_r2': r2_test,
      'all_r2': all_r2,
      'train_rmse': rmse_train,
      'test_rmse': rmse_test,
      'train_nrmse': train_nrmse,
      'test_nrmse' : all_nrmse,
      'all_nrmse': test_nrmse,
      'equation' : simplified_expr,
      'simplicity' : expr_simple
  }



if __name__ == "__main__":
  # import datasets
  df = pd.read_csv("/home/jpulido1/reu2024_team1/research/srbench_julian/srbench/experiment/datasets/all_cases_scaled.csv")

  # split predictors and target accordignly
  X = df[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv']]
  y = df['gauge_precipitation_matched']




  #create multiprocessing Pool object with 10 processors
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

  output.to_csv(f"benchmark_metrics_RILS_ROLS.csv")

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
  print("Train R2:", train_r2)
  print("Test R2:", test_r2)
  print("R2:", all_r2)
  print("Train NRMSE:", train_nrmse)
  print("Test NRMSE:", test_nrmse)
  print("NRMSE:", all_nrmse)
  print("Equations:", equations)
  print("Simplicity:", simplicity)

  