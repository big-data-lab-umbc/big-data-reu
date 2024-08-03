import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sympy as sp
from dso import DeepSymbolicRegressor
import multiprocessing
import time

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

def read_data():
    # import dataset
    df = pd.read_csv('/umbc/xfs1/cybertrn/reu2024/team1/research/data/cases/all_cases_scaled_v3_clustered.csv')

    X = df[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv']].values
    y = df['gauge_precipitation_matched'].values

    # read parameters from config file
    config = "config_regression_dso.json"

    return X, y, config

def model(seed: int):
    """runs model once;
    given a seed, it splits data into train/test based on that seed"""
    print(f"Run {seed} started.")

    # load radar and rainfall data and config file name with parameters
    X, y, config = read_data()

    # split data into train and test, with seed being the run of the for loop
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    # create model
    model = DeepSymbolicRegressor(config)

    # train model and predict
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    # calculate r2 scores
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_all = r2_score(y, y_pred_all)

    # calculate rmse scores
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))

    # normalize rmse with mean of y values
    nrmse_train = rmse_train / y_train.mean()
    nrmse_test = rmse_test / y_test.mean()
    nrmse_all = rmse_all / y.mean()

    # get best equation from model in string format
    expr_str = str(model.program_.sympy_expr)

    # ensure that equation string has consistent variable names: Z, ZDR, KDP, RhoHV
    feature_dict = {"x1":"Z", "x2":"ZDR", "x3":"KDP", "x4":"RhoHV"}
    for x, var in feature_dict.items():
        expr_str = expr_str.replace(x, var)

    # convert equation to SymPy format
    expr_sp = get_symbolic_model(expr_str)

    # calculate simplicity score
    simplicity = get_simplicity(expr_sp)

    print(f"Run {seed} complete.")

    # convert output to a dataframe
    return {
        'seed': seed,
        'train_r2': r2_train,
        'test_r2': r2_test,
        'all_r2': r2_all,
        'train_rmse': rmse_train,
        'test_rmse': rmse_test,
        'train_nrmse': nrmse_train,
        'test_nrmse': nrmse_test,
        'all_nrmse': nrmse_all,
        'equation': expr_sp,
        'simplicity': simplicity,
    }

if __name__ == "__main__":

    # create multiprocessing Pool object with 10 processors
    pool = multiprocessing.Pool() 
    pool = multiprocessing.Pool(processes=10) 

    # seeds for train/test split
    seed_inputs = [0,1,2,3,4,5,6,7,8,9]

    # time how long it takes
    start = time.time()

    # run all processors in parallel to perform 'model' method, with seed_inputs as arguments
    results = pool.map(model, seed_inputs) 

    # end time
    end = time.time()
    print(f"Time running all 10 trials: {end - start}" )

    # create lists of results for each run
    train_r2 = [result['train_r2'] for result in results]
    test_r2 = [result['test_r2'] for result in results]
    all_r2 = [result['all_r2'] for result in results]
    train_rmse = [result['train_rmse'] for result in results]
    test_rmse = [result['test_rmse'] for result in results]
    train_nrmse = [result['train_nrmse'].item() for result in results]
    test_nrmse = [result['test_nrmse'].item() for result in results]
    all_nrmse = [result['all_nrmse'].item() for result in results]
    equations = [result['equation'] for result in results]
    simplicity = [result['simplicity'].item() for result in results]

    # convert output to dataframe
    metrics = {
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

    output.to_csv("benchmark_metrics_dso.csv")

    #########################################################################################
    # print results

    # report the run with the approx. median (the 5th highest) test r2
    med_test_r2 = np.sort(output['test_r2'].values)[4]

    print("\nMetrics for Run with Median Test R2")
    print("Train R2", output.loc[output['test_r2'] == med_test_r2, 'train_r2'].values[0])
    print("Test R2", output.loc[output['test_r2'] == med_test_r2, 'test_r2'].values[0])
    print("Train NRMSE", output.loc[output['test_r2'] == med_test_r2, 'train_nrmse'].values[0])
    print("Test NRMSE", output.loc[output['test_r2'] == med_test_r2, 'test_nrmse'].values[0])
    print("Simplicity", output.loc[output['test_r2'] == med_test_r2, 'simplicity'].values[0])
    print("Equation:", output.loc[output['test_r2'] == med_test_r2, 'equation'].values[0])

    # report the run with best test r2
    max_test_r2 = output['test_r2'].max()
    print("\nMetrics for Run with Best Test R2")

    print("Overall R2", output.loc[output['test_r2'] == max_test_r2, 'all_r2'].values[0])
    print("Train R2", output.loc[output['test_r2'] == max_test_r2, 'train_r2'].values[0])
    print("Test R2", output.loc[output['test_r2'] == max_test_r2, 'test_r2'].values[0])
    print("Train NRMSE", output.loc[output['test_r2'] == max_test_r2, 'train_nrmse'].values[0])
    print("Test NRMSE", output.loc[output['test_r2'] == max_test_r2, 'test_nrmse'].values[0])
    print("Simplicity", output.loc[output['test_r2'] == max_test_r2, 'simplicity'].values[0])
    print("Equation:", output.loc[output['test_r2'] == max_test_r2, 'equation'].values[0])

    # print out each metric as list
    print("\nAll Metrics")
    print("Train R2:", train_r2)
    print("Test R2:", test_r2)
    print("Overall R2:", all_r2)
    print("Train NRMSE:", train_nrmse)
    print("Test NRMSE:", test_nrmse)
    print("Overall NRMSE:", all_nrmse)
    print("Equations:", equations)
    print("Simplicity:", simplicity)
