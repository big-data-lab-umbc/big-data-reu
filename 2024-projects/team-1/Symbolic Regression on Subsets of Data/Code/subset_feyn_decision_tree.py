import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import feyn

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

def run_feyn_1node(df_node, node_num):
    """Run Feyn model on one rain type,
    returning a Pandas DataFrame with the metrics and equations for 10 iterations"""

    # create lists of each model accuracy measurement
    train_r2 = []
    test_r2 = []
    all_r2 = []
    train_rmse = []
    test_rmse = []
    train_nrmse = []
    test_nrmse = []
    all_nrmse = []
    equations = []
    simplicity = []

    # run 10 iterations
    for i in range(10):

        print(f"Iteration {i} for node {node_num}")

        # split data into train and test, with seed being the iteration of the for loop
        train_df, test_df = train_test_split(df_node, test_size=0.25, random_state=i)

        ql = feyn.QLattice(random_seed=1)

        # fit and train model
        models = ql.auto_run(
            data=train_df,
            output_name='gauge_precipitation_matched',
            )

        # get best model
        best_model = models[0]

        # calculate train and test r2 and  append to list
        train_pred = best_model.predict(train_df)
        test_pred = best_model.predict(test_df)
        all_pred = best_model.predict(df_node)

        train_r2.append(r2_score(train_df['gauge_precipitation_matched'], train_pred))
        test_r2.append(r2_score(test_df['gauge_precipitation_matched'], test_pred))
        all_r2.append(r2_score(df_node['gauge_precipitation_matched'], all_pred))

        # calculate test and train rmse
        rmse_train = np.sqrt(mean_squared_error(train_df['gauge_precipitation_matched'], train_pred))
        train_rmse.append(rmse_train)
        rmse_test = (np.sqrt(mean_squared_error(test_df['gauge_precipitation_matched'], test_pred)))
        test_rmse.append(rmse_test)
        rmse_all = np.sqrt(mean_squared_error(df_node['gauge_precipitation_matched'], all_pred))

        # normalize rmse with mean of y values
        train_nrmse.append(rmse_train / train_df['gauge_precipitation_matched'].mean())
        test_nrmse.append(rmse_test / test_df['gauge_precipitation_matched'].mean())
        all_nrmse.append(rmse_all / df_node['gauge_precipitation_matched'].mean())

        # get best equation as string
        expr_str = str(best_model.sympify())

        # convert equation to SymPy, append to list
        expr_sp = get_symbolic_model(expr_str)
        equations.append(expr_sp)

        # append simplicity score to list
        expr_simplicity = get_simplicity(expr_sp)
        simplicity.append(expr_simplicity)

    metrics = {
        f'n{node_num}_train_r2': train_r2,
        f'n{node_num}_test_r2': test_r2,
        f'n{node_num}_all_r2': all_r2,
        f'n{node_num}_train_rmse': train_rmse,
        f'n{node_num}_test_rmse': test_rmse,
        f'n{node_num}_train_nrmse': train_nrmse,
        f'n{node_num}_test_nrmse': test_nrmse,
        f'n{node_num}_all_nrmse': all_nrmse,
        f'n{node_num}_equation': equations,
        f'n{node_num}_simplicity': simplicity,
    }

    output = pd.DataFrame(metrics)

    return output

def run_dt_feyn(df):
    df_feyn = df[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv', 'gauge_precipitation_matched']] \
                .rename(columns={'Reflectivity': 'Z', 'Zdr': 'ZDR', 'Kdp': 'KDP', 'Rhohv': 'RhoHV'})

    # data for decision tree
    X = df_feyn[['Z', 'ZDR', 'KDP', 'RhoHV']].values
    y = df_feyn['gauge_precipitation_matched'].values

    # run decision tree regressor with parameters defined to return 3 nodes
    tree = DecisionTreeRegressor(random_state=1, max_leaf_nodes=3, min_samples_leaf=400)
    tree.fit(X, y)

    plt.figure(figsize=(12, 12))
    plot_tree(tree, feature_names=['Z', 'ZDR', 'KDP', 'RhoHV'])
    plt.savefig('decision_tree_plot.png')

    # create new column with leaf node index
    df_feyn['node'] = tree.apply(X)

    # rename the nodes so that the ids are [0, 1, 2]
    rename_nodes = {df_feyn['node'].unique()[i]: i for i in range(3)}
    df_feyn['node'] = df_feyn['node'].replace(rename_nodes)

    all_nodes = []
    
    for node in df_feyn['node'].unique():
        df_node = df_feyn[df_feyn['node'] == node].drop(columns=['node'])
        node_metrics = run_feyn_1node(df_node, node)
        all_nodes.append(node_metrics)

    metrics = pd.concat(all_nodes, axis=1)

    metrics['mean_train_r2'] = metrics[['n0_train_r2', 'n1_train_r2', 'n2_train_r2']].mean(axis=1)
    metrics['mean_test_r2'] = metrics[['n0_test_r2', 'n1_test_r2', 'n2_test_r2']].mean(axis=1)
    metrics['mean_train_rmse'] = metrics[['n0_train_rmse', 'n1_train_rmse', 'n2_train_rmse']].mean(axis=1)
    metrics['mean_test_rmse'] = metrics[['n0_test_rmse', 'n1_test_rmse', 'n2_test_rmse']].mean(axis=1)
    metrics['mean_train_nrmse'] = metrics[['n0_train_nrmse', 'n1_train_nrmse', 'n2_train_nrmse']].mean(axis=1)
    metrics['mean_test_nrmse'] = metrics[['n0_test_nrmse', 'n1_test_nrmse', 'n2_test_nrmse']].mean(axis=1)
    metrics['mean_simplicity'] = metrics[['n0_simplicity', 'n1_simplicity', 'n2_simplicity']].mean(axis=1)

    return metrics

if __name__ == "__main__":

    df = pd.read_csv("dataset.csv")

    metrics = run_dt_feyn(df)
    metrics.to_csv("subset_metrics_feyn_decision_tree.csv")