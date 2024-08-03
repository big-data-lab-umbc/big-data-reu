import numpy as np
from pygpg.sk import GPGRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import argparse

df = pd.read_csv('all_cases_scaled_v3_clustered.csv')


def train_models(data, loss_method='regular', weight=1, cluster_name=None, bounds=(1.00, 3.05, 9.15, 102.00)):
    """
    Performs 10 trials of training the gpg symbolic regression model.
    :param data: The dataset to use.
    :param loss_method: Loss method to use.
    :param weight: Weight parameter associated with the knowledge-based loss term for the loss function.
    :param cluster_name: Column name of the dataset that has the cluster labels.
    :param bounds: Bounds for binning the rainfall rates into three groups. Only applicable for binned rainfall loss.
    :return: Results for all 10 trials as a list.
    """
    predictors = data[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv']]
    rain_rate = data['gauge_precipitation_matched'].values
    train_r2s = list()
    test_r2s = list()
    train_nrmses = list()
    test_nrmses = list()
    overall_r2s = list()
    overall_nrmses = list()
    all_models = list()
    all_simplicities = list()
    # Do 10 trials for training the model
    for i in range(10):
        # Split data into 75% training and 25% testing
        X_train, X_test, y_train, y_test = train_test_split(predictors, rain_rate, test_size=0.25, random_state=i)
        gpg = GPGRegressor(e=70_000, t=-1, g=-1, d=5, finetune=True, finetune_max_evals=8_000, verbose=False,
                           random_state=9)
        # Cluster-based loss
        if loss_method == 'clusters':
            category = data[cluster_name][X_train.index].values
            # Fit the model with the cluster labels
            gpg.fit(X_train, y_train, loss_method=loss_method, cluster_labels=category, weight=weight)
        # Binned rainfall loss
        elif loss_method == 'binned-rainfall':
            # Bin the data into 3 groups
            data['bin'] = pd.cut(data['gauge_precipitation_matched'], bins=list(bounds), include_lowest=True,
                                 labels=[0, 1, 2])
            category = data['bin'][X_train.index].astype(int).values
            # Create a dictionary of the location of each observation for each group
            indices_dict = {0: np.where(category == 0)[0], 1: np.where(category == 1)[0],
                            2: np.where(category == 2)[0]}
            # Fit the model with the indices dictionary
            gpg.fit(X_train, y_train, label_indices=indices_dict, loss_method=loss_method, weight=weight, bounds=bounds)
        # Z-R loss
        elif loss_method == 'Z-R':
            # Convert Z (dB) to Z (mm^6/mm^-3) to use for Z-R relation
            data['Z_mm6m3'] = 10 ** (df['Reflectivity'].values / 10)
            Z = data['Z_mm6m3'][X_train.index].values
            # Fit the model with the Z values
            gpg.fit(X_train, y_train, Z=Z, loss_method=loss_method, weight=weight)
        # Regular loss
        elif loss_method == 'regular':
            gpg.fit(X_train, y_train, loss_method=loss_method)
        else:
            print('Loss method not found, doing regular MSE.')
            gpg.fit(X_train, y_train, loss_method='regular')

        print(f'Done with iteration {i}.')

        # Compute metrics and append to lists
        test_r2 = np.round(r2_score(y_test, gpg.predict(X_test)), 4)
        print(f'Test R2: {test_r2}')
        test_nrmse = np.round(np.sqrt(mean_squared_error(y_test, gpg.predict(X_test))) / np.mean(y_test), 4)
        test_r2s.append(test_r2)
        test_nrmses.append(test_nrmse)
        train_r2 = np.round(r2_score(y_train, gpg.predict(X_train)), 4)
        train_nrmse = np.round(np.sqrt(mean_squared_error(y_train, gpg.predict(X_train))) / np.mean(y_train), 4)
        train_r2s.append(train_r2)
        train_nrmses.append(train_nrmse)
        overall_r2 = np.round(r2_score(rain_rate, gpg.predict(predictors)), 4)
        overall_r2s.append(overall_r2)
        overall_nrmse = np.round(np.sqrt(mean_squared_error(rain_rate, gpg.predict(predictors))) / np.mean(rain_rate), 4)
        overall_nrmses.append(overall_nrmse)
        str_model = str(gpg.model)
        all_models.append(str_model)
        num_components = 0
        for _ in sp.preorder_traversal(parse_expr(str_model)):
            num_components += 1
        simp = -np.round(np.log(num_components) / np.log(5), 1)
        all_simplicities.append(simp)

    # Return all results
    return [train_r2s, test_r2s, overall_r2s, train_nrmses, test_nrmses, overall_nrmses, all_models, all_simplicities]


if __name__ == '__main__':
    print('Starting code...')

    # Define script parameters
    parser = argparse.ArgumentParser(description='Train GPG models.')
    parser.add_argument('--loss', type=str, required=True, help='The type of loss to use when training the models')
    parser.add_argument('--weight', type=float, default=1, help='The weight parameter for the loss function')
    parser.add_argument('--clustername', type=str, help='The column name for the clusters')
    args = parser.parse_args()

    tr_r2s, te_r2s, all_r2s, tr_nrmses, te_nrmses, all_nrmses, all_mlds, all_simplicities = train_models(df,
                                                                                                         loss_method=args.loss,
                                                                                                         weight=args.weight,
                                                                                                         cluster_name=args.clustername)
    # Create pandas DataFrame out of the results
    results = {'Train R2s': tr_r2s, 'Test R2s': te_r2s, 'Overall R2s': all_r2s, 'Train NRMSES': tr_nrmses,
               'Test NRMSES': te_nrmses, 'Overall NRMSES': all_nrmses, 'Models': all_mlds,
               'Simplicities': all_simplicities}
    result_df = pd.DataFrame(results)

    # Save result DataFrame to CSV file
    if args.loss == 'regular':
        result_df.to_csv(f'regular_results.csv')
        print(f'Saved results to regular_results.csv.')
    if args.loss == 'clusters':
        result_df.to_csv(f'clusters_{args.clustername}_\u03BB={args.weight}_results.csv')
        print(f'Saved results to clusters_{args.clustername}_\u03BB={args.weight}_results.csv.')
    else:
        result_df.to_csv(f'{args.loss}_\u03BB={args.weight}_results.csv')
        print(f'Saved results to {args.loss}_\u03BB={args.weight}_results.csv.')

    # Print results for the best trial
    best_iteration = np.argmax(te_r2s)
    print('\n')
    print(f'Method: {args.loss}')
    print(f'Best Trial number: {best_iteration}')
    print(f'Corresponding Train R2: {tr_r2s[best_iteration]}')
    print(f'Corresponding Test R2: {te_r2s[best_iteration]}')
    print(f'Corresponding Train NRMSE: {tr_nrmses[best_iteration]}')
    print(f'Corresponding Test NRMSE: {te_nrmses[best_iteration]}')
    print(f'Corresponding model: {all_mlds[best_iteration]}')
    print(f'Corresponding model simplicity: {all_simplicities[best_iteration]}')

    # Print all results
    print(f'All Train R2s: {tr_r2s}')
    print(f'All Test R2s: {te_r2s}')
    print(f'All Overall R2s: {all_r2s}')
    print(f'All Train NRMSEs: {tr_nrmses}')
    print(f'All Test NRMSEs: {te_nrmses}')
    print(f'All Overall NRMSES: {all_nrmses}')
    print(f'All simplicities: {all_simplicities}')

    print('Done.')
