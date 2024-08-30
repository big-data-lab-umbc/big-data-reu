from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('all_cases_scaled_v3_clustered.csv')

predictors = df[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv']]
rain_rate = df['gauge_precipitation_matched'].values
train_r2s = list()
test_r2s = list()
train_nrmses = list()
test_nrmses = list()
overall_r2s = list()
overall_nrmses = list()
all_models = list()

for i in range(10):
    # Split data into 75% training and 25% testing
    X_train, X_test, y_train, y_test = train_test_split(predictors, rain_rate, test_size=0.25, random_state=i)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    all_models.append((lr.intercept_, lr.coef_))

    train_r2 = np.round(r2_score(y_true=y_train, y_pred=lr.predict(X_train)), 4)
    train_r2s.append(train_r2)
    test_r2 = np.round(r2_score(y_true=y_test, y_pred=lr.predict(X_test)), 4)
    test_r2s.append(test_r2)
    train_nrmse = np.round(np.sqrt(mean_squared_error(y_train, lr.predict(X_train))) / np.mean(y_train), 4)
    train_nrmses.append(train_nrmse)
    test_nrmse = np.round(np.sqrt(mean_squared_error(y_test, lr.predict(X_test))) / np.mean(y_test), 4)
    test_nrmses.append(test_nrmse)
    overall_r2 = np.round(r2_score(rain_rate, lr.predict(predictors)), 4)
    overall_r2s.append(overall_r2)
    overall_nrmse = np.round(np.sqrt(mean_squared_error(rain_rate, lr.predict(predictors))) / np.mean(rain_rate), 4)
    overall_nrmses.append(overall_nrmse)

results = {'Train R2s': train_r2s, 'Test R2s': test_r2s, 'Overall R2s': overall_r2s, 'Train NRMSES': train_nrmses,
           'Test NRMSES': test_nrmses, 'Overall NRMSES': overall_nrmses, 'Models': all_models}

result_df = pd.DataFrame(results)
result_df.to_csv('linear_regression_result.csv')
