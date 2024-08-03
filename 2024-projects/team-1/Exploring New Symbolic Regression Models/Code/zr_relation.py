from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('all_cases_scaled_clustered.csv')

# Convert Reflectivity from dBz to mm^6/mm^-3 to use in the Z-R relation
df['Z_mm6m-3'] = 10 ** (df['Reflectivity'] / 10)

# Define Z and R
Z = df['Z_mm6m-3'].values
R = df['gauge_precipitation_matched'].values

relations = [(200, 1.6), (300, 1.4)]
for relation in relations:
    a, b = relation
    R_pred = (Z / a) ** (1 / b)
    R2 = np.round(r2_score(y_true=R, y_pred=R_pred), 4)
    NRMSE = np.round(np.sqrt(mean_squared_error(y_true=R, y_pred=R_pred)) / np.mean(R), 4)
    print(f'R2 score for a={a}, b={b}: {R2}')
    print(f'NRMSE score for a={a}, b={b}: {NRMSE}')


# Objective function
def objective(params, Z, R):
    a, b = params
    term = (Z / a) ** (1 / b)
    return np.sum((R - term) ** 2)


initial_guess = [300, 1.4]

# Ensure a and b are positive
bounds = [(1e-6, None), (1e-6, None)]

# Optimization
result = minimize(objective, initial_guess, args=(Z, R), method='Powell', bounds=bounds)

# Extract optimal parameters
optimal_params = result.x
a, b = optimal_params

R_pred = (Z / a) ** (1 / b)
R2 = np.round(r2_score(y_true=R, y_pred=R_pred), 4)
NRMSE = np.round(np.sqrt(mean_squared_error(y_true=R, y_pred=R_pred)) / np.mean(R), 4)

print(f'R2 score for a={a}, b={b}: {R2}')
print(f'NRMSE score for a={a}, b={b}: {NRMSE}')


