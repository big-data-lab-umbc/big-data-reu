import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import seaborn as sns

df = pd.read_csv('all_cases_scaled_clustered.csv')

expression = '-0.009477709373552686*x_0*x_1*x_3*(x_1 - 5.078491)/(x_3 - 4.577389) + ' \
       '0.004744200993516793*x_0*x_3**2*(x_0 + x_2)*(x_2 + 0.5811773311853556)*' \
       '(x_3 + cos(x_1) + cos(x_2 - x_3) + 0.8610352745595037) + ' \
       '(0.004745682268064854*x_2 + 0.007022129103868347)*(-cos(x_3) ' \
       '+ x_2/x_1)/cos(x_0) - 4.549306'


def predicted_vs_actual_plot(data, equation, model_name='gpg'):
    eq = parse_expr(equation)
    variables = sorted(eq.free_symbols, key=lambda s: s.name)
    f_hat = lambdify(variables, eq, modules='numpy')

    predictors = data[['Reflectivity', 'Zdr', 'Kdp', 'Rhohv']]
    rain_rate = data['gauge_precipitation_matched'].values
    rain_rate_pred = list()
    # Compute the predicted rainfall rate for the equation and append to a list
    for idx in predictors.index:
        values = predictors.loc[idx, :]
        result = f_hat(values[0], values[1], values[2], values[3])
        rain_rate_pred.append(result)

    # Calculate R2 and NRMSE scores
    score = r2_score(y_true=rain_rate, y_pred=rain_rate_pred)
    nrmse = np.sqrt(mean_squared_error(y_true=rain_rate, y_pred=rain_rate_pred)) / np.mean(rain_rate)

    # Plot predicted vs. actual results with y=x line and linear fit line
    fig, ax = plt.subplots(figsize=(10, 8))
    x_lin = y_lin = np.arange(0, np.max(rain_rate))
    plt.plot(x_lin, y_lin, linestyle='dashed', label="Predicted = Actual")
    sns.regplot(x=rain_rate, y=rain_rate_pred, ci=95, color="#225ca7",
                line_kws=dict(color="#62d59e", label="Linear Fit (95% CI)"))

    metrics = f"$R^2$: {score:.4f}\n$NRMSE$: {nrmse:.4f}"
    plt.text(0.025, 0.8, metrics, fontsize=14, transform=ax.transAxes)
    plt.title("Predicted vs. Actual Rainfall Rate", loc='left', fontsize=16)
    plt.title(f'{model_name}', loc='right', fontsize=14)
    plt.xlabel("Actual Rainfall Rate (mm/hr)", fontsize=14)
    plt.ylabel("Predicted Rainfall Rate (mm/hr)", fontsize=14)

    plt.legend(fontsize=14)
    plt.show()


predicted_vs_actual_plot(data=df, equation=expression)
