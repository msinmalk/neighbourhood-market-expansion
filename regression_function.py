import pandas as pd
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def perform_ols_regression(data, predictors, response):
    """
    Performs a multivariable OLS regression.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the dataset.
    - predictors (list of str): The names of the predictor columns.
    - response (str): The name of the response column.

    Returns:
    - model_summary (str): A summary of the regression model.
    """
    # Selecting predictors and adding a constant for the intercept
    X = data[predictors]
    #X = sm.add_constant(X)

    # Selecting the response variable
    y = data[response]

    # Creating and fitting the OLS model
    model = sm.OLS(y, X).fit()

    # Returning the summary of the model
    return model


def analyze_correlations(variateDFT):
    # Creating a correlation matrix and displaying it as a heatmap
    corr_matrix = variateDFT.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix with Heatmap')
    plt.show()

    # Checking for multicollinearity using Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = variateDFT.columns
    vif_data["VIF"] = [variance_inflation_factor(variateDFT.values, i) for i in range(len(variateDFT.columns))]
    print(vif_data)
