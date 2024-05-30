import re

def base_mapping(financial_df, metrics={
    "assets": "asset", 
    "liabilities_and_equity": "liabilit|equit|fund", 
    "p_and_l": ""
}):
    if financial_df.index.name != 'Financial Metric':
        financial_df.set_index('Financial Metric', inplace=True)  # Setting 'Financial Metric' as index
    categorized_metrics = {}

    for key, maps in metrics.items():
        if maps:
            categorized_metrics[key] = [metric for metric in financial_df.index if re.search(maps, metric, re.IGNORECASE)]
        else:
            already_categorized = {item for sublist in categorized_metrics.values() for item in sublist}
            categorized_metrics[key] = [metric for metric in financial_df.index if metric not in already_categorized]

    # Create a base mapping dictionary
    base_mapping = {}

    # Map assets to 'Total assets'
    for item in categorized_metrics['assets']:
        base_mapping[item] = 'Total assets'

    # Map liabilities and equity to 'Total shareholders\' funds and liabilities'
    for item in categorized_metrics['liabilities_and_equity']:
        base_mapping[item] = 'Total shareholders\' funds and liabilities'

    # Map P&L items to 'Sales' (this could be adjusted to 'Operating revenue' if more appropriate)
    for item in categorized_metrics['p_and_l']:
        if 'Revenue' in item or 'Sales' in item:
            base_mapping[item] = 'Sales'
        else:
            base_mapping[item] = 'Sales'  # or 'Total operating revenue' if defined in your dataset

    return categorized_metrics, base_mapping

# Example usage:
# categorized_metrics, base_mapping = base_mapping(financial_df)
# print(base_mapping)


import pandas as pd

def forecast_financials(df, assumptions, base_year, forecast_years):
    """
    Generates a financial forecast based on given assumptions and historical data.

    Parameters:
        df (DataFrame): Historical financial data.
        assumptions (dict): Growth assumptions for various financial metrics.
        base_year (str): The base year for the forecast (last historical year).
        forecast_years (list): List of years for which the forecast is to be made.

    Returns:
        DataFrame: Forecasted financials including CAGR for the forecast period.
    """
    # Initialize a DataFrame to hold the forecasted values
    yoy_forecast_df = pd.DataFrame(index=df.index, columns=forecast_years)

    # Iterate over each forecast year and apply the assumptions to calculate forecast values
    for idx, year in enumerate(forecast_years):
        if 'Sales' in assumptions:
            if idx == 0:  # First forecast year, base it on the last historical year
                last_sales = df.loc[df.index.str.contains("Sales"), base_year].values[0]
            growth_rate = assumptions['Sales']['rates'][idx]
            forecast_sales = last_sales * (1 + growth_rate)
            yoy_forecast_df.loc['Sales', year] = forecast_sales
            last_sales = forecast_sales  # Update for next year's calculation

        if 'Costs of goods sold' in assumptions:
            cost_rate = assumptions['Costs of goods sold']['rates'][idx]
            forecast_costs = forecast_sales * cost_rate
            yoy_forecast_df.loc['Costs of goods sold', year] = forecast_costs

        if 'Gross profit' in assumptions:
            forecast_gross_profit = forecast_sales - forecast_costs
            yoy_forecast_df.loc['Gross profit', year] = forecast_gross_profit

        if 'Operating and SG&A costs' in assumptions:
            sgna_rate = assumptions['Operating and SG&A costs']['rates'][idx]
            forecast_sgna = forecast_sales * sgna_rate
            yoy_forecast_df.loc['Operating and SG&A costs', year] = forecast_sgna

    # Normalize the forecast to thousands for display
    yoy_forecast_df = yoy_forecast_df / 1e3

    # Calculate CAGR for the forecast period
    yoy_cagr_df = (yoy_forecast_df[forecast_years].astype(float).iloc[:, -1] /
                   yoy_forecast_df[forecast_years].astype(float).iloc[:, 0]) ** (1 / (len(forecast_years) - 1)) - 1
    yoy_forecast_df.loc[:, 'CAGR'] = yoy_cagr_df * 100

    return yoy_forecast_df

# Example usage:
# yoy_forecast_df = forecast_financials(df, assumptions, base_year, forecast_years)
# print(yoy_forecast_df)

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, model_cols=None):
    """
    Handles missing values in the DataFrame using specified imputation strategies.

    Parameters:
        df (DataFrame): The DataFrame with potential missing values.
        model_cols (list): Columns where model-based imputation should be used.

    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    if model_cols is None:
        model_cols = []

    # Transpose df if specific columns are not present
    if 'Costs of employees' not in df.columns:
        df = df.T

    # Identify columns with missing values
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    # Decide on an imputation strategy for each column
    imputation_strategies = {
        col: ('model' if col in model_cols and df[col].dtype.kind in 'biufc' else 'most_freuent')
        for col in cols_with_missing
    }

    # Apply imputation or model prediction
    for col, strategy in imputation_strategies.items():
        if strategy != 'model':
            # Simple imputation
            imputer = SimpleImputer(strategy=strategy)
            df[col] = imputer.fit_transform(df[[col]])
        else:
            # Setup for predictive modeling
            # Assuming you've already identified features to use
            features = df.columns.difference([col] + cols_with_missing).tolist()
            train_data = df.dropna(subset=[col] + features)
            target = train_data[col]
            train_features = train_data[features]

            # Scaling features
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            
            # Model fitting
            model = RandomForestRegressor(random_state=0)
            model.fit(train_features_scaled, target)
            
            # Predicting missing values
            test_features = df.loc[df[col].isnull(), features]
            test_features_scaled = scaler.transform(test_features)
            predicted_values = model.predict(test_features_scaled)
            
            # Fill in the missing values
            df.loc[df[col].isnull(), col] = predicted_values

    print("Missing values handled for columns:", cols_with_missing)
    print(df.info())

    return df

# Example usage:
# model_columns = ['Costs of employees']
# updated_df = handle_missing_values(df, model_columns)
# print(updated_df)

import pandas as pd
from sklearn.linear_model import LinearRegression

def linear_regression_forecast(df, forecast_years, transform_if_needed=True):
    """
    Forecasts financial metrics using linear regression based on historical data.

    Parameters:
        df (DataFrame): The DataFrame containing historical financial data.
        forecast_years (list): List of years to forecast (e.g., ['2022F', '2023F',...]).
        transform_if_needed (bool): Whether to transpose the DataFrame if necessary.

    Returns:
        DataFrame: DataFrame containing forecasted values and calculated CAGR.
    """
    # Check if transposition is needed
    if transform_if_needed and '2021' not in df.columns:
        df = df.T

    # Define base year and forecast years
    historical_years = df.columns[df.columns.str.isnumeric()]

    # Forecast function to apply to each row
    def forecast_metric(values, years, forecast_years):
        years_reshaped = years.values.reshape(-1, 1)
        values_reshaped = values.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(years_reshaped, values_reshaped)
        future_years = pd.Series(forecast_years).str[:-1].astype(int).values.reshape(-1, 1)
        predictions = model.predict(future_years).flatten()
        return pd.Series(predictions, index=forecast_years)

    # Apply forecasting
    historical_regression_forecast = df.apply(
        lambda x: forecast_metric(x, historical_years, forecast_years), axis=1) / 1e3

    # Calculate CAGR for the forecast period
    historical_cagr_values = (historical_regression_forecast[forecast_years].astype(float).iloc[:, -1] / 
                              historical_regression_forecast[forecast_years].astype(float).iloc[:, 0]) ** (1 / (len(forecast_years) - 1)) - 1
    historical_regression_forecast.loc[:, 'CAGR'] = historical_cagr_values * 100

    return historical_regression_forecast

# Example usage:
# forecast_years = ['2022F', '2023F', '2024F', '2025F', '2026F', '2027F']
# forecasted_df = linear_regression_forecast(df, forecast_years)
# print(forecasted_df)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def ratio_forecast_regression(df, historical_regression_forecast, base_mapping, forecast_years):
    """
    Performs regression on historical ratios and projects these into the future based on regression results and historical forecasts.

    Parameters:
        df (DataFrame): DataFrame containing historical financial data.
        historical_regression_forecast (DataFrame): Forecasted values for key metrics like 'Sales' and 'Total assets'.
        base_mapping (dict): Mapping of metrics to their respective bases.
        forecast_years (list): List of years to forecast.

    Returns:
        tuple: DataFrames containing forecasted financial metrics and percentages.
    """
    future_bases = {
        'Sales': historical_regression_forecast.loc['Sales'],
        'Total assets': historical_regression_forecast.loc['Total assets']
    }
    
    # Calculate historical ratios for regression
    historical_ratios = {}
    for metric, base in base_mapping.items():
        historical_ratios[metric] = df.loc[metric] / df.loc[base]
    historical_ratios = pd.DataFrame.from_dict(historical_ratios)
    historical_ratios = historical_ratios.dropna(axis='columns')

    # Perform regression and forecast future ratios
    forecasts = {}
    projected_ratios = {}
    yeardf = pd.DataFrame(df.columns)

    for metric, ratios in historical_ratios.items():
        model = LinearRegression()
        X = yeardf.values.reshape(-1, 1)  # Ensure X is correctly shaped
        y = ratios.values.reshape(-1, 1)

        # Fit the linear model
        model.fit(X, y)
        historical_signs = np.sign(ratios).tolist()
        last_valid_value = ratios[0]  # Start with the last historical value

        # Create X and y for model fitting
        for year in forecast_years:
            projected_ratio = model.predict(np.array([[int(year[:-1])]]))[0]
            projected_sign = np.sign(projected_ratio)
            
            # Check if the projected sign is not in historical signs
            if projected_sign not in historical_signs:
                projected_ratio = last_valid_value
            else:
                last_valid_value = projected_ratio
            
            # Store the forecasts
            forecasts[(metric, year)] = projected_ratio * future_bases[base_mapping[metric]][year]
            projected_ratios[(metric, year)] = projected_ratio

    # Convert dictionaries to DataFrames and pivot
    forecasts_data = pd.DataFrame([
        {'Financial Metric': metric, 'Year': year, 'Value': value}
        for (metric, year), value in forecasts.items()
    ])
    processed_data = pd.DataFrame([
        {'Financial Metric': metric, 'Year': year, 'Value': value}
        for (metric, year), value in projected_ratios.items()
    ])

    result_forecast = forecasts_data.pivot(index='Financial Metric', columns='Year', values='Value')
    result_percentages = processed_data.pivot(index='Financial Metric', columns='Year', values='Value') * 100

    # Calculate CAGR for forecasted values
    cagr_values = (result_forecast[forecast_years].astype(float).iloc[:, -1] / 
                   result_forecast[forecast_years].astype(float).iloc[:, 0]) ** (1 / (len(forecast_years) - 1)) - 1
    result_forecast.loc[:, 'CAGR'] = cagr_values * 100

    return result_forecast, result_percentages

# Example usage:
# forecast_years = ['2022F', '2023F', '2024F', '2025F', '2026F', '2027F']
# base_mapping = {'Sales': 'Total Revenue', 'Costs of employees': 'Total Costs'}
# result_forecast, result_percentages = ratio_forecast_regression(df, historical_regression_forecast, base_mapping, forecast_years)
# print(result_forecast)
# print(result_percentages)



