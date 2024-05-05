import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def base_mapping(financial_df, metrics=None):
    if metrics is None:
        metrics = {
            "assets": "asset", 
            "liabilities_and_equity": "liabilit|equit|fund", 
            "p_and_l": ""
        }

    if financial_df.index.name != 'Financial Metric':
        financial_df.set_index('Financial Metric', inplace=True)

    categorized_metrics = {}
    for key, maps in metrics.items():
        if maps:
            categorized_metrics[key] = [metric for metric in financial_df.index if re.search(maps, metric, re.IGNORECASE)]
        else:
            already_categorized = {item for sublist in categorized_metrics.values() for item in sublist}
            categorized_metrics[key] = [metric for metric in financial_df.index if metric not in already_categorized]

    base_mapping = {}
    for item in categorized_metrics['assets']:
        base_mapping[item] = 'Total assets'
    for item in categorized_metrics['liabilities_and_equity']:
        base_mapping[item] = 'Total shareholders\' funds and liabilities'
    for item in categorized_metrics['p_and_l']:
        base_mapping[item] = 'Sales' if 'Revenue' in item or 'Sales' in item else 'Sales'

    return categorized_metrics, base_mapping

def forecast_financials(df, assumptions, base_year, forecast_years):
    """
    Generates financial forecasts based on provided assumptions and historical data.
    
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
        # Calculate Sales
        if 'Sales' in assumptions:
            if idx == 0:  # First forecast year, base it on the last historical year
                last_sales = df.loc[df.index.str.contains("Sales"), base_year].values[0]
            growth_rate = assumptions['Sales']['rates'][idx]
            forecast_sales = last_sales * (1 + growth_rate)
            yoy_forecast_df.loc['Sales', year] = forecast_sales
            last_sales = forecast_sales  # Update for next year's calculation

        # Calculate Costs of Goods Sold
        if 'Costs of goods sold' in assumptions:
            cost_rate = assumptions['Costs of goods sold']['rates'][idx]
            forecast_costs = forecast_sales * cost_rate
            yoy_forecast_df.loc['Costs of goods sold', year] = forecast_costs

        # Calculate Gross Profit
        if 'Gross profit' in assumptions:
            forecast_gross_profit = forecast_sales - forecast_costs
            yoy_forecast_df.loc['Gross profit', year] = forecast_gross_profit

        # Calculate Operating and SG&A costs
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

# Example usage
# yoy_forecast_df = forecast_financials(df, assumptions, base_year, forecast_years)
# print(yoy_forecast_df)

def handle_missing_values(df, model_cols=None):
    if model_cols is None:
        model_cols = []

    if 'Costs of employees' not in df.columns:
        df = df.T

    cols_with_missing = df.columns[df.isnull().any()].tolist()
    imputation_strategies = {
        col: ('model' if df[col].dtype.kind in 'biufc' else 'most_frequent')
        for col in cols_with_missing
    }
    for col, strategy in imputation_strategies.items():
        if strategy != 'model':
            imputer = SimpleImputer(strategy=strategy)
            df[col] = imputer.fit_transform(df[[col]])
        else:
            features = df.columns.difference([col] + cols_with_missing).tolist()
            train_data = df.dropna(subset=[col] + features)
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_data[features])
            model = RandomForestRegressor(random_state=0)
            model.fit(train_features_scaled, train_data[col])
            test_features = scaler.transform(df.loc[df[col].isnull(), features])
            df.loc[df[col].isnull(), col] = model.predict(test_features)
    
    if 'Costs of employees' in df.columns:
        df = df.T

    return df


def forecast_metric(values, years, forecast_years):
    years_reshaped = years.values.reshape(-1, 1)
    values_reshaped = values.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(years_reshaped, values_reshaped)
    future_years = pd.Series(forecast_years).str[:-1].astype(int).values.reshape(-1, 1)
    predictions = model.predict(future_years).flatten()
    return pd.Series(predictions, index=forecast_years)

def linear_regression_forecast(df, forecast_years, transform_if_needed=True):
    if transform_if_needed and '2021' not in df.columns:
        df = df.T

    historical_years = df.columns[df.columns.str.isnumeric()]
    historical_regression_forecast = df.apply(
        lambda x: forecast_metric(x, historical_years, forecast_years), axis=1) / 1e3

    historical_cagr_values = ((historical_regression_forecast[forecast_years].astype(float).iloc[:, -1] /
                               historical_regression_forecast[forecast_years].astype(float).iloc[:, 0]) ** 
                              (1 / (len(forecast_years) - 1)) - 1) * 100
    historical_regression_forecast.loc[:, 'CAGR'] = historical_cagr_values

    if transform_if_needed and 'Non-current assets' not in df.columns:
        df = df.T
    return historical_regression_forecast


def apply_var_model(df, forecast_years):
    if len(df) <= 15:  # Check if there's enough data for the maximum lag
        raise ValueError("Insufficient data for the number of lags considered.")
    if not all(isinstance(year, (int, float)) for year in forecast_years):
        raise ValueError("Forecast years should be numeric.")

    model = VAR(df.dropna())  # Ensure data is complete
    results = model.fit(maxlags=15, ic='aic')
    forecasted_values = results.forecast(df.values[-results.k_ar:], steps=len(forecast_years))
    
    forecast_df = pd.DataFrame(forecasted_values, index=forecast_years, columns=df.columns)
    return forecast_df

def calculate_forecasts(historical_ratios, forecast_years, future_bases, base_mapping):
    forecasts = {}
    projected_ratios = {}
    
    for metric, ratios in historical_ratios.items():
        base = base_mapping[metric]
        if ratios.dropna().empty:
            continue

        # Using Ridge regression with polynomial features for more robust fitting
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge())
        X = np.arange(len(ratios)).reshape(-1, 1)
        y = ratios.values.flatten()

        model.fit(X, y)
        projected = model.predict(np.arange(len(ratios), len(ratios) + len(forecast_years)).reshape(-1, 1))
        
        # Use historical data to estimate variance and define a stability threshold
        historical_variance = np.var(y)
        stability_threshold = np.sqrt(historical_variance) * 2  # 2 standard deviations

        # Applying a stability check before accepting sign changes
        last_valid_value = ratios.iloc[-1]
        corrected_projected = np.where(
            np.abs(projected - last_valid_value) < stability_threshold,
            projected,
            last_valid_value + np.sign(projected - last_valid_value) * stability_threshold
        )

        forecasts[metric] = corrected_projected * future_bases[base].loc[forecast_years].values
        projected_ratios[metric] = corrected_projected

    forecast_df = pd.DataFrame(forecasts, index=forecast_years)
    projected_ratios_df = pd.DataFrame(projected_ratios, index=forecast_years) * 100
    
    return forecast_df, projected_ratios_df

def ratio_forecast_regression(df, historical_regression_forecast, base_mapping, forecast_years):
    future_bases = {base: historical_regression_forecast.loc[base] for base in set(base_mapping.values())}
    historical_ratios = {metric: df.loc[metric] / df.loc[base] for metric, base in base_mapping.items()}
    historical_ratios = pd.DataFrame.from_dict(historical_ratios).dropna(axis='columns')

    forecasts, projected_ratios = calculate_forecasts(historical_ratios, forecast_years, future_bases, base_mapping)
    # print(forecasts)
    # result_forecast = pd.DataFrame([{'Financial Metric': metric, 'Year': year, 'Value': value}
    #                                 for (metric, year), value in forecasts.items()]).pivot(
    #     index='Financial Metric', columns='Year', values='Value')
    # result_percentages = pd.DataFrame([{'Financial Metric': metric, 'Year': year, 'Value': value}
    #                                    for (metric, year), value in projected_ratios.items()]).pivot(
    #     index='Financial Metric', columns='Year', values='Value') * 100

    return forecasts, projected_ratios
